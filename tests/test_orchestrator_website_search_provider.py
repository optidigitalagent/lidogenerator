"""Offline orchestrator integration tests for task-scoped website search."""

import unittest
from unittest.mock import AsyncMock, patch

import orchestrator
from agents import website_resolver
from models import Business
from tests import test_orchestrator_policy as policy_tests
from website_pipeline import (
    LeadDecision,
    WebsiteAuditResult,
    WebsiteAuditStatus,
    qualify_lead,
)
from website_resolution import ResolutionStatus
from website_search_runtime import (
    BudgetedSearchProvider,
    UnavailableSearchProvider,
)


class EmptyProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, request):
        self.calls += 1
        return ()


class OrchestratorWebsiteSearchProviderTests(unittest.IsolatedAsyncioTestCase):
    def _helper(self) -> policy_tests.OrchestratorPolicyTests:
        return policy_tests.OrchestratorPolicyTests(methodName="runTest")

    async def test_off_mode_does_not_build_provider(self) -> None:
        with patch.object(orchestrator, "build_configured_search_provider") as factory:
            run = await self._helper()._run(
                target=1,
                max_candidates=2,
                batches=[[policy_tests._lead("one")]],
                resolver_mode="off",
            )
        factory.assert_not_called()
        self.assertEqual(run["resolver_calls"], [])

    async def test_shadow_config_none_builds_once_and_passes_none(self) -> None:
        with patch.object(
            orchestrator,
            "build_configured_search_provider",
            return_value=None,
        ) as factory:
            run = await self._helper()._run(
                target=1,
                max_candidates=2,
                batches=[[policy_tests._lead("one")]],
                resolver_mode="shadow",
            )
        factory.assert_called_once_with()
        self.assertIsNone(run["resolver_calls"][0][1])

    async def test_configured_provider_built_once_and_reused_across_batches(self) -> None:
        provider = EmptyProvider()
        with patch.object(
            orchestrator,
            "build_configured_search_provider",
            return_value=provider,
        ) as factory:
            run = await self._helper()._run(
                target=2,
                max_candidates=4,
                batches=[[policy_tests._lead("one")], [policy_tests._lead("two")]],
                resolver_mode="shadow",
            )
        factory.assert_called_once_with()
        self.assertEqual(len(run["resolver_calls"]), 2)
        self.assertTrue(all(call[1] is provider for call in run["resolver_calls"]))
        self.assertEqual(len(run["downstream"]["save"][0]), 2)

    async def test_injected_provider_bypasses_factory(self) -> None:
        provider = EmptyProvider()
        with patch.object(orchestrator, "build_configured_search_provider") as factory:
            run = await self._helper()._run(
                target=1,
                max_candidates=2,
                batches=[[policy_tests._lead("one")]],
                website_search_provider=provider,
            )
        factory.assert_not_called()
        self.assertIs(run["resolver_calls"][0][1], provider)

    async def test_budget_usage_is_in_progress_without_identity_data(self) -> None:
        provider = BudgetedSearchProvider(EmptyProvider(), 2)
        update = AsyncMock()
        with (
            patch.object(
                orchestrator,
                "build_configured_search_provider",
                return_value=provider,
            ),
            patch.object(orchestrator._Progress, "update", new=update),
        ):
            await self._helper()._run(
                target=1,
                max_candidates=2,
                batches=[[policy_tests._lead("private-business")]],
            )
        lines = [call.args[1] for call in update.await_args_list]
        self.assertIn("Пошуків офіційного сайту: 0/2", lines)
        budget_lines = [line for line in lines if "Пошуків офіційного сайту" in line]
        self.assertTrue(all("private-business" not in line for line in budget_lines))

    async def test_unavailable_search_stays_uncertain_and_is_not_strict_lead(self) -> None:
        business = Business(
            name="Business",
            city="Kyiv",
            instagram_url="https://instagram.com/business",
        )
        resolution = await website_resolver.resolve_business_website(
            business,
            UnavailableSearchProvider("Brave Search API key is not configured"),
        )
        self.assertIs(resolution.status, ResolutionStatus.UNCERTAIN)
        self.assertIsNot(resolution.status, ResolutionStatus.NOT_FOUND)
        qualification = qualify_lead(
            has_instagram=True,
            resolution=resolution,
            audit=WebsiteAuditResult(
                WebsiteAuditStatus.NOT_RUN,
                None,
                None,
                None,
            ),
        )
        self.assertIs(qualification.decision, LeadDecision.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
