"""Synthetic tests for website-resolution contracts and URL helpers."""

import ast
import dataclasses
import math
from pathlib import Path
import unittest

import website_resolution
from website_resolution import (
    CandidateEvidence,
    CandidateKind,
    CandidateSource,
    ResolutionStatus,
    WebsiteResolution,
    candidate_url_key,
    classify_candidate_url,
    deduplicate_candidate_urls,
    normalize_candidate_url,
    normalize_domain,
)


def _evidence(**overrides) -> CandidateEvidence:
    values = {
        "source": CandidateSource.MAPS,
        "candidate_url": "https://www.example.com",
        "normalized_url": "https://www.example.com/",
        "normalized_domain": "example.com",
        "final_domain": None,
        "kind": CandidateKind.UNKNOWN,
    }
    values.update(overrides)
    return CandidateEvidence(**values)


def _official_evidence(**overrides) -> CandidateEvidence:
    values = {
        "kind": CandidateKind.OFFICIAL_WEBSITE,
        "confidence": 0.9,
        "matched_signals": ("business name",),
    }
    values.update(overrides)
    return _evidence(**values)


class EnumContractTests(unittest.TestCase):
    def test_candidate_source_values_are_exact(self) -> None:
        self.assertEqual(
            tuple(item.value for item in CandidateSource),
            ("maps", "instagram_bio", "web_search"),
        )

    def test_candidate_kind_values_are_exact(self) -> None:
        self.assertEqual(
            tuple(item.value for item in CandidateKind),
            (
                "official_website",
                "social_profile",
                "link_in_bio",
                "marketplace_or_aggregator",
                "directory",
                "unknown",
            ),
        )

    def test_resolution_status_values_are_exact(self) -> None:
        self.assertEqual(
            tuple(item.value for item in ResolutionStatus),
            (
                "found_official",
                "social_only",
                "not_found",
                "uncertain",
                "resolution_error",
            ),
        )
        self.assertTrue(
            {"good", "bad", "dead"}.isdisjoint(
                item.value for item in ResolutionStatus
            )
        )


class NormalizeCandidateUrlTests(unittest.TestCase):
    def test_normalizes_scheme_host_default_ports_and_empty_path(self) -> None:
        cases = (
            ("HTTPS://EXAMPLE.COM", "https://example.com/"),
            ("http://Example.com:80", "http://example.com/"),
            ("https://Example.com:443", "https://example.com/"),
            ("https://Example.com:8443", "https://example.com:8443/"),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_candidate_url(value), expected)

    def test_removes_fragment_and_preserves_path(self) -> None:
        self.assertEqual(
            normalize_candidate_url("https://example.com/a//b#section"),
            "https://example.com/a//b",
        )

    def test_removes_tracking_and_sorts_meaningful_query_pairs(self) -> None:
        value = (
            "https://example.com/items?page=2&utm_source=x&id=9"
            "&UTM_Custom=y&fbclid=z&service=hair"
        )
        self.assertEqual(
            normalize_candidate_url(value),
            "https://example.com/items?id=9&page=2&service=hair",
        )

    def test_preserves_meaningful_blank_and_duplicate_query_values(self) -> None:
        self.assertEqual(
            normalize_candidate_url("https://example.com/?id=2&id=1&branch="),
            "https://example.com/?branch=&id=1&id=2",
        )

    def test_normalizes_idn_and_trailing_dot_hostname(self) -> None:
        self.assertEqual(
            normalize_candidate_url(
                "https://\u043f\u0440\u0438\u043c\u0435\u0440."
                "\u0443\u043a\u0440./"
            ),
            "https://xn--e1afmkfd.xn--j1amh/",
        )

    def test_trims_outer_whitespace(self) -> None:
        self.assertEqual(
            normalize_candidate_url(" \t https://example.com/path \n"),
            "https://example.com/path",
        )

    def test_preserves_percent_encoded_path(self) -> None:
        self.assertEqual(
            normalize_candidate_url("https://example.com/a%2Fb%20c"),
            "https://example.com/a%2Fb%20c",
        )


class UnsafeUrlTests(unittest.TestCase):
    def test_rejects_wrong_types_and_empty_values(self) -> None:
        for value in (None, 1, True, object()):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    normalize_candidate_url(value)  # type: ignore[arg-type]
        for value in ("", " \t\n "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(value)

    def test_rejects_relative_and_protocol_relative_urls(self) -> None:
        for value in ("/page", "//example.com", "example.com/page"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(value)

    def test_rejects_non_http_schemes(self) -> None:
        values = (
            "ftp://example.com",
            "file:///tmp/file",
            "javascript:alert(1)",
            "data:text/plain,x",
            "mailto:user@example.com",
        )
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(value)

    def test_rejects_missing_host_credentials_and_invalid_port(self) -> None:
        for value in (
            "https:///page",
            "https://user:pass@example.com",
            "https://example.com:",
            "https://example.com:notaport",
            "https://example.com:70000",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(value)

    def test_rejects_local_names(self) -> None:
        for host in ("localhost", "app.localhost", "printer.local", "local"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(f"https://{host}/")

    def test_rejects_unsafe_ipv4_literals(self) -> None:
        hosts = (
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "0.0.0.0",
            "224.0.0.1",
            "240.0.0.1",
        )
        for host in hosts:
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(f"https://{host}/")

    def test_rejects_unsafe_ipv6_literals(self) -> None:
        hosts = ("::1", "fc00::1", "fe80::1", "::", "ff02::1")
        for host in hosts:
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(f"https://[{host}]/")

    def test_rejects_invalid_idna_and_internal_whitespace(self) -> None:
        for value in ("https://bad_host.example/", "https://exa mple.com/"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_candidate_url(value)


class NormalizeDomainTests(unittest.TestCase):
    def test_normalizes_urls_bare_hosts_and_ports(self) -> None:
        cases = (
            ("https://WWW.Example.com/path", "example.com"),
            ("m.example.com", "example.com"),
            ("mobile.example.com", "example.com"),
            ("example.com:8443", "example.com"),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_domain(value), expected)

    def test_removes_repeated_presentation_prefixes(self) -> None:
        self.assertEqual(
            normalize_domain("mobile.www.m.example.com"),
            "example.com",
        )

    def test_preserves_real_subdomains(self) -> None:
        self.assertEqual(normalize_domain("shop.example.com"), "shop.example.com")

    def test_normalizes_trailing_dot_and_idn(self) -> None:
        self.assertEqual(normalize_domain("WWW.Example.com."), "example.com")
        self.assertEqual(
            normalize_domain(
                "https://\u043f\u0440\u0438\u043c\u0435\u0440."
                "\u0443\u043a\u0440/path"
            ),
            "xn--e1afmkfd.xn--j1amh",
        )

    def test_rejects_wrong_type_empty_and_non_host_values(self) -> None:
        for value in (None, 1, True):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    normalize_domain(value)  # type: ignore[arg-type]
        for value in ("", "  ", "example.com/path"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_domain(value)


class ClassificationTests(unittest.TestCase):
    def test_classifies_social_profiles(self) -> None:
        values = (
            "https://instagram.com/brand",
            "https://business.facebook.com/brand",
            "https://tiktok.com/@brand",
            "https://t.me/brand",
            "https://x.com/brand",
            "https://linkedin.com/company/brand",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertIs(
                    classify_candidate_url(value),
                    CandidateKind.SOCIAL_PROFILE,
                )

    def test_classifies_link_in_bio_services(self) -> None:
        for host in ("linktr.ee", "taplink.cc", "bio.link", "lnk.bio", "beacons.ai"):
            with self.subTest(host=host):
                self.assertIs(
                    classify_candidate_url(f"https://{host}/brand"),
                    CandidateKind.LINK_IN_BIO,
                )

    def test_classifies_marketplaces_and_aggregators(self) -> None:
        hosts = (
            "account.booksy.com",
            "booksy.com.ua",
            "treatwell.com",
            "fresha.com",
            "prom.ua",
            "olx.ua",
        )
        for host in hosts:
            with self.subTest(host=host):
                self.assertIs(
                    classify_candidate_url(f"https://{host}/brand"),
                    CandidateKind.MARKETPLACE_OR_AGGREGATOR,
                )

    def test_classifies_directories(self) -> None:
        for host in ("yellowpages.ua", "kyiv.locator.ua", "list.in.ua"):
            with self.subTest(host=host):
                self.assertIs(
                    classify_candidate_url(f"https://{host}/brand"),
                    CandidateKind.DIRECTORY,
                )

    def test_ordinary_and_carrd_domains_remain_unknown(self) -> None:
        for value in (
            "https://business.example/",
            "https://mybrand.example.com/",
            "https://brand.carrd.co/",
        ):
            with self.subTest(value=value):
                self.assertIs(classify_candidate_url(value), CandidateKind.UNKNOWN)

    def test_deceptive_suffixes_are_unknown(self) -> None:
        for value in (
            "https://instagram.com.example.org/",
            "https://fakebooksy.com/",
        ):
            with self.subTest(value=value):
                self.assertIs(classify_candidate_url(value), CandidateKind.UNKNOWN)

    def test_unsafe_url_raises_instead_of_classifying(self) -> None:
        with self.assertRaises(ValueError):
            classify_candidate_url("https://127.0.0.1/")


class DeduplicationTests(unittest.TestCase):
    def test_collapses_normalized_variants_and_preserves_first_order(self) -> None:
        values = [
            "https://EXAMPLE.com:443/?utm_source=x",
            "https://second.example/path",
            "https://example.com/",
            "https://second.example/path#fragment",
        ]
        snapshot = values.copy()
        self.assertEqual(
            deduplicate_candidate_urls(values),
            ("https://example.com/", "https://second.example/path"),
        )
        self.assertEqual(values, snapshot)

    def test_accepts_list_tuple_and_empty_sequence(self) -> None:
        expected = ("https://example.com/",)
        self.assertEqual(deduplicate_candidate_urls(["https://example.com"]), expected)
        self.assertEqual(deduplicate_candidate_urls(("https://example.com",)), expected)
        self.assertEqual(deduplicate_candidate_urls([]), ())

    def test_rejects_plain_string_and_invalid_elements(self) -> None:
        with self.assertRaises(TypeError):
            deduplicate_candidate_urls("https://example.com")
        with self.assertRaises(TypeError):
            deduplicate_candidate_urls(["https://example.com", None])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            deduplicate_candidate_urls(["https://example.com", "/relative"])

    def test_candidate_url_key_is_normalized_url(self) -> None:
        self.assertEqual(
            candidate_url_key("HTTPS://EXAMPLE.COM:443/?utm_source=x"),
            "https://example.com/",
        )


class CandidateEvidenceTests(unittest.TestCase):
    def test_valid_construction_normalizes_strings_and_confidence(self) -> None:
        evidence = _evidence(
            candidate_url="  https://www.example.com  ",
            final_domain="example.com",
            matched_signals=(" Business   name ", " City "),
            rejected_reason=" Not   enough evidence ",
            confidence=1,
            technical_error=" Provider   timeout ",
        )
        self.assertEqual(evidence.candidate_url, "https://www.example.com")
        self.assertEqual(evidence.matched_signals, ("Business name", "City"))
        self.assertEqual(evidence.rejected_reason, "Not enough evidence")
        self.assertEqual(evidence.technical_error, "Provider timeout")
        self.assertIs(type(evidence.confidence), float)

    def test_rejects_duplicate_or_mutable_signals(self) -> None:
        with self.assertRaises(ValueError):
            _evidence(matched_signals=("Business Name", " business   name "))
        with self.assertRaises(TypeError):
            _evidence(matched_signals=["name"])  # type: ignore[arg-type]

    def test_rejects_bad_source_kind_and_signal_elements(self) -> None:
        with self.assertRaises(TypeError):
            _evidence(source="maps")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            _evidence(kind="unknown")  # type: ignore[arg-type]
        for signal in (" ", None, 1):
            with self.subTest(signal=signal):
                with self.assertRaises((TypeError, ValueError)):
                    _evidence(matched_signals=(signal,))  # type: ignore[arg-type]

    def test_validates_confidence_boundaries_bool_and_non_finite_values(self) -> None:
        for value in (0, 0.0, 1, 1.0):
            with self.subTest(value=value):
                self.assertEqual(_evidence(confidence=value).confidence, float(value))
        for value in (True, False):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    _evidence(confidence=value)
        for value in (-0.1, 1.1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _evidence(confidence=value)

    def test_rejects_invalid_normalized_url_and_domains(self) -> None:
        with self.assertRaises(ValueError):
            _evidence(normalized_url="HTTPS://www.example.com")
        with self.assertRaises(ValueError):
            _evidence(normalized_domain="www.example.com")
        with self.assertRaises(ValueError):
            _evidence(final_domain="WWW.Example.com")

    def test_validates_optional_string_fields(self) -> None:
        for field in ("rejected_reason", "technical_error"):
            with self.subTest(field=field, value="empty"):
                with self.assertRaises(ValueError):
                    _evidence(**{field: " "})
            with self.subTest(field=field, value="wrong type"):
                with self.assertRaises(TypeError):
                    _evidence(**{field: 1})

    def test_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _evidence().kind = CandidateKind.DIRECTORY  # type: ignore[misc]


class WebsiteResolutionTests(unittest.TestCase):
    def test_valid_found_official(self) -> None:
        resolution = WebsiteResolution(
            status=ResolutionStatus.FOUND_OFFICIAL,
            resolved_url="https://www.example.com/",
            source=CandidateSource.MAPS,
            confidence=0.9,
            evidence=(_official_evidence(),),
        )
        self.assertIs(resolution.status, ResolutionStatus.FOUND_OFFICIAL)

    def test_found_official_requires_url_source_evidence_and_confidence(self) -> None:
        base = {
            "status": ResolutionStatus.FOUND_OFFICIAL,
            "resolved_url": "https://www.example.com/",
            "source": CandidateSource.MAPS,
            "confidence": 0.9,
            "evidence": (_official_evidence(),),
        }
        invalid_overrides = (
            {"resolved_url": None},
            {"source": None},
            {"confidence": 0.0},
            {"evidence": ()},
            {"error": "failure"},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    WebsiteResolution(**{**base, **overrides})

    def test_found_official_rejects_evidence_url_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            WebsiteResolution(
                ResolutionStatus.FOUND_OFFICIAL,
                "https://other.example/",
                CandidateSource.MAPS,
                0.9,
                (_official_evidence(),),
            )

    def test_valid_social_only(self) -> None:
        social = _evidence(
            candidate_url="https://instagram.com/brand",
            normalized_url="https://instagram.com/brand",
            normalized_domain="instagram.com",
            kind=CandidateKind.SOCIAL_PROFILE,
        )
        resolution = WebsiteResolution(
            ResolutionStatus.SOCIAL_ONLY,
            None,
            None,
            0.5,
            (social,),
        )
        self.assertIs(resolution.status, ResolutionStatus.SOCIAL_ONLY)

    def test_social_only_rejects_official_or_missing_social_evidence(self) -> None:
        for evidence in ((_official_evidence(),), (_evidence(),), ()):
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValueError):
                    WebsiteResolution(
                        ResolutionStatus.SOCIAL_ONLY,
                        None,
                        None,
                        0.5,
                        evidence,
                    )

    def test_valid_not_found_and_rejected_non_official_evidence(self) -> None:
        self.assertIs(
            WebsiteResolution(
                ResolutionStatus.NOT_FOUND,
                None,
                None,
                0,
                (_evidence(rejected_reason="not official"),),
            ).status,
            ResolutionStatus.NOT_FOUND,
        )
        self.assertEqual(
            WebsiteResolution(
                ResolutionStatus.NOT_FOUND,
                None,
                None,
                0,
                (),
            ).evidence,
            (),
        )

    def test_not_found_rejects_confidence_and_official_evidence(self) -> None:
        with self.assertRaises(ValueError):
            WebsiteResolution(ResolutionStatus.NOT_FOUND, None, None, 0.1, ())
        with self.assertRaises(ValueError):
            WebsiteResolution(
                ResolutionStatus.NOT_FOUND,
                None,
                None,
                0,
                (_official_evidence(),),
            )

    def test_valid_uncertain_does_not_force_defaults(self) -> None:
        resolution = WebsiteResolution(
            ResolutionStatus.UNCERTAIN,
            "https://example.com/",
            CandidateSource.WEB_SEARCH,
            0.4,
            (_evidence(),),
            error="identity inconclusive",
        )
        self.assertEqual(resolution.error, "identity inconclusive")

    def test_valid_resolution_error(self) -> None:
        resolution = WebsiteResolution(
            ResolutionStatus.RESOLUTION_ERROR,
            None,
            None,
            0,
            (),
            error="provider unavailable",
        )
        self.assertEqual(resolution.error, "provider unavailable")

    def test_resolution_error_requires_error_and_zero_unresolved_state(self) -> None:
        invalid_arguments = (
            {"error": None},
            {"confidence": 0.2, "error": "failure"},
            {"resolved_url": "https://example.com/", "error": "failure"},
            {"source": CandidateSource.MAPS, "error": "failure"},
        )
        base = {
            "status": ResolutionStatus.RESOLUTION_ERROR,
            "resolved_url": None,
            "source": None,
            "confidence": 0,
            "evidence": (),
        }
        for overrides in invalid_arguments:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    WebsiteResolution(**{**base, **overrides})

    def test_rejects_bad_evidence_container_and_item(self) -> None:
        with self.assertRaises(TypeError):
            WebsiteResolution(ResolutionStatus.UNCERTAIN, None, None, 0, [])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            WebsiteResolution(ResolutionStatus.UNCERTAIN, None, None, 0, ("item",))  # type: ignore[arg-type]

    def test_validates_confidence_and_is_frozen(self) -> None:
        for value, error in (
            (True, TypeError),
            (math.nan, ValueError),
            (math.inf, ValueError),
            (-0.1, ValueError),
            (1.1, ValueError),
        ):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    WebsiteResolution(
                        ResolutionStatus.UNCERTAIN,
                        None,
                        None,
                        value,
                        (),
                    )
        resolution = WebsiteResolution(
            ResolutionStatus.UNCERTAIN,
            None,
            None,
            0,
            (),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            resolution.status = ResolutionStatus.NOT_FOUND  # type: ignore[misc]


class BoundaryTests(unittest.TestCase):
    def test_classifier_never_calls_ordinary_domain_official(self) -> None:
        self.assertIs(
            classify_candidate_url("https://business.example"),
            CandidateKind.UNKNOWN,
        )

    def test_module_imports_no_network_library(self) -> None:
        source_path = Path(website_resolution.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])

        self.assertTrue(
            {"httpx", "playwright", "requests", "aiohttp"}.isdisjoint(
                imported_roots
            )
        )


if __name__ == "__main__":
    unittest.main()
