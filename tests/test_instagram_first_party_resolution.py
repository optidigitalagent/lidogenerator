"""Pure parser and contract tests for first-party Instagram evidence."""

import unittest

from instagram_first_party_resolution import (
    FirstPartyInstagramEvidenceSource as Source,
    FirstPartyInstagramStatus as Status,
    extract_instagram_profiles_from_html,
    resolution_from_extracted_profiles,
)


class FirstPartyInstagramExtractionTests(unittest.TestCase):
    def _extract(self, html: str):
        return extract_instagram_profiles_from_html(html)

    def test_direct_anchor_and_www_non_www(self) -> None:
        profiles = self._extract(
            '<a href="https://instagram.com/Direct.Brand/">one</a>'
            '<a href="http://www.instagram.com/direct.brand">two</a>'
        )
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].username, "direct.brand")
        self.assertEqual(
            profiles[0].resolved_url,
            "https://www.instagram.com/direct.brand/",
        )
        self.assertEqual(profiles[0].evidence_sources, (Source.HTML_ANCHOR,))

    def test_query_and_fragment_are_removed(self) -> None:
        profiles = self._extract(
            '<a href="https://www.instagram.com/query_brand/?hl=en#bio">Instagram</a>'
        )
        self.assertEqual(profiles[0].resolved_url, "https://www.instagram.com/query_brand/")

    def test_json_ld_same_as_string_list_and_graph(self) -> None:
        profiles = self._extract(
            '<script type="application/ld+json">'
            '{"sameAs":"https://instagram.com/graph_brand",'
            '"child":{"sameAs":["https://www.instagram.com/graph_brand/"]},'
            '"@graph":[{"sameAs":["https://instagram.com/graph_brand?x=1"]}]}'
            "</script>"
        )
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].username, "graph_brand")
        self.assertEqual(profiles[0].evidence_sources, (Source.JSON_LD_SAME_AS,))

    def test_duplicate_case_variants_collapse_and_combine_sources(self) -> None:
        profiles = self._extract(
            '<a href="https://instagram.com/Mixed.Case/">Instagram</a>'
            '<script type="application/ld+json">'
            '{"sameAs":["https://www.instagram.com/mixed.case/"]}'
            "</script>"
        )
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].username, "mixed.case")
        self.assertEqual(
            profiles[0].evidence_sources,
            (Source.HTML_ANCHOR, Source.JSON_LD_SAME_AS),
        )

    def test_reserved_routes_are_rejected(self) -> None:
        routes = (
            "p",
            "reel",
            "reels",
            "stories",
            "explore",
            "accounts",
            "direct",
            "login",
            "about",
        )
        html = "".join(
            f'<a href="https://instagram.com/{route}/">bad</a>' for route in routes
        )
        self.assertEqual(self._extract(html), ())

    def test_other_socials_and_arbitrary_text_are_ignored(self) -> None:
        html = (
            '<a href="https://facebook.com/not_instagram">Facebook</a>'
            '<p>Find us at https://instagram.com/text_only/ or @guessed_handle</p>'
            '<img alt="instagram.com/alt_text" onclick="openInstagram()">'
        )
        self.assertEqual(self._extract(html), ())

    def test_two_distinct_profiles_are_uncertain(self) -> None:
        profiles = self._extract(
            '<a href="https://instagram.com/first_brand">one</a>'
            '<a href="https://instagram.com/second_brand">two</a>'
        )
        result = resolution_from_extracted_profiles(
            profiles, pages_attempted=1, pages_succeeded=1
        )
        self.assertIs(result.status, Status.UNCERTAIN)
        self.assertIsNone(result.resolved_url)
        self.assertEqual(result.evidence_sources, (Source.HTML_ANCHOR,))

    def test_malformed_json_ld_is_safe(self) -> None:
        self.assertEqual(
            self._extract(
                '<script type="application/ld+json">{"sameAs": [broken}</script>'
            ),
            (),
        )

    def test_empty_is_not_found(self) -> None:
        result = resolution_from_extracted_profiles(
            self._extract("<html></html>"),
            pages_attempted=1,
            pages_succeeded=1,
        )
        self.assertIs(result.status, Status.NOT_FOUND)
        self.assertIsNone(result.resolved_url)
        self.assertEqual(result.evidence_sources, ())


if __name__ == "__main__":
    unittest.main()
