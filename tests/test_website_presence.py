import unittest

from website_presence import (
    WebsitePresenceResult,
    WebsitePresenceSource,
    WebsitePresenceStatus,
    classify_website_presence_url,
)


class WebsitePresenceTests(unittest.TestCase):
    def test_hosted_builders_are_present(self) -> None:
        for url in (
            "https://sites.google.com/view/example",
            "https://brand.wixsite.com/shop",
            "https://business.site/example",
            "https://brand.wordpress.com/",
            "https://brand.carrd.co/",
        ):
            with self.subTest(url=url):
                self.assertTrue(classify_website_presence_url(url))

    def test_custom_site_is_present(self) -> None:
        self.assertTrue(classify_website_presence_url("https://example-clinic.ua/"))

    def test_social_booking_directory_and_maps_are_not_sites(self) -> None:
        for url in (
            "https://instagram.com/example/",
            "https://linktr.ee/example",
            "https://booksy.com/uk-ua/example",
            "https://yellowpages.ua/example",
            "https://maps.google.com/?q=example",
        ):
            with self.subTest(url=url):
                self.assertFalse(classify_website_presence_url(url))

    def test_result_invariants_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            WebsitePresenceResult(WebsitePresenceStatus.PRESENT)
        with self.assertRaises(ValueError):
            WebsitePresenceResult(
                WebsitePresenceStatus.ABSENT_CONFIRMED,
                WebsitePresenceSource.WEB_SEARCH,
                error_category="timeout",
            )
        result = WebsitePresenceResult(
            WebsitePresenceStatus.PRESENT,
            WebsitePresenceSource.MAPS,
            "https://sites.google.com/view/example",
            ("maps_url_present",),
        )
        self.assertEqual(result.status, WebsitePresenceStatus.PRESENT)


if __name__ == "__main__":
    unittest.main()
