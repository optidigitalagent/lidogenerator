import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
import db
from contactability import (
    ContactChannel,
    contactability_from_business,
    lead_contact_bucket,
    normalize_email,
    normalize_phone,
    normalized_instagram_profile,
)
from models import Business
from website_pipeline import LeadDecision


class PhoneContractTests(unittest.TestCase):
    def test_valid_bounds_and_formatting(self):
        rows = {
            "+380 (50) 123-45-67": "+380501234567",
            "123-4567": "1234567",
            "+123 456 789 012 345": "+123456789012345",
        }
        for value, expected in rows.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_phone(value), expected)

    def test_invalid_phone_values(self):
        for value in (
            "123456",
            "1234567890123456",
            "0000000",
            "1111111111",
            "phone 380501234567",
            "++380501234567",
            "380+501234567",
            "+380 (50 1234567",
            "+380\t501234567",
            "",
            None,
        ):
            with self.subTest(value=value):
                self.assertIsNone(normalize_phone(value))

    def test_local_number_is_not_given_a_country_code(self):
        self.assertEqual(normalize_phone("050 123 45 67"), "0501234567")


class EmailContractTests(unittest.TestCase):
    def test_valid_email(self):
        self.assertEqual(normalize_email("Sales@Business.UA"), "Sales@business.ua")

    def test_invalid_email_values(self):
        for value in (
            "a b@business.ua",
            "a@b@business.ua",
            "business.ua",
            "a@localhost",
            "a@example.com",
            "a@example.org",
            "a@test.com",
            "a@invalid",
            "a@business.ua.",
            ".a@business.ua",
            "a..b@business.ua",
            f"{'a' * 65}@business.ua",
            f"a@{'b' * 246}.ua",
            "",
            None,
        ):
            with self.subTest(value=value):
                self.assertIsNone(normalize_email(value))


class ContactabilityTests(unittest.TestCase):
    def test_instagram_requires_a_direct_profile(self):
        self.assertEqual(
            normalized_instagram_profile("https://instagram.com/Direct.Profile/"),
            "https://www.instagram.com/Direct.Profile/",
        )
        for path in ("p/item", "reel/item", "stories/name/1", "explore"):
            with self.subTest(path=path):
                self.assertIsNone(
                    normalized_instagram_profile(f"https://instagram.com/{path}")
                )

    def test_priority_and_normalized_values(self):
        business = Business(
            instagram_url="https://instagram.com/brand/",
            phone="+380 (50) 123-45-67",
            email="Sales@Business.UA",
        )
        value = contactability_from_business(business)
        self.assertEqual(
            value.channels,
            (
                ContactChannel.INSTAGRAM,
                ContactChannel.PHONE,
                ContactChannel.EMAIL,
            ),
        )
        self.assertIs(value.preferred_channel, ContactChannel.INSTAGRAM)
        self.assertEqual(value.normalized_phone, "+380501234567")
        self.assertEqual(value.normalized_email, "Sales@business.ua")
        self.assertTrue(value.actionable)
        self.assertEqual(lead_contact_bucket(business), "multi_contact")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.normalized_phone = "1234567"

    def test_single_channels_and_none(self):
        rows = (
            (Business(phone="0501234567"), ContactChannel.PHONE, "phone_only"),
            (Business(email="hello@business.ua"), ContactChannel.EMAIL, "email_only"),
        )
        for business, channel, bucket in rows:
            with self.subTest(bucket=bucket):
                value = business.contactability
                self.assertEqual(value.channels, (channel,))
                self.assertIs(value.preferred_channel, channel)
                self.assertIs(business.preferred_contact_channel, channel)
                self.assertEqual(lead_contact_bucket(business), bucket)
        empty = Business().contactability
        self.assertEqual(empty.channels, ())
        self.assertIsNone(empty.preferred_channel)
        self.assertFalse(empty.actionable)
        self.assertEqual(lead_contact_bucket(Business()), "none")


class BusinessQualificationModeTests(unittest.TestCase):
    def test_instagram_only_preserves_historical_behavior(self):
        with patch.object(config, "LEAD_CONTACTABILITY_MODE", "instagram_only"):
            self.assertTrue(Business(instagram_url="legacy-value").is_lead)
            self.assertFalse(Business(phone="0501234567").is_lead)

    def test_multi_channel_website_matrix(self):
        contacts = (
            {"instagram_url": "https://instagram.com/brand/"},
            {"phone": "0501234567"},
            {"email": "hello@business.ua"},
        )
        with patch.object(config, "LEAD_CONTACTABILITY_MODE", "multi_channel"):
            for contact in contacts:
                with self.subTest(contact=contact):
                    self.assertTrue(Business(**contact).is_lead)
                    self.assertTrue(Business(site_quality="bad", **contact).is_lead)
                    self.assertFalse(Business(site_quality="good", **contact).is_lead)
                    self.assertFalse(
                        Business(site_quality="technical_error", **contact).is_lead
                    )
                    self.assertFalse(Business(site_quality="uncertain", **contact).is_lead)
            self.assertFalse(Business().is_lead)
            self.assertFalse(Business(phone="1111111").is_lead)

    def test_explicit_lead_decision_remains_authoritative(self):
        with patch.object(config, "LEAD_CONTACTABILITY_MODE", "multi_channel"):
            self.assertTrue(Business(lead_decision=LeadDecision.LEAD.value).is_lead)
            self.assertFalse(
                Business(
                    phone="0501234567",
                    lead_decision=LeadDecision.NOT_LEAD.value,
                ).is_lead
            )


class ContactabilityPersistenceTests(unittest.TestCase):
    def test_phone_only_lead_saves_and_reloads_without_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            config, "DB_PATH", str(Path(directory) / "contactability.db")
        ), patch.object(config, "LEAD_CONTACTABILITY_MODE", "multi_channel"):
            db.init_db()
            task_id = db.create_task("synthetic", "Test City", 1)
            business = Business(
                task_id=task_id,
                name="Synthetic Phone Lead",
                phone="050 123 45 67",
                site_quality="none",
            )
            self.assertEqual(db.save_businesses([business]), 1)
            loaded = db.get_businesses(task_id)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].phone, "050 123 45 67")
            self.assertTrue(loaded[0].is_lead)


if __name__ == "__main__":
    unittest.main()
