import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import config
import db
from integrations.opti_contract import (
    ContractError,
    build_payload,
    normalize_google_maps_url,
    normalize_instagram_handle,
    normalize_phone,
    normalize_website_domain,
    serialize_payload,
    stable_external_lead_id,
    validate_payload,
)
from models import Business


NOW = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)


def task(**overrides):
    value = {
        "external_batch_id": "batch-1",
        "niche": "dentistry",
        "city": "Kyiv",
        "count": 50,
        "created_at": "2026-08-08T09:00:00Z",
        "finished_at": "2026-08-08T10:00:00Z",
    }
    value.update(overrides)
    return value


def business(**overrides):
    value = {
        "name": "Bella Dent",
        "niche": "dentistry",
        "city": "Kyiv",
        "address": "Main 1",
        "phone": "+380 44 123 45 67",
        "instagram_url": "https://instagram.com/BellaDent/?utm_source=x",
        "google_maps_url": "https://www.google.com/maps/place/Bella/?hl=uk&query_place_id=ChIJ123",
        "site_quality": "none",
        "has_site": False,
        "collected_at": "2026-08-08T09:20:00Z",
        "rating": 4.8,
        "reviews_count": 120,
    }
    value.update(overrides)
    return Business(**value)


class OptiIdentityTests(unittest.TestCase):
    def test_prefers_existing_candidate_id(self):
        _, basis = stable_external_lead_id(business(external_candidate_id="global-7"))
        self.assertEqual("INTERNAL_CANDIDATE_ID", basis)

    def test_place_id_fallback(self):
        _, basis = stable_external_lead_id(business(external_candidate_id=""))
        self.assertEqual("GOOGLE_PLACE_ID", basis)

    def test_maps_url_normalization(self):
        one = "HTTPS://WWW.GOOGLE.COM/maps/place/A/?hl=uk&utm_source=x"
        two = "https://www.google.com/maps/place/A"
        self.assertEqual(normalize_google_maps_url(one), normalize_google_maps_url(two))

    def test_phone_normalization(self):
        self.assertEqual("+380441234567", normalize_phone("+38 (044) 123-45-67"))

    def test_instagram_normalization(self):
        self.assertEqual("belladent", normalize_instagram_handle("https://www.instagram.com/BellaDent/"))

    def test_website_domain_normalization(self):
        self.assertEqual("example.com", normalize_website_domain("HTTPS://WWW.Example.com/path"))

    def test_identity_fallback_bases(self):
        cases = (
            ({"google_maps_url": "https://maps.google.com/place/A"}, "GOOGLE_MAPS_URL"),
            ({"phone": "+380 1"}, "PHONE"),
            ({"instagram_url": "https://instagram.com/alpha"}, "INSTAGRAM"),
            ({"website": "https://www.example.com/path"}, "WEBSITE_DOMAIN"),
        )
        for values, expected in cases:
            candidate = business(
                google_place_id="", google_maps_url="", phone="", instagram_url="", website=""
            )
            for name, value in values.items():
                setattr(candidate, name, value)
            with self.subTest(expected=expected):
                self.assertEqual(expected, stable_external_lead_id(candidate)[1])

    def test_name_city_address_hash_is_deterministic(self):
        first = business(phone="", instagram_url="", website="", google_maps_url="", google_place_id="")
        second = business(
            name="  BELLA   DENT ", city="KYIV", address=" Main 1 ", phone="",
            instagram_url="", website="", google_maps_url="", google_place_id="",
        )
        self.assertEqual(stable_external_lead_id(first), stable_external_lead_id(second))
        self.assertEqual("NAME_CITY_ADDRESS_HASH", stable_external_lead_id(first)[1])


class OptiPayloadTests(unittest.TestCase):
    def test_canonical_fixture_sha_and_contract(self):
        path = Path(__file__).parents[1] / "contracts" / "opti-lead-import-v1.example.json"
        content = path.read_bytes()
        self.assertEqual(
            "7dd19ffcc99b5a79aff14898e5aa3e351792e1078438413331ef02ea94856130",
            hashlib.sha256(content).hexdigest(),
        )
        validate_payload(json.loads(content))

    def test_rank_and_nullable_fields(self):
        payload = build_payload(
            task(),
            [business(), business(name="Second", phone="+2", google_maps_url="")],
            generated_at=NOW,
        )
        self.assertEqual([1, 2], [lead["rank"] for lead in payload["leads"]])
        self.assertIsNone(payload["leads"][0]["email"])
        self.assertIsNone(payload["leads"][0]["websiteUrl"])

    def test_payload_bounds(self):
        payload = build_payload(task(), [business()], generated_at=NOW)
        too_long = copy.deepcopy(payload)
        too_long["leads"][0]["name"] = "x" * 301
        with self.assertRaises(ContractError):
            validate_payload(too_long)
        too_many = copy.deepcopy(payload)
        too_many["leads"] = [copy.deepcopy(payload["leads"][0]) for _ in range(201)]
        too_many["search"]["resultCount"] = 201
        with self.assertRaises(ContractError):
            validate_payload(too_many)
        unexpected = copy.deepcopy(payload)
        unexpected["rawScrape"] = {"html": "not allowed"}
        with self.assertRaises(ContractError):
            validate_payload(unexpected)

    def test_maximum_200_leads_is_accepted(self):
        leads = [
            business(
                name=f"Lead {index}", phone=f"+380000{index}", google_maps_url="",
                google_place_id="", instagram_url="",
            )
            for index in range(200)
        ]
        payload = build_payload(task(count=200), leads, generated_at=NOW)
        self.assertEqual(200, len(payload["leads"]))

    def test_serialization_is_deterministic_and_contains_no_token(self):
        payload = build_payload(task(), [business()], generated_at=NOW)
        first = serialize_payload(payload)
        second = serialize_payload(json.loads(first))
        self.assertEqual(first, second)
        self.assertNotIn(b"secret-token", first)
        self.assertNotIn("OPTI_IMPORT_TOKEN", payload)

    def test_only_persisted_final_businesses_are_used(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            config, "DB_PATH", str(Path(directory) / "bridge.db")
        ):
            db.init_db()
            task_id = db.create_task("dentistry", "Kyiv", 50)
            final = business(task_id=task_id)
            raw = business(name="Raw candidate", task_id=task_id, phone="+999")
            db.save_businesses([final])
            persisted = db.get_businesses_for_bridge(task_id)
            payload = build_payload(dict(db.get_task(task_id)), persisted, generated_at=NOW)
            self.assertEqual(["Bella Dent"], [lead["name"] for lead in payload["leads"]])
            self.assertNotIn(raw.name, serialize_payload(payload).decode("utf-8"))

    def test_contract_does_not_read_exports(self):
        payload = build_payload(task(), [business()], generated_at=NOW)
        self.assertEqual(1, payload["search"]["resultCount"])
        self.assertNotIn("csv", json.dumps(payload).casefold())
        self.assertNotIn("xlsx", json.dumps(payload).casefold())
