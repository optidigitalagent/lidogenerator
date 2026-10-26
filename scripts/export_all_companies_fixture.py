"""Print a provider-free Bridge fixture for the Opti local/mobile integration smoke."""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.opti_contract import build_payload
from models import Business

items = []
for index, quality in enumerate(("good", "bad", "none", "uncertain")):
    items.append(
        Business(
            name=f"Mobile {quality} company",
            category="Стоматология и семейная медицинская помощь",
            city="Київ",
            address="Хрещатик, 1",
            phone=f"+38050123456{index}",
            email=f"owner{index}@example.test",
            instagram_url=f"https://www.instagram.com/mobile_company_{index}/",
            google_place_id=f"mobile-smoke-{index}",
            google_maps_url=f"https://maps.google.com/?cid={index + 1}",
            website=f"https://company{index}.example/"
            if quality in {"good", "bad"}
            else "",
            has_site=quality in {"good", "bad"},
            site_quality=quality,
            website_presence_status="absent_confirmed" if quality == "none" else "",
            search_mode="ALL_COMPANIES",
        )
    )
print(
    json.dumps(
        build_payload(
            {
                "external_batch_id": "all-companies-mobile-smoke-v1",
                "niche": "dentistry",
                "city": "Kyiv",
                "count": 4,
                "created_at": "2026-10-24T08:00:00Z",
                "finished_at": "2026-10-24T08:05:00Z",
            },
            items,
            generated_at=datetime(2026, 10, 24, 8, 5, tzinfo=timezone.utc),
        ),
        ensure_ascii=False,
        indent=2,
    )
)
