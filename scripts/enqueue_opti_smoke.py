"""Enqueue a deterministic three-lead local smoke batch without network access."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db
from integrations.opti_contract import build_payload
from integrations.opti_outbox import enqueue
from models import Business


def main() -> None:
    db.init_db()
    fixed = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)
    task = {
        "external_batch_id": "local-opti-bridge-smoke-v0",
        "niche": "local smoke",
        "city": "Kyiv",
        "count": 3,
        "created_at": "2026-08-08T09:00:00Z",
        "finished_at": "2026-08-08T10:00:00Z",
    }
    leads = [
        Business(
            name=f"Opti smoke business {index}",
            city="Kyiv",
            phone=f"+38000000000{index}",
            instagram_url=f"https://instagram.com/opti_smoke_{index}",
            site_quality="none",
            collected_at=f"2026-08-08T09:0{index}:00Z",
        )
        for index in range(1, 4)
    ]
    row = enqueue(build_payload(task, leads, generated_at=fixed), now=fixed)
    print(f"Enqueued {row['externalBatchId']}: {row['status']}")


if __name__ == "__main__":
    main()
