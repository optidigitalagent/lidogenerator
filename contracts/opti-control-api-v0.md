# Opti Control API v0

The control API runs in the Telegram process and uses the same `SearchRuntime`
and global concurrency semaphore. SQLite remains the only source of truth for
search state and the existing Opti Bridge remains the only completed-lead import
path.

Public route: `GET /health`.

All `/internal/opti/v1` routes require
`Authorization: Bearer <LEAD_GENERATOR_CONTROL_TOKEN>`:

- `POST /searches` with `Idempotency-Key` and strict JSON
  `{"niche":"dentistry","city":"Kyiv","targetLeads":30}`
- `GET /searches?limit=50`
- `GET /searches/{searchId}`
- `POST /searches/{searchId}/stop`
- `POST /searches/{searchId}/retry-sync`

`searchId` is the existing stable `external_batch_id`, never the local SQLite
integer ID. Responses use the stable DTO described in `control_api.search_dto`.
They never include idempotency keys, export paths, outbox payloads or hashes, or
credentials. Errors have the shape
`{"error":{"code":"VALIDATION_ERROR","message":"..."}}`.

Example:

```http
POST /internal/opti/v1/searches HTTP/1.1
Authorization: Bearer <server-only-token>
Idempotency-Key: 0198-search-attempt
Content-Type: application/json

{"niche":"dentistry","city":"Kyiv","targetLeads":30}
```

The same key and normalized payload returns the same search with status 200. A
different payload with that key returns `409 IDEMPOTENCY_KEY_REUSED`.
