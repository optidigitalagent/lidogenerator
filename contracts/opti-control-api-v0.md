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

## Per-search selection mode

`POST /searches` additionally accepts optional `searchMode`:

```json
{"niche":"dentistry","city":"Kyiv","targetLeads":30,"searchMode":"ALL_COMPANIES"}
```

- `ALL_COMPANIES` is the default for new control-API searches. Website quality or
  unresolved presence alone cannot veto a candidate. The existing contact rule,
  collection, audits, scoring, bounds, deduplication and Bridge still apply.
- `WEBSITE_NEED` preserves the previous policy, including the existing
  `LEAD_WEBSITE_POLICY` and contactability settings. In the current production
  configuration (`verified_no_site_only`) this requires verified absence.
- Every search DTO contains `searchMode`; old task/business rows migrate to
  `WEBSITE_NEED`. The low-level `db.create_task` default remains legacy-compatible
  for existing Telegram callers. The Opti form defaults to `ALL_COMPANIES`.
- Mode is part of create-request idempotency: the same key with a different mode
  is a 409, including races. `SearchRuntime` runs the persisted task; there is no
  process-global selection override.
- Candidate history is scoped by mode so old website exclusions do not veto the
  expanded search. Business identities and import idempotency remain unchanged;
  a company encountered in both modes still has the same external lead ID.
- The existing `opti.lead-import.v1` contract already supports `GOOD_SITE` and
  `UNCERTAIN`. It does not need a new version or mutable outbox payloads. In the
  new mode `none`/`dead` without verified absence exports `UNCERTAIN`, never a
  confirmed `NO_SITE` fact. Historical exports and queued payloads are untouched.
