# Provider error diagnostics (draft)

The email source adapter emits one best-effort server JSON event for an original
SDK `APIError`. Successful discovery, query arguments, source selection, HTTP
error responses and retry policy are unchanged. The existing shared operation
coalescing means multiple callers receive the original outcome while only the
single SDK failure emits a diagnostic.

`EMAIL_SOURCE_PROVIDER_ERROR` version1 contains only:

- A canonical existing `operationKey` UUID and sanitized provider request ID
  (`req_` form or canonical UUID).
- HTTP status, bounded safe raw provider `code` and `type`, and a separate
  diagnostic classification.
- An allowlist of retry/rate-limit headers: bounded numeric counters,
  bounded duration reset values, and numeric/canonical HTTP-date `retry-after`.

No request/response body, exception message, query, business identity, credentials,
cookies, organization balance or payment data is logged. These fields stay in
server diagnostics and are not added to the tenant HTTP response or database.
Logging/extraction failure cannot replace the original product outcome and does
not start a second SDK call. Persistence still depends on the server log sink.

For HTTP429, exact recognized codes distinguish credit balance, organization or
project spend limits, organization usage limits, and temporary rate limiting.
Generic `insufficient_quota` remains generic. Missing, malformed or new codes
remain `UNKNOWN_429`; a broad `type` or a reset header cannot manufacture a
specific billing cause. Other HTTP statuses remain `PROVIDER_ERROR` internally.
The client still receives the existing `SOURCE_PROVIDER_QUOTA` for HTTP429.

The installed OpenAI SDK3.37.0 provides the inner error body, status, request ID
and response headers. Its `code` property can stringify malformed objects, so
diagnostics validate the raw inner body's fields without stringifying them.

Offline controls use real SDK exception objects and fake SDK transport. No
provider/site request, schema migration, model/SDK change or retry was introduced.
The historical November20 error remains UNKNOWN: this draft cannot reconstruct
fields which were not retained and is not a production/provider-account fix.

References: [error codes](https://developers.openai.com/api/docs/guides/error-codes),
[request IDs](https://developers.openai.com/api/reference/overview#debugging-requests),
[rate-limit headers](https://developers.openai.com/api/docs/guides/rate-limits).
