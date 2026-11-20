# RESPONSE13 — nullable provider sources

The selector previously raised `TypeError` when a completed provider web-search
item contained `sources: null`. It now treats missing/null source lists as empty,
retains valid completed-message citations, and sends malformed list/item shapes
through the existing `SOURCE_PROVIDER_ERROR` boundary. Completion rules, citation
priority, exact alias deduplication, cap8, URL guards and all request/SDK settings
are unchanged. No email-absence claim is made by an empty source list.

Verified locally: 28 focused tests, 961 full unit tests, Ruff check/format. All84
original ACCEPTANCE12 requests replayed offline in their originating arms with
exact complete fingerprints. Unaffected candidate responses are equal; three
otherwise usable eight-URL selections survive. Five completed-null RED/GREEN
parser controls include one explicitly labelled installed-origin component
fixture. Replays prove compatibility and source delivery, not email recovery.

The owner separately authorized repairs to the isolated measurement harness and
one NEW untouched20/40 after readiness. A12 remains closed/failed, all40 spent;
its20 are diagnostic/excluded. The corrected product is to be frozen before new
outcomes. Final independent gain, false-contact, preservation and release/live
gates remain unchanged. New measurement/release/live have not yet run at this
checkpoint. Evidence is in the RESPONSE13 executor package.
