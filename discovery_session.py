"""Task-scoped Google Maps discovery identity and counters."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import urllib.parse


_GOOGLE_HOST_PATTERN = re.compile(
    r"(?:^|\.)google\.(?:com|[a-z]{2}|com\.[a-z]{2})\Z",
    re.IGNORECASE,
)
_TRACKING_QUERY_KEYS = frozenset({"authuser", "entry", "g_ep", "hl"})


def _is_google_maps_place_url(parsed: urllib.parse.SplitResult) -> bool:
    host = (parsed.hostname or "").casefold().rstrip(".")
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and _GOOGLE_HOST_PATTERN.search(host) is not None
        and parsed.path.casefold().startswith("/maps/place/")
    )


def normalize_maps_place_url(value: str) -> str:
    """Return a conservative deterministic key for a Maps place link.

    Only harmless presentation/tracking parameters are removed from verified
    Google Maps place URLs. Unknown, non-Google, or malformed values fall back
    to their exact stripped text so unrelated links never collapse together.
    """

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, not {type(value).__name__}")

    raw = value.strip()
    if not raw:
        raise ValueError("value must not be empty")

    try:
        parsed = urllib.parse.urlsplit(raw)
        if not _is_google_maps_place_url(parsed):
            return f"raw:{raw}"

        host = (parsed.hostname or "").casefold().rstrip(".")
        if host.startswith("www."):
            host = host[4:]
        port = parsed.port
        if port is not None:
            host = f"{host}:{port}"

        path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/")
        retained_query = sorted(
            (key, item)
            for key, item in urllib.parse.parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            if key.casefold() not in _TRACKING_QUERY_KEYS
            and not key.casefold().startswith("utm_")
        )
        query = urllib.parse.urlencode(retained_query, doseq=True)
        return urllib.parse.urlunsplit(("https", host, path, query, ""))
    except (TypeError, ValueError, UnicodeError):
        return f"raw:{raw}"


@dataclass
class MapsDiscoverySession:
    """Mutable per-task Maps-link registry with identity-free counters."""

    dedupe_enabled: bool = True
    maps_links_discovered: int = 0
    maps_links_skipped_task_duplicate: int = 0
    maps_cards_actually_opened: int = 0
    _seen_link_keys: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.dedupe_enabled) is not bool:
            raise TypeError("dedupe_enabled must be a boolean")

    def claim_link(self, href: str) -> bool:
        """Register one stream-discovered href and return whether to open it."""

        key = normalize_maps_place_url(href)
        self.maps_links_discovered += 1
        if self.dedupe_enabled and key in self._seen_link_keys:
            self.maps_links_skipped_task_duplicate += 1
            return False
        if self.dedupe_enabled:
            self._seen_link_keys.add(key)
        return True

    def record_card_opened(self) -> None:
        """Record one actual card navigation after its link was claimed."""

        self.maps_cards_actually_opened += 1

    @property
    def task_duplicate_link_rate(self) -> float:
        if self.maps_links_discovered == 0:
            return 0.0
        return (
            self.maps_links_skipped_task_duplicate
            / self.maps_links_discovered
        )
