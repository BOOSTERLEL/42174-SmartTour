"""
Xiaohongshu CLI integration for popularity signals.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session

from smarttour.config import get_settings
from smarttour.models import XhsDiscovery, XhsPopularityRecord

logger = logging.getLogger(__name__)

_DISCOVERY_KEYWORDS: dict[str, list[str]] = {
    "attraction": [
        "{dest} 景点推荐",
        "{dest} 必去景点",
        "{dest} 打卡",
        "{dest} attractions",
        "{dest} must visit",
    ],
    "hotel": [
        "{dest} 酒店推荐",
        "{dest} 住宿攻略",
        "{dest} hotels",
        "{dest} accommodation",
    ],
    "restaurant": [
        "{dest} 美食推荐",
        "{dest} 餐厅推荐",
        "{dest} food guide",
        "{dest} best restaurants",
    ],
    "general": [
        "{dest} 旅游攻略",
        "{dest} travel guide",
    ],
}


class XiaohongshuClient:
    """
    Xiaohongshu search client for popularity signals.
    """

    def __init__(self) -> None:
        """
        Initialize the Xiaohongshu CLI wrapper.
        """

        self.settings = get_settings()
        self.enabled = self.settings.xiaohongshu_enabled
        self._vendor_dir = (
            Path(__file__).resolve().parents[2] / "vendor" / "xiaohongshu-skills"
        )
        self._cli_path = self._vendor_dir / "scripts" / "cli.py"

    def check_login_status(self) -> bool:
        """
        Return the current Xiaohongshu login state.

        Returns:
            ``True`` when the Xiaohongshu CLI reports an authenticated session.
        """

        if not self.enabled:
            return False
        return self._check_login()

    def search_popularity(
        self,
        session: Session,
        destination: str,
    ) -> dict[str, float]:
        """
        Search Xiaohongshu for destination popularity signals.

        Args:
            session: Active database session.
            destination: Destination name.

        Returns:
            Mapping of lowercase note titles to popularity scores.
        """

        discovery = self.discover_destination(session, destination)
        return discovery.popularity

    def discover_destination(
        self,
        session: Session,
        destination: str,
    ) -> XhsDiscovery:
        """
        Discover categorized Xiaohongshu hints for a destination.

        Args:
            session: Active database session.
            destination: Destination name.

        Returns:
            Discovery payload with categorized title hints and flat popularity.
        """

        logger.info("XHS discovery started for '%s'", destination)
        if not self.enabled:
            return XhsDiscovery(destination=destination)
        if not self._check_login():
            return XhsDiscovery(destination=destination)

        cached = self._load_cached_discovery(session, destination)
        if cached is not None:
            logger.info(
                "XHS discovery complete: %d categories, %d total hints",
                len(cached.hints_by_category),
                self._count_hint_titles(cached.hints_by_category),
            )
            return cached

        popularity: dict[str, float] = {}
        hints_by_category: dict[str, dict[str, float]] = {
            category: {} for category in _DISCOVERY_KEYWORDS
        }
        for category, query_templates in _DISCOVERY_KEYWORDS.items():
            for query_template in query_templates:
                feeds = self._search_feeds(
                    keyword=query_template.format(dest=destination),
                    sort_by="最多点赞",
                    note_type="图文",
                )
                logger.debug(
                    "XHS search keyword='%s' returned %d feeds",
                    query_template.format(dest=destination),
                    len(feeds),
                )
                for feed in feeds:
                    title = feed.get("displayTitle", "")
                    if not isinstance(title, str) or not title.strip():
                        continue
                    normalized_title = title.strip().lower()
                    score = self._compute_engagement_score(feed.get("interactInfo", {}))
                    popularity[normalized_title] = (
                        popularity.get(normalized_title, 0.0) + score
                    )
                    hints_by_category[category][normalized_title] = (
                        hints_by_category[category].get(normalized_title, 0.0) + score
                    )

        discovery = XhsDiscovery(
            destination=destination,
            hints_by_category={
                category: sorted(
                    entries.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                for category, entries in hints_by_category.items()
                if entries
            },
            popularity=popularity,
        )
        self._cache_discovery(session, discovery)
        logger.info(
            "XHS discovery complete: %d categories, %d total hints",
            len(discovery.hints_by_category),
            self._count_hint_titles(discovery.hints_by_category),
        )
        return discovery

    def search_place_notes(
        self,
        place_name: str,
        limit: int = 5,
    ) -> list[str]:
        """
        Search Xiaohongshu for notes about a specific place.

        Args:
            place_name: Name of the place to search for.
            limit: Maximum number of note titles to return.

        Returns:
            A list of relevant note titles for this place.
        """

        if not self.enabled or not self._check_login():
            return []

        feeds = self._search_feeds(
            keyword=place_name,
            sort_by="最多点赞",
            note_type="图文",
        )
        titles: list[str] = []
        seen: set[str] = set()
        for feed in feeds:
            title = feed.get("displayTitle", "")
            if not isinstance(title, str) or not title.strip():
                continue
            normalized = title.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            titles.append(title.strip())
            if len(titles) >= limit:
                break
        logger.debug(
            "XHS place notes search for '%s': %d titles",
            place_name,
            len(titles),
        )
        return titles

    def _check_login(self) -> bool:
        """
        Check Xiaohongshu login status via the vendor CLI.

        Returns:
            ``True`` when the CLI reports a logged-in state.
        """

        if not self._cli_path.exists():
            logger.warning("XHS CLI not found at %s", self._cli_path)
            return False

        command = self._build_cli_command("check-login")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except FileNotFoundError:
            logger.warning("uv is not available, skipping XHS login check")
            return False
        except Exception:
            logger.exception("XHS check-login error")
            return False

        if result.returncode == 0:
            return True
        if result.returncode == 1:
            logger.warning("XHS not logged in, skipping popularity search")
            return False
        logger.warning("XHS check-login failed (exit %d)", result.returncode)
        return False

    def _search_feeds(
        self,
        keyword: str,
        sort_by: str,
        note_type: str,
    ) -> list[dict]:
        """
        Call the vendor CLI `search-feeds` command.

        Args:
            keyword: Search keyword.
            sort_by: Feed sorting mode.
            note_type: Requested note type.

        Returns:
            Parsed feed dictionaries.
        """

        if not self._cli_path.exists():
            return []

        command = self._build_cli_command(
            "search-feeds",
            "--keyword",
            keyword,
            "--sort-by",
            sort_by,
            "--note-type",
            note_type,
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("XHS search timed out for keyword '%s'", keyword)
            return []
        except FileNotFoundError:
            logger.warning("uv is not available, skipping XHS search")
            return []
        except Exception:
            logger.exception("XHS search error for keyword '%s'", keyword)
            return []

        if result.returncode != 0:
            logger.warning(
                "XHS search failed (exit %d): %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
            return []

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("XHS search returned invalid JSON for keyword '%s'", keyword)
            return []

        feeds = payload.get("feeds", [])
        if not isinstance(feeds, list):
            return []
        return [item for item in feeds if isinstance(item, dict)]

    def _build_cli_command(self, *args: str) -> list[str]:
        """
        Build a vendor CLI command line.

        Args:
            *args: Command-specific arguments appended after the global options.

        Returns:
            A subprocess command list.
        """

        return [
            "uv",
            "run",
            "--directory",
            str(self._vendor_dir),
            "python",
            "scripts/cli.py",
            "--port",
            str(self.settings.xiaohongshu_chrome_port),
            *args,
        ]

    def _load_cached_discovery(
        self,
        session: Session,
        destination: str,
    ) -> XhsDiscovery | None:
        """
        Load cached discovery data when it is still fresh.

        Args:
            session: Active database session.
            destination: Destination name.

        Returns:
            Cached discovery payload or ``None`` when expired or missing.
        """

        record = session.get(
            XhsPopularityRecord, XhsPopularityRecord.cache_id(destination)
        )
        if record is None or not self._is_cache_fresh(record.fetched_at):
            return None
        return record.to_discovery()

    def _cache_discovery(
        self,
        session: Session,
        discovery: XhsDiscovery,
    ) -> None:
        """
        Persist a discovery payload into the local cache.

        Args:
            session: Active database session.
            discovery: Discovery payload to cache.
        """

        session.merge(XhsPopularityRecord.from_discovery(discovery))
        session.commit()

    def _is_cache_fresh(self, fetched_at: datetime | None) -> bool:
        """
        Return whether a cached popularity snapshot is still valid.

        Args:
            fetched_at: Cache timestamp.

        Returns:
            ``True`` when the snapshot is within the configured TTL.
        """

        if fetched_at is None:
            return False
        normalized = fetched_at
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=UTC)
        cutoff = datetime.now(UTC) - timedelta(
            hours=self.settings.xiaohongshu_cache_ttl_hours
        )
        return normalized >= cutoff

    def _compute_engagement_score(self, interact_info: object) -> float:
        """
        Compute a weighted engagement score from Xiaohongshu interact metrics.

        Args:
            interact_info: Raw `interactInfo` payload.

        Returns:
            Weighted engagement score.
        """

        if not isinstance(interact_info, dict):
            return 0.0
        likes = self._parse_count(interact_info.get("likedCount", ""))
        collected = self._parse_count(interact_info.get("collectedCount", ""))
        comments = self._parse_count(interact_info.get("commentCount", ""))
        return float(likes + 2 * collected + comments)

    def _parse_count(self, value: object) -> int:
        """
        Parse Xiaohongshu interaction counts including `万` suffixes.

        Args:
            value: Raw metric value.

        Returns:
            Normalized integer count.
        """

        if isinstance(value, (int, float)):
            return int(value)
        if not isinstance(value, str):
            return 0

        cleaned = value.strip().replace(",", "").replace("+", "")
        if not cleaned:
            return 0

        multiplier = 1.0
        if cleaned.endswith("万") or cleaned.endswith(("w", "W")):
            multiplier = 10_000.0
            cleaned = cleaned[:-1]

        try:
            return int(float(cleaned) * multiplier)
        except ValueError:
            return 0

    def _count_hint_titles(
        self,
        hints_by_category: dict[str, list[tuple[str, float]]],
    ) -> int:
        """
        Count total hint titles across discovery categories.

        Args:
            hints_by_category: Categorized hint mapping.

        Returns:
            Total number of hint entries.
        """

        return sum(len(entries) for entries in hints_by_category.values())
