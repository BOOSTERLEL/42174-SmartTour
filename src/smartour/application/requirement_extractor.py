"""Temporary rule-based requirement extraction for early backend development."""

import re
from typing import Protocol

from smartour.domain.requirement import Travelers, TravelRequirementUpdate


class RequirementExtractor(Protocol):
    """
    Extracts travel requirement updates from a user message.
    """

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Extract requirement updates from a user message.

        Args:
            message: The raw user message.

        Returns:
            The extracted requirement updates.
        """
        ...


class RuleBasedRequirementExtractor:
    """
    Extracts structured travel requirement updates from user text using simple rules.
    """

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Extract requirement updates from a user message.

        Args:
            message: The raw user message.

        Returns:
            The extracted requirement updates.
        """
        normalized_message = message.strip()
        lower_message = normalized_message.lower()
        return TravelRequirementUpdate(
            destination=self._extract_destination(normalized_message, lower_message),
            trip_dates=self._extract_trip_dates(normalized_message),
            trip_length_days=self._extract_trip_length_days(lower_message),
            travelers=self._extract_travelers(lower_message),
            budget_level=self._extract_budget_level(lower_message),
            travel_pace=self._extract_travel_pace(lower_message),
            interests=self._extract_interests(lower_message),
            hotel_area=self._extract_hotel_area(normalized_message, lower_message),
            transportation_mode=self._extract_transportation_mode(lower_message),
            language=self._extract_language(lower_message),
        )

    def _extract_destination(self, message: str, lower_message: str) -> str | None:
        """
        Extract the travel destination.

        Args:
            message: The raw user message.
            lower_message: The lowercase user message.

        Returns:
            The destination when detected.
        """
        patterns = [
            r"(?:destination|travel to|go to|visit)\s+([A-Za-z][A-Za-z\s.'-]{1,40})",
            r"(?:目的地|去|想去)\s*[:：]?\s*([\u4e00-\u9fffA-Za-z\s.'-]{2,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return self._clean_phrase(match.group(1))
        if "tokyo" in lower_message:
            return "Tokyo"
        if "sydney" in lower_message:
            return "Sydney"
        return None

    def _extract_trip_dates(self, message: str) -> str | None:
        """
        Extract explicit trip dates.

        Args:
            message: The raw user message.

        Returns:
            The trip dates when detected.
        """
        match = re.search(
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*(?:to|-|到)\s*\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            message,
        )
        if match:
            return match.group(1)
        return None

    def _extract_trip_length_days(self, lower_message: str) -> int | None:
        """
        Extract trip length in days.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The trip length when detected.
        """
        match = re.search(r"(\d{1,2})\s*(?:days?|天|日)", lower_message)
        if match:
            return int(match.group(1))
        return None

    def _extract_travelers(self, lower_message: str) -> Travelers | None:
        """
        Extract traveler count.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The travelers update when detected.
        """
        match = re.search(
            r"(\d{1,2})\s*(?:people|travelers?|adults?|人)", lower_message
        )
        if match:
            return Travelers(adults=int(match.group(1)))
        return None

    def _extract_budget_level(self, lower_message: str) -> str | None:
        """
        Extract budget level.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The budget level when detected.
        """
        if any(
            keyword in lower_message
            for keyword in ["moderate", "medium", "mid", "中等"]
        ):
            return "medium"
        if any(
            keyword in lower_message
            for keyword in ["luxury", "high", "premium", "豪华", "高端"]
        ):
            return "high"
        if any(
            keyword in lower_message
            for keyword in ["cheap", "budget-friendly", "low budget", "经济", "便宜"]
        ):
            return "low"
        return None

    def _extract_travel_pace(self, lower_message: str) -> str | None:
        """
        Extract travel pace.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The travel pace when detected.
        """
        if any(
            keyword in lower_message for keyword in ["relaxed", "slow", "轻松", "慢"]
        ):
            return "relaxed"
        if any(
            keyword in lower_message
            for keyword in ["packed", "intensive", "紧凑", "特种兵"]
        ):
            return "packed"
        if any(
            keyword in lower_message
            for keyword in ["balanced", "normal", "正常", "适中"]
        ):
            return "balanced"
        return None

    def _extract_interests(self, lower_message: str) -> list[str]:
        """
        Extract travel interests.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The detected interests.
        """
        interest_keywords = {
            "food": ["food", "restaurant", "美食", "吃"],
            "museums": ["museum", "museums", "博物馆"],
            "history": ["history", "historic", "历史"],
            "nature": ["nature", "park", "自然", "公园"],
            "shopping": ["shopping", "shop", "购物"],
            "nightlife": ["nightlife", "bar", "夜生活"],
            "family": ["family", "kids", "亲子"],
        }
        interests: list[str] = []
        for interest, keywords in interest_keywords.items():
            if any(keyword in lower_message for keyword in keywords):
                interests.append(interest)
        return interests

    def _extract_hotel_area(self, message: str, lower_message: str) -> str | None:
        """
        Extract preferred hotel area.

        Args:
            message: The raw user message.
            lower_message: The lowercase user message.

        Returns:
            The preferred hotel area when detected.
        """
        patterns = [
            r"(?:stay near|hotel near|hotel in)\s+([A-Za-z][A-Za-z\s.'-]{1,40})",
            r"(?:住在|酒店在|住到)\s*([\u4e00-\u9fffA-Za-z\s.'-]{2,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return self._clean_phrase(match.group(1))
        if (
            "near station" in lower_message
            or "near subway" in lower_message
            or "地铁" in lower_message
        ):
            return "near public transit"
        return None

    def _extract_transportation_mode(self, lower_message: str) -> str | None:
        """
        Extract preferred transportation mode.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The preferred transportation mode when detected.
        """
        if any(
            keyword in lower_message
            for keyword in ["transit", "subway", "metro", "public", "地铁", "公交"]
        ):
            return "transit"
        if any(keyword in lower_message for keyword in ["walk", "walking", "步行"]):
            return "walking"
        if any(
            keyword in lower_message for keyword in ["drive", "car", "自驾", "开车"]
        ):
            return "drive"
        return None

    def _extract_language(self, lower_message: str) -> str | None:
        """
        Extract preferred guide language.

        Args:
            lower_message: The lowercase user message.

        Returns:
            The preferred language when detected.
        """
        if any(keyword in lower_message for keyword in ["chinese", "中文", "汉语"]):
            return "zh"
        if "english" in lower_message:
            return "en"
        return None

    def _clean_phrase(self, value: str) -> str:
        """
        Clean a captured phrase.

        Args:
            value: The captured phrase.

        Returns:
            The cleaned phrase.
        """
        separators = [",", ".", "，", "。", " for", " with", " budget", " 预算"]
        cleaned = value.strip()
        for separator in separators:
            if separator in cleaned:
                cleaned = cleaned.split(separator, maxsplit=1)[0].strip()
        return cleaned or value.strip()
