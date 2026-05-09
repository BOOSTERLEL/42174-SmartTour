"""Normalize decoded requirement model spans into domain updates."""

import re
from typing import Any

from smartour.domain.requirement import Travelers, TravelRequirementUpdate
from smartour.integrations.requirement_model.decoder import RequirementSlotSpan

CHINESE_NUMBER_VALUES = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
INTEREST_CANONICAL_VALUES = {
    "food": "food",
    "restaurant": "food",
    "restaurants": "food",
    "美食": "food",
    "吃": "food",
    "museums": "museums",
    "museum": "museums",
    "博物馆": "museums",
    "history": "history",
    "historic": "history",
    "历史": "history",
    "nature": "nature",
    "park": "nature",
    "自然": "nature",
    "公园": "nature",
    "shopping": "shopping",
    "shop": "shopping",
    "购物": "shopping",
    "nightlife": "nightlife",
    "bar": "nightlife",
    "夜生活": "nightlife",
    "family": "family",
    "kids": "family",
    "亲子": "family",
}


def spans_to_requirement_update(
    spans: list[RequirementSlotSpan],
) -> TravelRequirementUpdate:
    """
    Convert decoded slot spans into a partial requirement update.

    Args:
        spans: The decoded requirement slot spans.

    Returns:
        The normalized requirement update.
    """
    values: dict[str, Any] = {
        "interests": [],
        "food_preferences": [],
    }
    adult_count: int | None = None
    child_count: int | None = None
    for span in sorted(spans, key=lambda value: value.confidence, reverse=True):
        if span.slot_name == "ADULTS":
            adult_count = normalize_count(span.text, allow_zero=False) or adult_count
            continue
        if span.slot_name == "CHILDREN":
            normalized_children = normalize_count(span.text, allow_zero=True)
            if normalized_children is not None:
                child_count = normalized_children
            continue
        add_span_value(values, span)
    if adult_count is not None or child_count is not None:
        values["travelers"] = Travelers(
            adults=adult_count,
            children=child_count or 0,
        )
    return TravelRequirementUpdate.model_validate(values)


def add_span_value(values: dict[str, Any], span: RequirementSlotSpan) -> None:
    """
    Add one normalized span value to a mutable update dictionary.

    Args:
        values: The mutable update dictionary.
        span: The decoded slot span.
    """
    normalized_value = normalize_span(span.slot_name, span.text)
    if normalized_value is None:
        return
    if span.slot_name == "INTEREST":
        append_unique(values, "interests", str(normalized_value))
        return
    if span.slot_name == "FOOD_PREFERENCE":
        append_unique(values, "food_preferences", str(normalized_value))
        return
    field_name = slot_name_to_field(span.slot_name)
    if field_name and field_name not in values:
        values[field_name] = normalized_value


def append_unique(values: dict[str, Any], field_name: str, value: str) -> None:
    """
    Append a normalized list value once.

    Args:
        values: The mutable update dictionary.
        field_name: The list field name.
        value: The canonical value.
    """
    values.setdefault(field_name, [])
    if value not in values[field_name]:
        values[field_name].append(value)


def slot_name_to_field(slot_name: str) -> str | None:
    """
    Convert an uppercase slot name into a domain field name.

    Args:
        slot_name: The model slot name.

    Returns:
        The domain field name when supported.
    """
    field_names = {
        "DESTINATION": "destination",
        "TRIP_DATES": "trip_dates",
        "TRIP_LENGTH_DAYS": "trip_length_days",
        "BUDGET_LEVEL": "budget_level",
        "TRAVEL_PACE": "travel_pace",
        "HOTEL_AREA": "hotel_area",
        "TRANSPORTATION_MODE": "transportation_mode",
        "LANGUAGE": "language",
    }
    return field_names.get(slot_name)


def normalize_span(slot_name: str, value: str) -> str | int | None:
    """
    Normalize a raw decoded slot span into a canonical value.

    Args:
        slot_name: The decoded model slot name.
        value: The raw decoded span.

    Returns:
        The canonical value when recognized.
    """
    cleaned_value = clean_phrase(value)
    lower_value = cleaned_value.lower()
    if slot_name == "TRIP_LENGTH_DAYS":
        return normalize_count(cleaned_value, allow_zero=False)
    if slot_name == "BUDGET_LEVEL":
        return normalize_budget(lower_value, cleaned_value)
    if slot_name == "TRAVEL_PACE":
        return normalize_pace(lower_value, cleaned_value)
    if slot_name == "TRANSPORTATION_MODE":
        return normalize_transportation(lower_value, cleaned_value)
    if slot_name == "LANGUAGE":
        return normalize_language(lower_value, cleaned_value)
    if slot_name == "INTEREST":
        return normalize_interest(lower_value, cleaned_value)
    if slot_name == "FOOD_PREFERENCE":
        return lower_value if cleaned_value.isascii() else cleaned_value
    if slot_name in {"DESTINATION", "TRIP_DATES", "HOTEL_AREA"}:
        return cleaned_value
    return None


def clean_phrase(value: str) -> str:
    """
    Clean punctuation and extra whitespace from a decoded phrase.

    Args:
        value: The raw phrase.

    Returns:
        The cleaned phrase.
    """
    return re.sub(r"\s+", " ", value.strip(" ,.;，。")).strip()


def normalize_count(value: str, allow_zero: bool) -> int | None:
    """
    Normalize Arabic and common Chinese count phrases.

    Args:
        value: The raw count phrase.
        allow_zero: Whether zero is a valid result.

    Returns:
        The count when detected and allowed.
    """
    lower_value = value.lower()
    if allow_zero and ("no " in lower_value or "不带" in value or "没有" in value):
        return 0
    digit_match = re.search(r"\d{1,2}", value)
    if digit_match:
        number_value = int(digit_match.group(0))
        if number_value > 0 or allow_zero:
            return number_value
    for text_value, number_value in CHINESE_NUMBER_VALUES.items():
        if text_value in value and (number_value > 0 or allow_zero):
            return number_value
    return None


def normalize_budget(lower_value: str, value: str) -> str | None:
    """
    Normalize a budget phrase.

    Args:
        lower_value: The lowercase phrase.
        value: The original phrase.

    Returns:
        The canonical budget level.
    """
    if any(keyword in lower_value for keyword in ("cheap", "budget-friendly")):
        return "low"
    if "low budget" in lower_value or any(
        keyword in value for keyword in ("经济", "便宜")
    ):
        return "low"
    if any(keyword in lower_value for keyword in ("moderate", "medium", "mid")):
        return "medium"
    if "中等" in value:
        return "medium"
    if any(keyword in lower_value for keyword in ("luxury", "high", "premium")):
        return "high"
    if any(keyword in value for keyword in ("豪华", "高端")):
        return "high"
    return None


def normalize_pace(lower_value: str, value: str) -> str | None:
    """
    Normalize a travel pace phrase.

    Args:
        lower_value: The lowercase phrase.
        value: The original phrase.

    Returns:
        The canonical travel pace.
    """
    if any(keyword in lower_value for keyword in ("relaxed", "slow")):
        return "relaxed"
    if any(keyword in value for keyword in ("轻松", "慢")):
        return "relaxed"
    if any(keyword in lower_value for keyword in ("balanced", "normal")):
        return "balanced"
    if any(keyword in value for keyword in ("正常", "适中")):
        return "balanced"
    if any(keyword in lower_value for keyword in ("packed", "intensive")):
        return "packed"
    if any(keyword in value for keyword in ("紧凑", "特种兵")):
        return "packed"
    return None


def normalize_transportation(lower_value: str, value: str) -> str | None:
    """
    Normalize a transportation mode phrase.

    Args:
        lower_value: The lowercase phrase.
        value: The original phrase.

    Returns:
        The canonical transportation mode.
    """
    if any(
        keyword in lower_value for keyword in ("transit", "subway", "metro", "public")
    ):
        return "transit"
    if any(keyword in value for keyword in ("地铁", "公交", "公共交通")):
        return "transit"
    if any(keyword in lower_value for keyword in ("walk", "walking")):
        return "walking"
    if "步行" in value:
        return "walking"
    if any(keyword in lower_value for keyword in ("drive", "car")):
        return "drive"
    if any(keyword in value for keyword in ("自驾", "开车")):
        return "drive"
    return None


def normalize_language(lower_value: str, value: str) -> str | None:
    """
    Normalize a requested guide language phrase.

    Args:
        lower_value: The lowercase phrase.
        value: The original phrase.

    Returns:
        The ISO 639-1 language code.
    """
    if "chinese" in lower_value or any(
        keyword in value for keyword in ("中文", "汉语")
    ):
        return "zh"
    if "english" in lower_value or "英文" in value:
        return "en"
    return None


def normalize_interest(lower_value: str, value: str) -> str | None:
    """
    Normalize a travel interest phrase.

    Args:
        lower_value: The lowercase phrase.
        value: The original phrase.

    Returns:
        The canonical interest.
    """
    if lower_value in INTEREST_CANONICAL_VALUES:
        return INTEREST_CANONICAL_VALUES[lower_value]
    for keyword, interest in INTEREST_CANONICAL_VALUES.items():
        if keyword in value or keyword in lower_value:
            return interest
    return lower_value or None
