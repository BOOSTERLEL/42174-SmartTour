"""BIO label definitions for the supervised requirement model."""

SLOT_NAMES: tuple[str, ...] = (
    "DESTINATION",
    "TRIP_DATES",
    "TRIP_LENGTH_DAYS",
    "ADULTS",
    "CHILDREN",
    "BUDGET_LEVEL",
    "TRAVEL_PACE",
    "INTEREST",
    "HOTEL_AREA",
    "TRANSPORTATION_MODE",
    "FOOD_PREFERENCE",
    "LANGUAGE",
)
LABEL_NAMES: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{slot_name}" for slot_name in SLOT_NAMES for prefix in ("B", "I")
)
LABEL_TO_ID: dict[str, int] = {
    label_name: label_index for label_index, label_name in enumerate(LABEL_NAMES)
}
ID_TO_LABEL: dict[int, str] = {
    label_index: label_name for label_name, label_index in LABEL_TO_ID.items()
}
