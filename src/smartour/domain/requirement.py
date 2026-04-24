"""Travel requirement domain models."""

from pydantic import BaseModel, Field


class Travelers(BaseModel):
    """
    Traveler counts for a trip.
    """

    adults: int | None = Field(default=None, ge=1)
    children: int = Field(default=0, ge=0)


class TravelRequirementUpdate(BaseModel):
    """
    Partial update for a travel requirement snapshot.
    """

    destination: str | None = None
    trip_dates: str | None = None
    trip_length_days: int | None = Field(default=None, ge=1)
    travelers: Travelers | None = None
    budget_level: str | None = None
    travel_pace: str | None = None
    interests: list[str] = Field(default_factory=list)
    hotel_area: str | None = None
    transportation_mode: str | None = None
    food_preferences: list[str] = Field(default_factory=list)
    language: str | None = None


class TravelRequirement(BaseModel):
    """
    Canonical travel requirement snapshot collected from conversation.
    """

    destination: str | None = None
    trip_dates: str | None = None
    trip_length_days: int | None = Field(default=None, ge=1)
    travelers: Travelers = Field(default_factory=Travelers)
    budget_level: str | None = None
    travel_pace: str | None = None
    interests: list[str] = Field(default_factory=list)
    hotel_area: str | None = None
    transportation_mode: str | None = None
    food_preferences: list[str] = Field(default_factory=list)
    language: str = "en"

    def merge(self, update: TravelRequirementUpdate) -> "TravelRequirement":
        """
        Merge a partial update into the requirement snapshot.

        Args:
            update: The partial requirement update.

        Returns:
            The merged requirement snapshot.
        """
        values = self.model_dump()
        for field_name in [
            "destination",
            "trip_dates",
            "trip_length_days",
            "budget_level",
            "travel_pace",
            "hotel_area",
            "transportation_mode",
            "language",
        ]:
            update_value = getattr(update, field_name)
            if update_value is not None:
                values[field_name] = update_value
        if update.travelers is not None:
            values["travelers"] = self._merge_travelers(update.travelers)
        values["interests"] = self._merge_list(self.interests, update.interests)
        values["food_preferences"] = self._merge_list(
            self.food_preferences, update.food_preferences
        )
        return TravelRequirement.model_validate(values)

    def missing_required_slots(self) -> list[str]:
        """
        Return required travel planning slots that are still missing.

        Returns:
            The missing required slot names.
        """
        missing_slots: list[str] = []
        if not self.destination:
            missing_slots.append("destination")
        if not self.trip_dates and not self.trip_length_days:
            missing_slots.append("trip_dates_or_length")
        if self.travelers.adults is None:
            missing_slots.append("travelers")
        if not self.budget_level:
            missing_slots.append("budget_level")
        if not self.travel_pace:
            missing_slots.append("travel_pace")
        if not self.interests:
            missing_slots.append("interests")
        if not self.hotel_area:
            missing_slots.append("hotel_area")
        if not self.transportation_mode:
            missing_slots.append("transportation_mode")
        return missing_slots

    def _merge_travelers(self, update: Travelers) -> Travelers:
        """
        Merge traveler count updates.

        Args:
            update: The traveler update.

        Returns:
            The merged travelers value.
        """
        return Travelers(
            adults=update.adults
            if update.adults is not None
            else self.travelers.adults,
            children=update.children
            if update.children != 0
            else self.travelers.children,
        )

    def _merge_list(
        self, current_values: list[str], update_values: list[str]
    ) -> list[str]:
        """
        Merge list values while preserving order.

        Args:
            current_values: The current list values.
            update_values: The update list values.

        Returns:
            The merged list values.
        """
        merged_values = list(current_values)
        for update_value in update_values:
            if update_value not in merged_values:
                merged_values.append(update_value)
        return merged_values
