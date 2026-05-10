"""Generate English synthetic data for the requirement understanding model."""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.llm_client import (
    LlmRequestError,
    OpenAiAugmentationClient,
    load_llm_settings,
)
from scripts.requirement_model.prompts import build_augmentation_messages
from scripts.requirement_model.schema import (
    SLOT_NAMES,
    SLOT_TO_FIELD,
    RequirementSlots,
    RequirementTrainingRecord,
    load_jsonl,
    tokenize_text,
    write_jsonl,
)

DEFAULT_OUTPUT_DIR = Path("data/requirement_model")
GENERATED_SPLITS: tuple[str, ...] = ("train", "validation", "test")
REQUIRED_SLOT_FIELDS: tuple[str, ...] = (
    "destination",
    "trip_dates",
    "trip_length_days",
    "adults",
    "children",
    "budget_level",
    "travel_pace",
    "interests",
    "hotel_area",
    "transportation_mode",
    "food_preferences",
    "language",
)
CJK_TEXT_PATTERN = re.compile(r"[\u4e00-\u9fff]")
MARKER_PATTERN = re.compile(r"\[([A-Z_]+)\|([^\[\]\|]+)\]")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SlotOption:
    """
    A surface phrase paired with its canonical slot value.
    """

    text: str
    value: str | int


@dataclass(frozen=True, slots=True)
class Segment:
    """
    A text segment that optionally carries one BIO slot label.
    """

    text: str
    slot_name: str | None = None
    value: str | int | None = None


RecordBuilder = Callable[[random.Random], RequirementTrainingRecord]


class DataGenerationError(ValueError):
    """
    Error raised when a generated data candidate is invalid.
    """


DESTINATIONS = (
    SlotOption("Tokyo", "Tokyo"),
    SlotOption("Sydney", "Sydney"),
    SlotOption("Kyoto", "Kyoto"),
    SlotOption("Paris", "Paris"),
    SlotOption("London", "London"),
    SlotOption("Barcelona", "Barcelona"),
    SlotOption("New York City", "New York City"),
    SlotOption("Singapore", "Singapore"),
    SlotOption("Vancouver", "Vancouver"),
    SlotOption("Rome", "Rome"),
    SlotOption("Amsterdam", "Amsterdam"),
    SlotOption("Seoul", "Seoul"),
)
TRIP_LENGTHS = (
    SlotOption("2 days", 2),
    SlotOption("3 days", 3),
    SlotOption("4 days", 4),
    SlotOption("5 days", 5),
    SlotOption("6 days", 6),
    SlotOption("7 days", 7),
    SlotOption("10 days", 10),
)
TRIP_DATES = (
    SlotOption("2026-07-01 to 2026-07-05", "2026-07-01 to 2026-07-05"),
    SlotOption("2026/08/10-2026/08/14", "2026/08/10-2026/08/14"),
    SlotOption("2026-09-20 to 2026-09-24", "2026-09-20 to 2026-09-24"),
    SlotOption("2026-10-03 through 2026-10-08", "2026-10-03 through 2026-10-08"),
    SlotOption("2026-11-12 to 2026-11-18", "2026-11-12 to 2026-11-18"),
)
ADULTS = (
    SlotOption("1 adult", 1),
    SlotOption("2 adults", 2),
    SlotOption("3 adults", 3),
    SlotOption("4 adults", 4),
    SlotOption("5 adults", 5),
)
CHILDREN = (
    SlotOption("no children", 0),
    SlotOption("no kids", 0),
    SlotOption("1 child", 1),
    SlotOption("2 children", 2),
    SlotOption("3 kids", 3),
)
BUDGETS = (
    SlotOption("budget-friendly", "low"),
    SlotOption("cheap", "low"),
    SlotOption("low budget", "low"),
    SlotOption("moderate", "medium"),
    SlotOption("medium budget", "medium"),
    SlotOption("mid-range", "medium"),
    SlotOption("luxury", "high"),
    SlotOption("high-end", "high"),
    SlotOption("premium", "high"),
)
PACES = (
    SlotOption("relaxed", "relaxed"),
    SlotOption("slow", "relaxed"),
    SlotOption("balanced", "balanced"),
    SlotOption("normal", "balanced"),
    SlotOption("packed", "packed"),
    SlotOption("intensive", "packed"),
)
INTERESTS = (
    SlotOption("food", "food"),
    SlotOption("restaurants", "food"),
    SlotOption("museums", "museums"),
    SlotOption("history", "history"),
    SlotOption("historic sites", "history"),
    SlotOption("nature", "nature"),
    SlotOption("parks", "nature"),
    SlotOption("shopping", "shopping"),
    SlotOption("nightlife", "nightlife"),
    SlotOption("family activities", "family"),
    SlotOption("architecture", "architecture"),
    SlotOption("local markets", "local markets"),
)
HOTEL_AREAS = (
    SlotOption("downtown", "downtown"),
    SlotOption("city center", "city center"),
    SlotOption("near the main station", "near the main station"),
    SlotOption("near public transit", "near public transit"),
    SlotOption("the old town", "the old town"),
    SlotOption("the waterfront", "the waterfront"),
    SlotOption("the museum district", "the museum district"),
    SlotOption("a quiet residential area", "a quiet residential area"),
)
TRANSPORTATION_MODES = (
    SlotOption("transit", "transit"),
    SlotOption("public transportation", "transit"),
    SlotOption("subway", "transit"),
    SlotOption("metro", "transit"),
    SlotOption("walking", "walking"),
    SlotOption("walkable routes", "walking"),
    SlotOption("drive", "drive"),
    SlotOption("rental car", "drive"),
)
FOOD_PREFERENCES = (
    SlotOption("ramen", "ramen"),
    SlotOption("seafood", "seafood"),
    SlotOption("vegetarian food", "vegetarian food"),
    SlotOption("local snacks", "local snacks"),
    SlotOption("street food", "street food"),
    SlotOption("coffee shops", "coffee shops"),
    SlotOption("desserts", "desserts"),
    SlotOption("fine dining", "fine dining"),
)
LANGUAGES = (SlotOption("English", "en"),)
SLOT_OPTIONS_BY_NAME: dict[str, tuple[SlotOption, ...]] = {
    "DESTINATION": DESTINATIONS,
    "TRIP_DATES": TRIP_DATES,
    "TRIP_LENGTH_DAYS": TRIP_LENGTHS,
    "ADULTS": ADULTS,
    "CHILDREN": CHILDREN,
    "BUDGET_LEVEL": BUDGETS,
    "TRAVEL_PACE": PACES,
    "INTEREST": INTERESTS,
    "HOTEL_AREA": HOTEL_AREAS,
    "TRANSPORTATION_MODE": TRANSPORTATION_MODES,
    "FOOD_PREFERENCE": FOOD_PREFERENCES,
    "LANGUAGE": LANGUAGES,
}
CANONICAL_SLOT_VALUES_BY_FIELD: dict[str, set[str]] = {
    SLOT_TO_FIELD[slot_name]: {
        str(option.value) for option in slot_options if isinstance(option.value, str)
    }
    for slot_name, slot_options in SLOT_OPTIONS_BY_NAME.items()
    if SLOT_TO_FIELD[slot_name]
    in {
        "budget_level",
        "travel_pace",
        "interests",
        "transportation_mode",
        "language",
    }
}
SURFACE_TO_CANONICAL: dict[str, dict[str, str | int]] = {
    slot_name: {option.text.lower(): option.value for option in options}
    for slot_name, options in SLOT_OPTIONS_BY_NAME.items()
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate English requirement model JSONL data."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42174)
    parser.add_argument("--language", choices=("en",), default="en")
    parser.add_argument("--template-only", action="store_true")
    parser.add_argument("--llm-augment", action="store_true")
    parser.add_argument("--llm-seed-count", type=int, default=8)
    parser.add_argument("--llm-target-count", type=int, default=24)
    parser.add_argument("--llm-max-rounds", type=int, default=200)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def make_record(segments: list[Segment]) -> RequirementTrainingRecord:
    """
    Build one validated training record from labeled text segments.

    Args:
        segments: The ordered text segments.

    Returns:
        The validated training record.
    """
    text = "".join(segment.text for segment in segments)
    tokens: list[str] = []
    labels: list[str] = []
    slot_values: dict[str, Any] = {
        "interests": [],
        "food_preferences": [],
    }
    for segment in segments:
        segment_tokens = tokenize_text(segment.text)
        tokens.extend(segment_tokens)
        labels.extend(make_segment_labels(segment.slot_name, len(segment_tokens)))
        if segment.slot_name is not None and segment.value is not None:
            add_slot_value(slot_values, segment.slot_name, segment.value)
    return RequirementTrainingRecord(
        text=text,
        tokens=tokens,
        labels=labels,
        slots=RequirementSlots.model_validate(slot_values),
    )


def make_record_from_marked_text(marked_text: str) -> RequirementTrainingRecord:
    """
    Convert inline marker text into a validated training record.

    Args:
        marked_text: Text containing markers such as `[DESTINATION|Tokyo]`.

    Returns:
        The parsed training record.

    Raises:
        DataGenerationError: Raised when markers are malformed or unsupported.
    """
    if not marked_text.strip():
        raise DataGenerationError("marked text is empty")
    segments: list[Segment] = []
    cursor = 0
    marker_count = 0
    for marker_match in MARKER_PATTERN.finditer(marked_text):
        prefix = marked_text[cursor : marker_match.start()]
        ensure_unmarked_text_is_plain(prefix)
        if prefix:
            segments.append(Segment(prefix))
        slot_name = marker_match.group(1)
        surface_text = marker_match.group(2).strip()
        canonical_value = normalize_marker_value(slot_name, surface_text)
        segments.append(Segment(surface_text, slot_name, canonical_value))
        cursor = marker_match.end()
        marker_count += 1
    suffix = marked_text[cursor:]
    ensure_unmarked_text_is_plain(suffix)
    if suffix:
        segments.append(Segment(suffix))
    if marker_count == 0:
        raise DataGenerationError("marked text did not contain slot markers")
    record = make_record(segments)
    validate_english_records("candidate", [record])
    return record


def ensure_unmarked_text_is_plain(text: str) -> None:
    """
    Validate that unmarked text does not contain stray marker brackets.

    Args:
        text: The unmarked text segment.

    Raises:
        DataGenerationError: Raised when bracket syntax is malformed.
    """
    if "[" in text or "]" in text:
        raise DataGenerationError("malformed marker syntax")


def normalize_marker_value(slot_name: str, value: str) -> str | int:
    """
    Normalize one marker value into its canonical slot value.

    Args:
        slot_name: The uppercase slot name.
        value: The marker surface value.

    Returns:
        The canonical value.

    Raises:
        DataGenerationError: Raised when the slot or value is unsupported.
    """
    if slot_name not in SLOT_NAMES:
        raise DataGenerationError(f"unsupported slot marker: {slot_name}")
    cleaned_value = WHITESPACE_PATTERN.sub(" ", value.strip())
    if not cleaned_value:
        raise DataGenerationError(f"{slot_name} marker value is empty")
    mapped_value = SURFACE_TO_CANONICAL.get(slot_name, {}).get(cleaned_value.lower())
    if mapped_value is not None:
        return mapped_value
    normalized_value = infer_marker_value(slot_name, cleaned_value)
    if normalized_value is None:
        raise DataGenerationError(f"unsupported {slot_name} marker value")
    return normalized_value


def infer_marker_value(slot_name: str, value: str) -> str | int | None:
    """
    Infer a canonical value for valid marker text not present in the catalog.

    Args:
        slot_name: The uppercase slot name.
        value: The marker surface value.

    Returns:
        The inferred canonical value, if supported.
    """
    lower_value = value.lower()
    if slot_name in {"DESTINATION", "TRIP_DATES", "HOTEL_AREA"}:
        return value if value.isascii() else None
    if slot_name in {"TRIP_LENGTH_DAYS", "ADULTS"}:
        return extract_count(value, allow_zero=False)
    if slot_name == "CHILDREN":
        return extract_count(value, allow_zero=True)
    if slot_name == "BUDGET_LEVEL":
        return infer_budget(lower_value)
    if slot_name == "TRAVEL_PACE":
        return infer_pace(lower_value)
    if slot_name == "TRANSPORTATION_MODE":
        return infer_transportation(lower_value)
    if slot_name == "LANGUAGE":
        return "en" if lower_value == "english" else None
    if slot_name in {"INTEREST", "FOOD_PREFERENCE"}:
        return lower_value if value.isascii() else None
    return None


def extract_count(value: str, allow_zero: bool) -> int | None:
    """
    Extract a numeric count from marker text.

    Args:
        value: The marker surface value.
        allow_zero: Whether zero is a supported count.

    Returns:
        The count, if detected.
    """
    lower_value = value.lower()
    if allow_zero and lower_value.startswith("no "):
        return 0
    count_match = re.search(r"\d{1,2}", value)
    if count_match is None:
        return None
    count = int(count_match.group(0))
    if count > 0 or allow_zero:
        return count
    return None


def infer_budget(lower_value: str) -> str | None:
    """
    Infer a canonical budget level.

    Args:
        lower_value: The lowercase marker value.

    Returns:
        The canonical budget level, if recognized.
    """
    if any(keyword in lower_value for keyword in ("cheap", "budget", "low")):
        return "low"
    if any(keyword in lower_value for keyword in ("moderate", "medium", "mid")):
        return "medium"
    if any(keyword in lower_value for keyword in ("luxury", "high", "premium")):
        return "high"
    return None


def infer_pace(lower_value: str) -> str | None:
    """
    Infer a canonical travel pace.

    Args:
        lower_value: The lowercase marker value.

    Returns:
        The canonical pace, if recognized.
    """
    if any(keyword in lower_value for keyword in ("relaxed", "slow", "easy")):
        return "relaxed"
    if any(keyword in lower_value for keyword in ("balanced", "normal", "moderate")):
        return "balanced"
    if any(keyword in lower_value for keyword in ("packed", "intensive", "busy")):
        return "packed"
    return None


def infer_transportation(lower_value: str) -> str | None:
    """
    Infer a canonical transportation mode.

    Args:
        lower_value: The lowercase marker value.

    Returns:
        The canonical transportation mode, if recognized.
    """
    if any(
        keyword in lower_value
        for keyword in ("transit", "subway", "metro", "public", "train", "bus")
    ):
        return "transit"
    if "walk" in lower_value:
        return "walking"
    if any(
        keyword in lower_value
        for keyword in ("drive", "car", "taxi", "cab", "scooter", "jeep")
    ):
        return "drive"
    return None


def make_segment_labels(slot_name: str | None, token_count: int) -> list[str]:
    """
    Create BIO labels for one tokenized segment.

    Args:
        slot_name: The slot name for the segment, if any.
        token_count: The number of segment tokens.

    Returns:
        The BIO labels.
    """
    if slot_name is None:
        return ["O"] * token_count
    if token_count == 0:
        return []
    return [f"B-{slot_name}"] + [f"I-{slot_name}"] * (token_count - 1)


def add_slot_value(
    slot_values: dict[str, Any], slot_name: str, value: str | int
) -> None:
    """
    Add one canonical value to the slots dictionary.

    Args:
        slot_values: The mutable slots dictionary.
        slot_name: The BIO slot name.
        value: The canonical slot value.
    """
    field_name = SLOT_TO_FIELD[slot_name]
    if field_name in {"interests", "food_preferences"}:
        slot_values.setdefault(field_name, [])
        if value not in slot_values[field_name]:
            slot_values[field_name].append(value)
        return
    slot_values[field_name] = value


def choose_two(
    options: tuple[SlotOption, ...], random_source: random.Random
) -> tuple[SlotOption, SlotOption]:
    """
    Choose two distinct slot options.

    Args:
        options: The candidate options.
        random_source: The deterministic random source.

    Returns:
        Two distinct options.
    """
    selected_options = random_source.sample(list(options), 2)
    return selected_options[0], selected_options[1]


def generate_records(
    count: int, seed: int, language: str = "en"
) -> list[RequirementTrainingRecord]:
    """
    Generate deterministic English synthetic training examples.

    Args:
        count: The target number of generated examples.
        seed: The deterministic random seed.
        language: The output language code.

    Returns:
        The generated training records.

    Raises:
        ValueError: Raised when a non-English language is requested.
    """
    if language != "en":
        raise ValueError("only English synthetic generation is supported")
    random_source = random.Random(seed)
    builders: tuple[RecordBuilder, ...] = (
        build_full_trip_record,
        build_date_trip_record,
        build_brief_trip_record,
        build_family_trip_record,
        build_food_first_record,
        build_transport_first_record,
    )
    records: list[RequirementTrainingRecord] = []
    while len(records) < count:
        for builder in builders:
            if len(records) >= count:
                break
            records.append(builder(random_source))
    random_source.shuffle(records)
    return records


def generate_llm_augmented_records(
    count: int,
    seed: int,
    language: str,
    output_dir: Path,
    checkpoint_path: Path | None,
    llm_seed_count: int,
    llm_target_count: int,
    llm_max_rounds: int,
    llm_temperature: float,
    llm_max_tokens: int,
) -> list[RequirementTrainingRecord]:
    """
    Generate template plus LLM-augmented English records.

    Args:
        count: The final target record count.
        seed: The deterministic random seed.
        language: The requested language code.
        output_dir: The dataset output directory.
        checkpoint_path: Optional accepted-record checkpoint path.
        llm_seed_count: Marked seeds sent per LLM round.
        llm_target_count: Requested utterance count per LLM round.
        llm_max_rounds: Maximum LLM augmentation rounds.
        llm_temperature: LLM sampling temperature.
        llm_max_tokens: Maximum output tokens per LLM request.

    Returns:
        The generated records.

    Raises:
        ValueError: Raised when the final target count cannot be reached.
    """
    if language != "en":
        raise ValueError("only English synthetic generation is supported")
    resolved_checkpoint_path = resolve_checkpoint_path(output_dir, checkpoint_path)
    records = load_checkpoint_records(resolved_checkpoint_path)
    if not records:
        records = []
    base_template_count = min(max(count // 2, 60), count)
    if len(records) < base_template_count:
        records = top_up_with_template_records(
            records, base_template_count, seed, language
        )
        write_jsonl(resolved_checkpoint_path, records)
    records = deduplicate_records(records)
    seen_texts = {normalize_text_for_dedupe(record.text) for record in records}
    random_source = random.Random(seed)
    client = OpenAiAugmentationClient(load_llm_settings())
    rejected_count = 0
    for round_index in range(1, llm_max_rounds + 1):
        if len(records) >= count:
            break
        marked_examples = [
            build_marked_seed_text(random_source)
            for _seed_index in range(llm_seed_count)
        ]
        target_count = min(llm_target_count, count - len(records))
        messages = build_augmentation_messages(marked_examples, target_count)
        try:
            marked_candidates = client.generate_marked_utterances(
                messages,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
        except LlmRequestError:
            rejected_count += target_count
            print(
                "llm augmentation "
                f"round={round_index} accepted={len(records)}/{count} "
                f"rejected={rejected_count}",
                flush=True,
            )
            continue
        accepted_this_round = False
        for marked_candidate in marked_candidates:
            try:
                record = make_record_from_marked_text(marked_candidate)
            except (DataGenerationError, ValueError):
                rejected_count += 1
                continue
            normalized_text = normalize_text_for_dedupe(record.text)
            if normalized_text in seen_texts:
                rejected_count += 1
                continue
            records.append(record)
            seen_texts.add(normalized_text)
            accepted_this_round = True
            if len(records) >= count:
                break
        if accepted_this_round:
            write_jsonl(resolved_checkpoint_path, records)
        print(
            "llm augmentation "
            f"round={round_index} accepted={len(records)}/{count} "
            f"rejected={rejected_count}",
            flush=True,
        )
    if len(records) < count:
        raise ValueError(
            "LLM augmentation accepted "
            f"{len(records)} records and rejected {rejected_count}; "
            f"target was {count}"
        )
    final_records = records[:count]
    write_jsonl(resolved_checkpoint_path, final_records)
    return final_records


def resolve_checkpoint_path(output_dir: Path, checkpoint_path: Path | None) -> Path:
    """
    Resolve the accepted-record checkpoint path.

    Args:
        output_dir: The dataset output directory.
        checkpoint_path: Optional explicit checkpoint path.

    Returns:
        The checkpoint path.
    """
    if checkpoint_path is not None:
        return checkpoint_path
    return output_dir / "accepted_candidates.jsonl"


def load_checkpoint_records(path: Path) -> list[RequirementTrainingRecord]:
    """
    Load accepted checkpoint records if present.

    Args:
        path: The checkpoint path.

    Returns:
        The checkpoint records.
    """
    if not path.exists():
        return []
    return load_jsonl(path)


def top_up_with_template_records(
    records: list[RequirementTrainingRecord],
    target_count: int,
    seed: int,
    language: str,
) -> list[RequirementTrainingRecord]:
    """
    Add deterministic template records until the base target is reached.

    Args:
        records: Existing accepted records.
        target_count: Desired minimum record count.
        seed: The deterministic random seed.
        language: The output language code.

    Returns:
        Records with deterministic template examples added.
    """
    accepted_records = deduplicate_records(records)
    seen_texts = {normalize_text_for_dedupe(record.text) for record in accepted_records}
    next_seed = seed
    while len(accepted_records) < target_count:
        next_seed += 1
        candidate_records = generate_records(target_count, next_seed, language)
        for record in candidate_records:
            normalized_text = normalize_text_for_dedupe(record.text)
            if normalized_text in seen_texts:
                continue
            accepted_records.append(record)
            seen_texts.add(normalized_text)
            if len(accepted_records) >= target_count:
                break
    return accepted_records


def build_full_trip_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build a complete English travel requirement example.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    return make_record(
        [
            Segment("I want to visit "),
            option_segment(random_source.choice(DESTINATIONS), "DESTINATION"),
            Segment(" for "),
            option_segment(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS"),
            Segment(" with "),
            option_segment(random_source.choice(ADULTS), "ADULTS"),
            Segment(" and "),
            option_segment(random_source.choice(CHILDREN), "CHILDREN"),
            Segment(", on a "),
            option_segment(random_source.choice(BUDGETS), "BUDGET_LEVEL"),
            Segment(" budget, with a "),
            option_segment(random_source.choice(PACES), "TRAVEL_PACE"),
            Segment(" pace. We like "),
            option_segment(first_interest, "INTEREST"),
            Segment(" and "),
            option_segment(second_interest, "INTEREST"),
            Segment(", want to stay in "),
            option_segment(random_source.choice(HOTEL_AREAS), "HOTEL_AREA"),
            Segment(", use "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES),
                "TRANSPORTATION_MODE",
            ),
            Segment(", prefer "),
            option_segment(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE"),
            Segment(", and receive the guide in "),
            option_segment(random_source.choice(LANGUAGES), "LANGUAGE"),
            Segment("."),
        ]
    )


def build_date_trip_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build an English request that uses explicit trip dates.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    return make_record(
        [
            Segment("Plan a "),
            option_segment(random_source.choice(BUDGETS), "BUDGET_LEVEL"),
            Segment(" trip to "),
            option_segment(random_source.choice(DESTINATIONS), "DESTINATION"),
            Segment(" from "),
            option_segment(random_source.choice(TRIP_DATES), "TRIP_DATES"),
            Segment(" for "),
            option_segment(random_source.choice(ADULTS), "ADULTS"),
            Segment(" plus "),
            option_segment(random_source.choice(CHILDREN), "CHILDREN"),
            Segment(". Keep it "),
            option_segment(random_source.choice(PACES), "TRAVEL_PACE"),
            Segment(", focus on "),
            option_segment(first_interest, "INTEREST"),
            Segment(" and "),
            option_segment(second_interest, "INTEREST"),
            Segment(", stay around "),
            option_segment(random_source.choice(HOTEL_AREAS), "HOTEL_AREA"),
            Segment(", get around by "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES),
                "TRANSPORTATION_MODE",
            ),
            Segment(", and include "),
            option_segment(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE"),
            Segment(". Please write it in "),
            option_segment(random_source.choice(LANGUAGES), "LANGUAGE"),
            Segment("."),
        ]
    )


def build_brief_trip_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build a compact English travel request.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    timing_segment = choose_timing_segment(random_source)
    return make_record(
        [
            Segment("Need "),
            timing_segment,
            Segment(" in "),
            option_segment(random_source.choice(DESTINATIONS), "DESTINATION"),
            Segment(" for "),
            option_segment(random_source.choice(ADULTS), "ADULTS"),
            Segment(", "),
            option_segment(random_source.choice(CHILDREN), "CHILDREN"),
            Segment(", "),
            option_segment(random_source.choice(BUDGETS), "BUDGET_LEVEL"),
            Segment(" hotels near "),
            option_segment(random_source.choice(HOTEL_AREAS), "HOTEL_AREA"),
            Segment(", "),
            option_segment(random_source.choice(PACES), "TRAVEL_PACE"),
            Segment(" days, "),
            option_segment(first_interest, "INTEREST"),
            Segment(", "),
            option_segment(second_interest, "INTEREST"),
            Segment(", "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES),
                "TRANSPORTATION_MODE",
            ),
            Segment(", "),
            option_segment(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE"),
            Segment(", "),
            option_segment(random_source.choice(LANGUAGES), "LANGUAGE"),
            Segment("."),
        ]
    )


def build_family_trip_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build an English request with traveler and family-oriented wording.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    return make_record(
        [
            Segment("Create a "),
            option_segment(random_source.choice(PACES), "TRAVEL_PACE"),
            Segment(" "),
            option_segment(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS"),
            Segment(" itinerary for "),
            option_segment(random_source.choice(ADULTS), "ADULTS"),
            Segment(" and "),
            option_segment(random_source.choice(CHILDREN), "CHILDREN"),
            Segment(" in "),
            option_segment(random_source.choice(DESTINATIONS), "DESTINATION"),
            Segment(". We want "),
            option_segment(random_source.choice(INTERESTS), "INTEREST"),
            Segment(", "),
            option_segment(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE"),
            Segment(", a "),
            option_segment(random_source.choice(BUDGETS), "BUDGET_LEVEL"),
            Segment(" plan, a hotel in "),
            option_segment(random_source.choice(HOTEL_AREAS), "HOTEL_AREA"),
            Segment(", "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES),
                "TRANSPORTATION_MODE",
            ),
            Segment(", and an "),
            option_segment(random_source.choice(LANGUAGES), "LANGUAGE"),
            Segment(" guide."),
        ]
    )


def build_food_first_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build an English request that foregrounds food preferences.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    return make_record(
        [
            Segment("We mainly care about "),
            option_segment(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE"),
            Segment(" and "),
            option_segment(first_interest, "INTEREST"),
            Segment(" for a "),
            option_segment(random_source.choice(BUDGETS), "BUDGET_LEVEL"),
            Segment(" "),
            option_segment(random_source.choice(DESTINATIONS), "DESTINATION"),
            Segment(" trip. It should last "),
            option_segment(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS"),
            Segment(" for "),
            option_segment(random_source.choice(ADULTS), "ADULTS"),
            Segment(" with "),
            option_segment(random_source.choice(CHILDREN), "CHILDREN"),
            Segment(", feel "),
            option_segment(random_source.choice(PACES), "TRAVEL_PACE"),
            Segment(", include "),
            option_segment(second_interest, "INTEREST"),
            Segment(", stay in "),
            option_segment(random_source.choice(HOTEL_AREAS), "HOTEL_AREA"),
            Segment(", use "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES),
                "TRANSPORTATION_MODE",
            ),
            Segment(", and be written in "),
            option_segment(random_source.choice(LANGUAGES), "LANGUAGE"),
            Segment("."),
        ]
    )


def build_transport_first_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build an English request that foregrounds transportation mode.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    return make_record(
        [
            Segment("For "),
            option_segment(random_source.choice(TRIP_DATES), "TRIP_DATES"),
            Segment(", arrange "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES),
                "TRANSPORTATION_MODE",
            ),
            Segment(" around "),
            option_segment(random_source.choice(DESTINATIONS), "DESTINATION"),
            Segment(" for "),
            option_segment(random_source.choice(ADULTS), "ADULTS"),
            Segment(" and "),
            option_segment(random_source.choice(CHILDREN), "CHILDREN"),
            Segment(". Keep lodging "),
            option_segment(random_source.choice(BUDGETS), "BUDGET_LEVEL"),
            Segment(" in "),
            option_segment(random_source.choice(HOTEL_AREAS), "HOTEL_AREA"),
            Segment(", make the days "),
            option_segment(random_source.choice(PACES), "TRAVEL_PACE"),
            Segment(", add "),
            option_segment(first_interest, "INTEREST"),
            Segment(", "),
            option_segment(second_interest, "INTEREST"),
            Segment(", and "),
            option_segment(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE"),
            Segment(". Use "),
            option_segment(random_source.choice(LANGUAGES), "LANGUAGE"),
            Segment("."),
        ]
    )


def choose_timing_segment(random_source: random.Random) -> Segment:
    """
    Choose either an explicit date range or a trip length segment.

    Args:
        random_source: The deterministic random source.

    Returns:
        A labeled timing segment.
    """
    if random_source.random() < 0.5:
        return option_segment(random_source.choice(TRIP_DATES), "TRIP_DATES")
    return option_segment(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS")


def build_marked_seed_text(random_source: random.Random) -> str:
    """
    Build one marked English seed utterance for LLM augmentation.

    Args:
        random_source: The deterministic random source.

    Returns:
        A marked seed utterance.
    """
    builders: tuple[Callable[[random.Random], str], ...] = (
        build_marked_full_trip,
        build_marked_date_trip,
        build_marked_brief_trip,
        build_marked_food_trip,
        build_marked_transport_trip,
    )
    return random_source.choice(builders)(random_source)


def build_marked_full_trip(random_source: random.Random) -> str:
    """
    Build a full marked seed request.

    Args:
        random_source: The deterministic random source.

    Returns:
        The marked seed request.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    destination = marked_option(random_source.choice(DESTINATIONS), "DESTINATION")
    trip_length = marked_option(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS")
    adults = marked_option(random_source.choice(ADULTS), "ADULTS")
    children = marked_option(random_source.choice(CHILDREN), "CHILDREN")
    budget = marked_option(random_source.choice(BUDGETS), "BUDGET_LEVEL")
    pace = marked_option(random_source.choice(PACES), "TRAVEL_PACE")
    hotel_area = marked_option(random_source.choice(HOTEL_AREAS), "HOTEL_AREA")
    transport = marked_option(
        random_source.choice(TRANSPORTATION_MODES), "TRANSPORTATION_MODE"
    )
    food = marked_option(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE")
    language = marked_option(random_source.choice(LANGUAGES), "LANGUAGE")
    return (
        f"I want to visit {destination} for {trip_length} with {adults} and "
        f"{children}, on a {budget} budget. Keep a {pace} pace, "
        f"include {marked_option(first_interest, 'INTEREST')} and "
        f"{marked_option(second_interest, 'INTEREST')}, stay in {hotel_area}, "
        f"use {transport}, prefer {food}, and write in {language}."
    )


def build_marked_date_trip(random_source: random.Random) -> str:
    """
    Build a marked seed request with explicit dates.

    Args:
        random_source: The deterministic random source.

    Returns:
        The marked seed request.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    trip_dates = marked_option(random_source.choice(TRIP_DATES), "TRIP_DATES")
    destination = marked_option(random_source.choice(DESTINATIONS), "DESTINATION")
    adults = marked_option(random_source.choice(ADULTS), "ADULTS")
    children = marked_option(random_source.choice(CHILDREN), "CHILDREN")
    budget = marked_option(random_source.choice(BUDGETS), "BUDGET_LEVEL")
    pace = marked_option(random_source.choice(PACES), "TRAVEL_PACE")
    transport = marked_option(
        random_source.choice(TRANSPORTATION_MODES), "TRANSPORTATION_MODE"
    )
    food = marked_option(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE")
    hotel_area = marked_option(random_source.choice(HOTEL_AREAS), "HOTEL_AREA")
    language = marked_option(random_source.choice(LANGUAGES), "LANGUAGE")
    return (
        f"Plan {trip_dates} in {destination} for {adults} and {children}. "
        f"The budget should be {budget}, the pace should be {pace}, "
        f"and the route should use {transport}. "
        f"Add {marked_option(first_interest, 'INTEREST')}, "
        f"{marked_option(second_interest, 'INTEREST')}, "
        f"{food}, a hotel near {hotel_area}, and an {language} guide."
    )


def build_marked_brief_trip(random_source: random.Random) -> str:
    """
    Build a compact marked seed request.

    Args:
        random_source: The deterministic random source.

    Returns:
        The marked seed request.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    transport = marked_option(
        random_source.choice(TRANSPORTATION_MODES), "TRANSPORTATION_MODE"
    )
    food = marked_option(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE")
    language = marked_option(random_source.choice(LANGUAGES), "LANGUAGE")
    return (
        f"Need {marked_timing_option(random_source)} in "
        f"{marked_option(random_source.choice(DESTINATIONS), 'DESTINATION')} for "
        f"{marked_option(random_source.choice(ADULTS), 'ADULTS')}, "
        f"{marked_option(random_source.choice(CHILDREN), 'CHILDREN')}, "
        f"{marked_option(random_source.choice(BUDGETS), 'BUDGET_LEVEL')} hotels in "
        f"{marked_option(random_source.choice(HOTEL_AREAS), 'HOTEL_AREA')}, "
        f"{marked_option(random_source.choice(PACES), 'TRAVEL_PACE')} days, "
        f"{marked_option(first_interest, 'INTEREST')}, "
        f"{marked_option(second_interest, 'INTEREST')}, "
        f"{transport}, {food}, {language}."
    )


def build_marked_food_trip(random_source: random.Random) -> str:
    """
    Build a food-forward marked seed request.

    Args:
        random_source: The deterministic random source.

    Returns:
        The marked seed request.
    """
    budget = marked_option(random_source.choice(BUDGETS), "BUDGET_LEVEL")
    destination = marked_option(random_source.choice(DESTINATIONS), "DESTINATION")
    adults = marked_option(random_source.choice(ADULTS), "ADULTS")
    children = marked_option(random_source.choice(CHILDREN), "CHILDREN")
    food = marked_option(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE")
    interest = marked_option(random_source.choice(INTERESTS), "INTEREST")
    trip_length = marked_option(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS")
    pace = marked_option(random_source.choice(PACES), "TRAVEL_PACE")
    hotel_area = marked_option(random_source.choice(HOTEL_AREAS), "HOTEL_AREA")
    transport = marked_option(
        random_source.choice(TRANSPORTATION_MODES), "TRANSPORTATION_MODE"
    )
    language = marked_option(random_source.choice(LANGUAGES), "LANGUAGE")
    return (
        f"Build a {budget} {destination} trip for {adults} and {children}. "
        f"We care about {food}, {interest}, {trip_length}, "
        f"a {pace} pace, {hotel_area}, {transport}, and {language}."
    )


def build_marked_transport_trip(random_source: random.Random) -> str:
    """
    Build a transportation-forward marked seed request.

    Args:
        random_source: The deterministic random source.

    Returns:
        The marked seed request.
    """
    first_interest, second_interest = choose_two(INTERESTS, random_source)
    transport = marked_option(
        random_source.choice(TRANSPORTATION_MODES), "TRANSPORTATION_MODE"
    )
    trip_dates = marked_option(random_source.choice(TRIP_DATES), "TRIP_DATES")
    destination = marked_option(random_source.choice(DESTINATIONS), "DESTINATION")
    adults = marked_option(random_source.choice(ADULTS), "ADULTS")
    children = marked_option(random_source.choice(CHILDREN), "CHILDREN")
    budget = marked_option(random_source.choice(BUDGETS), "BUDGET_LEVEL")
    hotel_area = marked_option(random_source.choice(HOTEL_AREAS), "HOTEL_AREA")
    pace = marked_option(random_source.choice(PACES), "TRAVEL_PACE")
    food = marked_option(random_source.choice(FOOD_PREFERENCES), "FOOD_PREFERENCE")
    language = marked_option(random_source.choice(LANGUAGES), "LANGUAGE")
    return (
        f"Use {transport} for a {trip_dates} trip to {destination}. "
        f"It is for {adults} and {children}, with {budget} lodging in "
        f"{hotel_area}. Keep it {pace}, "
        f"add {marked_option(first_interest, 'INTEREST')}, "
        f"{marked_option(second_interest, 'INTEREST')}, "
        f"{food}, and write it in {language}."
    )


def marked_timing_option(random_source: random.Random) -> str:
    """
    Choose a marked date range or trip length.

    Args:
        random_source: The deterministic random source.

    Returns:
        The marked timing option.
    """
    if random_source.random() < 0.5:
        return marked_option(random_source.choice(TRIP_DATES), "TRIP_DATES")
    return marked_option(random_source.choice(TRIP_LENGTHS), "TRIP_LENGTH_DAYS")


def marked_option(option: SlotOption, slot_name: str) -> str:
    """
    Convert a slot option into inline marker syntax.

    Args:
        option: The slot option.
        slot_name: The BIO slot name.

    Returns:
        The marked option.
    """
    return f"[{slot_name}|{option.text}]"


def option_segment(option: SlotOption, slot_name: str) -> Segment:
    """
    Convert a slot option into a labeled segment.

    Args:
        option: The slot option.
        slot_name: The BIO slot name.

    Returns:
        The labeled segment.
    """
    return Segment(option.text, slot_name, option.value)


def reviewed_records() -> list[RequirementTrainingRecord]:
    """
    Return the manually reviewed seed evaluation records.

    Returns:
        The reviewed records.
    """
    return [
        make_record(
            [
                Segment("We are going to "),
                Segment("Tokyo", "DESTINATION", "Tokyo"),
                Segment(" for "),
                Segment("4 days", "TRIP_LENGTH_DAYS", 4),
                Segment(" with "),
                Segment("2 adults", "ADULTS", 2),
                Segment(" and "),
                Segment("no children", "CHILDREN", 0),
                Segment(", a "),
                Segment("moderate", "BUDGET_LEVEL", "medium"),
                Segment(" budget, "),
                Segment("relaxed", "TRAVEL_PACE", "relaxed"),
                Segment(" days, "),
                Segment("food", "INTEREST", "food"),
                Segment(" and "),
                Segment("museums", "INTEREST", "museums"),
                Segment(", hotel near "),
                Segment("Shinjuku", "HOTEL_AREA", "Shinjuku"),
                Segment(", "),
                Segment("transit", "TRANSPORTATION_MODE", "transit"),
                Segment(", "),
                Segment("ramen", "FOOD_PREFERENCE", "ramen"),
                Segment(", and "),
                Segment("English", "LANGUAGE", "en"),
                Segment("."),
            ]
        ),
        make_record(
            [
                Segment("Book a "),
                Segment("luxury", "BUDGET_LEVEL", "high"),
                Segment(" "),
                Segment("Paris", "DESTINATION", "Paris"),
                Segment(" trip from "),
                Segment(
                    "2026-07-01 to 2026-07-05",
                    "TRIP_DATES",
                    "2026-07-01 to 2026-07-05",
                ),
                Segment(" for "),
                Segment("3 adults", "ADULTS", 3),
                Segment(" and "),
                Segment("1 child", "CHILDREN", 1),
                Segment(". We prefer "),
                Segment("walking", "TRANSPORTATION_MODE", "walking"),
                Segment(", "),
                Segment("history", "INTEREST", "history"),
                Segment(", "),
                Segment("vegetarian food", "FOOD_PREFERENCE", "vegetarian food"),
                Segment(", "),
                Segment("city center", "HOTEL_AREA", "city center"),
                Segment(", "),
                Segment("balanced", "TRAVEL_PACE", "balanced"),
                Segment(" pacing, and "),
                Segment("English", "LANGUAGE", "en"),
                Segment("."),
            ]
        ),
        make_record(
            [
                Segment("Please plan "),
                Segment("5 days", "TRIP_LENGTH_DAYS", 5),
                Segment(" in "),
                Segment("Sydney", "DESTINATION", "Sydney"),
                Segment(" for "),
                Segment("2 adults", "ADULTS", 2),
                Segment(" with "),
                Segment("2 children", "CHILDREN", 2),
                Segment(". Keep it "),
                Segment("budget-friendly", "BUDGET_LEVEL", "low"),
                Segment(" and "),
                Segment("packed", "TRAVEL_PACE", "packed"),
                Segment(", focus on "),
                Segment("nature", "INTEREST", "nature"),
                Segment(" and "),
                Segment("family activities", "INTEREST", "family"),
                Segment(", stay near "),
                Segment("the waterfront", "HOTEL_AREA", "the waterfront"),
                Segment(", use "),
                Segment("public transportation", "TRANSPORTATION_MODE", "transit"),
                Segment(", eat "),
                Segment("seafood", "FOOD_PREFERENCE", "seafood"),
                Segment(", and write in "),
                Segment("English", "LANGUAGE", "en"),
                Segment("."),
            ]
        ),
        make_record(
            [
                Segment("Arrange "),
                Segment(
                    "2026-10-03 through 2026-10-08",
                    "TRIP_DATES",
                    "2026-10-03 through 2026-10-08",
                ),
                Segment(" in "),
                Segment("Barcelona", "DESTINATION", "Barcelona"),
                Segment(" for "),
                Segment("4 adults", "ADULTS", 4),
                Segment(" and "),
                Segment("no kids", "CHILDREN", 0),
                Segment(". We want "),
                Segment("mid-range", "BUDGET_LEVEL", "medium"),
                Segment(" hotels in "),
                Segment("the old town", "HOTEL_AREA", "the old town"),
                Segment(", "),
                Segment("slow", "TRAVEL_PACE", "relaxed"),
                Segment(" mornings, "),
                Segment("architecture", "INTEREST", "architecture"),
                Segment(", "),
                Segment("local markets", "INTEREST", "local markets"),
                Segment(", "),
                Segment("street food", "FOOD_PREFERENCE", "street food"),
                Segment(", "),
                Segment("metro", "TRANSPORTATION_MODE", "transit"),
                Segment(", and "),
                Segment("English", "LANGUAGE", "en"),
                Segment("."),
            ]
        ),
    ]


def write_splits(
    output_dir: Path, records: list[RequirementTrainingRecord]
) -> dict[str, Path]:
    """
    Write generated records into train, validation, and test splits.

    Args:
        output_dir: The output directory.
        records: The generated records.

    Returns:
        A mapping from split name to output path.
    """
    train_count = max(1, int(len(records) * 0.8))
    validation_count = max(1, int(len(records) * 0.1))
    split_records = {
        "train": records[:train_count],
        "validation": records[train_count : train_count + validation_count],
        "test": records[train_count + validation_count :],
        "reviewed_test": reviewed_records(),
    }
    split_paths: dict[str, Path] = {}
    for split_name, split_values in split_records.items():
        split_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(split_path, split_values)
        split_paths[split_name] = split_path
    return split_paths


def validate_output(output_dir: Path) -> dict[str, int]:
    """
    Validate generated JSONL files, English-only content, and slot coverage.

    Args:
        output_dir: The output directory.

    Returns:
        The number of records in each split.

    Raises:
        ValueError: Raised when required coverage or split separation fails.
    """
    split_records_by_name: dict[str, list[RequirementTrainingRecord]] = {}
    for split_name in (*GENERATED_SPLITS, "reviewed_test"):
        split_records = load_jsonl(output_dir / f"{split_name}.jsonl")
        split_records_by_name[split_name] = split_records
    return validate_split_records(split_records_by_name)


def validate_split_records(
    split_records_by_name: dict[str, list[RequirementTrainingRecord]],
) -> dict[str, int]:
    """
    Validate split records for schema, English content, duplicates, and leakage.

    Args:
        split_records_by_name: Mapping of split name to records.

    Returns:
        The number of records in each split.

    Raises:
        ValueError: Raised when strict split validation fails.
    """
    split_counts: dict[str, int] = {}
    all_records: list[RequirementTrainingRecord] = []
    generated_texts: set[str] = set()
    reviewed_texts: set[str] = set()
    seen_texts: dict[str, str] = {}
    for split_name, split_records in split_records_by_name.items():
        validate_english_records(split_name, split_records)
        split_counts[split_name] = len(split_records)
        all_records.extend(split_records)
        for record in split_records:
            normalized_text = normalize_text_for_dedupe(record.text)
            existing_split = seen_texts.get(normalized_text)
            if existing_split is not None:
                if is_allowed_review_duplicate(existing_split, split_name):
                    continue
                raise ValueError(
                    "duplicate record text across splits: "
                    f"{existing_split} and {split_name}"
                )
            seen_texts[normalized_text] = split_name
        if split_name in {"train", "validation"}:
            generated_texts.update(
                normalize_text_for_dedupe(record.text) for record in split_records
            )
        if split_name == "reviewed_test":
            reviewed_texts.update(
                normalize_text_for_dedupe(record.text) for record in split_records
            )
    if generated_texts.intersection(reviewed_texts):
        raise ValueError(
            "reviewed test records must be separated from train and validation data"
        )
    covered_fields = collect_covered_fields(all_records)
    missing_fields = sorted(set(REQUIRED_SLOT_FIELDS).difference(covered_fields))
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        raise ValueError(f"missing slot coverage: {joined_fields}")
    return split_counts


def is_allowed_review_duplicate(existing_split: str, split_name: str) -> bool:
    """
    Return whether a duplicate is an allowed reviewed-test mirror.

    Args:
        existing_split: The split where the text was first seen.
        split_name: The current split name.

    Returns:
        Whether the duplicate is allowed.
    """
    return {existing_split, split_name} == {"test", "reviewed_test"}


def deduplicate_records(
    records: list[RequirementTrainingRecord],
) -> list[RequirementTrainingRecord]:
    """
    Remove exact normalized duplicate records while preserving order.

    Args:
        records: The candidate records.

    Returns:
        Unique records.
    """
    unique_records: list[RequirementTrainingRecord] = []
    seen_texts: set[str] = set()
    for record in records:
        normalized_text = normalize_text_for_dedupe(record.text)
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        unique_records.append(record)
    return unique_records


def normalize_text_for_dedupe(text: str) -> str:
    """
    Normalize text for duplicate and leakage checks.

    Args:
        text: The source text.

    Returns:
        The normalized text.
    """
    return WHITESPACE_PATTERN.sub(" ", text.strip().lower())


def validate_english_records(
    split_name: str, records: list[RequirementTrainingRecord]
) -> None:
    """
    Validate that records contain English-only text and English guide labels.

    Args:
        split_name: The split being validated.
        records: The records to validate.

    Raises:
        ValueError: Raised when a record contains unsupported language content.
    """
    for record_index, record in enumerate(records, start=1):
        if CJK_TEXT_PATTERN.search(record.text):
            raise ValueError(f"{split_name} record {record_index} is not English-only")
        if record.slots.language is not None and record.slots.language != "en":
            raise ValueError(
                f"{split_name} record {record_index} has non-English language slot"
            )
        validate_canonical_slot_values(split_name, record_index, record)


def validate_canonical_slot_values(
    split_name: str,
    record_index: int,
    record: RequirementTrainingRecord,
) -> None:
    """
    Validate that enumerated slot fields use known canonical values.

    Args:
        split_name: The split being validated.
        record_index: The one-based record index within the split.
        record: The record to validate.

    Raises:
        ValueError: Raised when a canonical slot value is unsupported.
    """
    slot_values = record.slots.model_dump()
    for field_name, allowed_values in CANONICAL_SLOT_VALUES_BY_FIELD.items():
        value = slot_values[field_name]
        values = value if isinstance(value, list) else [value]
        unsupported_values = [
            str(item)
            for item in values
            if item is not None and str(item) not in allowed_values
        ]
        if unsupported_values:
            joined_values = ", ".join(sorted(set(unsupported_values)))
            raise ValueError(
                f"{split_name} record {record_index} has unsupported "
                f"{field_name} value: {joined_values}"
            )


def collect_covered_fields(records: list[RequirementTrainingRecord]) -> set[str]:
    """
    Return slot fields that have at least one labeled canonical value.

    Args:
        records: The records to inspect.

    Returns:
        The covered slot field names.
    """
    covered_fields: set[str] = set()
    for record in records:
        slot_values = record.slots.model_dump()
        for field_name, value in slot_values.items():
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            covered_fields.add(field_name)
    return covered_fields


def main() -> None:
    """
    Generate and optionally validate requirement model data.
    """
    args = parse_args()
    if args.template_only and args.llm_augment:
        raise ValueError("--template-only and --llm-augment cannot be combined")
    if args.llm_augment:
        records = generate_llm_augmented_records(
            count=args.count,
            seed=args.seed,
            language=args.language,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            llm_seed_count=args.llm_seed_count,
            llm_target_count=args.llm_target_count,
            llm_max_rounds=args.llm_max_rounds,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
        )
    else:
        records = generate_records(args.count, args.seed, args.language)
    split_paths = write_splits(args.output_dir, records)
    if args.validate:
        split_counts = validate_output(args.output_dir)
    else:
        split_counts = {
            split_name: len(load_jsonl(path))
            for split_name, path in split_paths.items()
        }
    counts_text = ", ".join(
        f"{split_name}={split_count}"
        for split_name, split_count in sorted(split_counts.items())
    )
    print(f"wrote requirement model data to {args.output_dir}: {counts_text}")


if __name__ == "__main__":
    main()
