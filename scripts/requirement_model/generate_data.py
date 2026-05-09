"""Generate synthetic data for the supervised requirement understanding model."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schema import (
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


DESTINATIONS = (
    SlotOption("Tokyo", "Tokyo"),
    SlotOption("Sydney", "Sydney"),
    SlotOption("Kyoto", "Kyoto"),
    SlotOption("Paris", "Paris"),
    SlotOption("台北", "台北"),
    SlotOption("大阪", "大阪"),
    SlotOption("墨尔本", "墨尔本"),
)
TRIP_LENGTHS = (
    SlotOption("3 days", 3),
    SlotOption("4 days", 4),
    SlotOption("5 days", 5),
    SlotOption("三天", 3),
    SlotOption("四天", 4),
    SlotOption("五天", 5),
)
TRIP_DATES = (
    SlotOption("2026-07-01 to 2026-07-05", "2026-07-01 to 2026-07-05"),
    SlotOption("2026/08/10-2026/08/14", "2026/08/10-2026/08/14"),
    SlotOption("2026-09-20到2026-09-24", "2026-09-20到2026-09-24"),
)
ADULTS = (
    SlotOption("1 adult", 1),
    SlotOption("2 adults", 2),
    SlotOption("3 people", 3),
    SlotOption("两个人", 2),
    SlotOption("三位成人", 3),
)
CHILDREN = (
    SlotOption("no children", 0),
    SlotOption("1 child", 1),
    SlotOption("2 kids", 2),
    SlotOption("不带孩子", 0),
    SlotOption("一个小孩", 1),
)
BUDGETS = (
    SlotOption("budget-friendly", "low"),
    SlotOption("cheap", "low"),
    SlotOption("moderate", "medium"),
    SlotOption("mid-range", "medium"),
    SlotOption("luxury", "high"),
    SlotOption("经济型", "low"),
    SlotOption("中等预算", "medium"),
    SlotOption("高端", "high"),
)
PACES = (
    SlotOption("relaxed", "relaxed"),
    SlotOption("balanced", "balanced"),
    SlotOption("packed", "packed"),
    SlotOption("轻松", "relaxed"),
    SlotOption("适中", "balanced"),
    SlotOption("紧凑", "packed"),
)
INTERESTS = (
    SlotOption("food", "food"),
    SlotOption("museums", "museums"),
    SlotOption("history", "history"),
    SlotOption("nature", "nature"),
    SlotOption("shopping", "shopping"),
    SlotOption("nightlife", "nightlife"),
    SlotOption("美食", "food"),
    SlotOption("博物馆", "museums"),
    SlotOption("历史", "history"),
    SlotOption("自然", "nature"),
)
HOTEL_AREAS = (
    SlotOption("Shinjuku", "Shinjuku"),
    SlotOption("CBD", "CBD"),
    SlotOption("near the main station", "near the main station"),
    SlotOption("harbour area", "harbour area"),
    SlotOption("新宿", "新宿"),
    SlotOption("市中心", "市中心"),
    SlotOption("地铁站附近", "地铁站附近"),
)
TRANSPORTATION_MODES = (
    SlotOption("transit", "transit"),
    SlotOption("subway", "transit"),
    SlotOption("walking", "walking"),
    SlotOption("drive", "drive"),
    SlotOption("公共交通", "transit"),
    SlotOption("步行", "walking"),
    SlotOption("自驾", "drive"),
)
FOOD_PREFERENCES = (
    SlotOption("ramen", "ramen"),
    SlotOption("seafood", "seafood"),
    SlotOption("vegetarian food", "vegetarian food"),
    SlotOption("local snacks", "local snacks"),
    SlotOption("拉面", "拉面"),
    SlotOption("海鲜", "海鲜"),
    SlotOption("素食", "素食"),
)
LANGUAGES = (
    SlotOption("English", "en"),
    SlotOption("Chinese", "zh"),
    SlotOption("中文", "zh"),
    SlotOption("英文", "en"),
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic requirement model JSONL data."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42174)
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


def generate_records(count: int, seed: int) -> list[RequirementTrainingRecord]:
    """
    Generate deterministic synthetic training examples.

    Args:
        count: The target number of generated examples.
        seed: The deterministic random seed.

    Returns:
        The generated training records.
    """
    random_source = random.Random(seed)
    builders = (
        build_english_full_record,
        build_chinese_full_record,
        build_english_partial_record,
        build_chinese_partial_record,
    )
    records: list[RequirementTrainingRecord] = []
    while len(records) < count:
        for builder in builders:
            if len(records) >= count:
                break
            records.append(builder(random_source))
    random_source.shuffle(records)
    return records


def build_english_full_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build a full English travel requirement example.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS[:6], random_source)
    return make_record(
        [
            Segment("I want to visit "),
            option_segment(random_source.choice(DESTINATIONS[:4]), "DESTINATION"),
            Segment(" for "),
            option_segment(random_source.choice(TRIP_LENGTHS[:3]), "TRIP_LENGTH_DAYS"),
            Segment(" with "),
            option_segment(random_source.choice(ADULTS[:3]), "ADULTS"),
            Segment(" and "),
            option_segment(random_source.choice(CHILDREN[:3]), "CHILDREN"),
            Segment(", on a "),
            option_segment(random_source.choice(BUDGETS[:5]), "BUDGET_LEVEL"),
            Segment(" budget, with a "),
            option_segment(random_source.choice(PACES[:3]), "TRAVEL_PACE"),
            Segment(" pace. We like "),
            option_segment(first_interest, "INTEREST"),
            Segment(" and "),
            option_segment(second_interest, "INTEREST"),
            Segment(", want to stay in "),
            option_segment(random_source.choice(HOTEL_AREAS[:4]), "HOTEL_AREA"),
            Segment(", use "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES[:4]),
                "TRANSPORTATION_MODE",
            ),
            Segment(", prefer "),
            option_segment(
                random_source.choice(FOOD_PREFERENCES[:4]), "FOOD_PREFERENCE"
            ),
            Segment(", and receive the guide in "),
            option_segment(random_source.choice(LANGUAGES[:2]), "LANGUAGE"),
            Segment("."),
        ]
    )


def build_chinese_full_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build a full Chinese travel requirement example.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS[6:], random_source)
    return make_record(
        [
            Segment("我想去"),
            option_segment(random_source.choice(DESTINATIONS[4:]), "DESTINATION"),
            Segment("玩"),
            option_segment(random_source.choice(TRIP_LENGTHS[3:]), "TRIP_LENGTH_DAYS"),
            Segment("，"),
            option_segment(random_source.choice(ADULTS[3:]), "ADULTS"),
            Segment("，"),
            option_segment(random_source.choice(CHILDREN[3:]), "CHILDREN"),
            Segment("，预算"),
            option_segment(random_source.choice(BUDGETS[5:]), "BUDGET_LEVEL"),
            Segment("，节奏"),
            option_segment(random_source.choice(PACES[3:]), "TRAVEL_PACE"),
            Segment("，喜欢"),
            option_segment(first_interest, "INTEREST"),
            Segment("和"),
            option_segment(second_interest, "INTEREST"),
            Segment("，酒店想住"),
            option_segment(random_source.choice(HOTEL_AREAS[4:]), "HOTEL_AREA"),
            Segment("，交通用"),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES[4:]),
                "TRANSPORTATION_MODE",
            ),
            Segment("，想吃"),
            option_segment(
                random_source.choice(FOOD_PREFERENCES[4:]), "FOOD_PREFERENCE"
            ),
            Segment("，攻略用"),
            option_segment(random_source.choice(LANGUAGES[2:]), "LANGUAGE"),
            Segment("。"),
        ]
    )


def build_english_partial_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build a partial English travel requirement example.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    first_interest, second_interest = choose_two(INTERESTS[:6], random_source)
    if random_source.random() < 0.5:
        timing_segment = option_segment(
            random_source.choice(TRIP_DATES[:2]), "TRIP_DATES"
        )
    else:
        timing_segment = option_segment(
            random_source.choice(TRIP_LENGTHS[:3]), "TRIP_LENGTH_DAYS"
        )
    return make_record(
        [
            Segment("Plan "),
            timing_segment,
            Segment(" in "),
            option_segment(random_source.choice(DESTINATIONS[:4]), "DESTINATION"),
            Segment(" for "),
            option_segment(random_source.choice(ADULTS[:3]), "ADULTS"),
            Segment(". Focus on "),
            option_segment(first_interest, "INTEREST"),
            Segment(", "),
            option_segment(second_interest, "INTEREST"),
            Segment(", "),
            option_segment(random_source.choice(BUDGETS[:5]), "BUDGET_LEVEL"),
            Segment(" hotels near "),
            option_segment(random_source.choice(HOTEL_AREAS[:4]), "HOTEL_AREA"),
            Segment(", and "),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES[:4]),
                "TRANSPORTATION_MODE",
            ),
            Segment("."),
        ]
    )


def build_chinese_partial_record(
    random_source: random.Random,
) -> RequirementTrainingRecord:
    """
    Build a partial Chinese travel requirement example.

    Args:
        random_source: The deterministic random source.

    Returns:
        The generated training record.
    """
    if random_source.random() < 0.5:
        timing_segment = option_segment(
            random_source.choice(TRIP_DATES[1:]), "TRIP_DATES"
        )
    else:
        timing_segment = option_segment(
            random_source.choice(TRIP_LENGTHS[3:]), "TRIP_LENGTH_DAYS"
        )
    return make_record(
        [
            Segment("帮我安排"),
            option_segment(random_source.choice(DESTINATIONS[4:]), "DESTINATION"),
            Segment("旅行，时间是"),
            timing_segment,
            Segment("，"),
            option_segment(random_source.choice(ADULTS[3:]), "ADULTS"),
            Segment("，预算"),
            option_segment(random_source.choice(BUDGETS[5:]), "BUDGET_LEVEL"),
            Segment("，节奏"),
            option_segment(random_source.choice(PACES[3:]), "TRAVEL_PACE"),
            Segment("，住在"),
            option_segment(random_source.choice(HOTEL_AREAS[4:]), "HOTEL_AREA"),
            Segment("，交通"),
            option_segment(
                random_source.choice(TRANSPORTATION_MODES[4:]),
                "TRANSPORTATION_MODE",
            ),
            Segment("。"),
        ]
    )


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
                Segment(", and "),
                Segment("transit", "TRANSPORTATION_MODE", "transit"),
                Segment("."),
            ]
        ),
        make_record(
            [
                Segment("想去"),
                Segment("悉尼", "DESTINATION", "悉尼"),
                Segment("玩"),
                Segment("三天", "TRIP_LENGTH_DAYS", 3),
                Segment("，"),
                Segment("两个人", "ADULTS", 2),
                Segment("，"),
                Segment("中等预算", "BUDGET_LEVEL", "medium"),
                Segment("，"),
                Segment("适中", "TRAVEL_PACE", "balanced"),
                Segment("节奏，喜欢"),
                Segment("自然", "INTEREST", "nature"),
                Segment("和"),
                Segment("美食", "INTEREST", "food"),
                Segment("，住"),
                Segment("市中心", "HOTEL_AREA", "市中心"),
                Segment("，坐"),
                Segment("公共交通", "TRANSPORTATION_MODE", "transit"),
                Segment("。"),
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
                    "2026-07-01 to 2026-07-05", "TRIP_DATES", "2026-07-01 to 2026-07-05"
                ),
                Segment(" for "),
                Segment("3 people", "ADULTS", 3),
                Segment(" and "),
                Segment("1 child", "CHILDREN", 1),
                Segment(". We prefer "),
                Segment("walking", "TRANSPORTATION_MODE", "walking"),
                Segment(", "),
                Segment("history", "INTEREST", "history"),
                Segment(", "),
                Segment("vegetarian food", "FOOD_PREFERENCE", "vegetarian food"),
                Segment(", and "),
                Segment("English", "LANGUAGE", "en"),
                Segment("."),
            ]
        ),
        make_record(
            [
                Segment("安排"),
                Segment("大阪", "DESTINATION", "大阪"),
                Segment("，日期"),
                Segment(
                    "2026-09-20到2026-09-24", "TRIP_DATES", "2026-09-20到2026-09-24"
                ),
                Segment("，"),
                Segment("三位成人", "ADULTS", 3),
                Segment("，"),
                Segment("不带孩子", "CHILDREN", 0),
                Segment("，预算"),
                Segment("经济型", "BUDGET_LEVEL", "low"),
                Segment("，节奏"),
                Segment("紧凑", "TRAVEL_PACE", "packed"),
                Segment("，喜欢"),
                Segment("购物", "INTEREST", "shopping"),
                Segment("，想吃"),
                Segment("拉面", "FOOD_PREFERENCE", "拉面"),
                Segment("，用"),
                Segment("中文", "LANGUAGE", "zh"),
                Segment("。"),
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
    train_count = max(1, int(len(records) * 0.7))
    validation_count = max(1, int(len(records) * 0.15))
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
    Validate generated JSONL files and slot coverage.

    Args:
        output_dir: The output directory.

    Returns:
        The number of records in each split.

    Raises:
        ValueError: Raised when required slot coverage or split separation fails.
    """
    split_counts: dict[str, int] = {}
    all_records: list[RequirementTrainingRecord] = []
    generated_texts: set[str] = set()
    for split_name in (*GENERATED_SPLITS, "reviewed_test"):
        split_records = load_jsonl(output_dir / f"{split_name}.jsonl")
        split_counts[split_name] = len(split_records)
        all_records.extend(split_records)
        if split_name in GENERATED_SPLITS:
            generated_texts.update(record.text for record in split_records)
    reviewed_texts = {
        record.text for record in load_jsonl(output_dir / "reviewed_test.jsonl")
    }
    overlapping_texts = generated_texts.intersection(reviewed_texts)
    if overlapping_texts:
        raise ValueError("reviewed test records must be separated from generated data")
    covered_fields = collect_covered_fields(all_records)
    missing_fields = sorted(set(REQUIRED_SLOT_FIELDS).difference(covered_fields))
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        raise ValueError(f"missing slot coverage: {joined_fields}")
    return split_counts


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
    records = generate_records(args.count, args.seed)
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
