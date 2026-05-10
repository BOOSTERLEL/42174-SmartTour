"""Evaluate a trained requirement understanding model."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.clearml_tracking import (
    DEFAULT_CLEARML_PROJECT,
    ClearMlTracker,
    initialize_clearml_task,
)
from scripts.requirement_model.schema import (
    ID_TO_LABEL,
    LABEL_NAMES,
    RequirementSlots,
    load_jsonl,
)
from smartour.integrations.requirement_model.normalizer import (
    normalize_span as normalize_runtime_span,
)

DEFAULT_DATA_DIR = Path("data/requirement_model")
DEFAULT_MODEL_DIR = Path("models/requirement_model/quick")
SLOT_FIELDS: tuple[str, ...] = (
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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the requirement understanding model."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument(
        "--clearml-task-name",
        default="Requirement Model Evaluation",
    )
    return parser.parse_args()


def predict_labels(
    tokens: list[str],
    tokenizer: Any,
    model: Any,
    max_length: int,
) -> list[str]:
    """
    Predict one BIO label per source token.

    Args:
        tokens: The pre-tokenized source words.
        tokenizer: The Hugging Face tokenizer.
        model: The token-classification model.
        max_length: The maximum encoded sequence length.

    Returns:
        The predicted BIO labels aligned to the source tokens.
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    with torch.no_grad():
        logits = model(**encoding).logits[0]
    labels = ["O"] * len(tokens)
    seen_word_ids: set[int] = set()
    for encoded_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids or word_id >= len(tokens):
            continue
        predicted_id = int(torch.argmax(logits[encoded_index]).item())
        labels[word_id] = ID_TO_LABEL[predicted_id]
        seen_word_ids.add(word_id)
    return labels


def decode_slots(tokens: list[str], labels: list[str]) -> RequirementSlots:
    """
    Decode BIO token labels into canonical requirement slots.

    Args:
        tokens: The source tokens.
        labels: The BIO labels.

    Returns:
        The decoded slots.
    """
    slot_values: dict[str, Any] = {"interests": [], "food_preferences": []}
    current_slot: str | None = None
    current_tokens: list[str] = []
    for token, label in zip(tokens, labels, strict=False):
        if label == "O":
            flush_span(slot_values, current_slot, current_tokens)
            current_slot = None
            current_tokens = []
            continue
        prefix, slot_name = label.split("-", maxsplit=1)
        if prefix == "B" or current_slot != slot_name:
            flush_span(slot_values, current_slot, current_tokens)
            current_slot = slot_name
            current_tokens = [token]
        else:
            current_tokens.append(token)
    flush_span(slot_values, current_slot, current_tokens)
    return RequirementSlots.model_validate(slot_values)


def flush_span(
    slot_values: dict[str, Any],
    slot_name: str | None,
    tokens: list[str],
) -> None:
    """
    Normalize and store one decoded slot span.

    Args:
        slot_values: The mutable decoded slots dictionary.
        slot_name: The decoded slot name.
        tokens: The decoded slot tokens.
    """
    if slot_name is None or not tokens:
        return
    normalized_value = normalize_span(slot_name, join_tokens(tokens))
    if normalized_value is None:
        return
    if slot_name == "INTEREST":
        append_unique(slot_values, "interests", str(normalized_value))
        return
    if slot_name == "FOOD_PREFERENCE":
        append_unique(slot_values, "food_preferences", str(normalized_value))
        return
    field_name = slot_name.lower()
    if field_name == "trip_dates":
        slot_values["trip_dates"] = str(normalized_value)
    elif field_name == "trip_length_days":
        slot_values["trip_length_days"] = normalized_value
    elif field_name == "budget_level":
        slot_values["budget_level"] = str(normalized_value)
    elif field_name == "travel_pace":
        slot_values["travel_pace"] = str(normalized_value)
    elif field_name == "hotel_area":
        slot_values["hotel_area"] = str(normalized_value)
    elif field_name == "transportation_mode":
        slot_values["transportation_mode"] = str(normalized_value)
    else:
        slot_values[field_name] = normalized_value


def append_unique(slot_values: dict[str, Any], field_name: str, value: str) -> None:
    """
    Append a list slot value once.

    Args:
        slot_values: The mutable decoded slots dictionary.
        field_name: The list slot field name.
        value: The canonical value.
    """
    slot_values.setdefault(field_name, [])
    if value not in slot_values[field_name]:
        slot_values[field_name].append(value)


def join_tokens(tokens: list[str]) -> str:
    """
    Join tokens into a readable span.

    Args:
        tokens: The span tokens.

    Returns:
        The joined span text.
    """
    if any(re.search(r"[\u4e00-\u9fff]", token) for token in tokens):
        return "".join(tokens)
    span = " ".join(tokens)
    return (
        span.replace(" - ", "-")
        .replace(" / ", "/")
        .replace(" ,", ",")
        .replace(" .", ".")
    )


def normalize_span(slot_name: str, value: str) -> str | int | None:
    """
    Normalize one decoded span to a canonical slot value.

    Args:
        slot_name: The decoded slot name.
        value: The raw decoded span.

    Returns:
        The canonical value, or None when unsupported.
    """
    normalized_value = value.strip()
    if slot_name == "TRIP_LENGTH_DAYS":
        return normalize_number(normalized_value)
    if slot_name in {"ADULTS", "CHILDREN"}:
        if "no" in normalized_value.lower() or "不带" in normalized_value:
            return 0
        return normalize_number(normalized_value)
    return normalize_runtime_span(slot_name, normalized_value)


def normalize_number(value: str) -> int | None:
    """
    Normalize Arabic or common Chinese numbers.

    Args:
        value: The raw number phrase.

    Returns:
        The integer value when detected.
    """
    digit_match = re.search(r"\d{1,2}", value)
    if digit_match:
        return int(digit_match.group(0))
    for text_value, number_value in CHINESE_NUMBER_VALUES.items():
        if text_value in value:
            return number_value
    return None


def compute_metrics(
    gold_records: list[RequirementSlots],
    predicted_records: list[RequirementSlots],
    gold_labels: list[list[str]],
    predicted_labels: list[list[str]],
) -> dict[str, float]:
    """
    Compute slot accuracy, token F1, and exact match.

    Args:
        gold_records: The gold structured slots.
        predicted_records: The predicted structured slots.
        gold_labels: The gold BIO labels.
        predicted_labels: The predicted BIO labels.

    Returns:
        The metric values.
    """
    token_metrics = compute_token_f1(gold_labels, predicted_labels)
    slot_matches = 0
    slot_total = 0
    exact_matches = 0
    for gold_slots, predicted_slots in zip(
        gold_records, predicted_records, strict=True
    ):
        gold_values = normalize_slot_dict(gold_slots.model_dump())
        predicted_values = normalize_slot_dict(predicted_slots.model_dump())
        if gold_values == predicted_values:
            exact_matches += 1
        for field_name in SLOT_FIELDS:
            slot_total += 1
            if gold_values[field_name] == predicted_values[field_name]:
                slot_matches += 1
    return {
        "slot_accuracy": slot_matches / max(slot_total, 1),
        "micro_f1": token_metrics["micro_f1"],
        "macro_f1": token_metrics["macro_f1"],
        "exact_match_accuracy": exact_matches / max(len(gold_records), 1),
    }


def normalize_slot_dict(values: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize list field ordering for exact match comparison.

    Args:
        values: The slot values.

    Returns:
        The normalized slot values.
    """
    normalized_values = dict(values)
    normalized_values["interests"] = sorted(normalized_values["interests"])
    normalized_values["food_preferences"] = sorted(
        normalized_values["food_preferences"]
    )
    return normalized_values


def compute_token_f1(
    gold_labels: list[list[str]], predicted_labels: list[list[str]]
) -> dict[str, float]:
    """
    Compute micro and macro F1 for non-O BIO labels.

    Args:
        gold_labels: The gold BIO labels.
        predicted_labels: The predicted BIO labels.

    Returns:
        The F1 metric values.
    """
    true_positives: Counter[str] = Counter()
    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    for gold_sequence, predicted_sequence in zip(
        gold_labels, predicted_labels, strict=True
    ):
        for gold_label, predicted_label in zip(
            gold_sequence, predicted_sequence, strict=False
        ):
            if gold_label == predicted_label and gold_label != "O":
                true_positives[gold_label] += 1
            elif gold_label != predicted_label:
                if predicted_label != "O":
                    false_positives[predicted_label] += 1
                if gold_label != "O":
                    false_negatives[gold_label] += 1
    labels = [label_name for label_name in LABEL_NAMES if label_name != "O"]
    total_true_positives = sum(true_positives.values())
    total_false_positives = sum(false_positives.values())
    total_false_negatives = sum(false_negatives.values())
    micro_f1 = f1_score(
        total_true_positives,
        total_false_positives,
        total_false_negatives,
    )
    macro_f1_values = [
        f1_score(
            true_positives[label_name],
            false_positives[label_name],
            false_negatives[label_name],
        )
        for label_name in labels
    ]
    macro_f1 = sum(macro_f1_values) / max(len(macro_f1_values), 1)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


def f1_score(
    true_positive_count: int,
    false_positive_count: int,
    false_negative_count: int,
) -> float:
    """
    Compute F1 from confusion counts.

    Args:
        true_positive_count: The true-positive count.
        false_positive_count: The false-positive count.
        false_negative_count: The false-negative count.

    Returns:
        The F1 score.
    """
    denominator = (
        (2 * true_positive_count) + false_positive_count + false_negative_count
    )
    if denominator == 0:
        return 0.0
    return (2 * true_positive_count) / denominator


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    """
    Evaluate the trained model on one dataset split.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The metric values.
    """
    records = load_jsonl(args.data_dir / f"{args.split}.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()
    gold_slots: list[RequirementSlots] = []
    predicted_slots: list[RequirementSlots] = []
    gold_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []
    for record in records:
        labels = predict_labels(record.tokens, tokenizer, model, args.max_length)
        gold_slots.append(record.slots)
        predicted_slots.append(decode_slots(record.tokens, labels))
        gold_labels.append(record.labels)
        predicted_labels.append(labels)
    return compute_metrics(gold_slots, predicted_slots, gold_labels, predicted_labels)


def report_metrics_to_clearml(
    tracker: ClearMlTracker, metrics: dict[str, float]
) -> None:
    """
    Report evaluation metrics to ClearML.

    Args:
        tracker: The optional ClearML tracker.
        metrics: The computed evaluation metrics.
    """
    if not tracker.is_enabled:
        return
    for metric_name, metric_value in sorted(metrics.items()):
        tracker.report_scalar(
            title="evaluation",
            series=metric_name,
            value=metric_value,
            iteration=0,
        )
    tracker.upload_artifact("metrics", dict(metrics))


def main() -> None:
    """
    Print evaluation metrics for a trained requirement model.
    """
    args = parse_args()
    tracker = initialize_clearml_task(
        is_enabled=args.clearml,
        task_name=args.clearml_task_name,
        project_name=args.clearml_project,
        task_type="testing",
        tags=("requirement_model", "evaluation"),
        configuration={
            "data_dir": str(args.data_dir),
            "model_dir": str(args.model_dir),
            "split": args.split,
            "max_length": args.max_length,
        },
    )
    try:
        metrics = evaluate(args)
        report_metrics_to_clearml(tracker, metrics)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}={metric_value:.4f}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
