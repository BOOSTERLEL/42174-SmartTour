"""BIO span decoding for requirement model predictions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenPrediction:
    """
    One token-classification prediction with source text offsets.
    """

    label: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True, slots=True)
class RequirementSlotSpan:
    """
    One decoded requirement slot span.
    """

    slot_name: str
    text: str
    confidence: float


def decode_bio_spans(
    text: str,
    predictions: list[TokenPrediction],
    confidence_threshold: float,
) -> list[RequirementSlotSpan]:
    """
    Decode token BIO predictions into slot spans.

    Args:
        text: The original source text.
        predictions: Token predictions with character offsets.
        confidence_threshold: Minimum average confidence required for a span.

    Returns:
        The decoded requirement slot spans.
    """
    spans: list[RequirementSlotSpan] = []
    current_slot: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_confidences: list[float] = []
    for prediction in predictions:
        if prediction.label == "O" or prediction.start >= prediction.end:
            append_current_span(
                spans,
                text,
                current_slot,
                current_start,
                current_end,
                current_confidences,
                confidence_threshold,
            )
            current_slot = None
            current_start = None
            current_end = None
            current_confidences = []
            continue
        prefix, slot_name = split_label(prediction.label)
        should_start_span = (
            prefix == "B" or current_slot is None or current_slot != slot_name
        )
        if should_start_span:
            append_current_span(
                spans,
                text,
                current_slot,
                current_start,
                current_end,
                current_confidences,
                confidence_threshold,
            )
            current_slot = slot_name
            current_start = prediction.start
            current_end = prediction.end
            current_confidences = [prediction.confidence]
        else:
            current_end = prediction.end
            current_confidences.append(prediction.confidence)
    append_current_span(
        spans,
        text,
        current_slot,
        current_start,
        current_end,
        current_confidences,
        confidence_threshold,
    )
    return spans


def split_label(label: str) -> tuple[str, str]:
    """
    Split a BIO label into prefix and slot name.

    Args:
        label: The BIO label.

    Returns:
        The BIO prefix and slot name.
    """
    if "-" not in label:
        return "O", ""
    prefix, slot_name = label.split("-", maxsplit=1)
    return prefix, slot_name


def append_current_span(
    spans: list[RequirementSlotSpan],
    text: str,
    slot_name: str | None,
    start: int | None,
    end: int | None,
    confidences: list[float],
    confidence_threshold: float,
) -> None:
    """
    Append the active span if it is valid and confident enough.

    Args:
        spans: The mutable decoded spans list.
        text: The original source text.
        slot_name: The active slot name.
        start: The active span start offset.
        end: The active span end offset.
        confidences: Token confidences in the active span.
        confidence_threshold: Minimum average confidence required for a span.
    """
    if slot_name is None or start is None or end is None or start >= end:
        return
    confidence = sum(confidences) / max(len(confidences), 1)
    if confidence < confidence_threshold:
        return
    span_text = text[start:end].strip(" ,.;，。")
    if not span_text:
        return
    spans.append(
        RequirementSlotSpan(
            slot_name=slot_name,
            text=span_text,
            confidence=confidence,
        )
    )
