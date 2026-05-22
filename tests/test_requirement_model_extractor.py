"""Tests for the supervised requirement model extractor."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from smartour.integrations.requirement_model.decoder import (
    RequirementSlotSpan,
    TokenPrediction,
    decode_bio_spans,
)
from smartour.integrations.requirement_model.extractor import (
    RequirementModelExtractor,
    tokenize_source_text,
)
from smartour.integrations.requirement_model.labels import LABEL_NAMES, LABEL_TO_ID
from smartour.integrations.requirement_model.normalizer import (
    spans_to_requirement_update,
)


class StaticRequirementModelExtractor(RequirementModelExtractor):
    """
    Requirement model extractor that returns fixed spans for tests.
    """

    def __init__(self, spans: list[RequirementSlotSpan]) -> None:
        """
        Initialize the static test extractor.

        Args:
            spans: The fixed spans to return.
        """
        super().__init__(model_path=Path("missing-test-model"))
        self.spans = spans

    def predict_spans(self, message: str) -> list[RequirementSlotSpan]:
        """
        Return fixed spans without loading a model.

        Args:
            message: The raw user message.

        Returns:
            The fixed spans.
        """
        return self.spans


class FakeEncoding(dict[str, torch.Tensor]):
    """
    Minimal tokenizer encoding with Hugging Face-style word IDs.
    """

    def __init__(self, word_ids: list[int | None]) -> None:
        """
        Initialize the fake encoding.

        Args:
            word_ids: The encoded-token to source-word mapping.
        """
        super().__init__(
            {"input_ids": torch.zeros((1, len(word_ids)), dtype=torch.long)}
        )
        self.stored_word_ids = word_ids

    def word_ids(self, batch_index: int) -> list[int | None]:
        """
        Return encoded-token to source-word mappings.

        Args:
            batch_index: The batch index requested by the caller.

        Returns:
            The fake word ID sequence.
        """
        return self.stored_word_ids


class FakeTokenizer:
    """
    Minimal tokenizer that records source tokens and returns fixed word IDs.
    """

    def __init__(self, word_ids: list[int | None]) -> None:
        """
        Initialize the fake tokenizer.

        Args:
            word_ids: The encoded-token to source-word mapping.
        """
        self.word_id_values = word_ids
        self.received_tokens: list[str] = []
        self.did_receive_split_tokens = False

    def __call__(
        self,
        tokens: list[str],
        is_split_into_words: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> FakeEncoding:
        """
        Return a deterministic fake encoding.

        Args:
            tokens: The source tokens passed to the tokenizer.
            is_split_into_words: Whether the input is pre-tokenized.
            truncation: Whether truncation was requested.
            max_length: The maximum encoded sequence length.
            return_tensors: The requested tensor framework.

        Returns:
            A fake encoding.
        """
        self.received_tokens = tokens
        self.did_receive_split_tokens = is_split_into_words
        return FakeEncoding(self.word_id_values)


class FakeModel:
    """
    Minimal token classification model that returns fixed BIO labels.
    """

    def __init__(self, labels: list[str]) -> None:
        """
        Initialize the fake model.

        Args:
            labels: The label emitted for each encoded token.
        """
        self.labels = labels

    def __call__(self, **encoding: Any) -> SimpleNamespace:
        """
        Return fixed logits for each encoded token.

        Args:
            encoding: The fake model inputs.

        Returns:
            An object with a logits tensor.
        """
        logits = torch.full(
            (1, len(self.labels), len(LABEL_NAMES)),
            fill_value=-10.0,
        )
        for label_index, label in enumerate(self.labels):
            logits[0, label_index, LABEL_TO_ID[label]] = 10.0
        return SimpleNamespace(logits=logits)


def test_decode_bio_spans_repairs_invalid_i_label() -> None:
    """
    Verify that an I label without an active span starts a valid span.
    """
    spans = decode_bio_spans(
        "Tokyo relaxed",
        [
            TokenPrediction("I-DESTINATION", 0, 5, 0.91),
            TokenPrediction("O", 5, 6, 0.99),
            TokenPrediction("B-TRAVEL_PACE", 6, 13, 0.94),
        ],
        confidence_threshold=0.5,
    )

    assert spans == [
        RequirementSlotSpan("DESTINATION", "Tokyo", 0.91),
        RequirementSlotSpan("TRAVEL_PACE", "relaxed", 0.94),
    ]


def test_decode_bio_spans_omits_low_confidence_span() -> None:
    """
    Verify that low-confidence model predictions are not returned.
    """
    spans = decode_bio_spans(
        "Tokyo",
        [TokenPrediction("B-DESTINATION", 0, 5, 0.2)],
        confidence_threshold=0.5,
    )

    assert spans == []


def test_tokenize_source_text_preserves_date_and_word_offsets() -> None:
    """
    Verify source tokenization keeps full words and date tokens.
    """
    source_tokens = tokenize_source_text(
        "relaxed 2026-11-01 to 2026-11-04 vegetarian food"
    )

    assert [(token.text, token.start, token.end) for token in source_tokens] == [
        ("relaxed", 0, 7),
        ("2026-11-01", 8, 18),
        ("to", 19, 21),
        ("2026-11-04", 22, 32),
        ("vegetarian", 33, 43),
        ("food", 44, 48),
    ]


def test_requirement_model_extractor_uses_word_level_runtime_spans() -> None:
    """
    Verify runtime inference decodes source words instead of subword fragments.
    """
    message = "relaxed 2026-11-01 to 2026-11-04 vegetarian food"
    word_ids = [None, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, None]
    labels = [
        "O",
        "B-TRAVEL_PACE",
        "I-TRAVEL_PACE",
        "B-TRIP_DATES",
        "I-TRIP_DATES",
        "I-TRIP_DATES",
        "I-TRIP_DATES",
        "I-TRIP_DATES",
        "B-FOOD_PREFERENCE",
        "I-FOOD_PREFERENCE",
        "I-FOOD_PREFERENCE",
        "I-FOOD_PREFERENCE",
        "O",
    ]
    tokenizer = FakeTokenizer(word_ids)
    extractor = RequirementModelExtractor(
        model_path=Path("missing-test-model"),
        tokenizer=tokenizer,
        model=FakeModel(labels),
    )

    spans = extractor.predict_spans(message)

    assert tokenizer.did_receive_split_tokens
    assert tokenizer.received_tokens == [
        "relaxed",
        "2026-11-01",
        "to",
        "2026-11-04",
        "vegetarian",
        "food",
    ]
    assert spans == [
        RequirementSlotSpan("TRAVEL_PACE", "relaxed", 1.0),
        RequirementSlotSpan(
            "TRIP_DATES",
            "2026-11-01 to 2026-11-04",
            1.0,
        ),
        RequirementSlotSpan("FOOD_PREFERENCE", "vegetarian food", 1.0),
    ]


def test_spans_to_requirement_update_normalizes_supported_slots() -> None:
    """
    Verify that decoded spans map into the canonical requirement update model.
    """
    update = spans_to_requirement_update(
        [
            RequirementSlotSpan("DESTINATION", "Tokyo", 0.9),
            RequirementSlotSpan("TRIP_LENGTH_DAYS", "4 days", 0.9),
            RequirementSlotSpan("ADULTS", "2 adults", 0.9),
            RequirementSlotSpan("CHILDREN", "1 child", 0.9),
            RequirementSlotSpan("BUDGET_LEVEL", "moderate", 0.9),
            RequirementSlotSpan("TRAVEL_PACE", "relaxed", 0.9),
            RequirementSlotSpan("INTEREST", "food", 0.9),
            RequirementSlotSpan("INTEREST", "博物馆", 0.9),
            RequirementSlotSpan("HOTEL_AREA", "Shinjuku", 0.9),
            RequirementSlotSpan("TRANSPORTATION_MODE", "subway", 0.9),
            RequirementSlotSpan("FOOD_PREFERENCE", "ramen", 0.9),
            RequirementSlotSpan("LANGUAGE", "Chinese", 0.9),
        ]
    )

    assert update.destination == "Tokyo"
    assert update.trip_length_days == 4
    assert update.travelers is not None
    assert update.travelers.adults == 2
    assert update.travelers.children == 1
    assert update.budget_level == "medium"
    assert update.travel_pace == "relaxed"
    assert update.interests == ["food", "museums"]
    assert update.hotel_area == "Shinjuku"
    assert update.transportation_mode == "transit"
    assert update.food_preferences == ["ramen"]
    assert update.language == "zh"


def test_spans_to_requirement_update_omits_unknown_canonical_values() -> None:
    """
    Verify that unsupported scalar values do not corrupt requirement state.
    """
    update = spans_to_requirement_update(
        [
            RequirementSlotSpan("BUDGET_LEVEL", "surprising", 0.9),
            RequirementSlotSpan("TRAVEL_PACE", "chaotic", 0.9),
        ]
    )

    assert update.budget_level is None
    assert update.travel_pace is None


def test_spans_to_requirement_update_omits_lodging_words_as_interests() -> None:
    """
    Verify that hotel-area wording does not become a travel interest.
    """
    update = spans_to_requirement_update(
        [
            RequirementSlotSpan("INTEREST", "hotel", 0.9),
            RequirementSlotSpan("INTEREST", "area", 0.9),
            RequirementSlotSpan("INTEREST", "museums", 0.9),
        ]
    )

    assert update.interests == ["museums"]


def test_spans_to_requirement_update_normalizes_english_number_words() -> None:
    """
    Verify that English number words are normalized for traveler counts.
    """
    update = spans_to_requirement_update(
        [
            RequirementSlotSpan("ADULTS", "Two adults", 0.9),
            RequirementSlotSpan("CHILDREN", "three children", 0.9),
        ]
    )

    assert update.travelers is not None
    assert update.travelers.adults == 2
    assert update.travelers.children == 3


def test_requirement_model_extractor_preserves_contract() -> None:
    """
    Verify that the extractor returns a TravelRequirementUpdate.
    """
    extractor = StaticRequirementModelExtractor(
        [
            RequirementSlotSpan("DESTINATION", "Sydney", 0.9),
            RequirementSlotSpan("TRANSPORTATION_MODE", "public transit", 0.9),
        ]
    )

    update = extractor.extract("message")

    assert update.destination == "Sydney"
    assert update.transportation_mode == "transit"
