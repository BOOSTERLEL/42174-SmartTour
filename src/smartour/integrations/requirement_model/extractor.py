"""DistilBERT-backed supervised requirement extractor."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartour.domain.requirement import TravelRequirementUpdate
from smartour.integrations.requirement_model.decoder import (
    RequirementSlotSpan,
    TokenPrediction,
    decode_bio_spans,
)
from smartour.integrations.requirement_model.labels import ID_TO_LABEL
from smartour.integrations.requirement_model.normalizer import (
    spans_to_requirement_update,
)

DEFAULT_MAX_LENGTH = 192
SOURCE_TOKEN_PATTERN = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u4e00-\u9fff]|"
    r"[^\s]"
)


@dataclass(frozen=True, slots=True)
class SourceToken:
    """
    One model source token with offsets into the original message.
    """

    text: str
    start: int
    end: int


class RequirementModelExtractor:
    """
    Extracts travel requirements with a supervised token-classification model.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.35,
        max_length: int | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        """
        Initialize the requirement model extractor.

        Args:
            model_path: The local Hugging Face model artifact directory.
            confidence_threshold: Minimum average span confidence to accept.
            max_length: Optional maximum encoded sequence length override.
            tokenizer: Optional tokenizer used by tests.
            model: Optional model used by tests.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length or self._load_max_length()
        self.tokenizer = tokenizer
        self.model = model

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Extract requirement updates from a user message.

        Args:
            message: The raw user message.

        Returns:
            The extracted requirement update.
        """
        spans = self.predict_spans(message)
        return spans_to_requirement_update(spans)

    def predict_spans(self, message: str) -> list[RequirementSlotSpan]:
        """
        Predict decoded slot spans from a user message.

        Args:
            message: The raw user message.

        Returns:
            The decoded slot spans.
        """
        if not message.strip():
            return []
        self._ensure_model_loaded()
        tokenizer = self.tokenizer
        model = self.model
        if tokenizer is None or model is None:
            raise RuntimeError("Requirement model failed to load")
        torch = self._torch()
        source_tokens = tokenize_source_text(message)
        if not source_tokens:
            return []
        encoding = tokenizer(
            [token.text for token in source_tokens],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        with torch.no_grad():
            logits = model(**encoding).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
        predictions: list[TokenPrediction] = []
        seen_word_ids: set[int] = set()
        for encoded_index, word_id in enumerate(word_ids):
            if (
                word_id is None
                or word_id in seen_word_ids
                or word_id >= len(source_tokens)
            ):
                continue
            source_token = source_tokens[word_id]
            confidence_tensor, label_tensor = torch.max(
                probabilities[encoded_index], dim=-1
            )
            label_id = int(label_tensor.item())
            predictions.append(
                TokenPrediction(
                    label=self._label_for_id(label_id),
                    start=source_token.start,
                    end=source_token.end,
                    confidence=float(confidence_tensor.item()),
                )
            )
            seen_word_ids.add(word_id)
        return decode_bio_spans(
            message,
            predictions,
            confidence_threshold=self.confidence_threshold,
        )

    def _load_max_length(self) -> int:
        """
        Load the maximum sequence length from the local model metadata.

        Returns:
            The maximum sequence length.
        """
        config_path = self.model_path / "requirement_model_config.json"
        if not config_path.exists():
            return DEFAULT_MAX_LENGTH
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        return int(metadata.get("max_length", DEFAULT_MAX_LENGTH))

    def _ensure_model_loaded(self) -> None:
        """
        Lazily load the tokenizer and token-classification model.
        """
        if self.tokenizer is not None and self.model is not None:
            return
        transformers = self._transformers()
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
            )
        if self.model is None:
            self.model = transformers.AutoModelForTokenClassification.from_pretrained(
                self.model_path
            )
            self.model.eval()

    def _label_for_id(self, label_id: int) -> str:
        """
        Return a BIO label for a model label ID.

        Args:
            label_id: The model label ID.

        Returns:
            The BIO label name.
        """
        if self.model is not None:
            id2label = getattr(getattr(self.model, "config", None), "id2label", None)
            if id2label and label_id in id2label:
                return str(id2label[label_id])
        return ID_TO_LABEL.get(label_id, "O")

    def _torch(self) -> Any:
        """
        Import torch lazily.

        Returns:
            The torch module.
        """
        import torch

        return torch

    def _transformers(self) -> Any:
        """
        Import transformers lazily.

        Returns:
            The transformers module.
        """
        import transformers

        return transformers


def tokenize_source_text(text: str) -> list[SourceToken]:
    """
    Tokenize source text with offsets matching the model training labels.

    Args:
        text: The raw user message.

    Returns:
        The source tokens and their original character offsets.
    """
    return [
        SourceToken(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in SOURCE_TOKEN_PATTERN.finditer(text)
    ]
