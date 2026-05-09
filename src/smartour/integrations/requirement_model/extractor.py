"""DistilBERT-backed supervised requirement extractor."""

import json
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
        encoding = tokenizer(
            message,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**encoding).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
        predictions: list[TokenPrediction] = []
        for token_index, offsets in enumerate(offset_mapping):
            start, end = int(offsets[0]), int(offsets[1])
            if start >= end:
                continue
            confidence_tensor, label_tensor = torch.max(
                probabilities[token_index], dim=-1
            )
            label_id = int(label_tensor.item())
            predictions.append(
                TokenPrediction(
                    label=self._label_for_id(label_id),
                    start=start,
                    end=end,
                    confidence=float(confidence_tensor.item()),
                )
            )
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
