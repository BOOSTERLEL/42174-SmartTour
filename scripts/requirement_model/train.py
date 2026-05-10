"""Train a DistilBERT token-classification model for requirement extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
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
    LABEL_TO_ID,
    RequirementTrainingRecord,
    load_jsonl,
)

DEFAULT_DATA_DIR = Path("data/requirement_model")
DEFAULT_OUTPUT_DIR = Path("models/requirement_model/latest")
DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"
QUICK_MODEL_NAME = "sshleifer/tiny-distilbert-base-cased"


class RequirementTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """
    PyTorch dataset that aligns word-level BIO labels to tokenizer subtokens.
    """

    def __init__(
        self,
        records: list[RequirementTrainingRecord],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        """
        Initialize the token-classification dataset.

        Args:
            records: The labeled requirement records.
            tokenizer: The Hugging Face tokenizer.
            max_length: The maximum encoded sequence length.
        """
        self.features = [
            encode_record(record, tokenizer, max_length) for record in records
        ]

    def __len__(self) -> int:
        """
        Return the dataset size.

        Returns:
            The number of encoded examples.
        """
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Return one encoded feature set.

        Args:
            index: The feature index.

        Returns:
            The encoded tensors.
        """
        return self.features[index]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the supervised requirement understanding model."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument(
        "--clearml-task-name",
        default="Requirement Model Training",
    )
    return parser.parse_args()


def encode_record(
    record: RequirementTrainingRecord, tokenizer: Any, max_length: int
) -> dict[str, torch.Tensor]:
    """
    Encode one labeled record for token-classification training.

    Args:
        record: The labeled requirement record.
        tokenizer: The Hugging Face tokenizer.
        max_length: The maximum encoded sequence length.

    Returns:
        The encoded tensors including aligned labels.
    """
    encoding = tokenizer(
        record.tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    label_ids: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:
            label_ids.append(LABEL_TO_ID[record.labels[word_id]])
        else:
            label_ids.append(-100)
        previous_word_id = word_id
    feature = {
        name: tensor.squeeze(0)
        for name, tensor in encoding.items()
        if name != "offset_mapping"
    }
    feature["labels"] = torch.tensor(label_ids, dtype=torch.long)
    return feature


def collate_features(
    features: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Stack encoded features into one training batch.

    Args:
        features: The encoded examples.

    Returns:
        The batch tensors.
    """
    return {
        key: torch.stack([feature[key] for feature in features]) for key in features[0]
    }


def load_training_records(
    data_dir: Path, quick: bool
) -> list[RequirementTrainingRecord]:
    """
    Load training records for normal or quick training.

    Args:
        data_dir: The dataset directory.
        quick: Whether to use a small smoke-test subset.

    Returns:
        The training records.
    """
    records = load_jsonl(data_dir / "train.jsonl")
    if quick:
        return records[: min(16, len(records))]
    return records


def resolve_device(device_name: str) -> torch.device:
    """
    Resolve the requested training device.

    Args:
        device_name: The requested device name.

    Returns:
        The PyTorch device to use.

    Raises:
        RuntimeError: Raised when CUDA is requested but unavailable.
    """
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Move one batch of tensors to the training device.

    Args:
        batch: The CPU batch tensors from the data loader.
        device: The training device.

    Returns:
        The batch tensors on the training device.
    """
    return {
        key: tensor.to(device, non_blocking=device.type == "cuda")
        for key, tensor in batch.items()
    }


def train_model(
    args: argparse.Namespace, tracker: ClearMlTracker | None = None
) -> Path:
    """
    Train and save a DistilBERT token-classification model.

    Args:
        args: The parsed command-line arguments.
        tracker: Optional ClearML tracker.

    Returns:
        The model output directory.
    """
    active_tracker = tracker or ClearMlTracker()
    torch.manual_seed(42174)
    model_name = QUICK_MODEL_NAME if args.quick else args.model_name
    output_dir = (
        Path("models/requirement_model/quick") if args.quick else args.output_dir
    )
    records = load_training_records(args.data_dir, args.quick)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    device = resolve_device(args.device)
    print(f"training_device={device}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    dataset = RequirementTokenDataset(records, tokenizer, args.max_length)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_features,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch_number in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        average_loss = total_loss / max(len(data_loader), 1)
        print(f"epoch {epoch_number}: loss={average_loss:.4f}")
        active_tracker.report_scalar(
            title="loss",
            series="train",
            value=average_loss,
            iteration=epoch_number,
        )
        if args.quick:
            break
    save_model_artifacts(output_dir, tokenizer, model, model_name, args.max_length)
    active_tracker.upload_artifact("model", str(output_dir))
    return output_dir


def save_model_artifacts(
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    model_name: str,
    max_length: int,
) -> None:
    """
    Save tokenizer, model, and requirement-model metadata.

    Args:
        output_dir: The output directory.
        tokenizer: The trained tokenizer.
        model: The trained model.
        model_name: The base model name.
        max_length: The maximum sequence length.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    metadata = {
        "base_model_name": model_name,
        "label_names": list(LABEL_NAMES),
        "max_length": max_length,
    }
    (output_dir / "requirement_model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """
    Train the requirement understanding model from JSONL data.
    """
    args = parse_args()
    tracker = initialize_clearml_task(
        is_enabled=args.clearml,
        task_name=args.clearml_task_name,
        project_name=args.clearml_project,
        task_type="training",
        tags=("requirement_model", "training"),
        configuration={
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "model_name": args.model_name,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "quick": args.quick,
        },
    )
    try:
        output_dir = train_model(args, tracker)
        print(f"saved requirement model artifact to {output_dir}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
