"""Train a DistilBERT token-classification model for requirement extraction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from scripts.requirement_model.evaluate import evaluate_records
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
DEFAULT_CONVERGENCE_METRIC = "macro_f1"


@dataclass(frozen=True, slots=True)
class ConvergenceSettings:
    """
    Early-stopping settings for validation-driven training.
    """

    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float
    monitor_metric: str = DEFAULT_CONVERGENCE_METRIC


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """
    Metrics recorded after one training epoch.
    """

    epoch: int
    train_loss: float
    validation_metrics: dict[str, float]


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
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--validation-split", default="validation")
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
    parser.add_argument("--clearml-model-report", action="store_true")
    parser.add_argument("--clearml-register-model", action="store_true")
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


def build_convergence_settings(args: argparse.Namespace) -> ConvergenceSettings:
    """
    Build validated early-stopping settings from CLI arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The validated convergence settings.

    Raises:
        ValueError: Raised when the settings are inconsistent.
    """
    max_epochs = args.max_epochs if args.max_epochs is not None else args.epochs
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    if args.min_epochs < 1:
        raise ValueError("min_epochs must be at least 1")
    if args.min_epochs > max_epochs:
        raise ValueError("min_epochs cannot be greater than max_epochs")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    if args.min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    return ConvergenceSettings(
        max_epochs=max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )


def load_validation_records(
    data_dir: Path, split_name: str, quick: bool
) -> list[RequirementTrainingRecord]:
    """
    Load validation records when full convergence training is enabled.

    Args:
        data_dir: The dataset directory.
        split_name: The validation split name.
        quick: Whether quick smoke training is enabled.

    Returns:
        Validation records, or an empty list for quick smoke training.
    """
    if quick:
        return []
    return load_jsonl(data_dir / f"{split_name}.jsonl")


def is_metric_improved(
    metric_value: float, best_metric_value: float | None, min_delta: float
) -> bool:
    """
    Return whether a monitored metric improved enough to replace the best model.

    Args:
        metric_value: The current metric value.
        best_metric_value: The best metric value seen so far.
        min_delta: The minimum required improvement.

    Returns:
        Whether the metric improved.
    """
    if best_metric_value is None:
        return True
    return metric_value > best_metric_value + min_delta


def should_stop_for_convergence(
    epoch_number: int,
    best_epoch: int | None,
    settings: ConvergenceSettings,
) -> bool:
    """
    Return whether early stopping should stop after an epoch.

    Args:
        epoch_number: The completed epoch number.
        best_epoch: The epoch containing the current best validation metric.
        settings: The convergence settings.

    Returns:
        Whether training should stop.
    """
    if epoch_number < settings.min_epochs or best_epoch is None:
        return False
    return epoch_number - best_epoch >= settings.patience


def report_epoch_metrics_to_clearml(
    tracker: ClearMlTracker,
    epoch_number: int,
    train_loss: float,
    validation_metrics: dict[str, float],
) -> None:
    """
    Report one epoch of training and validation metrics.

    Args:
        tracker: The optional ClearML tracker.
        epoch_number: The completed epoch number.
        train_loss: The average training loss.
        validation_metrics: Validation metrics keyed by metric name.
    """
    tracker.report_scalar(
        title="loss",
        series="train",
        value=train_loss,
        iteration=epoch_number,
    )
    for metric_name, metric_value in sorted(validation_metrics.items()):
        tracker.report_scalar(
            title="validation",
            series=metric_name,
            value=metric_value,
            iteration=epoch_number,
        )


def build_training_report(
    model_name: str,
    output_dir: Path,
    settings: ConvergenceSettings,
    history: list[EpochMetrics],
    best_epoch: int | None,
    best_metric_value: float | None,
    stopped_early: bool,
) -> dict[str, Any]:
    """
    Build a JSON-compatible training report.

    Args:
        model_name: The base model name.
        output_dir: The saved model directory.
        settings: The convergence settings used for training.
        history: Per-epoch metric history.
        best_epoch: The epoch selected as the best checkpoint.
        best_metric_value: The best monitored validation metric value.
        stopped_early: Whether early stopping ended training before max epochs.

    Returns:
        A JSON-compatible report.
    """
    return {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "convergence": {
            "max_epochs": settings.max_epochs,
            "min_epochs": settings.min_epochs,
            "patience": settings.patience,
            "min_delta": settings.min_delta,
            "monitor_metric": settings.monitor_metric,
        },
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric_value,
        "stopped_early": stopped_early,
        "history": [
            {
                "epoch": epoch_metrics.epoch,
                "train_loss": epoch_metrics.train_loss,
                "validation_metrics": dict(epoch_metrics.validation_metrics),
            }
            for epoch_metrics in history
        ],
    }


def write_training_report(output_dir: Path, report: dict[str, Any]) -> None:
    """
    Write a training report into the model artifact directory.

    Args:
        output_dir: The model artifact directory.
        report: The JSON-compatible report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_epoch_metrics(
    epoch_metrics: EpochMetrics, monitor_metric: str
) -> str:
    """
    Format one epoch's metrics for command-line output.

    Args:
        epoch_metrics: The metrics recorded for the epoch.
        monitor_metric: The validation metric used for convergence.

    Returns:
        A compact one-line metric summary.
    """
    summary = f"epoch {epoch_metrics.epoch}: loss={epoch_metrics.train_loss:.4f}"
    if monitor_metric in epoch_metrics.validation_metrics:
        metric_value = epoch_metrics.validation_metrics[monitor_metric]
        summary += f" validation_{monitor_metric}={metric_value:.4f}"
    return summary


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
    convergence_settings = build_convergence_settings(args)
    records = load_training_records(args.data_dir, args.quick)
    validation_records = load_validation_records(
        args.data_dir, args.validation_split, args.quick
    )
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
    history: list[EpochMetrics] = []
    best_epoch: int | None = None
    best_metric_value: float | None = None
    stopped_early = False
    model.train()
    for epoch_number in range(1, convergence_settings.max_epochs + 1):
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
        validation_metrics: dict[str, float] = {}
        if validation_records:
            validation_metrics, _diagnostics = evaluate_records(
                records=validation_records,
                tokenizer=tokenizer,
                model=model,
                max_length=args.max_length,
                device=device,
            )
            model.train()
            metric_value = validation_metrics[convergence_settings.monitor_metric]
            if is_metric_improved(
                metric_value,
                best_metric_value,
                convergence_settings.min_delta,
            ):
                best_epoch = epoch_number
                best_metric_value = metric_value
                save_model_artifacts(
                    output_dir,
                    tokenizer,
                    model,
                    model_name,
                    args.max_length,
                )
        epoch_metrics = EpochMetrics(
            epoch=epoch_number,
            train_loss=average_loss,
            validation_metrics=validation_metrics,
        )
        history.append(epoch_metrics)
        report_epoch_metrics_to_clearml(
            active_tracker,
            epoch_number,
            average_loss,
            validation_metrics,
        )
        print(format_epoch_metrics(epoch_metrics, convergence_settings.monitor_metric))
        if args.quick:
            break
        if validation_records and should_stop_for_convergence(
            epoch_number,
            best_epoch,
            convergence_settings,
        ):
            stopped_early = True
            best_metric_text = (
                "unknown" if best_metric_value is None else f"{best_metric_value:.4f}"
            )
            print(
                "early_stopping="
                f"epoch_{epoch_number}; best_epoch={best_epoch}; "
                f"best_{convergence_settings.monitor_metric}="
                f"{best_metric_text}"
            )
            break
    if not validation_records:
        best_epoch = len(history) if history else None
        save_model_artifacts(output_dir, tokenizer, model, model_name, args.max_length)
    training_report = build_training_report(
        model_name=model_name,
        output_dir=output_dir,
        settings=convergence_settings,
        history=history,
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        stopped_early=stopped_early,
    )
    write_training_report(output_dir, training_report)
    active_tracker.upload_artifact("training_report", training_report)
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


def build_model_report(output_dir: Path) -> dict[str, Any]:
    """
    Build model configuration and file manifest details.

    Args:
        output_dir: The saved model directory.

    Returns:
        JSON-compatible model report details.
    """
    config_path = output_dir / "requirement_model_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = build_model_manifest(output_dir)
    total_size_bytes = sum(int(row["size_bytes"]) for row in manifest)
    return {
        "config": config,
        "label_map": dict(LABEL_TO_ID),
        "manifest": manifest,
        "total_size_bytes": total_size_bytes,
    }


def build_model_manifest(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a deterministic file manifest for a model directory.

    Args:
        output_dir: The saved model directory.

    Returns:
        File manifest rows.
    """
    if not output_dir.exists():
        return []
    rows: list[dict[str, str | int]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def report_model_details_to_clearml(
    tracker: ClearMlTracker,
    output_dir: Path,
    should_register_model: bool,
) -> None:
    """
    Report model metadata and optionally register an output model package.

    Args:
        tracker: The optional ClearML tracker.
        output_dir: The saved model directory.
        should_register_model: Whether to register a ClearML output model.
    """
    if not tracker.is_enabled:
        return
    model_report = build_model_report(output_dir)
    tracker.upload_artifact("model_report", model_report)
    tracker.report_single_value(
        "model_total_size_bytes",
        float(model_report["total_size_bytes"]),
    )
    tracker.report_table(
        title="model",
        series="file_manifest",
        rows=build_model_manifest_rows(model_report["manifest"]),
    )
    tracker.report_table(
        title="model",
        series="label_map",
        rows=build_label_map_rows(model_report["label_map"]),
    )
    if should_register_model:
        tracker.register_model_package(
            model_path=output_dir,
            name="requirement_model",
            config=model_report["config"],
            labels=model_report["label_map"],
        )


def build_model_manifest_rows(
    manifest: list[dict[str, str | int]]
) -> list[list[str | int]]:
    """
    Build ClearML table rows for a model manifest.

    Args:
        manifest: Model manifest dictionaries.

    Returns:
        Table rows including a header row.
    """
    rows: list[list[str | int]] = [["path", "size_bytes"]]
    for row in manifest:
        rows.append([row["path"], row["size_bytes"]])
    return rows


def build_label_map_rows(label_map: dict[str, int]) -> list[list[str | int]]:
    """
    Build ClearML table rows for label mappings.

    Args:
        label_map: Label identifiers keyed by label name.

    Returns:
        Table rows including a header row.
    """
    rows: list[list[str | int]] = [["label", "id"]]
    for label_name, label_id in sorted(label_map.items()):
        rows.append([label_name, label_id])
    return rows


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
            "max_epochs": args.max_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "validation_split": args.validation_split,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "quick": args.quick,
            "model_report": args.clearml_model_report,
            "register_model": args.clearml_register_model,
        },
    )
    try:
        output_dir = train_model(args, tracker)
        if args.clearml_model_report or args.clearml_register_model:
            report_model_details_to_clearml(
                tracker,
                output_dir,
                should_register_model=args.clearml_register_model,
            )
        print(f"saved requirement model artifact to {output_dir}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
