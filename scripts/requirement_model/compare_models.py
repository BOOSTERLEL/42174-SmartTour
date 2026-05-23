"""Compare converged requirement extraction models and promote the winner."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.clearml_tracking import (
    DEFAULT_CLEARML_PROJECT,
    ClearMlTracker,
    initialize_clearml_task,
)
from scripts.requirement_model.evaluate import (
    build_failure_table_rows,
    build_metric_table_rows,
    evaluate_records,
)
from scripts.requirement_model.schema import load_jsonl
from scripts.requirement_model.train import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    build_model_report,
    resolve_device,
    train_model,
)

DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    DEFAULT_MODEL_NAME,
    "distilbert-base-uncased",
    "bert-base-cased",
)
DEFAULT_EXPERIMENT_DIR = Path("models/requirement_model/experiments")
EVALUATION_SPLITS: tuple[str, ...] = ("validation", "test", "reviewed_test")
WINNER_SPLIT = "reviewed_test"
WINNER_METRIC_ORDER: tuple[tuple[str, str], ...] = (
    ("reviewed_test", "slot_accuracy"),
    ("reviewed_test", "exact_match_accuracy"),
    ("reviewed_test", "macro_f1"),
    ("validation", "macro_f1"),
)


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """
    Result bundle for one trained model candidate.
    """

    model_name: str
    model_slug: str
    output_dir: Path
    training_report: dict[str, Any]
    metrics_by_split: dict[str, dict[str, float]]
    diagnostics_by_split: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    """
    Result bundle for a full model comparison run.
    """

    run_id: str
    run_dir: Path
    results: list[ModelComparisonResult]
    winner: ModelComparisonResult
    summary: dict[str, Any]


class ScopedClearMlTracker(ClearMlTracker):
    """
    ClearML tracker that prefixes artifacts and chart titles for one model.
    """

    __slots__ = ("scope",)

    def __init__(self, tracker: ClearMlTracker, scope: str) -> None:
        """
        Initialize a scoped tracker.

        Args:
            tracker: The parent tracker.
            scope: The model-specific scope prefix.
        """
        super().__init__(task=tracker.task)
        self.scope = scope

    def report_scalar(
        self, title: str, series: str, value: float, iteration: int
    ) -> None:
        """
        Report a scoped scalar value to ClearML.

        Args:
            title: The scalar chart title.
            series: The chart series name.
            value: The scalar value.
            iteration: The scalar iteration.
        """
        super().report_scalar(
            title=f"{self.scope}/{title}",
            series=series,
            value=value,
            iteration=iteration,
        )

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        """
        Upload a scoped artifact.

        Args:
            name: The artifact name.
            artifact_object: The local path or serializable object to upload.
        """
        if name == "model" and isinstance(artifact_object, str):
            model_path = Path(artifact_object)
            if model_path.is_dir():
                super().upload_artifact(
                    name=f"{self.scope}_model_report",
                    artifact_object=build_model_report(model_path),
                )
                return
        super().upload_artifact(
            name=f"{self.scope}_{name}",
            artifact_object=artifact_object,
        )


TrainEvaluateFunction = Callable[
    [argparse.Namespace, str, Path, ClearMlTracker], ModelComparisonResult
]
PromoteFunction = Callable[[Path, Path], None]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train, evaluate, compare, and promote requirement models."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/requirement_model"))
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--default-model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", action="append", default=None)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--hpo-summary", type=Path, default=None)
    parser.add_argument(
        "--no-promote",
        dest="promote_winner",
        action="store_false",
        default=True,
    )
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument(
        "--clearml-task-name",
        default="Requirement Model Comparison",
    )
    parser.add_argument("--register-winner", action="store_true")
    return parser.parse_args()


def resolve_model_names(model_names: Sequence[str] | None) -> tuple[str, ...]:
    """
    Resolve model names from CLI arguments.

    Args:
        model_names: Optional model names supplied by the user.

    Returns:
        The model names to compare.

    Raises:
        ValueError: Raised when fewer than two models are selected.
    """
    resolved_names = tuple(model_names) if model_names else DEFAULT_MODEL_NAMES
    if len(resolved_names) < 2:
        raise ValueError("at least two models are required for comparison")
    return resolved_names


def build_run_id() -> str:
    """
    Build a timestamp run identifier.

    Returns:
        The run identifier.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify_model_name(model_name: str) -> str:
    """
    Convert a Hugging Face model name into a filesystem-safe slug.

    Args:
        model_name: The source model name.

    Returns:
        The slug value.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()
    return slug or "model"


def build_training_args(
    args: argparse.Namespace, model_name: str, output_dir: Path
) -> argparse.Namespace:
    """
    Build training arguments for one candidate model.

    Args:
        args: The comparison arguments.
        model_name: The Hugging Face model name.
        output_dir: The model-specific output directory.

    Returns:
        The training argument namespace expected by `train_model`.
    """
    training_parameters = resolve_training_parameters(args, model_name)
    return argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=output_dir,
        model_name=model_name,
        max_length=training_parameters["max_length"],
        epochs=training_parameters["max_epochs"],
        max_epochs=training_parameters["max_epochs"],
        min_epochs=training_parameters["min_epochs"],
        patience=training_parameters["patience"],
        min_delta=training_parameters["min_delta"],
        validation_split="validation",
        batch_size=training_parameters["batch_size"],
        learning_rate=training_parameters["learning_rate"],
        device=args.device,
        quick=getattr(args, "quick", False),
    )


def resolve_training_parameters(
    args: argparse.Namespace, model_name: str
) -> dict[str, int | float]:
    """
    Resolve training parameters from CLI defaults and optional HPO output.

    Args:
        args: The comparison arguments.
        model_name: The model candidate name.

    Returns:
        The resolved training parameters.
    """
    parameters: dict[str, int | float] = {
        "max_length": args.max_length,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    hpo_config = find_hpo_config_for_model(
        getattr(args, "hpo_summary_data", None),
        model_name,
    )
    if hpo_config is None:
        return parameters
    parameters.update(
        {
            "max_length": int(hpo_config["max_length"]),
            "max_epochs": int(hpo_config["max_epochs"]),
            "min_epochs": int(hpo_config["min_epochs"]),
            "patience": int(hpo_config["patience"]),
            "min_delta": float(hpo_config["min_delta"]),
            "batch_size": int(hpo_config["batch_size"]),
            "learning_rate": float(hpo_config["learning_rate"]),
        }
    )
    return parameters


def find_hpo_config_for_model(
    hpo_summary: dict[str, Any] | None,
    model_name: str,
) -> dict[str, Any] | None:
    """
    Find the best HPO trial config for a model candidate.

    Args:
        hpo_summary: The validated HPO summary, if provided.
        model_name: The model candidate name.

    Returns:
        The HPO trial config for this model, or None when not applicable.
    """
    if hpo_summary is None or hpo_summary["model_name"] != model_name:
        return None
    best_trial = cast(dict[str, Any], hpo_summary["best_trial"])
    return cast(dict[str, Any], best_trial["config"])


def train_and_evaluate_model(
    args: argparse.Namespace,
    model_name: str,
    output_dir: Path,
    tracker: ClearMlTracker,
) -> ModelComparisonResult:
    """
    Train and evaluate one model candidate.

    Args:
        args: The comparison arguments.
        model_name: The Hugging Face model name.
        output_dir: The candidate output directory.
        tracker: The optional ClearML tracker.

    Returns:
        The model comparison result.
    """
    model_slug = slugify_model_name(model_name)
    scoped_tracker = ScopedClearMlTracker(tracker, model_slug)
    training_args = build_training_args(args, model_name, output_dir)
    trained_output_dir = train_model(training_args, scoped_tracker)
    training_report = load_training_report(trained_output_dir)
    metrics_by_split, diagnostics_by_split = evaluate_model_splits(
        data_dir=args.data_dir,
        model_dir=trained_output_dir,
        max_length=training_args.max_length,
        device_name=args.device,
        tracker=tracker,
        model_slug=model_slug,
    )
    return ModelComparisonResult(
        model_name=model_name,
        model_slug=model_slug,
        output_dir=trained_output_dir,
        training_report=training_report,
        metrics_by_split=metrics_by_split,
        diagnostics_by_split=diagnostics_by_split,
    )


def load_training_report(output_dir: Path) -> dict[str, Any]:
    """
    Load a saved training report.

    Args:
        output_dir: The trained model output directory.

    Returns:
        The training report dictionary.
    """
    report_path = output_dir / "training_report.json"
    if not report_path.exists():
        return {
            "model_name": "unknown",
            "output_dir": str(output_dir),
            "epochs_ran": 0,
            "best_epoch": None,
            "best_validation_metric": None,
            "stopped_early": False,
            "history": [],
        }
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_hpo_summary(
    summary_path: Path | None, model_names: Sequence[str]
) -> dict[str, Any] | None:
    """
    Load and validate an optional HPO summary before comparison training starts.

    Args:
        summary_path: The HPO summary path, if one was provided.
        model_names: The model candidates selected for comparison.

    Returns:
        The validated HPO summary, or None when no summary was supplied.

    Raises:
        FileNotFoundError: Raised when the summary path does not exist.
        ValueError: Raised when the summary shape is invalid.
    """
    if summary_path is None:
        return None
    if not summary_path.exists():
        raise FileNotFoundError(f"missing HPO summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("HPO summary must be a JSON object")
    validate_hpo_summary(summary, model_names)
    return summary


def validate_hpo_summary(summary: dict[str, Any], model_names: Sequence[str]) -> None:
    """
    Validate an HPO summary used to tune comparison training.

    Args:
        summary: The decoded HPO summary.
        model_names: The model candidates selected for comparison.

    Raises:
        ValueError: Raised when required fields are missing or malformed.
    """
    model_name = summary.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("HPO summary must include model_name")
    if model_name not in model_names:
        raise ValueError("HPO summary model_name must match a selected model")
    best_trial = summary.get("best_trial")
    if not isinstance(best_trial, dict):
        raise ValueError("HPO summary must include best_trial")
    config = best_trial.get("config")
    if not isinstance(config, dict):
        raise ValueError("HPO summary best_trial must include config")
    require_positive_number(config, "learning_rate")
    require_positive_integer(config, "batch_size")
    require_positive_integer(config, "max_length")
    max_epochs = require_positive_integer(config, "max_epochs")
    min_epochs = require_positive_integer(config, "min_epochs")
    require_positive_integer(config, "patience")
    require_non_negative_number(config, "min_delta")
    if min_epochs > max_epochs:
        raise ValueError("HPO summary min_epochs cannot exceed max_epochs")


def require_positive_integer(config: dict[str, Any], field_name: str) -> int:
    """
    Read and validate a positive integer HPO config field.

    Args:
        config: The HPO trial config.
        field_name: The config field name.

    Returns:
        The integer field value.

    Raises:
        ValueError: Raised when the field is missing or invalid.
    """
    value = config.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"HPO summary config {field_name} must be a positive integer")
    return value


def require_positive_number(config: dict[str, Any], field_name: str) -> float:
    """
    Read and validate a positive numeric HPO config field.

    Args:
        config: The HPO trial config.
        field_name: The config field name.

    Returns:
        The numeric field value.

    Raises:
        ValueError: Raised when the field is missing or invalid.
    """
    value = config.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"HPO summary config {field_name} must be a positive number")
    return float(value)


def require_non_negative_number(config: dict[str, Any], field_name: str) -> float:
    """
    Read and validate a non-negative numeric HPO config field.

    Args:
        config: The HPO trial config.
        field_name: The config field name.

    Returns:
        The numeric field value.

    Raises:
        ValueError: Raised when the field is missing or invalid.
    """
    value = config.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(
            f"HPO summary config {field_name} must be a non-negative number"
        )
    return float(value)


def evaluate_model_splits(
    data_dir: Path,
    model_dir: Path,
    max_length: int,
    device_name: str,
    tracker: ClearMlTracker,
    model_slug: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    """
    Evaluate one trained model on every comparison split.

    Args:
        data_dir: The dataset directory.
        model_dir: The trained model directory.
        max_length: The maximum encoded sequence length.
        device_name: The requested device name.
        tracker: The optional ClearML tracker.
        model_slug: The model slug used in reports.

    Returns:
        Metrics and diagnostics keyed by split name.
    """
    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    metrics_by_split: dict[str, dict[str, float]] = {}
    diagnostics_by_split: dict[str, dict[str, Any]] = {}
    try:
        for split_name in EVALUATION_SPLITS:
            records = load_jsonl(data_dir / f"{split_name}.jsonl")
            metrics, diagnostics = evaluate_records(
                records=records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                device=device,
            )
            metrics_by_split[split_name] = metrics
            diagnostics_by_split[split_name] = diagnostics
            report_split_evaluation_to_clearml(
                tracker,
                model_slug,
                split_name,
                metrics,
                diagnostics,
            )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return metrics_by_split, diagnostics_by_split


def report_split_evaluation_to_clearml(
    tracker: ClearMlTracker,
    model_slug: str,
    split_name: str,
    metrics: dict[str, float],
    diagnostics: dict[str, Any],
) -> None:
    """
    Report one model split evaluation to ClearML.

    Args:
        tracker: The optional ClearML tracker.
        model_slug: The model slug used in report names.
        split_name: The evaluated split name.
        metrics: The computed scalar metrics.
        diagnostics: The computed detailed diagnostics.
    """
    if not tracker.is_enabled:
        return
    for metric_name, metric_value in sorted(metrics.items()):
        tracker.report_scalar(
            title=f"evaluation/{split_name}",
            series=f"{model_slug}/{metric_name}",
            value=metric_value,
            iteration=0,
        )
    tracker.report_confusion_matrix(
        title=f"{model_slug}/{split_name}",
        series="bio_confusion_matrix",
        matrix=diagnostics["confusion_matrix"],
        labels=diagnostics["labels"],
    )
    tracker.report_table(
        title=f"{model_slug}/{split_name}",
        series="label_metrics",
        rows=build_metric_table_rows(diagnostics["label_metrics"], "label"),
    )
    tracker.report_table(
        title=f"{model_slug}/{split_name}",
        series="slot_metrics",
        rows=build_metric_table_rows(diagnostics["slot_metrics"], "field"),
    )
    tracker.report_table(
        title=f"{model_slug}/{split_name}",
        series="failures",
        rows=build_failure_table_rows(diagnostics["failures"]),
    )
    tracker.upload_artifact(
        f"{model_slug}_{split_name}_metrics",
        dict(metrics),
    )
    tracker.upload_artifact(
        f"{model_slug}_{split_name}_diagnostics",
        dict(diagnostics),
    )


def rank_results(
    results: Sequence[ModelComparisonResult],
) -> list[ModelComparisonResult]:
    """
    Rank model results by the documented winner metric order.

    Args:
        results: Model results to rank.

    Returns:
        Results ordered from best to worst.
    """
    return sorted(results, key=result_sort_key)


def result_sort_key(
    result: ModelComparisonResult,
) -> tuple[float, float, float, float, str]:
    """
    Build a deterministic ascending sort key for one model result.

    Args:
        result: The model result.

    Returns:
        The sort key.
    """
    metric_values = [
        result.metrics_by_split[split_name][metric_name]
        for split_name, metric_name in WINNER_METRIC_ORDER
    ]
    return (
        -metric_values[0],
        -metric_values[1],
        -metric_values[2],
        -metric_values[3],
        result.model_name,
    )


def build_comparison_summary(
    run_id: str,
    run_dir: Path,
    results: Sequence[ModelComparisonResult],
    winner: ModelComparisonResult,
    promote_winner: bool,
    hpo_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build a JSON-compatible comparison summary.

    Args:
        run_id: The comparison run identifier.
        run_dir: The comparison run directory.
        results: The model results.
        winner: The selected winning result.
        promote_winner: Whether this run promoted the winner.
        hpo_summary: Optional HPO summary used for tuned parameters.

    Returns:
        The comparison summary.
    """
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "promoted": promote_winner,
        "hpo": build_hpo_summary_reference(hpo_summary),
        "winner": {
            "model_name": winner.model_name,
            "model_slug": winner.model_slug,
            "output_dir": str(winner.output_dir),
            "rank_metric_order": [
                {"split": split_name, "metric": metric_name}
                for split_name, metric_name in WINNER_METRIC_ORDER
            ],
            "metrics": winner.metrics_by_split,
            "training": winner.training_report,
        },
        "models": [
            {
                "model_name": result.model_name,
                "model_slug": result.model_slug,
                "output_dir": str(result.output_dir),
                "training": result.training_report,
                "metrics": result.metrics_by_split,
            }
            for result in results
        ],
    }


def build_hpo_summary_reference(
    hpo_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Build a compact HPO reference for comparison summaries.

    Args:
        hpo_summary: Optional HPO summary used for tuned parameters.

    Returns:
        A compact HPO reference, or None when HPO was not used.
    """
    if hpo_summary is None:
        return None
    best_trial = cast(dict[str, Any], hpo_summary["best_trial"])
    return {
        "run_id": hpo_summary.get("run_id"),
        "run_dir": hpo_summary.get("run_dir"),
        "model_name": hpo_summary["model_name"],
        "best_trial_id": best_trial.get("trial_id"),
        "best_config": best_trial["config"],
    }


def build_comparison_metric_rows(
    results: Sequence[ModelComparisonResult],
) -> list[list[str | float]]:
    """
    Build comparison metric table rows.

    Args:
        results: Model results to summarize.

    Returns:
        Table rows including a header row.
    """
    rows: list[list[str | float]] = [
        [
            "model_name",
            "model_slug",
            "split",
            "slot_accuracy",
            "exact_match_accuracy",
            "micro_f1",
            "macro_f1",
        ]
    ]
    for result in results:
        for split_name in EVALUATION_SPLITS:
            metrics = result.metrics_by_split[split_name]
            rows.append(
                [
                    result.model_name,
                    result.model_slug,
                    split_name,
                    metrics["slot_accuracy"],
                    metrics["exact_match_accuracy"],
                    metrics["micro_f1"],
                    metrics["macro_f1"],
                ]
            )
    return rows


def write_comparison_artifacts(
    run_dir: Path, summary: dict[str, Any], metric_rows: Sequence[Sequence[Any]]
) -> None:
    """
    Write local comparison summary artifacts.

    Args:
        run_dir: The comparison run directory.
        summary: The JSON-compatible comparison summary.
        metric_rows: CSV-compatible metric rows.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "comparison_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(metric_rows)


def report_comparison_to_clearml(
    tracker: ClearMlTracker,
    summary: dict[str, Any],
    metric_rows: Sequence[Sequence[Any]],
    winner: ModelComparisonResult,
    should_register_winner: bool,
    registration_model_dir: Path | None = None,
) -> None:
    """
    Report comparison summary artifacts to ClearML.

    Args:
        tracker: The optional ClearML tracker.
        summary: The JSON-compatible comparison summary.
        metric_rows: Table rows for scalar model metrics.
        winner: The selected winning result.
        should_register_winner: Whether to register the winning model package.
        registration_model_dir: Optional model directory copy used for registration.
    """
    if not tracker.is_enabled:
        return
    tracker.upload_artifact("comparison_summary", summary)
    tracker.report_table("comparison", "metrics", metric_rows)
    tracker.upload_artifact(
        "winner_model_report",
        build_model_report(winner.output_dir),
    )
    tracker.report_single_value(
        "winner_reviewed_test_slot_accuracy",
        winner.metrics_by_split[WINNER_SPLIT]["slot_accuracy"],
    )
    if should_register_winner:
        model_path = registration_model_dir or winner.output_dir
        winner_model_report = build_model_report(winner.output_dir)
        tracker.register_model_package(
            model_path=model_path,
            name="requirement_model_winner",
            config=winner_model_report["config"],
            labels=winner_model_report["label_map"],
        )


def promote_winner_model(source_dir: Path, target_dir: Path) -> None:
    """
    Promote a complete model artifact directory to the runtime default path.

    Args:
        source_dir: The winning model artifact directory.
        target_dir: The default model artifact directory.

    Raises:
        FileNotFoundError: Raised when the source is incomplete.
    """
    config_path = source_dir / "requirement_model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    target_parent = target_dir.parent
    staging_dir = target_parent / f".{target_dir.name}.staging"
    backup_dir = target_parent / f".{target_dir.name}.previous"
    remove_directory(staging_dir)
    remove_directory(backup_dir)
    shutil.copytree(source_dir, staging_dir)
    if target_dir.exists():
        target_dir.rename(backup_dir)
    staging_dir.rename(target_dir)
    remove_directory(backup_dir)


def remove_directory(path: Path) -> None:
    """
    Remove a directory if it exists.

    Args:
        path: The directory path.
    """
    if path.exists():
        shutil.rmtree(path)


def run_comparison(
    args: argparse.Namespace,
    tracker: ClearMlTracker | None = None,
    train_evaluate_model: TrainEvaluateFunction = train_and_evaluate_model,
    promote_model: PromoteFunction = promote_winner_model,
) -> ComparisonRun:
    """
    Run the full model comparison workflow.

    Args:
        args: The comparison arguments.
        tracker: Optional ClearML tracker.
        train_evaluate_model: Injectable train/evaluate function for tests.
        promote_model: Injectable promotion function for tests.

    Returns:
        The comparison run result.
    """
    active_tracker = tracker or ClearMlTracker()
    run_id = build_run_id()
    run_dir = args.experiment_dir / run_id
    model_names = resolve_model_names(args.model_name)
    hpo_summary = load_hpo_summary(getattr(args, "hpo_summary", None), model_names)
    comparison_args = argparse.Namespace(
        **vars(args),
        hpo_summary_data=hpo_summary,
    )
    results: list[ModelComparisonResult] = []
    for model_name in model_names:
        model_slug = slugify_model_name(model_name)
        output_dir = run_dir / model_slug
        result = train_evaluate_model(
            comparison_args,
            model_name,
            output_dir,
            active_tracker,
        )
        results.append(result)
    ranked_results = rank_results(results)
    winner = ranked_results[0]
    should_promote_winner = getattr(args, "promote_winner", True)
    summary = build_comparison_summary(
        run_id,
        run_dir,
        ranked_results,
        winner,
        promote_winner=should_promote_winner,
        hpo_summary=hpo_summary,
    )
    metric_rows = build_comparison_metric_rows(ranked_results)
    write_comparison_artifacts(run_dir, summary, metric_rows)
    if should_promote_winner:
        promote_model(winner.output_dir, args.default_model_dir)
    registration_model_dir: Path | None = None
    if args.register_winner and active_tracker.is_enabled and should_promote_winner:
        registration_model_dir = run_dir / f"{winner.model_slug}-clearml-registration"
        remove_directory(registration_model_dir)
        shutil.copytree(args.default_model_dir, registration_model_dir)
    try:
        report_comparison_to_clearml(
            active_tracker,
            summary,
            metric_rows,
            winner,
            should_register_winner=args.register_winner,
            registration_model_dir=registration_model_dir,
        )
    finally:
        if registration_model_dir is not None:
            remove_directory(registration_model_dir)
    return ComparisonRun(
        run_id=run_id,
        run_dir=run_dir,
        results=ranked_results,
        winner=winner,
        summary=summary,
    )


def main() -> None:
    """
    Run the model comparison workflow from the command line.
    """
    args = parse_args()
    tracker = initialize_clearml_task(
        is_enabled=args.clearml,
        task_name=args.clearml_task_name,
        project_name=args.clearml_project,
        task_type="training",
        tags=("requirement_model", "comparison"),
        configuration={
            "data_dir": str(args.data_dir),
            "experiment_dir": str(args.experiment_dir),
            "default_model_dir": str(args.default_model_dir),
            "model_names": list(resolve_model_names(args.model_name)),
            "max_length": args.max_length,
            "max_epochs": args.max_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "quick": args.quick,
            "hpo_summary": str(args.hpo_summary) if args.hpo_summary else None,
            "promote_winner": args.promote_winner,
            "register_winner": args.register_winner,
        },
    )
    try:
        comparison_run = run_comparison(args, tracker)
        winner_metrics = comparison_run.winner.metrics_by_split[WINNER_SPLIT]
        print(f"comparison_run_id={comparison_run.run_id}")
        print(f"winner={comparison_run.winner.model_name}")
        for metric_name, metric_value in sorted(winner_metrics.items()):
            print(f"winner_{WINNER_SPLIT}_{metric_name}={metric_value:.4f}")
        if args.promote_winner:
            print(f"promoted_default_model={args.default_model_dir}")
        else:
            print("promoted_default_model=false")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
