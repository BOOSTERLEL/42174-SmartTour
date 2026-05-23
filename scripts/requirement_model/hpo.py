"""Run deterministic hyperparameter optimization for the requirement model."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.clearml_tracking import (
    DEFAULT_CLEARML_PROJECT,
    ClearMlTracker,
    initialize_clearml_task,
)
from scripts.requirement_model.compare_models import (
    ScopedClearMlTracker,
    build_run_id,
    evaluate_model_splits,
    load_training_report,
)
from scripts.requirement_model.train import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_NAME,
    train_model,
)

DEFAULT_HPO_DIR = Path("models/requirement_model/hpo")
DEFAULT_OBJECTIVE_SPLIT = "validation"
DEFAULT_OBJECTIVE_METRIC = "macro_f1"
TrialStatus = Literal["completed", "failed"]
TRIAL_STATUS_COMPLETED: TrialStatus = "completed"
TRIAL_STATUS_FAILED: TrialStatus = "failed"


@dataclass(frozen=True, slots=True)
class HpoTrialConfig:
    """
    Hyperparameter values for one HPO trial.
    """

    trial_id: str
    learning_rate: float
    batch_size: int
    max_length: int
    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float

    def to_dict(self) -> dict[str, int | float | str]:
        """
        Convert the trial configuration to JSON-compatible values.

        Returns:
            The trial configuration dictionary.
        """
        return {
            "trial_id": self.trial_id,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


@dataclass(frozen=True, slots=True)
class HpoTrialResult:
    """
    Result bundle for one HPO trial.
    """

    trial_id: str
    status: TrialStatus
    config: HpoTrialConfig
    output_dir: Path
    objective_score: float | None
    training_report: dict[str, Any]
    metrics_by_split: dict[str, dict[str, float]]
    diagnostics_by_split: dict[str, dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the trial result to JSON-compatible values.

        Returns:
            The serialized trial result.
        """
        return {
            "trial_id": self.trial_id,
            "status": self.status,
            "config": self.config.to_dict(),
            "output_dir": str(self.output_dir),
            "objective_score": self.objective_score,
            "training": self.training_report,
            "metrics": self.metrics_by_split,
            "diagnostics": self.diagnostics_by_split,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class HpoRun:
    """
    Result bundle for a full HPO run.
    """

    run_id: str
    run_dir: Path
    results: list[HpoTrialResult]
    best_trial: HpoTrialResult | None
    summary: dict[str, Any]


TrialRunner = Callable[
    [argparse.Namespace, HpoTrialConfig, Path, ClearMlTracker], HpoTrialResult
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for requirement extraction."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--hpo-dir", type=Path, default=DEFAULT_HPO_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--learning-rate-values", default="5e-5")
    parser.add_argument("--batch-size-values", default="4")
    parser.add_argument("--max-length-values", default="192")
    parser.add_argument("--max-epochs-values", "--max-epochs", default="20")
    parser.add_argument("--min-epochs-values", "--min-epochs", default="3")
    parser.add_argument("--patience-values", "--patience", default="3")
    parser.add_argument("--min-delta-values", "--min-delta", default="0.001")
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--objective-split", default=DEFAULT_OBJECTIVE_SPLIT)
    parser.add_argument("--objective-metric", default=DEFAULT_OBJECTIVE_METRIC)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument(
        "--clearml-task-name",
        default="Requirement Model HPO",
    )
    return parser.parse_args()


def parse_positive_int_values(raw_values: str, field_name: str) -> tuple[int, ...]:
    """
    Parse a comma-separated list of positive integers.

    Args:
        raw_values: The comma-separated value string.
        field_name: The argument name used in validation errors.

    Returns:
        Parsed integer values.

    Raises:
        ValueError: Raised when the list is empty or contains invalid values.
    """
    try:
        values = tuple(
            int(raw_value.strip())
            for raw_value in raw_values.split(",")
            if raw_value.strip()
        )
    except ValueError as error:
        raise ValueError(f"{field_name} must contain integers") from error
    if not values:
        raise ValueError(f"{field_name} must include at least one value")
    if any(value < 1 for value in values):
        raise ValueError(f"{field_name} values must be at least 1")
    return values


def parse_float_values(
    raw_values: str,
    field_name: str,
    minimum: float,
) -> tuple[float, ...]:
    """
    Parse a comma-separated list of floats with a lower bound.

    Args:
        raw_values: The comma-separated value string.
        field_name: The argument name used in validation errors.
        minimum: The inclusive minimum value.

    Returns:
        Parsed float values.

    Raises:
        ValueError: Raised when the list is empty or contains invalid values.
    """
    try:
        values = tuple(
            float(raw_value.strip())
            for raw_value in raw_values.split(",")
            if raw_value.strip()
        )
    except ValueError as error:
        raise ValueError(f"{field_name} must contain numbers") from error
    if not values:
        raise ValueError(f"{field_name} must include at least one value")
    if any(value < minimum for value in values):
        raise ValueError(f"{field_name} values must be at least {minimum}")
    return values


def build_search_space(args: argparse.Namespace) -> tuple[HpoTrialConfig, ...]:
    """
    Build deterministic HPO trial configurations from CLI arguments.

    Args:
        args: The parsed HPO arguments.

    Returns:
        The trial configurations in execution order.

    Raises:
        ValueError: Raised when convergence settings are invalid.
    """
    if args.trial_limit is not None and args.trial_limit < 1:
        raise ValueError("trial_limit must be at least 1")
    learning_rate_values = parse_float_values(
        args.learning_rate_values, "learning_rate_values", 0.0
    )
    batch_size_values = parse_positive_int_values(
        args.batch_size_values, "batch_size_values"
    )
    max_length_values = parse_positive_int_values(
        args.max_length_values, "max_length_values"
    )
    max_epoch_values = parse_positive_int_values(
        args.max_epochs_values, "max_epochs_values"
    )
    min_epoch_values = parse_positive_int_values(
        args.min_epochs_values, "min_epochs_values"
    )
    patience_values = parse_positive_int_values(args.patience_values, "patience_values")
    min_delta_values = parse_float_values(
        args.min_delta_values, "min_delta_values", 0.0
    )
    trial_configs: list[HpoTrialConfig] = []
    value_product = itertools.product(
        learning_rate_values,
        batch_size_values,
        max_length_values,
        max_epoch_values,
        min_epoch_values,
        patience_values,
        min_delta_values,
    )
    for trial_index, values in enumerate(value_product, start=1):
        if args.trial_limit is not None and len(trial_configs) >= args.trial_limit:
            break
        (
            learning_rate,
            batch_size,
            max_length,
            max_epochs,
            min_epochs,
            patience,
            min_delta,
        ) = values
        if min_epochs > max_epochs:
            raise ValueError("min_epochs_values cannot exceed max_epochs_values")
        trial_configs.append(
            HpoTrialConfig(
                trial_id=f"trial-{trial_index:03d}",
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_length=max_length,
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                patience=patience,
                min_delta=min_delta,
            )
        )
    return tuple(trial_configs)


def build_training_args(
    args: argparse.Namespace, trial_config: HpoTrialConfig, output_dir: Path
) -> argparse.Namespace:
    """
    Build training arguments for one HPO trial.

    Args:
        args: The parsed HPO arguments.
        trial_config: The hyperparameter values for this trial.
        output_dir: The trial artifact directory.

    Returns:
        The training argument namespace expected by `train_model`.
    """
    return argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        max_length=trial_config.max_length,
        epochs=trial_config.max_epochs,
        max_epochs=trial_config.max_epochs,
        min_epochs=trial_config.min_epochs,
        patience=trial_config.patience,
        min_delta=trial_config.min_delta,
        validation_split=args.validation_split,
        batch_size=trial_config.batch_size,
        learning_rate=trial_config.learning_rate,
        device=args.device,
        quick=args.quick,
    )


def train_and_evaluate_trial(
    args: argparse.Namespace,
    trial_config: HpoTrialConfig,
    output_dir: Path,
    tracker: ClearMlTracker,
) -> HpoTrialResult:
    """
    Train and evaluate one HPO trial.

    Args:
        args: The parsed HPO arguments.
        trial_config: The hyperparameter values for this trial.
        output_dir: The trial artifact directory.
        tracker: The optional ClearML tracker.

    Returns:
        The completed HPO trial result.
    """
    scoped_tracker = ScopedClearMlTracker(tracker, trial_config.trial_id)
    training_args = build_training_args(args, trial_config, output_dir)
    train_model(training_args, scoped_tracker)
    training_report = load_training_report(output_dir)
    metrics_by_split, diagnostics_by_split = evaluate_model_splits(
        data_dir=args.data_dir,
        model_dir=output_dir,
        max_length=trial_config.max_length,
        device_name=args.device,
        tracker=tracker,
        model_slug=trial_config.trial_id,
    )
    return HpoTrialResult(
        trial_id=trial_config.trial_id,
        status=TRIAL_STATUS_COMPLETED,
        config=trial_config,
        output_dir=output_dir,
        objective_score=extract_objective_score(
            metrics_by_split,
            args.objective_split,
            args.objective_metric,
        ),
        training_report=training_report,
        metrics_by_split=metrics_by_split,
        diagnostics_by_split=diagnostics_by_split,
    )


def extract_objective_score(
    metrics_by_split: dict[str, dict[str, float]],
    objective_split: str,
    objective_metric: str,
) -> float | None:
    """
    Extract the objective score from split metrics.

    Args:
        metrics_by_split: Metrics keyed by split name.
        objective_split: The split used for objective ranking.
        objective_metric: The metric used for objective ranking.

    Returns:
        The objective score, or None when missing.
    """
    return metrics_by_split.get(objective_split, {}).get(objective_metric)


def build_failed_trial_result(
    trial_config: HpoTrialConfig,
    output_dir: Path,
    error: Exception,
) -> HpoTrialResult:
    """
    Build a failed trial result from a raised exception.

    Args:
        trial_config: The hyperparameter values for this trial.
        output_dir: The trial artifact directory.
        error: The trial exception.

    Returns:
        A failed HPO trial result.
    """
    return HpoTrialResult(
        trial_id=trial_config.trial_id,
        status=TRIAL_STATUS_FAILED,
        config=trial_config,
        output_dir=output_dir,
        objective_score=None,
        training_report={},
        metrics_by_split={},
        diagnostics_by_split={},
        error=f"{type(error).__name__}: {error}",
    )


def rank_trial_results(
    results: Sequence[HpoTrialResult],
    objective_split: str,
    objective_metric: str,
) -> list[HpoTrialResult]:
    """
    Rank completed trials by objective score and deterministic tie breakers.

    Args:
        results: Trial results to rank.
        objective_split: The split used for objective ranking.
        objective_metric: The metric used for objective ranking.

    Returns:
        Completed trial results ordered from best to worst.
    """
    eligible_results = [
        result
        for result in results
        if result.status == TRIAL_STATUS_COMPLETED
        and extract_objective_score(
            result.metrics_by_split,
            objective_split,
            objective_metric,
        )
        is not None
    ]
    return sorted(
        eligible_results,
        key=lambda result: build_trial_sort_key(
            result,
            objective_split,
            objective_metric,
        ),
    )


def build_trial_sort_key(
    result: HpoTrialResult,
    objective_split: str,
    objective_metric: str,
) -> tuple[float, float, float, float, str]:
    """
    Build a deterministic ascending sort key for one trial.

    Args:
        result: The trial result.
        objective_split: The split used for objective ranking.
        objective_metric: The metric used for objective ranking.

    Returns:
        The sort key.
    """
    metrics = result.metrics_by_split.get(objective_split, {})
    objective_score = metrics[objective_metric]
    return (
        -objective_score,
        -metrics.get("exact_match_accuracy", 0.0),
        -metrics.get("slot_accuracy", 0.0),
        -metrics.get("micro_f1", 0.0),
        result.trial_id,
    )


def build_hpo_summary(
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
    results: Sequence[HpoTrialResult],
    best_trial: HpoTrialResult | None,
) -> dict[str, Any]:
    """
    Build a JSON-compatible HPO summary.

    Args:
        run_id: The HPO run identifier.
        run_dir: The HPO artifact directory.
        args: The parsed HPO arguments.
        results: Trial results in execution order.
        best_trial: The selected best trial, if any trial completed.

    Returns:
        The HPO summary.
    """
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_name": args.model_name,
        "objective": {
            "split": args.objective_split,
            "metric": args.objective_metric,
            "direction": "maximize",
            "tie_breakers": [
                "exact_match_accuracy",
                "slot_accuracy",
                "micro_f1",
                "trial_id",
            ],
        },
        "best_trial": best_trial.to_dict() if best_trial is not None else None,
        "trials": [result.to_dict() for result in results],
    }


def build_hpo_trial_rows(
    results: Sequence[HpoTrialResult],
    objective_split: str,
    objective_metric: str,
) -> list[list[Any]]:
    """
    Build CSV and table rows for HPO trials.

    Args:
        results: Trial results to summarize.
        objective_split: The split used for objective ranking.
        objective_metric: The metric used for objective ranking.

    Returns:
        Rows including a header row.
    """
    metric_columns = collect_metric_columns(results)
    rows: list[list[Any]] = [
        [
            "trial_id",
            "status",
            "objective_split",
            "objective_metric",
            "objective_score",
            "learning_rate",
            "batch_size",
            "max_length",
            "max_epochs",
            "min_epochs",
            "patience",
            "min_delta",
            "output_dir",
            "error",
            *[
                format_metric_column(split_name, metric_name)
                for split_name, metric_name in metric_columns
            ],
        ]
    ]
    for result in results:
        config = result.config
        metric_values = [
            result.metrics_by_split.get(split_name, {}).get(metric_name, "")
            for split_name, metric_name in metric_columns
        ]
        rows.append(
            [
                result.trial_id,
                result.status,
                objective_split,
                objective_metric,
                result.objective_score if result.objective_score is not None else "",
                config.learning_rate,
                config.batch_size,
                config.max_length,
                config.max_epochs,
                config.min_epochs,
                config.patience,
                config.min_delta,
                str(result.output_dir),
                result.error or "",
                *metric_values,
            ]
        )
    return rows


def collect_metric_columns(
    results: Sequence[HpoTrialResult],
) -> tuple[tuple[str, str], ...]:
    """
    Collect deterministic metric columns from trial results.

    Args:
        results: Trial results to inspect.

    Returns:
        Split and metric name pairs.
    """
    columns = {
        (split_name, metric_name)
        for result in results
        for split_name, metrics in result.metrics_by_split.items()
        for metric_name in metrics
    }
    return tuple(sorted(columns))


def format_metric_column(split_name: str, metric_name: str) -> str:
    """
    Format one metric column name.

    Args:
        split_name: The metric split name.
        metric_name: The metric name.

    Returns:
        A CSV-safe column name.
    """
    return f"{split_name}_{metric_name}"


def write_hpo_artifacts(
    run_dir: Path,
    summary: dict[str, Any],
    trial_rows: Sequence[Sequence[Any]],
) -> None:
    """
    Write local HPO summary artifacts.

    Args:
        run_dir: The HPO artifact directory.
        summary: The JSON-compatible HPO summary.
        trial_rows: CSV-compatible trial rows.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hpo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "hpo_trials.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(trial_rows)


def report_hpo_to_clearml(
    tracker: ClearMlTracker,
    run_dir: Path,
    summary: dict[str, Any],
    trial_rows: Sequence[Sequence[Any]],
    results: Sequence[HpoTrialResult],
    best_trial: HpoTrialResult | None,
) -> None:
    """
    Report HPO summary artifacts to ClearML.

    Args:
        tracker: The optional ClearML tracker.
        run_dir: The HPO artifact directory.
        summary: The JSON-compatible HPO summary.
        trial_rows: Table rows for HPO results.
        results: Trial results in execution order.
        best_trial: The selected best trial, if any trial completed.
    """
    if not tracker.is_enabled:
        return
    tracker.upload_artifact("hpo_summary", summary)
    tracker.upload_artifact("hpo_trials_csv", str(run_dir / "hpo_trials.csv"))
    tracker.report_table("hpo", "trials", trial_rows)
    for iteration, result in enumerate(results, start=1):
        if result.objective_score is None:
            continue
        tracker.report_scalar(
            title="hpo/objective",
            series=result.trial_id,
            value=result.objective_score,
            iteration=iteration,
        )
        for split_name, metrics in sorted(result.metrics_by_split.items()):
            for metric_name, metric_value in sorted(metrics.items()):
                tracker.report_scalar(
                    title=f"hpo/{split_name}",
                    series=f"{result.trial_id}/{metric_name}",
                    value=metric_value,
                    iteration=iteration,
                )
    if best_trial is not None and best_trial.objective_score is not None:
        tracker.report_single_value("hpo_best_objective", best_trial.objective_score)


def run_hpo(
    args: argparse.Namespace,
    tracker: ClearMlTracker | None = None,
    trial_runner: TrialRunner = train_and_evaluate_trial,
) -> HpoRun:
    """
    Run the full HPO workflow.

    Args:
        args: The parsed HPO arguments.
        tracker: Optional ClearML tracker.
        trial_runner: Injectable trial runner for tests.

    Returns:
        The HPO run result.
    """
    active_tracker = tracker or ClearMlTracker()
    run_id = args.run_id or build_run_id()
    run_dir = args.hpo_dir / run_id
    trial_configs = build_search_space(args)
    results: list[HpoTrialResult] = []
    for trial_config in trial_configs:
        output_dir = run_dir / trial_config.trial_id
        try:
            result = trial_runner(args, trial_config, output_dir, active_tracker)
        except Exception as error:
            result = build_failed_trial_result(trial_config, output_dir, error)
            results.append(result)
            if args.fail_fast:
                raise
        else:
            results.append(result)
    ranked_results = rank_trial_results(
        results,
        args.objective_split,
        args.objective_metric,
    )
    best_trial = ranked_results[0] if ranked_results else None
    summary = build_hpo_summary(run_id, run_dir, args, results, best_trial)
    trial_rows = build_hpo_trial_rows(
        results,
        args.objective_split,
        args.objective_metric,
    )
    write_hpo_artifacts(run_dir, summary, trial_rows)
    report_hpo_to_clearml(
        active_tracker,
        run_dir,
        summary,
        trial_rows,
        results,
        best_trial,
    )
    return HpoRun(
        run_id=run_id,
        run_dir=run_dir,
        results=results,
        best_trial=best_trial,
        summary=summary,
    )


def build_clearml_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build secret-free ClearML task configuration.

    Args:
        args: The parsed HPO arguments.

    Returns:
        The ClearML configuration dictionary.
    """
    return {
        "data_dir": str(args.data_dir),
        "hpo_dir": str(args.hpo_dir),
        "run_id": args.run_id,
        "model_name": args.model_name,
        "learning_rate_values": args.learning_rate_values,
        "batch_size_values": args.batch_size_values,
        "max_length_values": args.max_length_values,
        "max_epochs_values": args.max_epochs_values,
        "min_epochs_values": args.min_epochs_values,
        "patience_values": args.patience_values,
        "min_delta_values": args.min_delta_values,
        "trial_limit": args.trial_limit,
        "objective_split": args.objective_split,
        "objective_metric": args.objective_metric,
        "validation_split": args.validation_split,
        "device": args.device,
        "quick": args.quick,
        "fail_fast": args.fail_fast,
    }


def main() -> None:
    """
    Run the HPO workflow from the command line.
    """
    args = parse_args()
    tracker = initialize_clearml_task(
        is_enabled=args.clearml,
        task_name=args.clearml_task_name,
        project_name=args.clearml_project,
        task_type="training",
        tags=("requirement_model", "hpo"),
        configuration=build_clearml_configuration(args),
    )
    try:
        hpo_run = run_hpo(args, tracker)
        print(f"hpo_run_id={hpo_run.run_id}")
        print(f"hpo_run_dir={hpo_run.run_dir}")
        if hpo_run.best_trial is None:
            raise RuntimeError("no completed HPO trials")
        print(f"best_trial={hpo_run.best_trial.trial_id}")
        print(f"best_objective={hpo_run.best_trial.objective_score:.4f}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
