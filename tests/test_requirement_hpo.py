"""Tests for requirement model hyperparameter optimization orchestration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.requirement_model.clearml_tracking import ClearMlTracker
from scripts.requirement_model.hpo import (
    HpoTrialConfig,
    HpoTrialResult,
    build_clearml_configuration,
    build_search_space,
    parse_args,
    rank_trial_results,
    report_hpo_to_clearml,
    run_hpo,
)


class RecordingHpoTracker(ClearMlTracker):
    """
    ClearML tracker test double that records HPO reports.
    """

    __slots__ = ("artifacts", "scalars", "single_values", "tables")

    def __init__(self) -> None:
        """
        Initialize recorded ClearML calls.
        """
        super().__init__()
        self.artifacts: list[tuple[str, Any]] = []
        self.scalars: list[tuple[str, str, float, int]] = []
        self.single_values: list[tuple[str, float]] = []
        self.tables: list[tuple[str, str, Sequence[Sequence[Any]]]] = []

    @property
    def is_enabled(self) -> bool:
        """
        Return that this test double should receive ClearML calls.

        Returns:
            Always true for recording tests.
        """
        return True

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        """
        Record one artifact upload.

        Args:
            name: The artifact name.
            artifact_object: The uploaded object.
        """
        self.artifacts.append((name, artifact_object))

    def report_scalar(
        self, title: str, series: str, value: float, iteration: int
    ) -> None:
        """
        Record one scalar report.

        Args:
            title: The scalar chart title.
            series: The chart series name.
            value: The scalar value.
            iteration: The scalar iteration.
        """
        self.scalars.append((title, series, value, iteration))

    def report_single_value(self, name: str, value: float) -> None:
        """
        Record one summary value.

        Args:
            name: The metric name.
            value: The metric value.
        """
        self.single_values.append((name, value))

    def report_table(
        self,
        title: str,
        series: str,
        rows: Sequence[Sequence[Any]],
        iteration: int = 0,
    ) -> None:
        """
        Record one table report.

        Args:
            title: The table title.
            series: The table series name.
            rows: The reported table rows.
            iteration: The reporting iteration.
        """
        self.tables.append((title, series, rows))


def make_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    """
    Build a complete HPO argument namespace for tests.

    Args:
        tmp_path: The temporary artifact root.
        overrides: Argument values to override.

    Returns:
        The HPO argument namespace.
    """
    values: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "hpo_dir": tmp_path / "hpo",
        "run_id": "test-run",
        "model_name": "tiny-model",
        "learning_rate_values": "5e-5",
        "batch_size_values": "4",
        "max_length_values": "64",
        "max_epochs_values": "3",
        "min_epochs_values": "1",
        "patience_values": "2",
        "min_delta_values": "0.001",
        "trial_limit": None,
        "objective_split": "validation",
        "objective_metric": "macro_f1",
        "validation_split": "validation",
        "device": "cpu",
        "quick": False,
        "fail_fast": False,
        "clearml": False,
        "clearml_project": "Smartour",
        "clearml_task_name": "Requirement Model HPO",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_config(trial_id: str = "trial-001") -> HpoTrialConfig:
    """
    Build one HPO trial configuration.

    Args:
        trial_id: The trial identifier.

    Returns:
        The trial configuration.
    """
    return HpoTrialConfig(
        trial_id=trial_id,
        learning_rate=5e-5,
        batch_size=4,
        max_length=64,
        max_epochs=3,
        min_epochs=1,
        patience=2,
        min_delta=0.001,
    )


def make_result(
    trial_id: str,
    output_dir: Path,
    macro_f1: float,
    exact_match_accuracy: float,
    slot_accuracy: float,
    micro_f1: float,
) -> HpoTrialResult:
    """
    Build a completed HPO trial result.

    Args:
        trial_id: The trial identifier.
        output_dir: The trial output directory.
        macro_f1: The validation macro F1 score.
        exact_match_accuracy: The validation exact-match accuracy.
        slot_accuracy: The validation slot accuracy.
        micro_f1: The validation micro F1 score.

    Returns:
        The HPO trial result.
    """
    metrics = {
        "validation": {
            "macro_f1": macro_f1,
            "exact_match_accuracy": exact_match_accuracy,
            "slot_accuracy": slot_accuracy,
            "micro_f1": micro_f1,
        }
    }
    return HpoTrialResult(
        trial_id=trial_id,
        status="completed",
        config=make_config(trial_id),
        output_dir=output_dir,
        objective_score=macro_f1,
        training_report={"epochs_ran": 2},
        metrics_by_split=metrics,
        diagnostics_by_split={"validation": {}},
    )


def test_build_search_space_is_deterministic_and_limited(tmp_path: Path) -> None:
    """
    Verify search-space expansion keeps deterministic product order and limits.
    """
    args = make_args(
        tmp_path,
        learning_rate_values="5e-5,3e-5",
        batch_size_values="2,4",
        max_length_values="64",
        max_epochs_values="3",
        min_epochs_values="1",
        patience_values="1",
        min_delta_values="0,0.01",
        trial_limit=3,
    )

    configs = build_search_space(args)

    assert [config.trial_id for config in configs] == [
        "trial-001",
        "trial-002",
        "trial-003",
    ]
    assert [
        (config.learning_rate, config.batch_size, config.min_delta)
        for config in configs
    ] == [(5e-5, 2, 0.0), (5e-5, 2, 0.01), (5e-5, 4, 0.0)]


def test_parse_args_accepts_singular_convergence_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify single-value convergence aliases map to value-list arguments.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hpo.py",
            "--max-epochs",
            "8",
            "--min-epochs",
            "3",
            "--patience",
            "2",
            "--min-delta",
            "0.01",
        ],
    )

    args = parse_args()

    assert args.max_epochs_values == "8"
    assert args.min_epochs_values == "3"
    assert args.patience_values == "2"
    assert args.min_delta_values == "0.01"


def test_build_search_space_rejects_invalid_convergence(tmp_path: Path) -> None:
    """
    Verify HPO refuses trial grids where minimum epochs exceed maximum epochs.
    """
    args = make_args(
        tmp_path,
        max_epochs_values="1",
        min_epochs_values="2",
    )

    with pytest.raises(ValueError, match="min_epochs_values"):
        build_search_space(args)


def test_rank_trial_results_uses_objective_and_tie_breakers(tmp_path: Path) -> None:
    """
    Verify trial ranking uses objective, exact match, slot accuracy, and trial ID.
    """
    lower_objective = make_result(
        "trial-001",
        tmp_path / "trial-001",
        macro_f1=0.70,
        exact_match_accuracy=0.99,
        slot_accuracy=0.99,
        micro_f1=0.99,
    )
    lower_exact_match = make_result(
        "trial-002",
        tmp_path / "trial-002",
        macro_f1=0.80,
        exact_match_accuracy=0.40,
        slot_accuracy=0.99,
        micro_f1=0.99,
    )
    best_result = make_result(
        "trial-003",
        tmp_path / "trial-003",
        macro_f1=0.80,
        exact_match_accuracy=0.50,
        slot_accuracy=0.10,
        micro_f1=0.10,
    )

    ranked_results = rank_trial_results(
        [lower_objective, lower_exact_match, best_result],
        objective_split="validation",
        objective_metric="macro_f1",
    )

    assert ranked_results[0] == best_result


def test_run_hpo_records_failed_trial_and_continues(tmp_path: Path) -> None:
    """
    Verify failed trials are summarized while later trials can still win.
    """
    args = make_args(
        tmp_path,
        learning_rate_values="5e-5,3e-5",
    )

    def trial_runner(
        hpo_args: argparse.Namespace,
        trial_config: HpoTrialConfig,
        output_dir: Path,
        tracker: ClearMlTracker,
    ) -> HpoTrialResult:
        """
        Fail the first trial and complete the second.

        Args:
            hpo_args: The HPO arguments.
            trial_config: The trial configuration.
            output_dir: The trial output directory.
            tracker: The optional ClearML tracker.

        Returns:
            The completed trial result for the second trial.
        """
        if trial_config.trial_id == "trial-001":
            raise RuntimeError("training failed")
        return make_result(
            trial_config.trial_id,
            output_dir,
            macro_f1=0.83,
            exact_match_accuracy=0.30,
            slot_accuracy=0.90,
            micro_f1=0.82,
        )

    hpo_run = run_hpo(args, trial_runner=trial_runner)

    assert [result.status for result in hpo_run.results] == ["failed", "completed"]
    assert hpo_run.best_trial is not None
    assert hpo_run.best_trial.trial_id == "trial-002"
    summary_path = tmp_path / "hpo" / "test-run" / "hpo_summary.json"
    trial_csv_path = tmp_path / "hpo" / "test-run" / "hpo_trials.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with trial_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert summary["best_trial"]["trial_id"] == "trial-002"
    assert summary["trials"][0]["error"] == "RuntimeError: training failed"
    assert rows[0]["status"] == "failed"
    assert rows[1]["validation_macro_f1"] == "0.83"


def test_run_hpo_reports_summary_to_clearml(tmp_path: Path) -> None:
    """
    Verify enabled ClearML tracking receives summary artifacts and scalars.
    """
    args = make_args(
        tmp_path,
        learning_rate_values="5e-5,3e-5",
    )
    tracker = RecordingHpoTracker()

    def trial_runner(
        hpo_args: argparse.Namespace,
        trial_config: HpoTrialConfig,
        output_dir: Path,
        tracker: ClearMlTracker,
    ) -> HpoTrialResult:
        """
        Return deterministic completed trial results.

        Args:
            hpo_args: The HPO arguments.
            trial_config: The trial configuration.
            output_dir: The trial output directory.
            tracker: The optional ClearML tracker.

        Returns:
            The completed trial result.
        """
        score = 0.70 if trial_config.trial_id == "trial-001" else 0.90
        return make_result(
            trial_config.trial_id,
            output_dir,
            macro_f1=score,
            exact_match_accuracy=score,
            slot_accuracy=score,
            micro_f1=score,
        )

    hpo_run = run_hpo(args, tracker=tracker, trial_runner=trial_runner)

    assert hpo_run.best_trial is not None
    assert hpo_run.best_trial.trial_id == "trial-002"
    assert ("hpo_best_objective", 0.90) in tracker.single_values
    assert ("hpo_summary", hpo_run.summary) in tracker.artifacts
    assert any(table[:2] == ("hpo", "trials") for table in tracker.tables)
    assert ("hpo/objective", "trial-002", 0.90, 2) in tracker.scalars


def test_noop_clearml_tracker_is_safe(tmp_path: Path) -> None:
    """
    Verify the HPO ClearML reporter is safe when tracking is disabled.
    """
    report_hpo_to_clearml(
        ClearMlTracker(),
        tmp_path,
        {"run_id": "test"},
        [["trial_id"], ["trial-001"]],
        [],
        None,
    )


def test_build_clearml_configuration_is_secret_free(tmp_path: Path) -> None:
    """
    Verify ClearML configuration contains only run settings.
    """
    args = make_args(tmp_path)

    configuration = build_clearml_configuration(args)

    assert "CLEARML_API_SECRET_KEY" not in configuration
    assert configuration["model_name"] == "tiny-model"
