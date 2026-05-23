"""Tests for requirement model training report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.requirement_model.clearml_tracking import ClearMlTracker
from scripts.requirement_model.train import (
    ConvergenceSettings,
    EpochMetrics,
    build_label_map_rows,
    build_model_manifest,
    build_model_manifest_rows,
    build_model_report,
    build_training_report,
    is_metric_improved,
    report_epoch_metrics_to_clearml,
    should_stop_for_convergence,
)


class RecordingTracker(ClearMlTracker):
    """
    ClearML tracker test double that records scalar reports.
    """

    __slots__ = ("scalars",)

    def __init__(self) -> None:
        """
        Initialize recorded scalar calls.
        """
        super().__init__()
        self.scalars: list[tuple[str, str, float, int]] = []

    def report_scalar(
        self, title: str, series: str, value: float, iteration: int
    ) -> None:
        """
        Record one scalar value.

        Args:
            title: The scalar chart title.
            series: The chart series name.
            value: The scalar value.
            iteration: The scalar iteration.
        """
        self.scalars.append((title, series, value, iteration))


def test_build_model_manifest_lists_files_deterministically(tmp_path: Path) -> None:
    """
    Verify model manifests include relative file paths and sizes.
    """
    (tmp_path / "b.txt").write_text("bb", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    manifest = build_model_manifest(tmp_path)

    assert manifest == [
        {"path": "a.txt", "size_bytes": 1},
        {"path": "b.txt", "size_bytes": 2},
    ]


def test_build_model_report_loads_config_and_manifest(tmp_path: Path) -> None:
    """
    Verify model reports include config, labels, manifest, and total size.
    """
    config = {"base_model_name": "tiny", "max_length": 16}
    (tmp_path / "requirement_model_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_text("weights", encoding="utf-8")

    report = build_model_report(tmp_path)

    assert report["config"] == config
    assert report["label_map"]["O"] == 0
    assert report["total_size_bytes"] > 0
    assert report["manifest"][0]["path"] == "model.safetensors"


def test_table_row_builders_include_headers() -> None:
    """
    Verify model report table helpers include header rows.
    """
    manifest_rows = build_model_manifest_rows(
        [{"path": "model.safetensors", "size_bytes": 7}]
    )
    label_rows = build_label_map_rows({"O": 0})

    assert manifest_rows == [["path", "size_bytes"], ["model.safetensors", 7]]
    assert label_rows == [["label", "id"], ["O", 0]]


def test_metric_improvement_uses_minimum_delta() -> None:
    """
    Verify convergence only replaces the best metric after enough improvement.
    """
    assert is_metric_improved(0.5, None, 0.001)
    assert is_metric_improved(0.502, 0.5, 0.001)
    assert not is_metric_improved(0.5005, 0.5, 0.001)


def test_should_stop_for_convergence_respects_min_epochs_and_patience() -> None:
    """
    Verify early stopping waits for both the minimum epoch and patience windows.
    """
    settings = ConvergenceSettings(
        max_epochs=20,
        min_epochs=3,
        patience=2,
        min_delta=0.001,
    )

    assert not should_stop_for_convergence(2, 1, settings)
    assert should_stop_for_convergence(3, 1, settings)
    assert not should_stop_for_convergence(3, 2, settings)


def test_build_training_report_records_convergence_history(tmp_path: Path) -> None:
    """
    Verify training reports capture convergence settings and epoch history.
    """
    settings = ConvergenceSettings(
        max_epochs=20,
        min_epochs=3,
        patience=3,
        min_delta=0.001,
    )

    report = build_training_report(
        model_name="distilbert-base-uncased",
        output_dir=tmp_path,
        settings=settings,
        history=[
            EpochMetrics(
                epoch=1,
                train_loss=0.8,
                validation_metrics={"macro_f1": 0.4},
            )
        ],
        best_epoch=1,
        best_metric_value=0.4,
        stopped_early=False,
    )

    assert report["model_name"] == "distilbert-base-uncased"
    assert report["convergence"]["monitor_metric"] == "macro_f1"
    assert report["best_epoch"] == 1
    assert report["history"][0]["validation_metrics"]["macro_f1"] == 0.4


def test_report_epoch_metrics_to_clearml_reports_training_and_validation() -> None:
    """
    Verify per-epoch training and validation metrics are reported to ClearML.
    """
    tracker = RecordingTracker()

    report_epoch_metrics_to_clearml(
        tracker,
        epoch_number=2,
        train_loss=0.25,
        validation_metrics={"macro_f1": 0.7, "slot_accuracy": 0.9},
    )

    assert ("loss", "train", 0.25, 2) in tracker.scalars
    assert ("validation", "macro_f1", 0.7, 2) in tracker.scalars
    assert ("validation", "slot_accuracy", 0.9, 2) in tracker.scalars
