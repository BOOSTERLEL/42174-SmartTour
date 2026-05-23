"""Tests for requirement model comparison orchestration helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.requirement_model.clearml_tracking import ClearMlTracker
from scripts.requirement_model.compare_models import (
    EVALUATION_SPLITS,
    ModelComparisonResult,
    build_comparison_metric_rows,
    promote_winner_model,
    rank_results,
    run_comparison,
    slugify_model_name,
)


def make_result(
    model_name: str,
    output_dir: Path,
    reviewed_slot_accuracy: float,
    reviewed_exact_match: float,
    reviewed_macro_f1: float,
    validation_macro_f1: float,
) -> ModelComparisonResult:
    """
    Build a minimal model comparison result for tests.

    Args:
        model_name: The model name.
        output_dir: The model output directory.
        reviewed_slot_accuracy: The reviewed-test slot accuracy.
        reviewed_exact_match: The reviewed-test exact-match accuracy.
        reviewed_macro_f1: The reviewed-test macro F1.
        validation_macro_f1: The validation macro F1.

    Returns:
        A model comparison result.
    """
    metrics_by_split = {
        "validation": {
            "slot_accuracy": 0.7,
            "exact_match_accuracy": 0.2,
            "micro_f1": 0.6,
            "macro_f1": validation_macro_f1,
        },
        "test": {
            "slot_accuracy": 0.8,
            "exact_match_accuracy": 0.3,
            "micro_f1": 0.7,
            "macro_f1": 0.6,
        },
        "reviewed_test": {
            "slot_accuracy": reviewed_slot_accuracy,
            "exact_match_accuracy": reviewed_exact_match,
            "micro_f1": 0.8,
            "macro_f1": reviewed_macro_f1,
        },
    }
    return ModelComparisonResult(
        model_name=model_name,
        model_slug=slugify_model_name(model_name),
        output_dir=output_dir,
        training_report={"epochs_ran": 4},
        metrics_by_split=metrics_by_split,
        diagnostics_by_split={split_name: {} for split_name in EVALUATION_SPLITS},
    )


def test_rank_results_uses_declared_metric_tie_breakers(tmp_path: Path) -> None:
    """
    Verify ranking uses reviewed slot accuracy before later tie breakers.
    """
    weaker_result = make_result(
        "model-a",
        tmp_path / "a",
        reviewed_slot_accuracy=0.90,
        reviewed_exact_match=0.90,
        reviewed_macro_f1=0.90,
        validation_macro_f1=0.90,
    )
    stronger_result = make_result(
        "model-b",
        tmp_path / "b",
        reviewed_slot_accuracy=0.91,
        reviewed_exact_match=0.10,
        reviewed_macro_f1=0.10,
        validation_macro_f1=0.10,
    )

    ranked_results = rank_results([weaker_result, stronger_result])

    assert ranked_results[0] == stronger_result


def test_rank_results_uses_exact_match_tie_breaker(tmp_path: Path) -> None:
    """
    Verify exact-match accuracy breaks reviewed slot-accuracy ties.
    """
    first_result = make_result(
        "model-a",
        tmp_path / "a",
        reviewed_slot_accuracy=0.90,
        reviewed_exact_match=0.50,
        reviewed_macro_f1=0.95,
        validation_macro_f1=0.95,
    )
    second_result = make_result(
        "model-b",
        tmp_path / "b",
        reviewed_slot_accuracy=0.90,
        reviewed_exact_match=0.60,
        reviewed_macro_f1=0.10,
        validation_macro_f1=0.10,
    )

    ranked_results = rank_results([first_result, second_result])

    assert ranked_results[0] == second_result


def test_build_comparison_metric_rows_contains_every_split(tmp_path: Path) -> None:
    """
    Verify comparison rows expose every top-level metric for every split.
    """
    result = make_result(
        "model-a",
        tmp_path / "a",
        reviewed_slot_accuracy=0.90,
        reviewed_exact_match=0.50,
        reviewed_macro_f1=0.80,
        validation_macro_f1=0.70,
    )

    rows = build_comparison_metric_rows([result])

    assert rows[0] == [
        "model_name",
        "model_slug",
        "split",
        "slot_accuracy",
        "exact_match_accuracy",
        "micro_f1",
        "macro_f1",
    ]
    assert {row[2] for row in rows[1:]} == set(EVALUATION_SPLITS)


def test_promote_winner_model_replaces_default_atomically(tmp_path: Path) -> None:
    """
    Verify promotion copies a complete winner over the default artifact path.
    """
    source_dir = tmp_path / "winner"
    target_dir = tmp_path / "latest"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "requirement_model_config.json").write_text(
        '{"base_model_name": "winner"}',
        encoding="utf-8",
    )
    (target_dir / "requirement_model_config.json").write_text(
        '{"base_model_name": "old"}',
        encoding="utf-8",
    )

    promote_winner_model(source_dir, target_dir)

    assert "winner" in (target_dir / "requirement_model_config.json").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / ".latest.previous").exists()


def test_promote_winner_model_requires_config(tmp_path: Path) -> None:
    """
    Verify incomplete artifacts are not promoted.
    """
    source_dir = tmp_path / "winner"
    target_dir = tmp_path / "latest"
    source_dir.mkdir()
    target_dir.mkdir()
    (target_dir / "requirement_model_config.json").write_text(
        '{"base_model_name": "old"}',
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        promote_winner_model(source_dir, target_dir)

    assert "old" in (target_dir / "requirement_model_config.json").read_text(
        encoding="utf-8"
    )


def test_run_comparison_does_not_promote_when_candidate_fails(
    tmp_path: Path,
) -> None:
    """
    Verify default promotion waits until every candidate finishes.
    """
    did_promote = False
    args = argparse.Namespace(
        experiment_dir=tmp_path / "experiments",
        default_model_dir=tmp_path / "latest",
        model_name=("good-model", "bad-model"),
        register_winner=False,
    )

    def train_evaluate_model(
        comparison_args: argparse.Namespace,
        model_name: str,
        output_dir: Path,
        tracker: ClearMlTracker,
    ) -> ModelComparisonResult:
        """
        Return one successful result, then fail the second candidate.

        Args:
            comparison_args: The comparison arguments.
            model_name: The candidate model name.
            output_dir: The output directory.
            tracker: The optional ClearML tracker.

        Returns:
            A model result for the successful candidate.
        """
        if model_name == "bad-model":
            raise RuntimeError("training failed")
        return make_result(
            model_name,
            output_dir,
            reviewed_slot_accuracy=0.9,
            reviewed_exact_match=0.5,
            reviewed_macro_f1=0.8,
            validation_macro_f1=0.7,
        )

    def promote_model(source_dir: Path, target_dir: Path) -> None:
        """
        Record promotion attempts.

        Args:
            source_dir: The winner source directory.
            target_dir: The default target directory.
        """
        nonlocal did_promote
        did_promote = True

    with pytest.raises(RuntimeError, match="training failed"):
        run_comparison(
            args,
            tracker=ClearMlTracker(),
            train_evaluate_model=train_evaluate_model,
            promote_model=promote_model,
        )

    assert not did_promote
