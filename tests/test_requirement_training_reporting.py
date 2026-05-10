"""Tests for requirement model training report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.requirement_model.train import (
    build_label_map_rows,
    build_model_manifest,
    build_model_manifest_rows,
    build_model_report,
)


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
