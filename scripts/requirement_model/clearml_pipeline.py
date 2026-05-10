"""Run a local ClearML pipeline for requirement model reporting."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.clearml_tracking import (
    DEFAULT_CLEARML_PROJECT,
    clear_task_script_diff,
    ensure_clearml_environment,
    force_clearml_requirements_env_freeze,
)

DEFAULT_DATA_DIR = Path("data/requirement_model")
DEFAULT_LATEST_MODEL_DIR = Path("models/requirement_model/latest")
DEFAULT_QUICK_MODEL_DIR = Path("models/requirement_model/quick")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the requirement model ClearML reporting pipeline."
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--evaluation-model-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--clearml-register-model", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> str:
    """
    Run one pipeline step command.

    Args:
        command: The command argument list.

    Returns:
        A short completion marker.
    """
    print("running: " + " ".join(command))
    subprocess.run(command, check=True)
    return "completed"


def run_audit_step(
    data_dir: str,
    project_name: str,
    dataset_version: str,
) -> str:
    """
    Run the detailed data audit step.

    Args:
        data_dir: The dataset directory.
        project_name: The ClearML project name.
        dataset_version: The ClearML dataset version.

    Returns:
        A short completion marker.
    """
    return run_command(
        [
            sys.executable,
            "scripts/requirement_model/audit_data.py",
            "--data-dir",
            data_dir,
            "--reviewed-test",
            "--strict",
            "--clearml",
            "--clearml-project",
            project_name,
            "--clearml-report-data-profile",
            "--clearml-dataset-version",
            dataset_version,
            "--clearml-task-name",
            "Requirement Model Pipeline Data Audit",
        ]
    )


def run_training_step(
    project_name: str,
    quick: bool,
    device: str,
    batch_size: int,
    epochs: int,
    should_register_model: bool,
) -> str:
    """
    Run the detailed training report step.

    Args:
        project_name: The ClearML project name.
        quick: Whether to run quick smoke training.
        device: The training device.
        batch_size: The training batch size.
        epochs: The training epoch count.
        should_register_model: Whether to register the trained model.

    Returns:
        A short completion marker.
    """
    command = [
        sys.executable,
        "scripts/requirement_model/train.py",
        "--clearml",
        "--clearml-project",
        project_name,
        "--clearml-model-report",
        "--clearml-task-name",
        "Requirement Model Pipeline Training",
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
    ]
    if quick:
        command.append("--quick")
    if should_register_model:
        command.append("--clearml-register-model")
    return run_command(command)


def run_evaluation_step(
    project_name: str,
    model_dir: str,
) -> str:
    """
    Run the detailed evaluation report step.

    Args:
        project_name: The ClearML project name.
        model_dir: The model directory to evaluate.

    Returns:
        A short completion marker.
    """
    return run_command(
        [
            sys.executable,
            "scripts/requirement_model/evaluate.py",
            "--model-dir",
            model_dir,
            "--split",
            "reviewed_test",
            "--clearml",
            "--clearml-project",
            project_name,
            "--clearml-detailed-report",
            "--clearml-task-name",
            "Requirement Model Pipeline Evaluation",
        ]
    )


def build_pipeline(
    project_name: str,
    data_dir: Path,
    model_dir: Path,
    quick: bool,
    device: str,
    batch_size: int,
    epochs: int,
    should_register_model: bool,
) -> Any:
    """
    Build the local ClearML pipeline controller.

    Args:
        project_name: The ClearML project name.
        data_dir: The dataset directory.
        model_dir: The model directory to evaluate.
        quick: Whether the training step uses quick mode.
        device: The training device.
        batch_size: The training batch size.
        epochs: The training epoch count.
        should_register_model: Whether to register the trained model.

    Returns:
        The configured ClearML pipeline controller.
    """
    from clearml import PipelineController, Task

    force_clearml_requirements_env_freeze(Task)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pipeline = PipelineController(
        name="Requirement Model Detailed Workflow",
        project=project_name,
        version=timestamp,
        add_pipeline_tags=True,
        target_project=project_name,
    )
    current_task: Any | None = Task.current_task()
    if current_task is not None:
        clear_task_script_diff(current_task)
    pipeline.add_function_step(
        name="audit_data",
        function=run_audit_step,
        function_kwargs={
            "data_dir": str(data_dir),
            "project_name": project_name,
            "dataset_version": f"pipeline-detailed-{timestamp}",
        },
        function_return=["status"],
        project_name=project_name,
        task_name="Requirement Model Pipeline Data Audit Step",
        helper_functions=[run_command],
    )
    pipeline.add_function_step(
        name="train_model",
        function=run_training_step,
        function_kwargs={
            "project_name": project_name,
            "quick": quick,
            "device": device,
            "batch_size": batch_size,
            "epochs": epochs,
            "should_register_model": should_register_model,
        },
        function_return=["status"],
        project_name=project_name,
        task_name="Requirement Model Pipeline Training Step",
        parents=["audit_data"],
        helper_functions=[run_command],
    )
    pipeline.add_function_step(
        name="evaluate_model",
        function=run_evaluation_step,
        function_kwargs={
            "project_name": project_name,
            "model_dir": str(model_dir),
        },
        function_return=["status"],
        project_name=project_name,
        task_name="Requirement Model Pipeline Evaluation Step",
        parents=["train_model"],
        helper_functions=[run_command],
    )
    return pipeline


def resolve_model_dir(args: argparse.Namespace) -> Path:
    """
    Resolve the model directory used by the evaluation step.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The model directory.
    """
    if args.evaluation_model_dir is not None:
        return args.evaluation_model_dir
    if args.quick:
        return DEFAULT_QUICK_MODEL_DIR
    return DEFAULT_LATEST_MODEL_DIR


def main() -> None:
    """
    Run the ClearML pipeline.
    """
    args = parse_args()
    if not args.local:
        raise ValueError("this workflow only supports --local execution")
    ensure_clearml_environment()
    pipeline = build_pipeline(
        project_name=args.clearml_project,
        data_dir=args.data_dir,
        model_dir=resolve_model_dir(args),
        quick=args.quick,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        should_register_model=args.clearml_register_model,
    )
    pipeline.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    main()
