"""
Runner for collecting coverage.
"""

import json
import pathlib
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.console_logging import get_child_logger
from config.lab_settings import LabSettings

logger = get_child_logger(__file__)


class CoverageRunError(Exception):
    """
    Error for coverage collection.
    """


class CoverageCreateReportError(Exception):
    """
    Error for report creation.
    """


def get_target_score(lab_path: Path) -> int:
    """
    Get student's expectations on a final mark.

    Args:
        lab_path (Path): Path to lab

    Returns:
        int: Desired score
    """
    settings = LabSettings(lab_path / "settings.json")
    return settings.target_score


def extract_percentage_from_report(report_path: Path) -> int:
    """
    Load previous run value.

    Args:
        report_path (Path): Path to report

    Returns:
        int: Previous run value
    """
    with report_path.open(encoding="utf-8") as f:
        content = json.load(f)
    return int(content["totals"]["percent_covered_display"])


@handles_console_error()
def run_coverage_subprocess(
    lab_path: pathlib.Path, python_exe_path: pathlib.Path, mark_label: str
) -> tuple[str, str, int]:
    """
    Entrypoint for a single lab coverage collection.

    Args:
        lab_path (Path): Path to lab
        python_exe_path (Path): Path to python executable
        mark_label (str): Target score label

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    args = [
        "-m",
        "coverage",
        "run",
        "--include",
        f"{lab_path.name}/main.py,"
        f"{lab_path.name}/scraper.py,"
        f"{lab_path.name}/pipeline.py,"
        f"{lab_path.name}/pos_pipeline.py",
        "-m",
        "pytest",
        "-m",
        f"{lab_path.name}{mark_label}",
    ]
    return _run_console_tool(str(python_exe_path), args, debug=True, cwd=str(lab_path.parent))


@handles_console_error()
def run_coverage_collection(
    lab_path: Path, artifacts_path: Path, check_target_score: bool = True
) -> tuple[str, str, int]:
    """
    Entrypoint for a single lab coverage collection.

    Args:
        lab_path (Path): Path to lab
        artifacts_path (Path): Path to artifacts
        check_target_score (bool): Target score check

    Returns:
        tuple[str, str, int]: stdout, stderr, return code
    """
    logger.info(f"Processing {lab_path} ...")

    python_exe_path = choose_python_exe()
    mark = ""
    if check_target_score:
        target_score = get_target_score(lab_path)
        mark = f" and mark{target_score}"

    run_coverage_subprocess(lab_path, python_exe_path, mark)

    report_path = artifacts_path / f"{lab_path.name}.json"
    args = ["-m", "coverage", "json", "-o", str(report_path)]
    return _run_console_tool(str(python_exe_path), args, debug=True, cwd=str(lab_path.parent))
