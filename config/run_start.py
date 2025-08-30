"""
Run start.
"""

from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.collect_coverage.run_coverage import get_target_score
from config.console_logging import get_child_logger
from config.constants import CONFIG_PACKAGE_PATH, PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig

logger = get_child_logger(__file__)


@handles_console_error()
def run_start(lab_name: str) -> tuple[str, str, int]:
    """
    Run start.py script in the specified lab directory.

    Args:
        lab_name (str): Name of the lab directory.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool(
        str(choose_python_exe()), [str("start.py")], cwd=PROJECT_ROOT / lab_name, debug=True
    )


@handles_console_error()
def check_start_content(lab_name: str) -> tuple[str, str, int]:
    """
    Check the content of start.py script using check_start_content.py script.

    Args:
        lab_name (str): Name of the lab directory.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    start_py_file = PROJECT_ROOT / lab_name / "start.py"
    with start_py_file.open() as f:
        start_py_content = f.read()

    return _run_console_tool(
        str(choose_python_exe()),
        [
            str(Path(CONFIG_PACKAGE_PATH, "check_start_content.py")),
            "--start_py_content",
            start_py_content,
        ],
        cwd=PROJECT_ROOT,
        debug=True,
    )


def main() -> None:
    """
    Main function to run start.py checks for each lab.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs = project_config.get_labs_names()

    for lab_name in labs:
        logger.info(f"Running start.py checks for lab {lab_name}")

        target_score = get_target_score(PROJECT_ROOT / lab_name)

        if target_score == 0:
            logger.info("Skipping stage. Target score is 0.")
            continue
        run_start(lab_name)

        logger.info(f"Check calling lab {lab_name} passed")

        check_start_content(lab_name)

    logger.info("All start.py checks passed.")


if __name__ == "__main__":
    main()
