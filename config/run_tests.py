"""
Run tests for each lab using pytest.
"""

from tap import Tap

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.collect_coverage.run_coverage import get_target_score
from config.console_logging import get_child_logger
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig

logger = get_child_logger(__file__)


class CommandLineInterface(Tap):
    """
    Types for the argument parser.
    """

    pr_name: str
    pr_author: str
    lab_path: str | None = None
    pytest_label: str | None = None


def prepare_pytest_args(
    lab_path: str, target_score: int, pytest_label: str | None = None
) -> list[str]:
    """
    Build the arguments for running pytest.

    Args:
        lab_path (str): Path to the lab.
        target_score (int): Target score for the lab.
        pytest_label (str | None): Label for pytest.

    Returns:
        list[str]: List of arguments for pytest.
    """
    if pytest_label is None:
        pytest_label = lab_path

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    lab_config = project_config.get_lab_config(lab_path)

    pytest_args = [
        "-m",
        f"mark{target_score} and {pytest_label}" if lab_path else pytest_label,
        "--capture=no",
    ]
    if lab_config and lab_config.settings:
        if ignored_folders := lab_config.settings.ignore:
            pytest_args.extend(f"--ignore={folder}" for folder in ignored_folders)

    logger.info(pytest_args)
    return pytest_args


@handles_console_error(ok_codes=(0, 5))
def run_pytest(pytest_args: list[str]) -> tuple[str, str, int]:
    """
    Run pytest with the given arguments.

    Args:
        pytest_args (list[str]): Arguments for pytest.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    args = ["-m", "pytest", *pytest_args]
    return _run_console_tool(str(choose_python_exe()), args, cwd=PROJECT_ROOT, debug=True)


def check_skip(lab_path: str) -> bool:
    """
    Exit if skip conditions are met.

    Args:
        lab_path (str): Path to the lab.

    Returns:
        bool: True if should be skipped
    """
    if lab_path:
        score_path = PROJECT_ROOT / lab_path
        score = get_target_score(lab_path=score_path)
        if score == 0:
            logger.info(f"Skipping check due to no mark for lab {lab_path}.")
            return True

    logger.info(f"No special reasons for skipping {lab_path}!")
    return False


def main() -> None:
    """
    Main function to run tests for only one lab or for one by one.
    """
    args = CommandLineInterface(underscores_to_dashes=True).parse_args()

    if args.lab_path:
        if check_skip(args.lab_path):
            return
        target_score = get_target_score(PROJECT_ROOT / args.lab_path)
        pytest_args = prepare_pytest_args(args.lab_path, target_score, args.pytest_label)

        _, _, return_code = run_pytest(pytest_args)
        if return_code == 5:
            logger.info(
                f"This combination of mark and label "
                f"doesn't match any tests for {args.lab_path}."
            )

    else:
        project_config = ProjectConfig(PROJECT_CONFIG_PATH)
        labs = project_config.get_labs_names()

        logger.info(f"Current scope: {labs}")

        for lab_name in labs:
            if check_skip(lab_name):
                continue
            logger.info(f"Running tests for lab {lab_name}")

            target_score = get_target_score(PROJECT_ROOT / lab_name)
            pytest_args = prepare_pytest_args(lab_name, target_score)

            _, _, return_code = run_pytest(pytest_args)
            if return_code == 5:
                logger.info(
                    f"This combination of mark and label "
                    f"doesn't match any tests for {lab_name}."
                )


if __name__ == "__main__":
    main()
