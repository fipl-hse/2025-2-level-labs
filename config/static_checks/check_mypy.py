"""
Check mypy for type checking in Python code.
"""

# pylint: disable=duplicate-code
from os import listdir
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.console_logging import get_child_logger
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.lab_settings import LabSettings
from config.project_config import ProjectConfig

logger = get_child_logger(__file__)


@handles_console_error()
def check_mypy_on_paths(paths: list[Path], path_to_config: Path) -> tuple[str, str, int]:
    """
    Run mypy checks for the project.

    Args:
        paths (list[Path]): Paths to the projects.
        path_to_config (Path): Path to the config.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    mypy_args = [
        "-m",
        "mypy",
        *map(str, filter(lambda x: x.exists(), paths)),
        "--config-file",
        str(path_to_config),
    ]

    return _run_console_tool(str(choose_python_exe()), mypy_args, debug=True)


def main() -> None:
    """
    Run mypy checks for the project.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths()
    addons = project_config.get_addons_names()

    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    if addons:
        logger.info(f"Running mypy on {' '.join(addons)}")
        check_mypy_on_paths(
            [PROJECT_ROOT / addon for addon in addons],
            pyproject_path,
        )

    for lab_name in labs_list:
        lab_path = PROJECT_ROOT / lab_name
        if "settings.json" in listdir(lab_path):
            target_score = LabSettings(PROJECT_ROOT / f"{lab_path}/settings.json").target_score

            if target_score > 7:
                logger.info(f"Running mypy for lab {lab_path}")
                check_mypy_on_paths([lab_path], pyproject_path)


if __name__ == "__main__":
    main()
