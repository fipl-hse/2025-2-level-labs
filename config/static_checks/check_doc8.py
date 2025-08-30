"""
Check doc8 for style checking of rst files.
"""

# pylint: disable=duplicate-code
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.console_logging import get_child_logger
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig

logger = get_child_logger(__file__)


@handles_console_error()
def check_doc8_on_paths(paths: list[Path], path_to_config: Path) -> tuple[str, str, int]:
    """
    Run doc8 checks for the project.

    Args:
        paths (list[Path]): Paths to the projects.
        path_to_config (Path): Path to the config.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    doc8_args = [
        "-m",
        "doc8",
        *map(str, filter(lambda x: x.exists(), paths)),
        "--config",
        str(path_to_config),
    ]

    return _run_console_tool(str(choose_python_exe()), doc8_args, debug=True)


def main() -> None:
    """
    Run doc8 checks for the project.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths(include_addons=False)

    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    logger.info("Running doc8 for main docs")
    rst_main_files = list(PROJECT_ROOT.glob("*rst"))
    check_doc8_on_paths(rst_main_files, pyproject_path)

    logger.info("Running doc8 for other docs")
    docs_path = PROJECT_ROOT / "docs"
    rst_files = list(docs_path.rglob("*.rst"))

    check_doc8_on_paths(rst_files, pyproject_path)

    for lab_name in labs_list:
        lab_path = PROJECT_ROOT / lab_name
        rst_labs_files = lab_path.rglob("*.rst")
        logger.info(f"Running doc8 for lab {lab_path}")
        check_doc8_on_paths(rst_labs_files, pyproject_path)


if __name__ == "__main__":
    main()
