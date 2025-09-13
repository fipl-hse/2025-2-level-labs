"""
Check newline at the end of a file.
"""

import sys

from config.console_logging import get_child_logger
from config.constants import PROJECT_ROOT

logger = get_child_logger(__file__)


def get_paths() -> list:
    """
    Get paths to non-python files.

    Returns:
        list: Paths to non-python files
    """
    paths_to_exclude = {
        "venv",
        ".venv",
        ".git",
        ".idea",
        ".coverage",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "build",
        "russian-syntagrus-ud-2.0-170801.udpipe",
        "tmp",
        "intersphinx",
        "test_tmp",
        "dist",
    }

    only_sources = [file for file in PROJECT_ROOT.iterdir() if file.name not in paths_to_exclude]

    list_with_paths = []
    for source_file in only_sources:
        if not source_file.is_dir():
            list_with_paths.append(source_file)
            continue

        list_with_paths.extend(
            [
                file
                for file in source_file.rglob("*")
                if not set(i.name for i in file.parents) & paths_to_exclude
            ]
        )
    return list_with_paths


def check_paths(list_with_paths: list) -> list:
    """
    Check if the path is correct.

    Args:
        list_with_paths (list): Paths to non-python files

    Returns:
        list: Appropriate paths
    """
    paths_to_exclude = ["1_raw.txt", "__init__.cpython-310.pyc", "test_params.cpython-310.pyc"]
    bad_endings = [".jpg", ".png", ".pkl", ".udpipe", ".zip"]
    paths = []
    for path in sorted(list_with_paths):
        is_file = path.is_file() and path.stat().st_size != 0
        is_ok_file = (
            path.name not in paths_to_exclude
            and "__pycache__" not in str(path)
            and path.suffix not in bad_endings
        )
        if is_file and is_ok_file:
            paths.append(path)
    return paths


def has_newline(paths: list) -> bool:
    """
    Check for a newline at the end.

    Args:
        paths (list): Appropriate paths

    Returns:
        bool: Has newline or not
    """
    bad_paths = []
    check_is_good = True
    for path in paths:
        logger.info(f"Analyzing {path}")
        with open(path, encoding="utf-8") as file:
            lines = file.readlines()
        if lines[-1][-1] != "\n":
            bad_paths.append(path)
            check_is_good = False
    if check_is_good:
        logger.info("All files conform to the template.")
    else:
        for bad_path in bad_paths:
            logger.info(f"No newline at the end of the {bad_path}")
    return check_is_good


def main() -> None:
    """
    Entrypoint for module.
    """
    list_with_paths = get_paths()
    paths = check_paths(list_with_paths)
    result = has_newline(paths)
    sys.exit(not result)


if __name__ == "__main__":
    main()
