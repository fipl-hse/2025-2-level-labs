"""
Check availability of __init__.py in every directory except for excluded ones.
"""

import sys
from pathlib import Path

from config.console_logging import get_child_logger
from config.constants import PROJECT_ROOT

logger = get_child_logger(__file__)


def main() -> None:
    """
    Checks that each directory has __init__.py.
    """
    excluded_dirs = ["venv", ".git", "__pycache__"]
    project_root = Path(PROJECT_ROOT)

    missing_init = []

    for directory in project_root.rglob("*"):
        if directory.is_dir() and any(excluded in str(directory) for excluded in excluded_dirs):
            continue
        python_files = list(directory.glob("*.py"))
        if python_files and not (directory / "__init__.py").exists():
            missing_init.append(str(directory))

    if missing_init:
        logger.error("Error: __init__.py was not found in following directories:")
        for path in missing_init:
            print(f"- {path}")
        sys.exit(1)
    logger.info("All directories with Python-files contain __init__.py.")
    sys.exit(0)


if __name__ == "__main__":
    main()
