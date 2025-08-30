"""
Root logger configuration etc.
"""

from logging import getLogger, Logger
from os.path import sep
from pathlib import Path

from logging518.config import fileConfig

from config.constants import PROJECT_ROOT

fileConfig(PROJECT_ROOT / "pyproject.toml")


def get_root_logger() -> Logger:
    """
    Get the root logger of the project.

    Returns:
        Logger: Instance of root logger
    """
    return getLogger(" ")


def get_child_logger(file_path: str) -> Logger:
    """
    Get the child logger from root logger by path.

    Args:
        file_path (str): File path

    Returns:
        Logger: Instance of child logger
    """
    root_logger = get_root_logger()
    child_suffix = file_path
    if Path(file_path).is_relative_to(PROJECT_ROOT):
        child_suffix = str(Path(file_path).relative_to(PROJECT_ROOT))
    return root_logger.getChild(f"{sep}{child_suffix}")
