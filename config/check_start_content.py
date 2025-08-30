"""
Some simple checks for start.py lab files.
"""

import argparse
import sys

from config.console_logging import get_child_logger

logger = get_child_logger(__file__)


def check_assert_line(content: str) -> bool:
    """
    Check assert line.

    Args:
        content (str): Content

    Returns:
        bool: Is expected in content or not
    """
    expected = "assert result"
    expected_alternative = "assert RESULT"
    return expected in content or expected_alternative in content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checks start.py files and tests them")
    parser.add_argument("--start_py_content", type=str, help="Content of start.py for each lab")
    args: argparse.Namespace = parser.parse_args()

    if check_assert_line(args.start_py_content):
        logger.info("Passed")
        sys.exit(0)
    logger.info("Make sure you made assert result in start.py file")
    sys.exit(1)

# Test: a) assert RESULT is in file
# Test: b) call start.py, expect it does not throw Error
