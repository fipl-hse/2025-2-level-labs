"""
Check docstrings for conformance to the Google-style-docstrings.
"""

from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.console_logging import get_child_logger
from config.constants import PROJECT_ROOT

logger = get_child_logger(__file__)


@handles_console_error()
def check_with_pydoctest(path_to_config: Path) -> tuple[str, str, int]:
    """
    Check docstrings in files with pydoctest module.

    Args:
        path_to_config (Path): Path to pydoctest config

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    pydoctest_args = ["--config", str(path_to_config), "--verbosity", "2"]
    return _run_console_tool("pydoctest", pydoctest_args, debug=True, cwd=PROJECT_ROOT)


@handles_console_error()
def check_with_pydocstyle() -> tuple[str, str, int]:
    """
    Check docstrings in files with pydocstyle module.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    pydocstyle_args = ["-m", "pydocstyle", "--count"]
    return _run_console_tool(
        str(choose_python_exe()), pydocstyle_args, debug=True, cwd=PROJECT_ROOT
    )


def main() -> None:
    """
    Check docstrings for labs, config and core_utils packages.
    """
    pydoctest_config = PROJECT_ROOT / "config" / "static_checks" / "pydoctest.json"

    check_with_pydoctest(pydoctest_config)


if __name__ == "__main__":
    main()
