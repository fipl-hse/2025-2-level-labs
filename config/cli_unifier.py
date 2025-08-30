"""
CLI commands.
"""

# pylint: disable=subprocess-run-check

import functools
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from config.console_logging import get_child_logger
from config.constants import CONFIG_PACKAGE_PATH

logger = get_child_logger(__file__)

BAD_OUTPUTS = ("", "0\n", "\r\n", "\r", b"0\r\n", "\n")


def convert_raw_output_to_str(content: bytes) -> str:
    """
    Converts result of the command-line process output to str.

    Args:
        content(bytes): raw result from the subprocess call

    Returns:
        str: string representation
    """
    return content.decode("utf-8").replace("\r", "")


def log_output(output_type: str, content: bytes | str) -> None:
    """
    Prints result of the command-line process in specific template.

    Args:
        output_type(str): type of output, for example stdout or stderr
        content(bytes | str): raw result from the subprocess call
    """
    with open(
        str(Path(CONFIG_PACKAGE_PATH) / "assets" / "console_tool_log_template.txt"),
        encoding="utf-8",
    ) as f:
        output_template = f.read()

    logger.info(
        output_template.format(
            output_type=output_type,
            content=convert_raw_output_to_str(content) if isinstance(content, bytes) else content,
        )
    )


def choose_python_exe() -> Path:
    """
    Select python binary path depending on current OS.

    Returns:
        Path: A path to python exe
    """
    lab_path = Path(__file__).parent.parent
    if platform.system() == "Windows":
        python_exe_path = lab_path / "venv" / "Scripts" / "python.exe"
    else:
        python_exe_path = lab_path / "venv" / "bin" / "python"
    return python_exe_path


def prepare_args_for_shell(args: list[object]) -> str:
    """
    Prepare argument for CLI.

    Args:
        args (list[object]): arguments to join

    Returns:
        str: arguments for CLI
    """
    return " ".join(map(str, args))


def modify_path(path: str) -> str:
    """
    Run CLI commands.

    Args:
        path (str): A path modify

    Returns:
        str: Modified path in str format
    """
    pattern_to_remove = r"/home/runner/work/[^/]+/[^/]+/"

    pattern_python_end = r"python$"

    if re.search(pattern_python_end, path):
        return "python"

    modified_path = re.sub(pattern_to_remove, "", path)

    return modified_path


def _run_console_tool(exe: str, /, args: list[str], **kwargs: Any) -> tuple[str, str, int]:
    """
    Run CLI commands.

    Args:
        exe (str): A path to python exe
        args (list[str]): Arguments
        **kwargs (Any): Options

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    kwargs_processed: list[str] = []
    for item in kwargs.items():
        if item[0] in ("env", "debug", "cwd"):
            continue
        kwargs_processed.extend(map(str, item))

    options = [str(exe), *args, *kwargs_processed]

    if kwargs.get("debug", False):
        arguments = []
        for index, option in enumerate(options[1:]):
            arguments.append(
                f'"{modify_path(option)}"'
                if "--" in options[index] or "-m" in options[index]
                else modify_path(option)
            )
        logger.info(
            f"Attempting to run with the following arguments: "
            f'{" ".join([modify_path(str(exe)), *arguments])}'
        )

    env = kwargs.get("env")
    if env:
        result = subprocess.run(options, capture_output=True, env=env)
    elif kwargs.get("cwd"):
        result = subprocess.run(options, capture_output=True, cwd=kwargs.get("cwd"))
    else:
        result = subprocess.run(options, capture_output=True)
    return (
        convert_raw_output_to_str(result.stdout),
        convert_raw_output_to_str(result.stderr),
        result.returncode,
    )


def handles_console_error(
    exit_code_on_error: int = 1, ok_codes: tuple[int, ...] = (0,)
) -> Callable:
    """
    Decorator to handle console tool errors.

    Args:
        exit_code_on_error (int): Exit code to use when an error occurs.
        ok_codes (tuple[int, ...]): Exit codes considered as success. Defaults to (0,).

    Returns:
        Callable: The wrapped function.
    """

    def decorator(func: Callable[[Any], tuple[str, str, int]]) -> Callable:
        """
        Decorator to handle console tool errors.

        Args:
            func (Callable): The function to be decorated, expected to return a `CompletedProcess`.

        Returns:
            Callable: The wrapped function with error handling.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[str, str, int]:
            """
            Wrapper function to handle console tool errors.

            Args:
                *args (Any): Variable length argument list to pass to the decorated function.
                **kwargs (Any): Arbitrary keyword arguments to pass to the decorated function.
            """
            try:
                logger.info(f"Call to {func.__name__}")
                stdout, stderr, return_code = func(*args, **kwargs)
            except subprocess.CalledProcessError as error:
                logger.info(f"Exit code: {error.returncode}.\n")
                if error.output and error.output not in BAD_OUTPUTS:
                    log_output("Console run stdout", error.output)
                # return code is not 0, but sometimes it still can be OK
                if error.returncode in ok_codes:
                    logger.info(f"Exit code: {error.returncode}.\n")
                    if error.output and error.output not in BAD_OUTPUTS:
                        log_output("Console run stdout", error.output)
                    if error.stderr and error.stderr not in BAD_OUTPUTS:
                        log_output("Console run stderr", error.stderr)
                    return (
                        convert_raw_output_to_str(error.output),
                        convert_raw_output_to_str(error.stderr),
                        error.returncode,
                    )

                logger.error(f"Check failed with exit code {error.returncode}.")
                if error.stderr and error.stderr not in BAD_OUTPUTS:
                    log_output("Console run stderr", error.stderr)
                sys.exit(exit_code_on_error)
            else:
                logger.info(f"Exit code: {return_code}.\n")
                if stdout and stdout not in BAD_OUTPUTS:
                    log_output("Console run stdout", stdout)
                return stdout, stderr, return_code

        return wrapper

    return decorator
