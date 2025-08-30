"""
Script for updating student`s forks.
"""

import json
from pathlib import Path

from tap import Tap

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.console_logging import get_child_logger
from config.constants import CONFIG_PACKAGE_PATH

logger = get_child_logger(__file__)


class CommandLineInterface(Tap):
    """
    Script to update student`s forks with latest main branch.
    """

    config: Path  # Path to json configuration


@handles_console_error()
def update_fork(
    python: Path,
    repositories: dict[str, str],
    authentication: dict[str, str],
    strategy: str,
    paths_to_keep: dict[str, list[str]],
) -> tuple[str, str, int]:
    """
    Update fork to upstream/main with update_fork.py.

    Args:
        python (Path): Path to python executable
        repositories (dict[str, str]): dict with URLs to repositories
        authentication (dict[str, str]): dict with username and GitHub token
        strategy (str): strategy to update student`s repository
        paths_to_keep (dict[str, list[str]]): dict with files to keep as is from upstream and fork

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    args = [
        str(CONFIG_PACKAGE_PATH / "github" / "update_fork.py"),
        "--fork",
        repositories["fork"],
        "--strategy",
        strategy,
        "--user",
        authentication["user"],
        "--auth",
        authentication["token"],
        "--upstream",
        repositories["upstream"],
    ]
    if paths_to_keep["fork"]:
        args.extend(["--paths-keep-from-fork", *paths_to_keep["fork"]])

    if paths_to_keep["upstream"]:
        args.extend(["--paths-keep-from-upstream", *paths_to_keep["upstream"]])
    return _run_console_tool(str(python), args=args, debug=True)


def update_forks(
    python: Path,
    authentication: dict[str, str],
    repositories: dict,
    strategy: str,
    paths_to_keep: dict[str, list[str]],
) -> None:
    """
    Update forks to upstream/main with update_fork.py.

    Args:
        python (Path): Path to python executable
        authentication (dict[str, str]): dict with username and GitHub token
        repositories (dict): dict with upstream and forks
        strategy (str): strategy to update student`s repository
        paths_to_keep (dict[str, list[str]]): dict with files to keep as is from upstream and fork
    """
    upstream = repositories["upstream"]
    for fork in repositories["forks"]:
        logger.info(f"Start update fork: {fork}")
        update_fork(
            python=python,
            repositories={
                "fork": fork,
                "upstream": upstream,
            },
            authentication=authentication,
            strategy=strategy,
            paths_to_keep=paths_to_keep,
        )


def main(configuration_path: Path) -> None:
    """
    Main function.

    Args:
        configuration_path (Path): Path to json configuration
    """
    with configuration_path.open() as configuration_file:
        configuration = json.load(configuration_file)

    python_exe_path = choose_python_exe()
    authentication = configuration["authentication"]
    upstream = configuration["upstream"]

    winners = configuration["winners"]
    update_forks(
        python=python_exe_path,
        authentication=authentication,
        repositories={
            "upstream": upstream,
            "forks": winners["forks"],
        },
        strategy="winner",
        paths_to_keep=winners["pathsToKeep"],
    )

    losers = configuration["losers"]
    update_forks(
        python=python_exe_path,
        authentication=authentication,
        strategy="loser",
        repositories={
            "upstream": upstream,
            "forks": losers["forks"],
        },
        paths_to_keep=losers["pathsToKeep"],
    )


if __name__ == "__main__":
    ARGUMENTS = CommandLineInterface(underscores_to_dashes=True).parse_args()
    main(configuration_path=ARGUMENTS.config)
