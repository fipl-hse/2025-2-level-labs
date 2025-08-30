"""
Script for updating student`s fork.
"""

import sys
import tempfile
from enum import Enum
from pathlib import Path

from tap import Tap

from config.cli_unifier import _run_console_tool, handles_console_error
from config.console_logging import get_child_logger

logger = get_child_logger(__file__)


class Strategies(Enum):
    """
    Strategies to update fork.
    """

    WINNER = "winner"
    LOSER = "loser"


class CommandLineInterface(Tap):
    """
    Script to update student fork with latest main branch.
    """

    fork: str
    upstream: str
    strategy: Strategies

    paths_keep_from_fork: tuple[str, ...] = ()
    paths_keep_from_upstream: tuple[str, ...] = ()

    merge_commit_message: str = "Update repository with upstream/main"
    user: str
    auth: str


class RemoteBranches(Enum):
    """
    Remote branches.
    """

    UPSTREAM = "upstream/main"
    ORIGIN = "origin/main"


def create_fork_url_with_auth(fork_url: str, authentication_token: str) -> str:
    """
    Create a link to fetch and push student`s repository.

    Args:
        fork_url (str): An URL to student`s fork
        authentication_token (str): GitHub access token

    Returns:
        str: the URL to student`s fork with embedded GitHub token to work with fork
    """
    http_prefix = "https://"
    return f"https://x-access-token:{authentication_token}@{fork_url[len(http_prefix):]}"


@handles_console_error()
def clone_fork(fork_url: str, root_dir: Path) -> tuple[str, str, int]:
    """
    Clone a repository.

    Args:
        fork_url (str): An URL to student`s fork
        root_dir (Path): Path where will be cloned the repository

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool("git", args=["clone", fork_url], cwd=root_dir, debug=True)


@handles_console_error()
def setup_repository(fork_path: Path, user: str) -> tuple[str, str, int]:
    """
    Setup the local repository: set username, email and https protocol instead of ssh.

    Args:
        fork_path (Path): Path to the local repository
        user (str): Username

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    _run_console_tool(
        "git", args=["config", "--local", "user.name", user], cwd=fork_path, debug=True
    )
    _run_console_tool(
        "git",
        args=["config", "--local", "user.email", f"${user}@users.noreply.github.com"],
        cwd=fork_path,
        debug=True,
    )
    return _run_console_tool(
        "git",
        args=["config", "--local", "url.https://github.com/.insteadOf", "git://github.com/"],
        cwd=fork_path,
        debug=True,
    )


@handles_console_error()
def add_upstream(fork_path: Path, upstream: str) -> tuple[str, str, int]:
    """
    Add remote with name upstream.

    Args:
        fork_path (Path): Path to the local repository
        upstream (str): An URL to the main repository

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    _run_console_tool(
        "git", args=["remote", "add", "upstream", upstream], cwd=fork_path, debug=True
    )

    return _run_console_tool("git", args=["fetch", "upstream"], cwd=fork_path, debug=True)


def get_merge_strategy_option(strategy: Strategies) -> tuple[str, str]:
    """
    Provide arguments for merge strategy.

    Args:
        strategy (Strategies): strategy to update student`s repository

    Returns:
        tuple[str, str]: tuple of arguments for merge command
    """
    strategies_to_git_option = {Strategies.WINNER: "theirs", Strategies.LOSER: "ours"}
    return "--strategy-option", strategies_to_git_option[strategy]


def get_repository_path(root_path: Path) -> Path:
    """
    Get path to repository.

    Args:
        root_path (Path): Path to the folder contains the repository

    Returns:
        Path: path to the repository
    """
    folders = list(root_path.glob("*"))
    if len(folders) != 1:
        raise ValueError(f"Cannot find repository in {root_path}")
    return folders[0]


@handles_console_error(ok_codes=(0, 1))
def checkout_path(
    fork_path: Path,
    paths_to_checkout: tuple[str, ...],
    branch: RemoteBranches,
) -> tuple[str, str, int]:
    """
    Revert files to branch.

    Args:
        fork_path (Path): Path to the local repository
        paths_to_checkout (tuple[str, ...]): Paths to check out to a branch
        branch (RemoteBranches): Branch name

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool(
        "git",
        args=["checkout", branch.value, "--", *paths_to_checkout],
        cwd=fork_path,
        debug=True,
    )


@handles_console_error()
def push_head_to_origin(fork_path: Path) -> tuple[str, str, int]:
    """
    Push head to origin/main.

    Args:
        fork_path (Path): Path to the local repository

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool(
        "git",
        args=["push", "origin", "HEAD:main"],
        cwd=fork_path,
        debug=True,
    )


@handles_console_error()
def git_status(fork_path: Path) -> tuple[str, str, int]:
    """
    Get a git status of the repository.

    Args:
        fork_path (Path): Path to the local repository

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool(
        "git",
        args=["status"],
        cwd=fork_path,
        debug=True,
    )


@handles_console_error()
def git_commit(fork_path: Path, commit_message: str) -> tuple[str, str, int]:
    """
    Create a commit.

    Args:
        fork_path (Path): Path to the local repository
        commit_message (str): Commit message

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool(
        "git",
        args=["commit", "-m", f'"{commit_message}"'],
        cwd=fork_path,
        debug=True,
    )


@handles_console_error()
def update_with_upstream(
    fork_path: Path,
    strategy: Strategies,
) -> tuple[str, str, int]:
    """
    Update with upstream/main.

    Args:
        fork_path (Path): Path to the local repository
        strategy (Strategies): strategy to update student`s repository

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    _run_console_tool(
        "git",
        args=[
            "checkout",
            RemoteBranches.ORIGIN.value,
        ],
        cwd=fork_path,
        debug=True,
    )

    merge_strategy_options = get_merge_strategy_option(strategy)

    return _run_console_tool(
        "git",
        args=["merge", *merge_strategy_options, "--no-edit", RemoteBranches.UPSTREAM.value],
        cwd=fork_path,
        debug=True,
    )


def main(
    repo_settings: dict[str, str],
    authentication: dict[str, str],
    paths_to_keep: dict[str, tuple[str, ...]],
    strategy: Strategies,
    merge_commit_message: str,
) -> None:
    """
    Update student`s fork with upstream.

    Args:
        repo_settings (dict[str, str]): Dict with URLs to fork and upstream
        authentication (dict[str, str]): Dict with username and token
        paths_to_keep (dict[str, tuple[str, ...]]): Dict with path to keep from fork and upstream
        strategy (Strategies): strategy to update student`s repository
        merge_commit_message (str): Merge commit message
    """
    fork_url = create_fork_url_with_auth(repo_settings["fork"], authentication["token"])
    with tempfile.TemporaryDirectory() as temp_dir:
        tempporary_dir = Path(temp_dir)
        clone_fork(fork_url=fork_url, root_dir=tempporary_dir)
        fork_path = get_repository_path(tempporary_dir)
        setup_repository(fork_path, authentication["user"])
        add_upstream(fork_path=fork_path, upstream=repo_settings["upstream"])
        update_with_upstream(
            fork_path=fork_path,
            strategy=strategy,
        )

        paths_keep_from_fork = paths_to_keep["origin"]
        paths_keep_from_upstream = paths_to_keep["upstream"]

        if paths_keep_from_fork:
            stdout, stderr, exit_code = checkout_path(
                fork_path=fork_path,
                paths_to_checkout=paths_keep_from_fork,
                branch=RemoteBranches.ORIGIN,
            )
            if exit_code:
                logger.error("Cannot checkout path to the upstream")
                sys.exit(1)

        if paths_keep_from_upstream:
            stdout, stderr, exit_code = checkout_path(
                fork_path=fork_path,
                paths_to_checkout=paths_keep_from_upstream,
                branch=RemoteBranches.UPSTREAM,
            )
            if exit_code and "did not match any file" in stderr:
                logger.error(
                    "\n[WARNING] Cannot checkout path to the origin.\n"
                    "[WARNING] Probably the fork does not contain it.\n"
                )

        stdout, stderr, exit_code = git_status(fork_path=fork_path)
        skip_commit_message = "nothing to commit, working tree clean"
        if skip_commit_message not in stdout:
            git_commit(fork_path, merge_commit_message)
        push_head_to_origin(fork_path)


if __name__ == "__main__":
    ARGUMENTS = CommandLineInterface(underscores_to_dashes=True).parse_args()
    REPO_SETTINGS = {"fork": ARGUMENTS.fork, "upstream": ARGUMENTS.upstream}
    PATH_TO_KEEP = {
        "origin": ARGUMENTS.paths_keep_from_fork,
        "upstream": ARGUMENTS.paths_keep_from_upstream,
    }
    AUTH_ARGS = {"user": ARGUMENTS.user, "token": ARGUMENTS.auth}
    main(
        repo_settings=REPO_SETTINGS,
        strategy=ARGUMENTS.strategy,
        authentication=AUTH_ARGS,
        paths_to_keep=PATH_TO_KEEP,
        merge_commit_message=ARGUMENTS.merge_commit_message,
    )
