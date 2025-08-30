"""
A module for eliminating old workflows.
"""

from datetime import datetime

from ghapi.all import GhApi

from config.console_logging import get_child_logger

logger = get_child_logger(__file__)

if __name__ == "__main__":
    # GITHUB_TOKEN should be set, otherwise it would not work
    api = GhApi()

    OWNER = "fipl-hse"
    REPO = "2021-2-level-ctlr-admin"
    EXPIRATION_DAYS = 3
    PER_PAGE = 100

    _ = api.actions.list_workflow_runs_for_repo(OWNER, REPO, per_page=PER_PAGE)
    max_page_idx = api.last_page()
    logger.info(f"Max page is {max_page_idx}")

    for page_idx in range(max_page_idx + 1):
        runs = api.actions.list_workflow_runs_for_repo(
            OWNER, REPO, per_page=PER_PAGE, page=page_idx
        )
        for run in runs.workflow_runs:
            if run.event == "push" and run.head_branch == "main":
                logger.info(f"Skipping #{run.id} as it was run for main branch")
                continue

            delta = datetime.utcnow() - datetime.strptime(run.updated_at, "%Y-%m-%dT%H:%M:%SZ")

            if delta.days > EXPIRATION_DAYS:
                logger.info(f"Removing workflow run #{run.id}.\n" f"Author: {run.actor.login}")
                api.actions.delete_workflow_run(OWNER, REPO, run.id)
            else:
                logger.info(
                    f"Skipping #{run.id}\n" f"as it was run earlier than {delta.days} days ago"
                )
