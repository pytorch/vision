"""
This script finds the merger responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.

Note: we ping the merger only, not the reviewers, as the reviewers can sometimes be external to torchvision
with no labeling responsibility, so we don't want to bother them.
"""

import sys
from typing import Any, Optional, Set, Tuple

import requests

# For a PR to be properly labeled it should have one primary label and one secondary label
PRIMARY_LABELS = {
    "new feature",
    "bug",
    "code quality",
    "enhancement",
    "bc-breaking",
    "deprecation",
    "other",
    "prototype",
}

SECONDARY_LABELS = {
    "dependency issue",
    "module: c++ frontend",
    "module: ci",
    "module: datasets",
    "module: documentation",
    "module: io",
    "module: models.quantization",
    "module: models",
    "module: onnx",
    "module: ops",
    "module: reference scripts",
    "module: rocm",
    "module: tests",
    "module: transforms",
    "module: utils",
    "module: video",
    "Perf",
    "Revert(ed)",
    "topic: build",
}


def query_torchvision(cmd: str, *, accept) -> Any:
    response = requests.get(f"https://api.github.com/repos/pytorch/vision/{cmd}", headers=dict(Accept=accept))
    return response.json()


def get_pr_number(commit_hash: str) -> Optional[int]:
    # See https://docs.github.com/en/rest/reference/repos#list-pull-requests-associated-with-a-commit
    data = query_torchvision(f"commits/{commit_hash}/pulls", accept="application/vnd.github.groot-preview+json")
    if not data:
        return None
    return data[0]["number"]


def get_pr_merger_and_labels(pr_number: int) -> Tuple[str, Set[str]]:
    # See https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
    data = query_torchvision(f"pulls/{pr_number}", accept="application/vnd.github.v3+json")
    merger = data["merged_by"]["login"]
    labels = {label["name"] for label in data["labels"]}
    return merger, labels


if __name__ == "__main__":
    commit_hash = sys.argv[1]
    pr_number = get_pr_number(commit_hash)
    if not pr_number:
        sys.exit(0)

    merger, labels = get_pr_merger_and_labels(pr_number)
    is_properly_labeled = bool(PRIMARY_LABELS.intersection(labels) and SECONDARY_LABELS.intersection(labels))

    if not is_properly_labeled:
        print(f"@{merger}")
