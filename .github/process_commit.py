"""
This script finds all responsible users for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.
"""

import sys
from typing import Any, Optional, Set, Tuple

import requests

# If the PR has any of these labels, we accept it as properly labeled.
REQUIRED_LABELS = {
    "new feature",
    "bug",
    "code quality",
    "enhancement",
    "bc-breaking",
    "dependency issue",
    "deprecation",
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
}


def find_responsible_users(commit_hash: str) -> Set[str]:
    pr_number = get_pr_number(commit_hash)
    if not pr_number:
        return set()

    merger, labels = get_pr_merger_and_labels(pr_number)
    is_properly_labeled = bool(REQUIRED_LABELS.intersection(labels))
    if is_properly_labeled:
        return set()

    return {merger, *get_pr_reviewers(pr_number)}


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


def get_pr_reviewers(pr_number: int) -> Set[str]:
    # See https://docs.github.com/en/rest/reference/pulls#list-reviews-for-a-pull-request
    data = query_torchvision(f"pulls/{pr_number}/reviews", accept="application/vnd.github.v3+json")
    return {review["user"]["login"] for review in data if review["state"] == "APPROVED"}


if __name__ == "__main__":
    commit_hash = sys.argv[1]
    users = find_responsible_users(commit_hash)
    print(", ".join(sorted([f"@{user}" for user in users])))
