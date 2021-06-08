import json
import sys
from typing import Any, Dict, Set, Tuple

import requests

# If the PR has any of these labels, we mark it as properly labeled.
REQUIRED_LABELS = {
    "TODO",
}


def main(commit_hash: str) -> Dict[str, Any]:
    pr_number = get_pr_number(commit_hash)
    merger, labels = get_pr_merger_and_labels(pr_number)
    is_properly_labeled = bool(REQUIRED_LABELS.intersection(labels))
    if not is_properly_labeled:
        users = {merger, *get_pr_reviewers(pr_number)}
    else:
        users = ()
    return dict(
        is_properly_labeled=is_properly_labeled,
        responsible_users=", ".join(sorted([f"@{user}" for user in users])),
    )


def _query_torchvision(cmd: str, *, accept) -> Any:
    response = requests.get(f"https://api.github.com/repos/pytorch/vision/{cmd}", headers=dict(Accept=accept))
    return response.json()


def get_pr_number(commit_hash: str) -> int:
    # See https://docs.github.com/en/rest/reference/repos#list-pull-requests-associated-with-a-commit
    data = _query_torchvision(f"commits/{commit_hash}/pulls", accept="application/vnd.github.groot-preview+json")
    return data[0]["number"]


def get_pr_merger_and_labels(pr_number: int) -> Tuple[str, Set[str]]:
    # See https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
    data = _query_torchvision(f"pulls/{pr_number}", accept="application/vnd.github.v3+json")
    merger = data["merged_by"]["login"]
    labels = {label["name"] for label in data["labels"]}
    return merger, labels


def get_pr_reviewers(pr_number: int) -> Set[str]:
    # See https://docs.github.com/en/rest/reference/pulls#list-reviews-for-a-pull-request
    data = _query_torchvision(f"pulls/{pr_number}/reviews", accept="application/vnd.github.v3+json")
    return {review["user"]["login"] for review in data if review["state"] == "APPROVED"}


if __name__ == "__main__":
    commit_hash = sys.argv[1]
    data = main(commit_hash)
    print(json.dumps(data))
