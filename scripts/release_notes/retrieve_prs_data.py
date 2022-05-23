import json
import locale
import os
import re
import subprocess
from collections import namedtuple
from os.path import expanduser

import requests


Features = namedtuple(
    "Features",
    [
        "title",
        "body",
        "pr_number",
        "files_changed",
        "labels",
    ],
)


def dict_to_features(dct):
    return Features(
        title=dct["title"],
        body=dct["body"],
        pr_number=dct["pr_number"],
        files_changed=dct["files_changed"],
        labels=dct["labels"],
    )


def features_to_dict(features):
    return dict(features._asdict())


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()


def commit_body(commit_hash):
    cmd = f"git log -n 1 --pretty=format:%b {commit_hash}"
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_title(commit_hash):
    cmd = f"git log -n 1 --pretty=format:%s {commit_hash}"
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_files_changed(commit_hash):
    cmd = f"git diff-tree --no-commit-id --name-only -r {commit_hash}"
    ret, out, err = run(cmd)
    return out.split("\n") if ret == 0 else None


def parse_pr_number(body, commit_hash, title):
    regex = r"(#[0-9]+)"
    matches = re.findall(regex, title)
    if len(matches) == 0:
        if "revert" not in title.lower() and "updating submodules" not in title.lower():
            print(f"[{commit_hash}: {title}] Could not parse PR number, ignoring PR")
        return None
    if len(matches) > 1:
        print(f"[{commit_hash}: {title}] Got two PR numbers, using the last one")
        return matches[-1][1:]
    return matches[0][1:]


def get_ghstack_token():
    pattern = "github_oauth = (.*)"
    with open(expanduser("~/.ghstackrc"), "r+") as f:
        config = f.read()
    matches = re.findall(pattern, config)
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    return matches[0]


token = get_ghstack_token()
headers = {"Authorization": f"token {token}"}


def run_query(query):
    request = requests.post("https://api.github.com/graphql", json={"query": query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(f"Query failed to run by returning code of {request.status_code}. {query}")


def gh_labels(pr_number):
    query = f"""
    {{
      repository(owner: "pytorch", name: "vision") {{
        pullRequest(number: {pr_number}) {{
          labels(first: 10) {{
            edges {{
              node {{
                name
              }}
            }}
          }}
        }}
      }}
    }}
    """
    query = run_query(query)
    edges = query["data"]["repository"]["pullRequest"]["labels"]["edges"]
    return [edge["node"]["name"] for edge in edges]


def get_features(commit_hash, return_dict=False):
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash),
    )
    pr_number = parse_pr_number(body, commit_hash, title)
    labels = []
    if pr_number is not None:
        labels = gh_labels(pr_number)
    result = Features(title, body, pr_number, files_changed, labels)
    if return_dict:
        return features_to_dict(result)
    return result


class CommitDataCache:
    def __init__(self, path="results/data.json"):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            self.data = self.read_from_disk()

    def get(self, commit):
        if commit not in self.data.keys():
            # Fetch and cache the data
            self.data[commit] = get_features(commit)
            self.write_to_disk()
        return self.data[commit]

    def read_from_disk(self):
        with open(self.path) as f:
            data = json.load(f)
            data = {commit: dict_to_features(dct) for commit, dct in data.items()}
        return data

    def write_to_disk(self):
        data = {commit: features._asdict() for commit, features in self.data.items()}
        with open(self.path, "w") as f:
            json.dump(data, f)


def get_commits_between(base_version, new_version):
    cmd = f"git merge-base {base_version} {new_version}"
    rc, merge_base, _ = run(cmd)
    assert rc == 0

    # Returns a list of something like
    # b33e38ec47 Allow a higher-precision step type for Vec256::arange (#34555)
    cmd = f"git log --reverse --oneline {merge_base}..{new_version}"
    rc, commits, _ = run(cmd)
    assert rc == 0

    log_lines = commits.split("\n")
    hashes, titles = zip(*[log_line.split(" ", 1) for log_line in log_lines])
    return hashes, titles


def convert_to_dataframes(feature_list):
    import pandas as pd

    df = pd.DataFrame.from_records(feature_list, columns=Features._fields)
    return df


def main(base_version, new_version):
    hashes, titles = get_commits_between(base_version, new_version)

    cdc = CommitDataCache("data.json")
    for idx, commit in enumerate(hashes):
        if idx % 10 == 0:
            print(f"{idx} / {len(hashes)}")
        cdc.get(commit)

    return cdc


if __name__ == "__main__":
    # d = get_features('2ab93592529243862ce8ad5b6acf2628ef8d0dc8')
    # print(d)
    # hashes, titles = get_commits_between("tags/v0.9.0", "fc852f3b39fe25dd8bf1dedee8f19ea04aa84c15")

    # Usage: change the tags below accordingly to the current release, then save the json with
    # cdc.write_to_disk().
    # Then you can use classify_prs.py (as a notebook)
    # to open the json and generate the release notes semi-automatically.
    cdc = main("tags/v0.9.0", "fc852f3b39fe25dd8bf1dedee8f19ea04aa84c15")
    from IPython import embed

    embed()
