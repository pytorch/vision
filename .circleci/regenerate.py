#!/usr/bin/env python3

"""
This script should use a very simple, functional programming style.
Avoid Jinja macros in favor of native Python functions.

Don't go overboard on code generation; use Python only to generate
content that can't be easily declared statically using CircleCI's YAML API.

Data declarations (e.g. the nested loops for defining the configuration matrix)
should be at the top of the file for easy updating.

See this comment for design rationale:
https://github.com/pytorch/vision/pull/1321#issuecomment-531033978
"""

import jinja2
import yaml
import os.path


def workflows(prefix='', filter_branch=None, upload=False, indentation=6):
    w = []
    for btype in ["wheel", "conda"]:
        for os_type in ["linux", "macos"]:
            for python_version in ["3.5", "3.6", "3.7", "3.8"]:
                for cu_version in (["cpu", "cu92", "cu100", "cu101"] if os_type == "linux" else ["cpu"]):
                    for unicode in ([False, True] if btype == "wheel" and python_version == "2.7" else [False]):
                        w += workflow_pair(
                            btype, os_type, python_version, cu_version,
                            unicode, prefix, upload, filter_branch=filter_branch)

    return indent(indentation, w)


def workflow_pair(btype, os_type, python_version, cu_version, unicode, prefix='', upload=False, *, filter_branch=None):

    w = []
    unicode_suffix = "u" if unicode else ""
    base_workflow_name = f"{prefix}binary_{os_type}_{btype}_py{python_version}{unicode_suffix}_{cu_version}"

    w.append(generate_base_workflow(
        base_workflow_name, python_version, cu_version,
        unicode, os_type, btype, filter_branch=filter_branch))

    if upload:
        w.append(generate_upload_workflow(base_workflow_name, os_type, btype, cu_version, filter_branch=filter_branch))

    return w


def generate_base_workflow(base_workflow_name, python_version, cu_version,
                           unicode, os_type, btype, *, filter_branch=None):

    d = {
        "name": base_workflow_name,
        "python_version": python_version,
        "cu_version": cu_version,
    }

    if unicode:
        d["unicode_abi"] = '1'

    if cu_version == "cu92":
        d["wheel_docker_image"] = "pytorch/manylinux-cuda92"
    elif cu_version == "cu100":
        d["wheel_docker_image"] = "pytorch/manylinux-cuda100"

    if filter_branch is not None:
        d["filters"] = {"branches": {"only": filter_branch}}

    return {f"binary_{os_type}_{btype}": d}


def generate_upload_workflow(base_workflow_name, os_type, btype, cu_version, *, filter_branch=None):
    d = {
        "name": f"{base_workflow_name}_upload",
        "context": "org-member",
        "requires": [base_workflow_name],
    }

    if btype == 'wheel':
        d["subfolder"] = "" if os_type == 'macos' else cu_version + "/"

    if filter_branch is not None:
        d["filters"] = {"branches": {"only": filter_branch}}

    return {f"binary_{btype}_upload": d}


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(
        yaml.dump(data_list, default_flow_style=False).splitlines())


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=False,
    )

    with open(os.path.join(d, 'config.yml'), 'w') as f:
        f.write(env.get_template('config.yml.in').render(workflows=workflows))
