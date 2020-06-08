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


def workflows(prefix='', filter_branch=None, upload=False, indentation=6, windows_latest_only=False):
    w = []
    for btype in ["wheel", "conda"]:
        for os_type in ["linux", "macos", "win"]:
            python_versions = ["3.6", "3.7", "3.8"]
            cu_versions = (["cpu", "cu92", "cu101", "cu102"] if os_type == "linux" or os_type == "win" else ["cpu"])
            for python_version in python_versions:
                for cu_version in cu_versions:
                    for unicode in ([False, True] if btype == "wheel" and python_version == "2.7" else [False]):
                        fb = filter_branch
                        if windows_latest_only and os_type == "win" and filter_branch is None and \
                            (python_version != python_versions[-1] or
                             (cu_version not in [cu_versions[0], cu_versions[-1]])):
                            fb = "master"
                        w += workflow_pair(
                            btype, os_type, python_version, cu_version,
                            unicode, prefix, upload, filter_branch=fb)

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


manylinux_images = {
    "cu92": "pytorch/manylinux-cuda92",
    "cu101": "pytorch/manylinux-cuda101",
    "cu102": "pytorch/manylinux-cuda102",
}


def get_manylinux_image(cu_version):
    cu_suffix = "102"
    if cu_version.startswith('cu'):
        cu_suffix = cu_version[len('cu'):]
    return f"pytorch/manylinux-cuda{cu_suffix}"


def generate_base_workflow(base_workflow_name, python_version, cu_version,
                           unicode, os_type, btype, *, filter_branch=None):

    d = {
        "name": base_workflow_name,
        "python_version": python_version,
        "cu_version": cu_version,
    }

    if os_type != "win" and unicode:
        d["unicode_abi"] = '1'

    if os_type != "win":
        d["wheel_docker_image"] = get_manylinux_image(cu_version)

    if filter_branch is not None:
        d["filters"] = {"branches": {"only": filter_branch}}

    w = f"binary_{os_type}_{btype}_release" if os_type == "win" else f"binary_{os_type}_{btype}"
    return {w: d}


def generate_upload_workflow(base_workflow_name, os_type, btype, cu_version, *, filter_branch=None):
    d = {
        "name": f"{base_workflow_name}_upload",
        "context": "org-member",
        "requires": [base_workflow_name],
    }

    if btype == 'wheel':
        d["subfolder"] = "" if os_type == 'macos' else cu_version + "/"

    if filter_branch is not None:
        d["filters"] = {
            "branches": {
                "only": filter_branch
            },
            "tags": {
                # Using a raw string here to avoid having to escape
                # anything
                "only": r"/v[0-9]+(\.[0-9]+)*-rc[0-9]+/"
            }
        }

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
