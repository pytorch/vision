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

import os.path

import jinja2
import yaml
from jinja2 import select_autoescape


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(yaml.dump(data_list, default_flow_style=False).splitlines())


def cmake_workflows(indentation=6):
    jobs = []
    python_version = "3.8"
    for os_type in ["linux", "windows", "macos"]:
        # Skip OSX CUDA
        device_types = ["cpu", "gpu"] if os_type != "macos" else ["cpu"]
        for device in device_types:
            job = {"name": f"cmake_{os_type}_{device}", "python_version": python_version}

            job["cu_version"] = "cu117" if device == "gpu" else "cpu"
            if device == "gpu" and os_type == "linux":
                job["wheel_docker_image"] = "pytorch/manylinux-cuda117"
            jobs.append({f"cmake_{os_type}_{device}": job})
    return indent(indentation, jobs)


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        keep_trailing_newline=True,
    )

    with open(os.path.join(d, "config.yml"), "w") as f:
        f.write(
            env.get_template("config.yml.in").render(
                cmake_workflows=cmake_workflows,
            )
        )
