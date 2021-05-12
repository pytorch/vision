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
from jinja2 import select_autoescape
import yaml
import os.path


PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9"]

RC_PATTERN = r"/v[0-9]+(\.[0-9]+)*-rc[0-9]+/"


def build_workflows(prefix='', filter_branch=None, upload=False, indentation=6, windows_latest_only=False):
    w = []
    for btype in ["wheel", "conda"]:
        for os_type in ["linux", "macos", "win"]:
            python_versions = PYTHON_VERSIONS
            cu_versions_dict = {"linux": ["cpu", "cu102", "cu111", "rocm4.0.1", "rocm4.1"],
                                "win": ["cpu", "cu102", "cu111"],
                                "macos": ["cpu"]}
            cu_versions = cu_versions_dict[os_type]
            for python_version in python_versions:
                for cu_version in cu_versions:
                    # ROCm conda packages not yet supported
                    if cu_version.startswith('rocm') and btype == "conda":
                        continue
                    for unicode in [False]:
                        fb = filter_branch
                        if windows_latest_only and os_type == "win" and filter_branch is None and \
                            (python_version != python_versions[-1] or
                             (cu_version not in [cu_versions[0], cu_versions[-1]])):
                            fb = "master"
                        w += workflow_pair(
                            btype, os_type, python_version, cu_version,
                            unicode, prefix, upload, filter_branch=fb)

    if not filter_branch:
        # Build on every pull request, but upload only on nightly and tags
        w += build_doc_job(None)
        w += upload_doc_job('nightly')
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
        if filter_branch == 'nightly' and os_type in ['linux', 'win']:
            pydistro = 'pip' if btype == 'wheel' else 'conda'
            w.append(generate_smoketest_workflow(pydistro, base_workflow_name, filter_branch, python_version, os_type))

    return w


def build_doc_job(filter_branch):
    job = {
        "name": "build_docs",
        "python_version": "3.7",
        "requires": ["binary_linux_wheel_py3.7_cpu", ],
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    return [{"build_docs": job}]


def upload_doc_job(filter_branch):
    job = {
        "name": "upload_docs",
        "context": "org-member",
        "python_version": "3.7",
        "requires": ["build_docs", ],
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch,
                                                tags_list=RC_PATTERN)
    return [{"upload_docs": job}]


manylinux_images = {
    "cu92": "pytorch/manylinux-cuda92",
    "cu101": "pytorch/manylinux-cuda101",
    "cu102": "pytorch/manylinux-cuda102",
    "cu110": "pytorch/manylinux-cuda110",
    "cu111": "pytorch/manylinux-cuda111",
    "cu112": "pytorch/manylinux-cuda112",
}


def get_manylinux_image(cu_version):
    if cu_version == "cpu":
        return "pytorch/manylinux-cuda102"
    elif cu_version.startswith('cu'):
        cu_suffix = cu_version[len('cu'):]
        return f"pytorch/manylinux-cuda{cu_suffix}"
    elif cu_version.startswith('rocm'):
        rocm_suffix = cu_version[len('rocm'):]
        return f"pytorch/manylinux-rocm:{rocm_suffix}"


def get_conda_image(cu_version):
    if cu_version == "cpu":
        return "pytorch/conda-builder:cpu"
    elif cu_version.startswith('cu'):
        cu_suffix = cu_version[len('cu'):]
        return f"pytorch/conda-builder:cuda{cu_suffix}"


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
        # ROCm conda packages not yet supported
        if "rocm" not in cu_version:
            d["conda_docker_image"] = get_conda_image(cu_version)

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

    w = f"binary_{os_type}_{btype}"
    return {w: d}


def gen_filter_branch_tree(*branches, tags_list=None):
    filter_dict = {"branches": {"only": [b for b in branches]}}
    if tags_list is not None:
        filter_dict["tags"] = {"only": tags_list}
    return filter_dict


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


def generate_smoketest_workflow(pydistro, base_workflow_name, filter_branch, python_version, os_type):

    required_build_suffix = "_upload"
    required_build_name = base_workflow_name + required_build_suffix

    smoke_suffix = f"smoke_test_{pydistro}"
    d = {
        "name": f"{base_workflow_name}_{smoke_suffix}",
        "requires": [required_build_name],
        "python_version": python_version,
    }

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {f"smoke_test_{os_type}_{pydistro}": d}


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(
        yaml.dump(data_list, default_flow_style=False).splitlines())


def unittest_workflows(indentation=6):
    jobs = []
    for os_type in ["linux", "windows", "macos"]:
        for device_type in ["cpu", "gpu"]:
            if os_type == "macos" and device_type == "gpu":
                continue
            for i, python_version in enumerate(PYTHON_VERSIONS):
                job = {
                    "name": f"unittest_{os_type}_{device_type}_py{python_version}",
                    "python_version": python_version,
                }

                if device_type == 'gpu':
                    if python_version != "3.8":
                        job['filters'] = gen_filter_branch_tree('master', 'nightly')
                    job['cu_version'] = 'cu101'
                else:
                    job['cu_version'] = 'cpu'

                jobs.append({f"unittest_{os_type}_{device_type}": job})

    return indent(indentation, jobs)


def cmake_workflows(indentation=6):
    jobs = []
    python_version = '3.8'
    for os_type in ['linux', 'windows', 'macos']:
        # Skip OSX CUDA
        device_types = ['cpu', 'gpu'] if os_type != 'macos' else ['cpu']
        for device in device_types:
            job = {
                'name': f'cmake_{os_type}_{device}',
                'python_version': python_version
            }

            job['cu_version'] = 'cu101' if device == 'gpu' else 'cpu'
            if device == 'gpu' and os_type == 'linux':
                job['wheel_docker_image'] = 'pytorch/manylinux-cuda101'
            jobs.append({f'cmake_{os_type}_{device}': job})
    return indent(indentation, jobs)


def ios_workflows(indentation=6, nightly=False):
    jobs = []
    build_job_names = []
    name_prefix = "nightly_" if nightly else ""
    env_prefix = "nightly-" if nightly else ""
    for arch, platform in [('x86_64', 'SIMULATOR'), ('arm64', 'OS')]:
        name = f'{name_prefix}binary_libtorchvision_ops_ios_12.0.0_{arch}'
        build_job_names.append(name)
        build_job = {
            'build_environment': f'{env_prefix}binary-libtorchvision_ops-ios-12.0.0-{arch}',
            'ios_arch': arch,
            'ios_platform': platform,
            'name': name,
        }
        if nightly:
            build_job['filters'] = gen_filter_branch_tree('nightly')
        jobs.append({'binary_ios_build': build_job})

    if nightly:
        upload_job = {
            'build_environment': f'{env_prefix}binary-libtorchvision_ops-ios-12.0.0-upload',
            'context': 'org-member',
            'filters': gen_filter_branch_tree('nightly'),
            'requires': build_job_names,
        }
        jobs.append({'binary_ios_upload': upload_job})
    return indent(indentation, jobs)


def android_workflows(indentation=6, nightly=False):
    jobs = []
    build_job_names = []
    name_prefix = "nightly_" if nightly else ""
    env_prefix = "nightly-" if nightly else ""

    name = f'{name_prefix}binary_libtorchvision_ops_android'
    build_job_names.append(name)
    build_job = {
        'build_environment': f'{env_prefix}binary-libtorchvision_ops-android',
        'name': name,
    }

    if nightly:
        upload_job = {
            'build_environment': f'{env_prefix}binary-libtorchvision_ops-android-upload',
            'context': 'org-member',
            'filters': gen_filter_branch_tree('nightly'),
            'name': f'{name_prefix}binary_libtorchvision_ops_android_upload'
        }
        jobs.append({'binary_android_upload': upload_job})
    else:
        jobs.append({'binary_android_build': build_job})
    return indent(indentation, jobs)


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=('html', 'xml')),
        keep_trailing_newline=True,
    )

    with open(os.path.join(d, 'config.yml'), 'w') as f:
        f.write(env.get_template('config.yml.in').render(
            build_workflows=build_workflows,
            unittest_workflows=unittest_workflows,
            cmake_workflows=cmake_workflows,
            ios_workflows=ios_workflows,
            android_workflows=android_workflows,
        ))
