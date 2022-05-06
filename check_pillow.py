import sys
import tempfile
import unittest.mock
from collections import namedtuple

from pkg_resources import Requirement
from setuptools.extern.packaging.tags import _version_nodot, sys_tags
from setuptools.package_index import PackageIndex, distros_for_url


SortKey = namedtuple("SortKey", ("parsed_version", "precedence", "key", "location", "py_version", "platform"))

print(f"sys.version_info: {sys.version_info}")
print(f"sys.version_info[:2]: {sys.version_info[:2]}")
print(f"_version_nodot(sys.version_info[:2]): {_version_nodot(sys.version_info[:2])}")
print(f"''.join(map(str, (3, 10))): {''.join(map(str, (3, 10)))}")
print(f"list(sys_tags())[0]: {list(sys_tags())[0]}")

print("#" * 80)

with unittest.mock.patch("setuptools.wheel.Wheel.is_compatible", return_value=True):
    for dist in distros_for_url(
        "https://files.pythonhosted.org/packages/a8/df/1177786a2d1c0bf732ba6d5f05a2fa40f016e81e1c16d62f1101e35d271e/Pillow-9.1.0-cp310-cp310-win_amd64.whl#sha256=97bda660702a856c2c9e12ec26fc6d187631ddfd896ff685814ab21ef0597033"
    ):
        print(SortKey(*dist.hashcmp))

print("#" * 80)

index = PackageIndex()
requirement = Requirement.parse("pillow >= 5.3.0, !=8.3.*")
tmpdir = tempfile.mkdtemp()

index.fetch_distribution(requirement, tmpdir)

dists = [dist for dist in index._distmap["pillow"] if dist.version.split(".")[0] == "9"]

sort_keys = [SortKey(*dist.hashcmp) for dist in dists]
for sort_key in sort_keys:
    print(sort_key)

print("#" * 80)
