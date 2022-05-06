import tempfile
from collections import namedtuple

from pkg_resources import Requirement
from setuptools.package_index import PackageIndex

SortKey = namedtuple("SortKey", ("parsed_version", "precedence", "key", "location", "py_version", "platform"))

index = PackageIndex()
requirement = Requirement.parse("pillow >= 5.3.0, !=8.3.*")
tmpdir = tempfile.mkdtemp()

index.fetch_distribution(requirement, tmpdir)

print(f"entries in distmap: {len(index._distmap)}")
print(f"pillow in distmap: {'pillow' in index._distmap}")
print(f"Pillow in distmap: {'Pillow' in index._distmap}")

dists = [dist for dist in index._distmap["pillow"] if dist.version.split(".")[0] == "9"]

print(f"Total number of pillow dists: {len(index._distmap['pillow'])}")
print(f"Number of pillow > 9 dists: {len(dists)}")

sort_keys = [SortKey(*dist.hashcmp) for dist in dists]
for sort_key in sort_keys:
    print(sort_key)

print("#" * 80)
