import tempfile
from collections import namedtuple

from pkg_resources import Requirement
from setuptools.package_index import PackageIndex

SortKey = namedtuple("SortKey", ("parsed_version", "precedence", "key", "location", "py_version", "platform"))

index = PackageIndex()
requirement = Requirement.parse("pillow >= 5.3.0, !=8.3.*")
tmpdir = tempfile.mkdtemp()

index.fetch_distribution(requirement, tmpdir)

dists = [dist for dist in index._distmap["pillow"] if dist.version.split(".")[0] == "9"]
sort_keys = [SortKey(*dist.hashcmp) for dist in dists]
for sort_key in sort_keys:
    print(sort_key)
