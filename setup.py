from __future__ import print_function
import os
import io
import re
import sys
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
import subprocess


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


version = '0.2.3a0'
sha = 'Unknown'
package_name = os.getenv('TORCHVISION_PACKAGE_NAME', 'torchvision')

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('TORCHVISION_BUILD_VERSION'):
    assert os.getenv('TORCHVISION_BUILD_NUMBER') is not None
    build_number = int(os.getenv('TORCHVISION_BUILD_NUMBER'))
    version = os.getenv('TORCHVISION_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'torchvision', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


write_version_file()

readme = open('README.rst').read()

pytorch_package_name = os.getenv('TORCHVISION_PYTORCH_DEPENDENCY_NAME', 'torch')

requirements = [
    'numpy',
    'six',
    pytorch_package_name,
]

pillow_ver = ' >= 4.1.1'
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req + pillow_ver)


setup(
    # Metadata
    name=package_name,
    version=version,
    author='PyTorch Core Team',
    author_email='soumith@pytorch.org',
    url='https://github.com/pytorch/vision',
    description='image and video datasets and models for torch deep learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,
    extras_require={
        "scipy": ["scipy"],
    },
)
