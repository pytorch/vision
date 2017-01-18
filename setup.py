#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.1.6'

long_description = '''torch-vision provides DataLoaders, Pre-trained models
and common transforms for torch for images and videos'''

setup_info = dict(
    # Metadata
    name='torchvision',
    version=VERSION,
    author='PyTorch Core Team',
    author_email='soumith@pytorch.org',
    url='https://github.com/pytorch/vision',
    description='image and video datasets and models for torch deep learning',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
)

setup(**setup_info)
