#!/usr/bin/env python

"""
Copyright 2023, EPFL/Blue Brain Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import find_packages, setup

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()


EXTRA_LUIGI = [
    "luigi>=3.0",
    "luigi-tools>=0.0.12",
]

EXTRA_TEST = [
    "pytest>=6.2",
    "dictdiffer>=0.8"
]

EXTRA_DOC = [
    "graphviz",
    "sphinx",
    "sphinx-bluebrain-theme",
]


setup(
    name="bluepyemodel",
    use_scm_version={
        'version_scheme': 'python-simplified-semver',
        'local_scheme': 'no-local-version'
    },
    setup_requires=['setuptools_scm'],
    author="Blue Brain Project, EPFL",
    author_email="",
    description="Blue Brain Python Electrical Modeling Pipeline",
    long_description=README,
    long_description_content_type="text/x-rst",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "ipyparallel>=6.3",
        "tqdm",
        "pyyaml",
        "gitpython",
        "bluepyopt>=1.14.10",
        "bluepyefe>=2.2.0",
        "neurom>=3.0,<4.0",
        "efel>=3.1",
        "configparser",
        "neuron>=8.0",
        "morph_tool>=2.8",
        "morphio",
        "fasteners>=0.16",
        "jinja2>=3.0.3",
        "currentscape>=0.0.11"
    ],
    extras_require={
        "luigi": EXTRA_LUIGI,
        "all": EXTRA_LUIGI + EXTRA_TEST,
        "docs": EXTRA_DOC + EXTRA_LUIGI,
        "test": EXTRA_TEST,
    },
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    keywords=[
        'computational neuroscience',
        'simulation',
        'analysis',
        'parameters',
        'Blue Brain Project'],
    url="https://github.com/BlueBrain/BluePyEModel",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
)
