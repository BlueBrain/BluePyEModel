#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "bluepyemodel/version.py").__version__

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    README = f.read()


EXTRA_LUIGI = [
    "luigi>=3.0",
    "luigi-tools>=0.0.12",
    "bbp-workflow>=2.1.19",
    "bbp-workflow-cli",
]

EXTRA_NEXUS = [
    "icselector",
    "nexusforge>=0.7.1",
    "entity_management>=1.2",
    "pyJWT>=2.1.0"
]

EXTRA_CURRENTSCAPE = [
    "currentscape>=0.0.11"
]

EXTRA_TEST = [
    "pytest>=6.2",
    "dictdiffer>=0.8"
]

EXTRA_DOC = [
    "graphviz",
    "sphinx",
    "sphinx-autoapi",
    "sphinx-bluebrain-theme",
    "myst_parser",
]


setup(
    name="bluepyemodel",
    author="BlueBrain cells",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    version=VERSION,
    description="Electrical modeling pipeline",
    long_description=README,
    long_description_content_type="text/x-rst",
    license="BBP-internal-confidential",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "ipyparallel>=6.3",
        "tqdm",
        "pyyaml",
        "gitpython",
        "bluepyopt>=1.12.12",
        "bluepyefe>=2.0.0",
        "neurom>=3.0,<4.0",
        "efel>=3.1,<=4.1.91",
        "configparser",
        "neuron>=8.0",
        "morph_tool>=2.8",
        "fasteners>=0.16",
        "jinja2==3.0.3"
    ],
    extras_require={
        "luigi": EXTRA_LUIGI + EXTRA_CURRENTSCAPE,
        "nexus": EXTRA_NEXUS,
        "currentscape": EXTRA_CURRENTSCAPE,
        "all": EXTRA_LUIGI + EXTRA_NEXUS + EXTRA_TEST + EXTRA_CURRENTSCAPE,
        "docs": EXTRA_DOC + EXTRA_LUIGI,
        "test": EXTRA_TEST,
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["data/*.npy"]},
)
