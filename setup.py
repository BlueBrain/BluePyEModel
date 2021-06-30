#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "bluepyemodel/version.py").__version__

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()


EXTRA_LUIGI = ["luigi", "luigi-tools", "bbp-workflow>=2.1.19", "bbp-workflow-cli"]
EXTRA_GENERALISATION = ["bluepyparallel>=0.0.3"]
EXTRA_NEXUS = ["nexusforge", "entity_management", "pyJWT==1.7.1"]
EXTRA_TEST = ["pytest", "dictdiffer"]
EXTRA_DOC = [
    "graphviz",
    "sphinx",
    "sphinx-autoapi",
    "sphinx-bluebrain-theme",
]
EXTRA_CMA = ["bluepyopt @ git+http://github.com/BlueBrain/BluePyOpt@CMA_clean#egg=bluepyopt"]


setup(
    name="BluePyEModel",
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
        "ipyparallel",
        "tqdm",
        "pyyaml",
        "gitpython",
        "bluepyopt",
        "bluepyefe @ git+http://github.com/BlueBrain/BluePyEfe@BPE2#egg=bluepyefe",
        "efel",
        "psycopg2",
        "bluepy",
        "neuron",
        "morph_tool",
        "fasteners",
    ],
    extras_require={
        "luigi": EXTRA_LUIGI,
        "generalisation": EXTRA_GENERALISATION,
        "nexus": EXTRA_NEXUS,
        "all": EXTRA_LUIGI + EXTRA_GENERALISATION + EXTRA_NEXUS + EXTRA_TEST + EXTRA_CMA,
        "docs": EXTRA_DOC + EXTRA_LUIGI,
        "test": EXTRA_TEST,
        "cma": EXTRA_CMA,
    },
    packages=find_packages(),
    entry_points={"console_scripts": ["BluePyEModel=bluepyemodel.apps.emodel_release:cli"]},
    include_package_data=True,
    package_data={"": ["data/*.npy"]},
)
