#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "bluepyemodel/version.py").__version__

EXTRA_LUIGI = ["luigi", "luigi-tools"]
EXTRA_GENERALISATION = ["bluepyparallel"]
EXTRA_NEXUS = ["nexusforge"]

setup(
    name="BluePyEModel",
    author="BlueBrain cells",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    version=VERSION,
    description="",
    license="BBP-internal-confidential",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "ipyparallel",
        "tqdm",
        "pyyaml",
        "gitpython",
        "bluepyopt @ git+http://github.com/BlueBrain/BluePyOpt@CMA_clean#egg=bluepyopt",
        "bluepyefe @ git+http://github.com/BlueBrain/BluePyEfe@BPE2#egg=bluepyefe",
        "efel",
        "psycopg2",
        "bluepy",
        "neuron",
        "morph_tool",
    ],
    extras_require={
        "luigi": EXTRA_LUIGI,
        "generalisation": EXTRA_GENERALISATION,
        "nexus": EXTRA_NEXUS,
        "all": EXTRA_LUIGI + EXTRA_GENERALISATION + EXTRA_NEXUS,
    },
    packages=find_packages(),
    entry_points={"console_scripts": ["BluePyEModel=bluepyemodel.apps.emodel_release:cli"]},
    include_package_data=True,
    package_data={"": ["data/*.npy"]},
)
