#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "bluepyemodel/version.py").__version__

setup(
    name="BluePyEModel",
    author="BlueBrain cells",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    version=VERSION,
    description="",
    license="BBP-internal-confidential",
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'ipyparallel',
        'dask[distributed]>=2.30',
        'dask_mpi>=2.20',
        'tqdm',
        'pyyaml',
        'luigi',
        'bluepyopt @ git+http://github.com/BlueBrain/BluePyOpt@CMA_clean#egg=bluepyopt',
        'bluepyefe @ git+ssh://bbpcode.epfl.ch/analysis/BluePyEfe@BPE2#egg=bluepyefe',
        'efel',
        'psycopg2',
        'nexusforge',
        'bluepy',
        'neuron',
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": ["BluePyEModel=bluepyemodel.apps.emodel_release:cli"]
    },
    include_package_data=True,
    package_data={'': ['data/*.npy']},
)
