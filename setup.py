#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = "0.0.1dev1" # imp.load_source("", "cell_optimisation_analysis/version.py").__version__

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
        'nose',
        'ipyparallel',
        'tqdm',
        'pyyaml',
        'luigi',
        'bluepyopt @ git+http://github.com/BlueBrain/BluePyOpt@CMA_clean#egg=bluepyopt',
        'bluepyefe @ git+ssh://bbpcode.epfl.ch/analysis/BluePyEfe@BPE2#egg=bluepyefe',
        'efel',
        'psycopg2',
        'nexusforge',
        'bluepy',
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": ["cell-optimisation-analysis=cell_optimisation_analysis.apps.emodel_release:cli"],
    },
)
