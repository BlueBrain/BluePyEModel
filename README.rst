|banner|

BluePyEModel: Blue Brain Python Electrical Modeling Pipeline
============================================================

+----------------+------------+
| Latest Release | |pypi|     |
+----------------+------------+
| Documentation  | |docs|     |
+----------------+------------+
| License        | |license|  |
+----------------+------------+
| Build Status 	 | |tests|    |
+----------------+------------+
| Coverage       | |coverage| |
+----------------+------------+
| Citation       | |zenodo|   |
+----------------+------------+


Introduction
------------

The Blue Brain Python Electrical Modeling Pipeline (BluePyEModel) is a Python package facilitating the configuration and execution of electrical neuron model (e-model) building tasks. It covers tasks such as extraction of electrical features from electrophysiology data, e-model parameters optimisation and model validation. As such, it builds on top of `eFEL <https://github.com/BlueBrain/eFEL>`_, `BluePyEfe <https://github.com/BlueBrain/BluePyEfe>`_ and `BluePyOpt <https://github.com/BlueBrain/BluePyOpt>`_.

For a general overview and example of electrical model building, please refer to the paper: `A universal workflow for creation, validation and generalization of detailed neuronal models <https://doi.org/10.1016/j.patter.2023.100855>`_.

Note that this package only covers e-model building based on patch-clamp data and that it relies solely on the `NEURON <https://www.neuron.yale.edu/neuron/>`_ simulator.

Citation
--------

When you use the BluePyEModel software or method for your research, we ask you to cite the following publication (this includes poster presentations):

.. code-block::

    @software{bluepyemodel_zenodo,
      author       = {Damart, Tanguy and Jaquier, Aurélien and Arnaudon, Alexis and Mandge, Darshan and Van Geit, Werner and Kilic, Ilkan},
      title        = {BluePyEModel},
      month        = aug,
      year         = 2023,
      publisher    = {Zenodo},
      doi          = {8283490},
      url          = {https://doi.org/10.5281/zenodo.8283490}
    }

Installation
------------

BluePyEModel can be pip installed with the following command:

.. code-block:: python

    pip install bluepyemodel[all]

If you do not wish to install all dependencies, specific dependencies can be selected by indicating which ones to install between brackets in place of 'all' (If you want multiple dependencies, they have to be separated by commas). The available dependencies are:

* luigi
* nexus
* all

To get started with the E-Model building pipeline
-------------------------------------------------

.. image:: https://raw.githubusercontent.com/BlueBrain/BluePyEModel/main/doc/images/pipeline.png
   :alt: E-Model building pipeline

This section presents a general picture of the pipeline. For a detailed picture and how to use it, please refer to the `example directory <https://github.com/BlueBrain/BluePyEModel/tree/main/examples/L5PC/>`_ and its `README <https://github.com/BlueBrain/BluePyEModel/tree/main/examples/L5PC/README.rst>`_.

The pipeline is divided in 6 steps:

* ``extraction``: extracts e-features from ephys recordings and averages the results e-feature values along the requested targets.
* ``optimisation``: builds a NEURON cell model and optimises its parameters using as targets the efeatures computed during e-feature extraction.
* ``storage of the model``: reads the results of the extraction and stores the models (best set of parameters) in a local json file.
* ``validation``: reads the models and runs the optimisation protocols and/or validation protocols on them. The e-feature scores obtained on these protocols are then passed to a validation function that decides if the model is good enough.
* ``plotting``: reads the models and runs the optimisation protocols and/or validation protocols on them. Then, plots the resulting traces along the e-feature scores and parameter distributions.
* ``exporting``: read the parameter of the best models and export them in files that can be used either in NEURON or for circuit building.

These six steps are to be run in order as for example validation cannot be run if no models have been stored. Steps ``validation``, ``plotting`` and ``exporting`` are optional. Step ``extraction`` can also be optional in the case where the file containing the protocols and optimisation targets is created by hand or if it is obtained from an older project.

Schematics of BluePyEModel classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/BlueBrain/BluePyEModel/main/doc/images/classes_schema.png
   :alt: Schematics of BluePyEModel classes

Acknowledgment
~~~~~~~~~~~~~~

This work was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology. This work has been partially funded by the European Union Seventh Framework Program (FP7/2007­2013) under grant agreement no. 604102 (HBP), and by the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 720270 (Human Brain Project SGA1) and No. 785907 (Human Brain Project SGA2) and by the EBRAINS research infrastructure, funded from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3).

Copyright
~~~~~~~~~

Copyright (c) 2023-2024 Blue Brain Project/EPFL

This work is licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`_


.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
                :target: https://github.com/BlueBrain/BluePyEModel/blob/main/LICENSE.txt

.. |tests| image:: https://github.com/BlueBrain/BluepyEModel/actions/workflows/test.yml/badge.svg
   :target: https://github.com/BlueBrain/BluepyEModel/actions/workflows/test.yml
   :alt: CI

.. |pypi| image:: https://img.shields.io/pypi/v/bluepyemodel.svg
               :target: https://pypi.org/project/bluepyemodel/
               :alt: latest release

.. |docs| image:: https://readthedocs.org/projects/bluepyemodel/badge/?version=latest
               :target: https://bluepyemodel.readthedocs.io/
               :alt: latest documentation

.. |coverage| image:: https://codecov.io/github/BlueBrain/BluePyEModel/coverage.svg?branch=main
                   :target: https://codecov.io/gh/BlueBrain/bluepyemodel
                   :alt: coverage

.. |zenodo| image:: https://zenodo.org/badge/651152332.svg
                 :target: https://zenodo.org/badge/latestdoi/651152332

..
    The following image is also defined in the index.rst file, as the relative path is
    different, depending from where it is sourced.
    The following location is used for the github README
    The index.rst location is used for the docs README; index.rst also defined an end-marker,
    to skip content after the marker 'substitutions'.

.. substitutions
.. |banner| image:: https://raw.githubusercontent.com/BlueBrain/BluePyEModel/main/doc/source/logo/BluePyEModelBanner.jpg