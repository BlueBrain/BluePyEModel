======================
Miscellaneous Examples
======================

This folder contains various examples demonstrating different functionalities and use cases.

run_emodel: Run EModel on BlueCelluLab
=======================================

This example demonstrates how to run a simulation of an EModel stored on Nexus using BlueCelluLab to explore the single cell behaviour.

For detailed instructions, refer to the `README <./run_emodel/README.rst>`_.

memodel: Modify and Upload MEModel
==================================

This example demonstrates how to create an MEModel, modify its morphology, run and perform plot analysis, and upload the modified MEModel.

To run the example, edit the `memodel.py` script to specify the MEModel ID and run the script:

.. code-block:: python

    python ./memodel/memodel.py

local2nexus: Export Local E-Model to Nexus
===========================================

This example demonstrates how to store a locally built EModel (using LocalAccessPoint) to the `BlueBrain Nexus <https://github.com/BlueBrain/nexus>`_ knowledge graph.

For detailed instructions, refer to the `README <./local2nexus/README.md>`_.

icselector : Ion Channel Selector
=================================

ICSelector is a tool for selecting ion channel models based on specific sets of ion channel genes, assigning optimization parameter bounds and fixed values for a model. The provided example includes gene/ion channel mapping files to help you test the tool.

To test ICSelector, run the following command:

.. code-block:: bash

    sh ./icselector/test_icselector.sh