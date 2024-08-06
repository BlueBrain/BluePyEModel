======================
Miscellaneous Examples
======================

This folder contains various examples demonstrating different functionalities and use cases.

icselector : Ion Channel Selector
=================================

ICSelector is a tool for selecting ion channel models based on specific sets of ion channel genes. The provided example includes gene/ion channel mapping files to help you test the tool.

To test ICSelector, run the following command:

.. code-block:: bash

    sh ./icselector/test_icselector.sh


local2nexus: Export Local E-Model to Nexus
===========================================

This example demonstrates how to export a locally built e-model to Nexus.

For detailed instructions, refer to the `README <./local2nexus/README.md>`_.


memodel: Modify and Upload MEModel
==================================

This example demonstrates how to retrieve an MEModel, modify its morphology, perform plot analysis, and upload the modified MEModel.

To run the example, edit the `memodel.py` script to specify the MEModel ID and run the script:

.. code-block:: python

    python ./memodel/memodel.py


run_emodel: Run EModel on BlueCelluLab
=======================================

This example demonstrates how to run a simulation of an EModel stored on Nexus using BlueCelluLab.

For detailed instructions, refer to the `README <./run_emodel/README.rst>`_.