Running an emodel on BlueCelluLab
=================================

The ``run_emodel.py`` script provides an example to run a simulation of an emodel stored on Nexus with BlueCellulab. The script takes care of downloading all the resources related to the emodel, including hoc templates, morphologies, and mod files.

Prerequisites
-------------
Before running the script, ensure that you have the necessary Python packages installed. It is recommended to create a new virtual environment for this purpose:

1. Create a new virtual environment: ``python -m venv venv``
2. Activate the virtual environment: ``source venv/bin/activate``

With the virtual environment activated, install the following packages:

- ``bluepyemodel``: Install using pip with the command ``pip install bluepyemodel``
- ``bluecellulab``: Install using pip with the command ``pip install bluecellulab```

Usage
-----
To execute the script, you must provide the ``emodel_id``, which is the Nexus ``Resource ID`` for the emodel resource.

To run the script:

.. code-block:: shell

   python run_emodel.py --emodel_id="emodel_id"

After running the script, the simulation results will be saved in the ``figures`` directory.

Configuration
-------------
The script can be customised to alter various model parameters, including stimulus amplitudes, and also change other model parameters such as ``v_init`` and ``temperature``. Modify these parameters directly within the script as needed.
If you want to run a model that uses threshold-based optimization, you need to specify the amplitudes as a percentage of the threshold value.