# BluePyEModel Examples

This directory includes various examples demonstrating how to use BluePyEModel:

- [**L5PC**](./L5PC/README.rst): Provides a guide for setting up and running the E-Model building pipeline locally on a personal computer or on a Slurm cluster. This is an example based on the juvenile (P14) rat somatosensory cortex layer 5 thick-tufted pyramidal cell (L5TPC) e-model optimisation of the [SSCxEModelExamples](https://github.com/BlueBrain/SSCxEModelExamples/tree/main/optimization) repository.

- [**simplecell**](./simplecell/simplecell.ipynb): Demonstrates creating a single-compartment neuron model with two parameters using rheobase independent optimisation.

- [**nexus**](./nexus/README.md): Demonstrates the use of BluePyEModel through the Nexus access point.

- [**others**](./others/README.rst): Contains various examples demonstrating different functionalities and use cases.
    - [**run_emodel**](./others/run_emodel/README.rst): Runs an EModel simulation on BlueCelluLab to explore the single cell behaviour.
    - [**memodel**](./others/memodel/README.md): Creates, runs, analyses, and uploads a MEModel.
    - [**local2nexus**](./others/local2nexus/README.md): Stores a locally built EModel (using LocalAccessPoint) to the [BlueBrain Nexus](https://github.com/BlueBrain/nexus) knowledge graph.
    - [**ICSelector**](./others/icselector/icselector_example.py): Selects ion channel models using gene mappings; assigns optimisation parameter bounds and fixed values for a model.