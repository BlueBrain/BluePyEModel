# MEModel example

## Purpose

This example demonstrates how to create a MEModel (morpho-electric) model using an existing EModel and a new morphology. The example will create and run the MEModel and plot the results as model traces, feature scores, and currentscape, among others.

## Usage

1. **Set Configuration:**
   - Update `memodel_id` with the ID of the MEModel you want to modify.

2. **Run the Script:**
   ```bash
   python memodel.py

You will be prompted to enter your Nexus access token.

## Notes

- Clean the figures directory before running the script to avoid old data conflicts.
- Ensure that `forge.yml` and `forge_ontology_path.yml` files are present in the script directory.
- Verify that the `memodel_id` is correctly specified to avoid errors.
