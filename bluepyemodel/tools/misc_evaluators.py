"""Module with some custom evaluator functions."""
import os
import pickle
from copy import copy
from pathlib import Path

from bluepyemodel.ais_synthesis.evaluators import get_emodel_data
from bluepyemodel.evaluation.evaluator import create_evaluator


def save_traces(trace_folder, responses, filename="trace.pkl"):
    """Save traces in a pickle using hashed id from combo data."""
    if not Path(trace_folder).exists():
        os.mkdir(trace_folder)
    pickle.dump(responses, open(filename, "wb"))


def trace_evaluation(
    combo,
    emodel_db,
    morph_modifiers=None,
    morphology_path="morphology_path",
    stochasticity=False,
    timeout=1000,
):
    """Evaluating single protocol and save traces."""

    cell, protocols, features, parameters = get_emodel_data(
        emodel_db, combo, morphology_path, copy(morph_modifiers)
    )
    _evaluator = create_evaluator(
        cell_model=cell,
        protocols_definition=protocols,
        features_definition=features,
        stochasticity=stochasticity,
        timeout=timeout,
    )

    stimuli = _evaluator.fitness_protocols["main_protocol"].subprotocols()
    responses = _evaluator.run_protocols(_evaluator.fitness_protocols.values(), parameters)
    return protocols, stimuli, responses
