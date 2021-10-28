"""Module with some custom evaluator functions."""
import json
import pickle
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
from bluepyparallel import evaluate

from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.modifiers import synth_axon


def single_feature_evaluation(
    combo,
    emodel_db,
    morphology_path="morphology_path",
    stochasticity=False,
    timeout=1000000,
    trace_data_path=None,
    score_threshold=12.0,
    max_threshold_voltage=0,
    nseg_frequency=40,
    dt=None,
):
    """Evaluating single protocol and save traces."""
    emodel_db.set_emodel(combo["emodel"])
    if morphology_path in combo:
        emodel_db.morph_path = combo[morphology_path]
    if "AIS_scaler" in combo and "AIS_params" in combo:
        emodel_db.pipeline_settings.morph_modifiers = [
            partial(synth_axon, params=combo["AIS_params"], scale=combo["AIS_scaler"])
        ]

    emodel_db.emodel = combo["emodel"]
    evaluator = get_evaluator_from_access_point(
        emodel_db,
        stochasticity=stochasticity,
        timeout=timeout,
        score_threshold=score_threshold,
        max_threshold_voltage=max_threshold_voltage,
        nseg_frequency=nseg_frequency,
        dt=dt,
        strict_holding_bounds=False

    )
    params = emodel_db.get_emodel()["parameters"]
    if "new_parameters" in combo:
        params.update(combo["new_parameters"])

    responses = evaluator.run_protocols(evaluator.fitness_protocols.values(), params)
    features = evaluator.fitness_calculator.calculate_values(responses)
    scores = evaluator.fitness_calculator.calculate_scores(responses)

    for f, val in features.items():
        if isinstance(val, np.ndarray) and len(val) > 0:
            try:
                features[f] = np.nanmean(val)
            except AttributeError:
                features[f] = None
        else:
            features[f] = None

    if trace_data_path is not None:
        Path(trace_data_path).mkdir(exist_ok=True, parents=True)
        stimuli = evaluator.fitness_protocols["main_protocol"].subprotocols()
        hash_id = sha256(json.dumps(combo).encode()).hexdigest()
        trace_data_path = f"{trace_data_path}/trace_data_{hash_id}.pkl"
        with open(trace_data_path, "wb") as handle:
            pickle.dump([stimuli, responses], handle)

    return {
        "features": json.dumps(features),
        "scores": json.dumps(scores),
        "trace_data": trace_data_path,
    }


def feature_evaluation(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    resume=False,
    db_url=None,
    parallel_factory=None,
    trace_data_path=None,
    stochasticity=False,
    score_threshold=12.0,
    timeout=1000000,
    nseg_frequency=40,
    dt=None,
):
    """Compute the features and the scores on the combos dataframe.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance
        dt (float): if not None, cvode will be disabled and fixed timesteps used.

    Returns:
        pandas.DataFrame: original combos with computed scores and features
    """
    evaluation_function = partial(
        single_feature_evaluation,
        emodel_db=emodel_db,
        morphology_path=morphology_path,
        trace_data_path=trace_data_path,
        stochasticity=stochasticity,
        score_threshold=score_threshold,
        timeout=timeout,
        nseg_frequency=nseg_frequency,
        dt=dt,
    )

    return evaluate(
        morphs_combos_df,
        evaluation_function,
        new_columns=[["features", ""], ["scores", ""], ["trace_data", ""]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )
