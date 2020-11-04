"""Evaluations functions for scores, rin and rho factors."""
import json
import logging
import os
import pickle
from functools import partial
from hashlib import sha256
from pathlib import Path

from ..evaluation.evaluator import create_evaluator
from ..evaluation.model import create_cell_model
from ..evaluation.protocols import SearchRinHoldingCurrent
from ..evaluation.modifiers import (
    isolate_axon,
    remove_axon,
    replace_axon_with_taper,
    synth_axon,
)
from .tools.evaluator import evaluate_combos

logger = logging.getLogger(__name__)
RIN_RESPONSE = "rin_noholding"  # "rin_holding" or "rin_noholding"


def _get_synth_modifiers(combo, morph_modifiers=None):
    """Insert synth_axon to start of morph_modifiers if AIS info in combo,
    else use replace_axon_with_taper.
    """
    if morph_modifiers is None:
        morph_modifiers = []
    if "AIS_scale" in combo and combo["AIS_scale"] is not None:
        morph_modifiers.insert(
            0,
            partial(
                synth_axon,
                params=json.loads(combo.AIS_model)["popt"],
                scale=combo.AIS_scale,
            ),
        )
    else:
        morph_modifiers.insert(0, replace_axon_with_taper)
    return morph_modifiers


def _single_evaluation(
    combo,
    emodel_db,
    morph_modifiers=None,
    key="scores",
    morphology_path="morphology_path",
    save_traces=False,
    trace_folder="traces",
    stochasticity=False,
    timeout=1000,
):
    """Evaluating single protocol."""

    cell, protocols, features, emodel_params = get_emodel_data(
        emodel_db, combo, morphology_path, morph_modifiers
    )
    _evaluator = create_evaluator(
        cell_model=cell,
        protocols_definition=protocols,
        features_definition=features,
        stochasticity=stochasticity,
        timeout=timeout,
    )

    responses = _evaluator.run_protocols(
        _evaluator.fitness_protocols.values(),
        emodel_params,
    )
    scores = _evaluator.fitness_calculator.calculate_scores(responses)

    if save_traces:
        _save_traces(trace_folder, responses, get_combo_hash(combo))

    return {key: json.dumps(scores)}


def get_combo_hash(combo):
    """Convert combo values to hash for saving traces."""
    msg = json.dumps(combo[["name", "emodel", "mtype", "etype"]].tolist())
    return sha256(msg.encode()).hexdigest()


def _save_traces(trace_folder, responses, combo_hash):
    """Save traces in a pickle using hashed id from combo data."""
    if not Path(trace_folder).exists():
        os.mkdir(trace_folder)
    pickle.dump(
        responses,
        open(Path(trace_folder) / ("trace_id_" + str(combo_hash) + ".pkl"), "wb"),
    )


def evaluate_scores(
    morphs_combos_df,
    emodel_db,
    task_ids=None,
    morphology_path="morphology_path",
    save_traces=False,
    trace_folder="traces",
    continu=False,
    combos_db_filename="scores_db.sql",
    parallel_factory=None,
):
    """Compute the scores on the combos dataframe.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        task_ids (int): index of combos_original to compute, if None, all will be computed
        morphology_path (str): entry from dataframe with morphology paths
        save_traces (bool): save responses as pickles to plot traces
        trace_folder (str): folder name to save traces pickles
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        combos_db_filename (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed scores
    """
    evaluation_function = partial(
        _single_evaluation,
        emodel_db=emodel_db,
        morphology_path=morphology_path,
        save_traces=save_traces,
        trace_folder=trace_folder,
    )

    return evaluate_combos(
        morphs_combos_df,
        evaluation_function,
        new_columns=[["scores", ""]],
        task_ids=task_ids,
        continu=continu,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )


def get_emodel_data(
    emodel_db,
    combo,
    morphology_path,
    morph_modifiers,
):
    """Gather needed emodel data and build cell model for evaluation."""
    parameters, mechanisms, _ = emodel_db.get_parameters(combo["emodel"])
    protocols = emodel_db.get_protocols(combo["emodel"])
    features = emodel_db.get_features(combo["emodel"])
    emodel_params = emodel_db.get_emodel(combo["emodel"])["parameters"]

    cell = create_cell_model(
        "cell",
        combo[morphology_path],
        mechanisms,
        parameters,
        morph_modifiers=_get_synth_modifiers(combo, morph_modifiers=morph_modifiers),
    )
    return cell, protocols, features, emodel_params


def run_custom(self, cell_model, param_values, rmp, sim=None, isolate=None, timeout=None):
    """Run function for SearchRinHoldingCurrent that saves rin_holding and rin_noholding."""
    # Calculate Rin without holding current
    protocol = self.create_protocol(holding_current=0.0)
    response = protocol.run(cell_model, param_values, sim, isolate, timeout=timeout)
    rin = self.target_Rin.calculate_feature(response)

    holding_current = self.search_holding_current(cell_model, param_values, rmp, rin, sim, isolate)
    if holding_current is None:
        return None

    # Return the response of the final estimate of the holding current
    protocol = self.create_protocol(holding_current=holding_current)
    response = protocol.run(cell_model, param_values, sim, isolate, timeout=timeout)
    rin_holding = self.target_Rin.calculate_feature(response)
    response["bpo_holding_current"] = holding_current
    response["rin_noholding"] = rin
    response["rin_holding"] = rin_holding

    return response


def _rin_evaluation(
    combo,
    emodel_db,
    morph_modifiers=None,
    key="rin",
    morphology_path="morphology_path",
    with_currents=False,
    stochasticity=False,
    timeout=1000,
):
    """Evaluating rin protocol."""

    cell, protocols, features, emodel_params = get_emodel_data(
        emodel_db, combo, morphology_path, morph_modifiers
    )

    _evaluator = create_evaluator(
        cell_model=cell,
        protocols_definition={prot: protocols[prot] for prot in ["RMP", "RinHoldCurrent"]},
        features_definition=features,
        stochasticity=stochasticity,
        timeout=timeout,
    )
    _evaluator.fitness_protocols[
        "main_protocol"
    ].Rin_protocol.run = run_custom.__get__(  # pylint: disable=E1120
        _evaluator.fitness_protocols["main_protocol"].Rin_protocol,
        SearchRinHoldingCurrent,
    )
    responses = _evaluator.run_protocols(_evaluator.fitness_protocols.values(), emodel_params)
    if with_currents:
        return {
            key + "holding_current": responses["bpo_holding_current"],
            key + "threshold_current": responses["bpo_threshold_current"],
        }
    return {key: responses[RIN_RESPONSE]}


def evaluate_ais_rin(
    morphs_combos_df,
    emodel_db,
    task_ids=None,
    morphology_path="morphology_path",
    continu=False,
    combos_db_filename="eval_db.sql",
    parallel_factory=None,
):
    """Compute the input resistance of the ais (axon).

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        task_ids (int): index of combos_original to compute, if None, all will be computed
        morphology_path (str): entry from dataframe with morphology paths
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        combos_db_filename (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin of ais
    """

    key = "rin_ais"
    rin_ais_evaluation = partial(
        _rin_evaluation,
        emodel_db=emodel_db,
        morph_modifiers=[isolate_axon],
        key=key,
        morphology_path=morphology_path,
    )
    return evaluate_combos(
        morphs_combos_df,
        rin_ais_evaluation,
        new_columns=[[key, 0.0]],
        task_ids=task_ids,
        continu=continu,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )


def evaluate_somadend_rin(
    morphs_combos_df,
    emodel_db,
    task_ids=None,
    morphology_path="morphology_path",
    continu=False,
    combos_db_filename="eval_db.sql",
    parallel_factory=None,
):
    """Compute the input resistance of the soma and dentrites.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        task_ids (int): index of combos_original to compute, if None, all will be computed
        morphology_path (str): entry from dataframe with morphology paths
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        combos_db_filename (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin or soma+dendrite
    """
    key = "rin_no_axon"
    rin_dendrite_evaluation = partial(
        _rin_evaluation,
        emodel_db=emodel_db,
        morph_modifiers=[remove_axon],
        key=key,
        morphology_path=morphology_path,
    )
    return evaluate_combos(
        morphs_combos_df,
        rin_dendrite_evaluation,
        new_columns=[[key, 0.0]],
        task_ids=task_ids,
        continu=continu,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )


def evaluate_rho_axon(
    morphs_combos_df,
    emodel_db,
    task_ids=None,
    morphology_path="morphology_path",
    continu=False,
    combos_db_filename="eval_db.sql",
    parallel_factory=None,
):
    """Compute the input resistances and rho factor.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        task_ids (int): index of combos_original to compute, if None, all will be computed
        morphology_path (str): entry from dataframe with morphology paths
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        combos_db_filename (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rho axon
    """
    morphs_combos_df = evaluate_somadend_rin(
        morphs_combos_df,
        emodel_db,
        task_ids=task_ids,
        morphology_path=morphology_path,
        continu=continu,
        combos_db_filename=combos_db_filename,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df = evaluate_ais_rin(
        morphs_combos_df,
        emodel_db,
        task_ids=task_ids,
        morphology_path=morphology_path,
        continu=continu,
        combos_db_filename=combos_db_filename,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df["rho_axon"] = morphs_combos_df.rin_ais / morphs_combos_df.rin_no_axon
    return morphs_combos_df


def evaluate_combos_rho(
    morphs_combos_df,
    emodel_db,
    emodels=None,
    morphology_path="morphology_path",
    save_traces=False,
    trace_folder="traces",
    continu=False,
    combos_db_filename="eval_db.sql",
    parallel_factory=None,
):
    """Evaluate me-combos and rho axons.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        save_traces (bool): save responses as pickles to plot traces
        trace_folder (str): folder name to save traces pickles
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        combos_db_filename (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed scores
    """
    if emodels is None:
        task_ids = morphs_combos_df.index
    else:
        task_ids = morphs_combos_df[morphs_combos_df.emodel.isin(emodels)].index
    morphs_combos_df = evaluate_rho_axon(
        morphs_combos_df,
        emodel_db,
        task_ids,
        continu=continu,
        morphology_path=morphology_path,
        combos_db_filename=combos_db_filename + ".rho",
        parallel_factory=parallel_factory,
    )

    morphs_combos_df = evaluate_scores(
        morphs_combos_df,
        emodel_db,
        task_ids,
        save_traces=save_traces,
        trace_folder=trace_folder,
        continu=continu,
        combos_db_filename=combos_db_filename + ".scores",
        morphology_path=morphology_path,
        parallel_factory=parallel_factory,
    )

    return morphs_combos_df


def evaluate_currents(
    morphs_combos_df,
    emodel_db,
    task_ids=None,
    morphology_path="morphology_path",
    continu=False,
    combos_db_filename="eval_db.sql",
    parallel_factory=None,
):
    """Compute the threshold and holding currents.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        task_ids (int): index of combos_original to compute, if None, all will be computed
        morphology_path (str): entry from dataframe with morphology paths
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        combos_db_filename (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin of ais
    """
    key = ""
    current_evaluation = partial(
        _rin_evaluation,
        emodel_db=emodel_db,
        key=key,
        morphology_path=morphology_path,
        with_currents=True,
    )
    return evaluate_combos(
        morphs_combos_df,
        current_evaluation,
        new_columns=[[key + "holding_current", 0.0], [key + "threshold_current", 0.0]],
        task_ids=task_ids,
        continu=continu,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )
