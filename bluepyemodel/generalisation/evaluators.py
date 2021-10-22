"""Evaluators."""
import json
import logging
import os
import pickle
from copy import copy
from functools import partial
from hashlib import sha256
from pathlib import Path

from bluepyparallel import evaluate

from bluepyemodel.evaluation.evaluator import create_evaluator
from bluepyemodel.evaluation.evaluator import define_main_protocol
from bluepyemodel.evaluation.evaluator import get_simulator
from bluepyemodel.evaluation.modifiers import isolate_axon
from bluepyemodel.evaluation.modifiers import remove_axon
from bluepyemodel.evaluation.modifiers import replace_axon_with_taper
from bluepyemodel.evaluation.modifiers import synth_axon
from bluepyemodel.model.model import create_cell_model

logger = logging.getLogger(__name__)


def _get_synth_modifiers(combo, morph_modifiers=None):
    """Insert synth_axon to start of morph_modifiers if AIS info in combo,
    else use replace_axon_with_taper.
    """
    if morph_modifiers is None:
        morph_modifiers = []
    if "AIS_scaler" in combo and combo["AIS_scaler"] is not None:
        morph_modifiers.insert(
            0,
            partial(
                synth_axon,
                params=json.loads(combo["AIS_model"])["popt"],
                scale=combo["AIS_scaler"],
            ),
        )
    elif replace_axon_with_taper not in morph_modifiers:
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

    responses = _evaluator.run_protocols(_evaluator.fitness_protocols.values(), parameters)
    scores = _evaluator.fitness_calculator.calculate_scores(responses)

    if save_traces:
        _save_traces(trace_folder, responses, get_combo_hash(combo))

    return {key: json.dumps(scores)}


def get_combo_hash(combo):
    """Convert combo values to hash for saving traces."""
    return sha256(json.dumps(combo).encode()).hexdigest()


def _save_traces(trace_folder, responses, combo_hash):
    """Save traces in a pickle using hashed id from combo data."""
    if not Path(trace_folder).exists():
        os.mkdir(trace_folder)
    with open(Path(trace_folder) / ("trace_id_" + str(combo_hash) + ".pkl"), "wb") as traces_file:
        pickle.dump(responses, traces_file)


def evaluate_scores(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    save_traces=False,
    trace_folder="traces",
    resume=False,
    db_url="scores_db.sql",
    parallel_factory=None,
):
    """Compute the scores on the combos dataframe.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        save_traces (bool): save responses as pickles to plot traces
        trace_folder (str): folder name to save traces pickles
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
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

    return evaluate(
        morphs_combos_df,
        evaluation_function,
        new_columns=[["scores", ""]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def get_emodel_data(emodel_db, combo, morphology_path, morph_modifiers):
    """Gather needed emodel data and build cell model for evaluation."""
    emodel_db.emodel = "_".join(combo["emodel"].split("_")[:2])
    emodel_db.morph_path = combo[morphology_path]
    model_configuration = emodel_db.get_model_configuration()

    protocols = emodel_db.get_protocols()
    features = emodel_db.get_features()

    emodel_db.emodel = combo["emodel"]  # to get the hash from the final
    emodel_params = emodel_db.get_emodel()["parameters"]

    cell = create_cell_model(
        "cell",
        model_configuration=model_configuration,
        morph_modifiers=_get_synth_modifiers(combo, morph_modifiers=morph_modifiers),
    )
    return cell, protocols, features, emodel_params


def _rin_evaluation(
    combo,
    emodel_db,
    morph_modifiers=None,
    key="rin",
    morphology_path="morphology_path",
    with_currents=False,
    stochasticity=False,
    timeout=1000,
    ais_recording=False,
):
    """Evaluating rin protocol."""

    cell_model, _, features, emodel_params = get_emodel_data(
        emodel_db, combo, morphology_path, copy(morph_modifiers)
    )
    main_protocol, features = define_main_protocol(
        {}, features, stochasticity, ais_recording=ais_recording
    )
    cell_model.freeze(emodel_params)
    sim = get_simulator(stochasticity, cell_model)

    if with_currents:
        responses = {}
        for pre_run in [
            main_protocol.run_RMP,
            main_protocol.run_holding,
            main_protocol.run_rin,
            main_protocol.run_threshold,
        ]:
            responses.update(pre_run(cell_model, responses, sim=sim, timeout=timeout)[0])

        cell_model.unfreeze(emodel_params.keys())
        return {
            key + "holding_current": responses["bpo_holding_current"],
            key + "threshold_current": responses["bpo_threshold_current"],
        }
    responses = main_protocol.run_rin(cell_model, {}, sim=sim, timeout=timeout)[0]
    cell_model.unfreeze(emodel_params.keys())
    return {key: responses["bpo_rin"]}


def evaluate_ais_rin(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    resume=False,
    db_url="eval_db.sql",
    parallel_factory=None,
):
    """Compute the input resistance of the ais (axon).

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
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
        ais_recording=True,
    )
    return evaluate(
        morphs_combos_df,
        rin_ais_evaluation,
        new_columns=[[key, 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def evaluate_somadend_rin(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    resume=False,
    db_url="eval_db.sql",
    parallel_factory=None,
):
    """Compute the input resistance of the soma and dentrites.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
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
    return evaluate(
        morphs_combos_df,
        rin_dendrite_evaluation,
        new_columns=[[key, 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def evaluate_rho_axon(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    resume=False,
    db_url="eval_db.sql",
    parallel_factory=None,
):
    """Compute the input resistances and rho factor.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        rersume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rho axon
    """
    morphs_combos_df = evaluate_somadend_rin(
        morphs_combos_df,
        emodel_db,
        morphology_path=morphology_path,
        resume=resume,
        db_url=db_url,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df = evaluate_ais_rin(
        morphs_combos_df,
        emodel_db,
        morphology_path=morphology_path,
        resume=resume,
        db_url=db_url,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df["rho_axon"] = morphs_combos_df.rin_ais / morphs_combos_df.rin_no_axon
    return morphs_combos_df


def evaluate_combos_rho(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    save_traces=False,
    trace_folder="traces",
    resume=False,
    db_url="eval_db.sql",
    parallel_factory=None,
):
    """Evaluate me-combos and rho axons.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        save_traces (bool): save responses as pickles to plot traces
        trace_folder (str): folder name to save traces pickles
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed scores
    """
    morphs_combos_df = evaluate_rho_axon(
        morphs_combos_df,
        emodel_db,
        resume=resume,
        morphology_path=morphology_path,
        db_url=str(db_url) + ".rho",
        parallel_factory=parallel_factory,
    )

    morphs_combos_df = evaluate_scores(
        morphs_combos_df,
        emodel_db,
        save_traces=save_traces,
        trace_folder=trace_folder,
        resume=resume,
        db_url=str(db_url) + ".scores",
        morphology_path=morphology_path,
        parallel_factory=parallel_factory,
    )

    return morphs_combos_df


def evaluate_currents(
    morphs_combos_df,
    emodel_db,
    morphology_path="morphology_path",
    resume=False,
    db_url="eval_db.sql",
    parallel_factory=None,
):
    """Compute the threshold and holding currents.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
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

    return evaluate(
        morphs_combos_df,
        current_evaluation,
        new_columns=[[key + "holding_current", 0.0], [key + "threshold_current", 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )
