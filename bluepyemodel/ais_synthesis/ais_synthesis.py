"""Main functions for AIS synthesis."""
import json
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
from bluepyparallel import evaluate

from .evaluators import evaluate_somadend_rin
from .utils import get_emodels

logger = logging.getLogger(__name__)


def _debug_plot(p, scale_min, scale_max, rin_ais, scale, mtype, task_id):
    if not Path("figures_debug").exists():
        os.mkdir("figures_debug")

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    plt.figure()
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 1000)
    plt.plot(scales, 10 ** p(np.log10(scales)))
    plt.axhline(10 ** p(np.log10(scale_min)), c="r")
    plt.axhline(10 ** p(np.log10(scale_max)), c="g")
    plt.axhline(rin_ais)
    plt.axvline(scale)
    plt.suptitle(mtype)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("figures_debug/AIS_scale_" + str(task_id) + ".png")
    plt.close()


def _synth_combo(combo, ais_models, target_rhos, scale_min, scale_max):
    """compute AIS  scale."""
    mtype = combo["mtype"]
    emodel = combo["emodel"]
    if mtype not in ais_models:
        mtype = "all"

    rin_ais = combo["rin_no_axon"] * target_rhos[emodel][mtype]
    p = np.poly1d(ais_models[mtype]["resistance"][emodel]["polyfit_params"])

    # first ensures we are within the rin range of the fit
    if rin_ais > 10 ** p(np.log10(scale_min)):
        scale = scale_min
        ais_failed = 1
    elif rin_ais < 10 ** p(np.log10(scale_max)):
        scale = scale_max
        ais_failed = 1
    else:
        roots_all = (p - np.log10(rin_ais)).r
        roots_real = roots_all[np.imag(roots_all) == 0]
        roots = roots_real[(np.log10(scale_min) < roots_real) & (roots_real < np.log10(scale_max))]
        if len(roots) == 0:
            scale = 0
            logger.info("could not find the roots in : %s ", str(roots_real))
            ais_failed = 1
        else:
            # if multiple root, use the one with scale closest to unity
            scale = 10 ** np.real(roots[np.argmin(abs(roots - 1))])
            ais_failed = 0

    #  if debug_plots:
    #    _debug_plot(p, scale_min, scale_max, rin_ais, scale, mtype, task_id)
    return {
        "ais_failed": ais_failed,
        "AIS_scale": scale,
        "AIS_model": json.dumps(ais_models[mtype]["AIS"]),
    }


def _clean_ais_model(ais_models):
    """Remove unnecessary entries in ais_model dict to speed up parallelisation"""
    ais_models_clean = {}
    for mtype in ais_models:
        ais_models_clean[mtype] = {}
        ais_models_clean[mtype]["AIS"] = ais_models[mtype]["AIS"]
        ais_models_clean[mtype]["resistance"] = {}
        for emodel in ais_models[mtype]["resistance"]:
            ais_models_clean[mtype]["resistance"][emodel] = {}
            ais_models_clean[mtype]["resistance"][emodel]["polyfit_params"] = ais_models[mtype][
                "resistance"
            ][emodel]["polyfit_params"]
    return ais_models_clean


def synthesize_ais(
    morphs_combos_df,
    emodel_db,
    ais_models,
    target_rhos,
    emodels=None,
    morphology_path="morphology_path",
    continu=False,
    parallel_factory=None,
    scales_params=None,
    combos_db_filename="synth_db.sql",
):
    """Synthesize AIS to match target rho_axon.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        ais_models (dict): dict with ais models
        target_rhos (dict): dict with target rhos
        emodels (list/str): list of emodels to consider, or 'all'
        continu (bool): to ecrase previous AIS Rin computations
        scales_params (dict): parmeter for scales of AIS to use
        parallel_factory (ParallelFactory): parallel factory instance
    """
    emodels = get_emodels(morphs_combos_df, emodels)

    if scales_params["lin"]:
        scale_min = scales_params["min"]
        scale_max = scales_params["max"]
    else:
        scale_min = 10 ** scales_params["min"]
        scale_max = 10 ** scales_params["max"]

    task_ids = morphs_combos_df[morphs_combos_df.emodel.isin(emodels)].index
    morphs_combos_df = evaluate_somadend_rin(
        morphs_combos_df,
        emodel_db,
        task_ids=task_ids,
        morphology_path=morphology_path,
        continu=continu,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )

    synth_combo = partial(
        _synth_combo,
        ais_models=_clean_ais_model(ais_models),
        target_rhos=target_rhos,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return evaluate(
        morphs_combos_df,
        synth_combo,
        new_columns=[["ais_failed", 1], ["AIS_scale", 1.0], ["AIS_model", ""]],
        task_ids=task_ids,
        continu=continu,
        parallel_factory=parallel_factory,
        no_sql=True,  # each evaluation is fast, so we won't use sql backend for massive speedup
    )
