"""Main functions for AIS synthesis."""
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .evaluators import evaluate_somadend_rin
from .utils import get_emodels

logger = logging.getLogger(__name__)


def _debug_plot(p, scale_min, scale_max, rin_ais, scale, mtype, task_id):
    if not Path("figures_debug").exists():
        os.mkdir("figures_debug")

    import matplotlib.pyplot as plt
    import matplotlib

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


def synthesize_ais(
    morphs_combos_df,
    emodel_db,
    ais_models,
    target_rhos,
    emodels="all",
    morphology_path="morphology_path",
    continu=False,
    ipyp_profile=None,
    parallel_factory=None,
    scales_params=None,
    debug_plots=False,
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
        debug_plots (bool): set to tru to save a figure for each AIS scale
        ipyp_profile (str): name of ipyparallel profile
        parallel_factory (ParallelFactory): parallel factory instance (alternative to ipyp_profile)
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
        ipyp_profile=ipyp_profile,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )

    for task_id in tqdm(task_ids, file=sys.stdout):
        mtype = morphs_combos_df.mtype[task_id]
        emodel = morphs_combos_df.emodel[task_id]
        if mtype not in ais_models:
            mtype = "all"
        ais_model = deepcopy(ais_models[mtype])

        rin_ais = morphs_combos_df.rin_no_axon[task_id] * target_rhos[emodel][mtype]
        p = np.poly1d(ais_model["resistance"][emodel]["polyfit_params"])

        # first ensures we are within the rin range of the fit
        if rin_ais > 10 ** p(np.log10(scale_min)):
            scale = scale_min
            morphs_combos_df.loc[task_id, "ais_failed"] = 1
        elif rin_ais < 10 ** p(np.log10(scale_max)):
            scale = scale_max
            morphs_combos_df.loc[task_id, "ais_failed"] = 1
        else:
            roots_all = (p - np.log10(rin_ais)).r
            roots_real = roots_all[np.imag(roots_all) == 0]
            roots = roots_real[
                (np.log10(scale_min) < roots_real) & (roots_real < np.log10(scale_max))
            ]
            if len(roots) == 0:
                scale = 0
                logger.info("could not find the roots in : %s ", str(roots_real))
                morphs_combos_df.loc[task_id, "ais_failed"] = 1
            else:
                # if multiple root, use the one with scale closest to unity
                scale = 10 ** np.real(roots[np.argmin(abs(roots - 1))])
                morphs_combos_df.loc[task_id, "ais_failed"] = 0

        morphs_combos_df.loc[task_id, "AIS_scale"] = scale
        morphs_combos_df.loc[task_id, "AIS_model"] = json.dumps(ais_model["AIS"])

        if debug_plots:
            _debug_plot(p, scale_min, scale_max, rin_ais, scale, mtype, task_id)

    return morphs_combos_df
