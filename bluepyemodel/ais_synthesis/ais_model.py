"""Repair functions for cell morphologies, given an emodel."""
import json
import logging

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

from morphio.mut import Morphology
from morphio import SectionType

from .evaluators import evaluate_ais_rin, evaluate_rho_axon, evaluate_scores
from .utils import get_mtypes, get_scores

logger = logging.getLogger(__name__)
POLYFIT_DEGREE = 10


def get_ais(neuron):
    """Get the axon initial section of a neuron."""
    for neurite in neuron.root_sections:
        if neurite.type == SectionType.axon:
            return neurite
    raise Exception("AIS not found")


def taper_function(length, strength, scale, terminal_diam):
    """Function to model tappers AIS."""
    return strength * np.exp(-length / scale) + terminal_diam


def extract_ais_diameters(morphologies):
    """Produce an iterator on ais diameters."""
    for neuron in morphologies:
        ais = get_ais(neuron)
        yield ais.diameters


def extract_ais_path_distances(morphologies):
    """Produce an iterator on ais diameters."""
    for neuron in morphologies:
        ais = get_ais(neuron)
        yield np.insert(
            np.linalg.norm(np.cumsum(np.diff(ais.points, axis=0), axis=0), axis=1), 0, 0
        )


def build_ais_diameter_model(morphology_paths, bin_size=2, total_length=60):
    """Build the AIS model by fitting first sections of axons.

    Args:
        morphology_paths (list): list of paths to morphologies
        bin_size (float): size of bins (in unit length) for smoothing of diameters
        total_length (flow): length of AIS
    """
    morphologies = [Morphology(str(morphology)) for morphology in morphology_paths]

    distances, diameters = [], []
    for dist, diams in zip(
        extract_ais_path_distances(morphologies), extract_ais_diameters(morphologies)
    ):
        distances += list(dist)
        diameters += list(diams)

    all_bins = np.arange(0, total_length, bin_size)
    indices = np.digitize(np.array(distances), all_bins, right=False)

    means, stds, bins = [], [], []
    for i in list(set(indices)):
        diams = np.array(diameters)[indices == i]
        means.append(np.mean(diams))
        stds.append(np.std(diams))
        bins.append(all_bins[i - 1])

    popt, _ = curve_fit(taper_function, np.array(bins), np.array(means))[:2]
    logger.debug("Taper strength: %s, scale: %s, terminal diameter: %s", *popt)

    model = {}
    # first value is the length of AIS
    model["AIS"] = {
        "popt_names": ["length"] + list(taper_function.__code__.co_varnames[1:]),
        "popt": [total_length] + popt.tolist(),
    }

    model["data"] = {
        "distances": np.array(distances).tolist(),
        "diameters": np.array(diameters).tolist(),
        "bins": np.array(bins).tolist(),
        "means": np.array(means).tolist(),
        "stds": np.array(stds).tolist(),
    }
    return model


def build_ais_diameter_models(
    morphs_df, mtypes="all", morphology_path="morphology_path", mtype_dependent=False
):
    """Build ais diameter models from data, and plot the results.

    Args:
        morphs_df (dataframe): dataframe with morphologies data
        mtypes (str): mtypes to use, can be == 'all'
        morphology_path (str): column name of dataframe for paths to morphologies
        mtyoe_dependent (bool): if True, we will try to build a model per mtype
    Returns:
        (dict, dict): dictionaries of models and data for plotting
    """
    logger.info("Building AIS diameter models from data")

    mtypes = get_mtypes(morphs_df, mtypes)
    logger.info("Extracting model from all axons")
    morphologies_all = morphs_df.loc[:, morphology_path].to_list()

    models = {}
    models["all"] = build_ais_diameter_model(morphologies_all)
    if not mtype_dependent:
        return models

    for mtype in mtypes:
        logger.info("Extracting %s", mtype)

        morphologies = morphs_df.loc[
            morphs_df.mtype == mtype, morphology_path
        ].to_list()
        if len(morphologies) > 5:
            model = build_ais_diameter_model(morphologies)
            if model["AIS"]["popt"][0] > 0.0:
                models[mtype] = model
                models[mtype]["used_all_data"] = False
            else:
                logger.info("Negative tapering, we use all the data instead.")
                models[mtype] = models["all"]
                models[mtype]["used_all_data"] = True
        else:
            logger.info(
                "m-type not used because of less than 5 cells, we use all the data instead"
            )
            models[mtype] = models["all"]
            models[mtype]["used_all_data"] = True

    return models


def _get_scales(scales_params, with_unity=False):
    """Create scale array from parameters."""
    if scales_params["lin"]:
        scales = np.linspace(
            scales_params["min"], scales_params["max"], scales_params["n"]
        )
    else:
        scales = np.logspace(
            scales_params["min"], scales_params["max"], scales_params["n"]
        )

    if with_unity:
        return np.insert(scales, 0, 1)
    return scales


def _prepare_scaled_combos(
    morphs_combos_df, ais_models, scales_params, emodel, mtypes="all"
):
    """Prepare combos with scaled AIS."""
    scales = _get_scales(scales_params)
    mtypes = get_mtypes(morphs_combos_df, mtypes)

    if mtypes[0] not in ais_models:
        mtypes = ["all"]

    fit_df = pd.DataFrame()
    for mtype in mtypes:
        me_mask = morphs_combos_df.emodel == emodel
        if mtypes[0] != "all":
            me_mask = me_mask & (morphs_combos_df.mtype == mtype)

        if len(morphs_combos_df[me_mask]) > 0:
            logger.info("Creating combos for %s, %s", emodel, mtype)
            df_tmp = morphs_combos_df[me_mask & morphs_combos_df.use_axon].iloc[0]
            for scale in scales:
                df_tmp["AIS_scale"] = scale
                df_tmp["AIS_model"] = json.dumps(ais_models[mtype]["AIS"])
                df_tmp["for_optimisation"] = False
                fit_df = fit_df.append(df_tmp.copy())
    return fit_df.reset_index()


def build_ais_resistance_models(
    morphs_combos_df,
    emodel_db,
    emodel,
    ais_models,
    scales_params,
    morphology_path="morphology_path",
    ipyp_profile=None,
    continu=False,
    combos_db_filename="eval_db.sql",
):
    """Build AIS resistance models for an emodel.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        emodel (str): emodel to consider
        ais_models: (dict): dict with ais models
        scales_params (dict): parmeter for scales of AIS to use
        continu (bool): to ecrase previous AIS Rin computations
    Returns:
        (dataframe, dict): dataframe with Rin results and models for plotting
    """
    fit_df = _prepare_scaled_combos(morphs_combos_df, ais_models, scales_params, emodel)
    fit_df = evaluate_ais_rin(
        fit_df,
        emodel_db,
        continu=continu,
        combos_db_filename=combos_db_filename,
        morphology_path=morphology_path,
        ipyp_profile=ipyp_profile,
    )

    mtypes = get_mtypes(morphs_combos_df, "all")
    if mtypes[0] not in ais_models:
        mtypes = ["all"]

    mask = fit_df.emodel == emodel
    for mtype in mtypes:
        if mtypes[0] != "all":
            mask = mask & (fit_df.mtype == mtype)
        if len(fit_df[mask]) > 0:
            if "resistance" not in ais_models[mtype]:
                ais_models[mtype]["resistance"] = {}
            ais_models[mtype]["resistance"][emodel] = {
                "polyfit_params": np.polyfit(
                    np.log10(fit_df[mask].AIS_scale.to_numpy()),
                    np.log10(fit_df[mask].rin_ais.to_numpy()),
                    POLYFIT_DEGREE,
                ).tolist()
            }

    return fit_df, ais_models


def _prepare_scan_rho_combos(
    morphs_combos_df, ais_models, scales_params, emodel, mtypes="all"
):
    """Prepare the combos for scaning rho."""
    scales = _get_scales(scales_params, with_unity=True)

    mtypes = get_mtypes(morphs_combos_df, mtypes)

    if mtypes[0] not in ais_models:
        mtypes = ["all"]

    mask = morphs_combos_df.emodel == emodel
    rho_scan_df = pd.DataFrame()
    for mtype in mtypes:
        if mtypes[0] != "all":
            me_mask = mask & (morphs_combos_df.mtype == mtype)
        else:
            me_mask = mask
        if len(morphs_combos_df[me_mask]) > 0:
            logger.info("creating rows for %s, %s", emodel, mtype)
            new_row = morphs_combos_df[mask & morphs_combos_df.for_optimisation].copy()
            if len(new_row) > 1:
                if mtype in new_row.mtype:
                    new_row = new_row[new_row.mtype == mtype]
                else:
                    new_row = new_row.head(1)
                    logger.info(
                        "Multiple candidates but no matching mtypes, we take the first %s.",
                        new_row.mtype.to_numpy(),
                    )
            if len(new_row) > 0:
                for scale in scales:
                    new_row["mtype"] = mtype
                    new_row["AIS_scale"] = scale
                    new_row["AIS_model"] = json.dumps(ais_models[mtype]["AIS"])
                    new_row["for_optimisation"] = False
                    rho_scan_df = rho_scan_df.append(new_row.copy())
            else:
                logger.info("no cell for %s, %s", emodel, mtype)

    return rho_scan_df.reset_index()


def get_rho_targets(final):
    """Use emodel parameters to estimate target rho axon."""

    axon_params = [
        "gNaTgbar_NaTg.axonal",
        "gK_Tstbar_K_Tst.axonal",
        "gK_Pstbar_K_Pst.axonal",
        "gCa_HVAbar_Ca_HVA2.axonal",
    ]
    fit_params = {"param_min": 0, "param_max": 1, "target_min": 20, "target_max": 100}

    df = pd.DataFrame()
    for emodel in final:
        for name, val in final[emodel]["params"].items():
            if name in axon_params:
                df.loc[emodel, name] = val

    df = (df - df.min()) / (df.max() - df.min())
    df["mean_axon"] = df.mean(axis=1)
    df["target"] = fit_params["target_max"] * (
        df["mean_axon"] - fit_params["param_min"]
    ) / (fit_params["param_max"] - fit_params["param_min"]) + fit_params[
        "target_min"
    ] * (
        df["mean_axon"] - fit_params["param_max"]
    ) / (
        fit_params["param_min"] - fit_params["param_max"]
    )
    return df["target"]


def find_target_rho_axon(
    morphs_combos_df,
    emodel_db,
    emodel,
    ais_models,
    scales_params,
    morphology_path="morphology_path",
    continu=False,
    ipyp_profile=None,
    filter_sigma=1,
    combos_db_filename="rho_scan_db.sql",
):
    """Find the target rho axons for an emodel.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        emodel (str): emodel to consider
        ais_models_file (str): path to yaml with ais models
        scales_params (dict): parameter for scales of AIS to use
        continu (bool): to ecrase previous AIS Rin computations
        filter_sigma (float): sigma for guassian smoothing of mean scores,
            using (scipy.ndimage.filters.gaussian_filter)

    Returns:
        (dataframe, dict): dataframe with results and dict target rhos for plots
    """
    rho_scan_df = _prepare_scan_rho_combos(
        morphs_combos_df, ais_models, scales_params, emodel
    )
    rho_scan_df = evaluate_rho_axon(
        rho_scan_df,
        emodel_db,
        morphology_path=morphology_path,
        continu=continu,
        combos_db_filename=combos_db_filename + ".rho",
        ipyp_profile=ipyp_profile,
    )

    rho_scan_df = evaluate_scores(
        rho_scan_df,
        emodel_db,
        save_traces=False,
        continu=continu,
        combos_db_filename=combos_db_filename + ".scores",
        morphology_path=morphology_path,
        ipyp_profile=ipyp_profile,
    )

    rho_scan_df = get_scores(rho_scan_df)
    mtypes = get_mtypes(morphs_combos_df, "all")

    if mtypes[0] not in ais_models:
        mtypes = ["all"]

    mask = rho_scan_df.emodel == emodel
    target_rhos = {}
    for mtype in mtypes:
        if mtypes[0] != "all":
            mask = mask & (rho_scan_df.mtype == mtype)
        if len(rho_scan_df[mask]) > 0:
            scale_mask = rho_scan_df.AIS_scale == 1
            median_scores = rho_scan_df[mask & ~scale_mask].median_score.to_numpy()
            smooth_score = gaussian_filter(median_scores, filter_sigma)
            rho_scan_df.loc[mask & ~scale_mask, "smooth_score"] = smooth_score
            best_score_id = rho_scan_df[mask & ~scale_mask].smooth_score.idxmin()

            if emodel not in target_rhos:
                target_rhos[emodel] = {}
            target_rhos[emodel][mtype] = float(
                rho_scan_df.loc[best_score_id, "rho_axon"]
            )

    return rho_scan_df, target_rhos
