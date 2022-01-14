"""Repair functions for cell morphologies, given an emodel."""
import json
import logging

import neurom as nm
import numpy as np
import pandas as pd
from morph_tool.neuron_surface import get_NEURON_surface
from morphio import SectionType
from morphio.mut import Morphology
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from tqdm import tqdm

from bluepyemodel.generalisation.evaluators import evaluate_ais_rin
from bluepyemodel.generalisation.evaluators import evaluate_rho
from bluepyemodel.generalisation.evaluators import evaluate_rho_axon
from bluepyemodel.generalisation.evaluators import evaluate_scores
from bluepyemodel.generalisation.evaluators import evaluate_soma_rin
from bluepyemodel.generalisation.utils import get_mtypes
from bluepyemodel.generalisation.utils import get_scores

logger = logging.getLogger(__name__)
POLYFIT_DEGREE = 10
RHO_FACTOR_FEATURES = [
    "ohmic_input_resistance_vb_ssse",
    "bpo_holding_current",
    "bpo_threshold_current",
    "mean_frequency",
    "inv_time_to_first_spike",
    "AHP_depth",
]


def build_soma_model(morphology_paths):
    """Build soma model.

    Using only surface area for now.
    """
    soma_surfaces = [float(get_NEURON_surface(path)) for path in tqdm(morphology_paths)]
    soma_radii = [
        float(nm.get("soma_radius", nm.load_morphology(path))) for path in morphology_paths
    ]
    return {
        "soma_model": {
            "soma_surface": float(np.mean(soma_surfaces)),
            "soma_radius": float(np.mean(soma_radii)),
        },
        "soma_data": {
            "soma_radii": soma_radii,
            "soma_surfaces": soma_surfaces,
        },
    }


def build_soma_models(
    morphs_df, mtypes="all", morphology_path="morphology_path", mtype_dependent=False
):
    """Build soma models from data, and plot the results.

    Args:
        morphs_df (dataframe): dataframe with morphologies data
        mtypes (str): mtypes to use, can be == 'all'
        morphology_path (str): column name of dataframe for paths to morphologies
        mtype_dependent (bool): if True, we will try to build a model per mtype
    Returns:
        (dict, dict): dictionaries of models and data for plotting
    """
    logger.info("Building soma models from data")

    if not mtype_dependent:
        if mtypes is not None:
            morphs_df = morphs_df[morphs_df.mtype.isin(mtypes) & morphs_df.for_optimisation]
        return {"all": build_soma_model(morphs_df[morphology_path].to_list())}

    models = {}
    mtypes = get_mtypes(morphs_df, mtypes)
    for mtype in tqdm(mtypes):
        morphologies = morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list()
        model = build_soma_model(morphologies)
        models[mtype] = model

    return models


def _prepare_soma_scaled_combos(morphs_combos_df, soma_models, scales_params, emodel):
    """Prepare combos with scaled soma."""
    scales = get_scales(scales_params)

    fit_df = pd.DataFrame()
    mask = morphs_combos_df.emodel == emodel
    if len(morphs_combos_df[mask]) > 0:
        logger.info("Creating combos for %s", emodel)
        df_tmp = morphs_combos_df[mask].iloc[0]
        for scale in scales:
            df_tmp["soma_scaler"] = scale
            df_tmp["soma_model"] = json.dumps(soma_models["all"]["soma_model"])
            df_tmp["for_optimisation"] = False
            fit_df = fit_df.append(df_tmp.copy())
    return fit_df.reset_index(drop=True)


def build_soma_resistance_models(
    morphs_combos_df,
    emodel_db,
    emodel,
    soma_models,
    scales_params,
    morphology_path="morphology_path",
    parallel_factory=None,
    resume=False,
    db_url="eval_db.sql",
):
    """Build Soma resistance models for an emodel.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        emodel (str): emodel to consider
        ais_models: (dict): dict with ais models
        scales_params (dict): parmeter for scales of AIS to use
        resume (bool): to ecrase previous AIS Rin computations
    Returns:
        (dataframe, dict): dataframe with Rin results and models for plotting
    """
    fit_df = _prepare_soma_scaled_combos(morphs_combos_df, soma_models, scales_params, emodel)
    fit_df = evaluate_soma_rin(
        fit_df,
        emodel_db,
        resume=resume,
        db_url=db_url,
        morphology_path=morphology_path,
        parallel_factory=parallel_factory,
    )

    mtype = "all"

    mask = fit_df.emodel == emodel
    if len(fit_df[mask]) > 0:
        if "resistance" not in soma_models[mtype]:
            soma_models[mtype]["resistance"] = {}
        soma_scaler = fit_df[mask].soma_scaler.to_numpy()
        rin_soma = fit_df[mask].rin_soma.to_numpy()
        soma_scaler = soma_scaler[rin_soma > 0]
        rin_soma = rin_soma[rin_soma > 0]
        if len(rin_soma) != len(fit_df[mask].soma_scaler.to_numpy()):
            logger.warning("Some soma Rin are < 0, we will drop these!")
        soma_models[mtype]["resistance"][emodel] = {
            "polyfit_params": np.polyfit(
                np.log10(soma_scaler),
                np.log10(rin_soma),
                POLYFIT_DEGREE,
            ).tolist()
        }

    return fit_df, soma_models


def find_target_rho(
    morphs_combos_df,
    emodel_db,
    emodel,
    morphology_path="morphology_path",
    resume=False,
    parallel_factory=None,
):
    """Find the target rho for an emodel.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        emodel (str): emodel to consider
        soma_models (str): path to yaml with ais models
        scales_params (dict): parameter for scales of AIS to use
        resume (bool): to ecrase previous AIS Rin computations
        filter_sigma (float): sigma for guassian smoothing of mean scores,
            using (scipy.ndimage.filters.gaussian_filter)

    Returns:
        (dataframe, dict): dataframe with results and dict target rhos for plots
    """
    rho_df = evaluate_rho(
        morphs_combos_df,
        emodel_db,
        morphology_path=morphology_path,
        resume=resume,
        db_url=None,
        parallel_factory=parallel_factory,
    )

    return {emodel: {"all": float(rho_df.rho.median())}}, rho_df


def get_ais(neuron):
    """Get the axon initial section of a neuron."""
    for neurite in neuron.root_sections:
        if neurite.type == SectionType.axon:
            return neurite
    raise Exception("AIS not found")


def taper_function(length, strength, taper_scale, terminal_diameter):
    """Function to model tappers AIS."""
    return strength * np.exp(-length / taper_scale) + terminal_diameter


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


def build_ais_diameter_model(morphology_paths, bin_size=2, total_length=60, with_taper=False):
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
    bounds = [3 * [-np.inf], 3 * [np.inf]]
    if not with_taper:
        bounds[0][0] = 0.0
        bounds[1][0] = 0.000001

    popt, _ = curve_fit(taper_function, np.array(bins), np.array(means), bounds=bounds)[:2]
    logger.debug("Taper strength: %s, scale: %s, terminal diameter: %s", *popt)

    model = {}
    # first value is the length of AIS
    model["AIS_model"] = {
        "popt_names": ["length"] + list(taper_function.__code__.co_varnames[1:]),
        "popt": [total_length] + popt.tolist(),
    }

    model["AIS_data"] = {
        "distances": np.array(distances).tolist(),
        "diameters": np.array(diameters).tolist(),
        "bins": np.array(bins).tolist(),
        "means": np.array(means).tolist(),
        "stds": np.array(stds).tolist(),
    }
    return model


def build_ais_diameter_models(
    morphs_df,
    mtypes="all",
    morphology_path="morphology_path",
    mtype_dependent=False,
    with_taper=True,
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
    morphologies_all = morphs_df[morphs_df.for_optimisation][morphology_path].to_list()

    models = {}
    models["all"] = build_ais_diameter_model(morphologies_all, with_taper=with_taper)
    if not mtype_dependent:
        return models

    for mtype in tqdm(mtypes):
        morphologies = morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list()
        if len(morphologies) > 5:
            model = build_ais_diameter_model(morphologies, with_taper=with_taper)
            if model["AIS"]["popt"][0] > 0.0:
                models[mtype] = model
                models[mtype]["used_all_data"] = False
            else:
                logger.info("Negative tapering, we use all the data instead.")
                models[mtype] = models["all"]
                models[mtype]["used_all_data"] = True
        else:
            logger.info("m-type not used because of less than 5 cells, we use all the data instead")
            models[mtype] = models["all"]
            models[mtype]["used_all_data"] = True

    return models


def get_scales(scales_params, with_unity=False):
    """Create scale array from parameters."""
    if scales_params["lin"]:
        scales = np.linspace(scales_params["min"], scales_params["max"], scales_params["n"])
    else:
        scales = np.logspace(scales_params["min"], scales_params["max"], scales_params["n"])

    if with_unity:
        return np.insert(scales, 0, 1)
    return scales


def _prepare_ais_scaled_combos(morphs_combos_df, ais_models, scales_params, emodel):
    """Prepare combos with scaled AIS."""
    scales = get_scales(scales_params)

    fit_df = pd.DataFrame()
    mask = morphs_combos_df.emodel == emodel
    if len(morphs_combos_df[mask]) > 0:
        logger.info("Creating combos for %s", emodel)
        df_tmp = morphs_combos_df[mask].iloc[0]
        for scale in scales:
            df_tmp["AIS_scaler"] = scale
            df_tmp["AIS_model"] = json.dumps(ais_models["all"]["AIS_model"])
            df_tmp["for_optimisation"] = False
            fit_df = fit_df.append(df_tmp.copy())
    return fit_df.reset_index(drop=True)


def build_ais_resistance_models(
    morphs_combos_df,
    emodel_db,
    emodel,
    ais_models,
    scales_params,
    morphology_path="morphology_path",
    parallel_factory=None,
    resume=False,
    db_url="eval_db.sql",
):
    """Build AIS resistance models for an emodel.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        emodel (str): emodel to consider
        ais_models: (dict): dict with ais models
        scales_params (dict): parmeter for scales of AIS to use
        resume (bool): to ecrase previous AIS Rin computations
    Returns:
        (dataframe, dict): dataframe with Rin results and models for plotting
    """
    fit_df = _prepare_ais_scaled_combos(morphs_combos_df, ais_models, scales_params, emodel)
    fit_df = evaluate_ais_rin(
        fit_df,
        emodel_db,
        resume=resume,
        db_url=db_url,
        morphology_path=morphology_path,
        parallel_factory=parallel_factory,
    )

    mtype = "all"

    mask = fit_df.emodel == emodel
    if len(fit_df[mask]) > 0:
        if "resistance" not in ais_models[mtype]:
            ais_models[mtype]["resistance"] = {}
        AIS_scaler = fit_df[mask].AIS_scaler.to_numpy()
        rin_ais = fit_df[mask].rin_ais.to_numpy()
        AIS_scaler = AIS_scaler[rin_ais > 0]
        rin_ais = rin_ais[rin_ais > 0]
        if len(rin_ais) != len(fit_df[mask].AIS_scaler.to_numpy()):
            logger.warning("Some AIS Rin are < 0, we will drop these!")
        ais_models[mtype]["resistance"][emodel] = {
            "polyfit_params": np.polyfit(
                np.log10(AIS_scaler),
                np.log10(rin_ais),
                POLYFIT_DEGREE,
            ).tolist()
        }

    return fit_df, ais_models


def _prepare_scan_rho_combos(morphs_combos_df, ais_models, scales_params, emodel):
    """Prepare the combos for scaning rho."""
    scales = get_scales(scales_params, with_unity=True)

    mask = morphs_combos_df.emodel == emodel
    rho_scan_df = pd.DataFrame()
    if len(morphs_combos_df[mask]) > 0:
        logger.info("creating rows for %s", emodel)
        new_row = morphs_combos_df[mask & morphs_combos_df.for_optimisation].copy()
        if len(new_row) > 1:
            new_row = new_row.head(1)
            logger.info("Multiple candidates, we take the first.")
        if len(new_row) == 1:
            for scale in scales:
                new_row["mtype"] = "all"
                new_row["AIS_scaler"] = scale
                new_row["AIS_model"] = json.dumps(ais_models["all"]["AIS_model"])
                new_row["for_optimisation"] = False
                rho_scan_df = rho_scan_df.append(new_row.copy())
        else:
            logger.info("no cell for %s", emodel)

    return rho_scan_df.reset_index(drop=True)


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
        for name, val in final[emodel]["parameters"].items():
            if name in axon_params:
                df.loc[emodel, name] = val

    df = (df - df.min()) / (df.max() - df.min())
    df["mean_axon"] = df.mean(axis=1)
    df["target"] = fit_params["target_max"] * (df["mean_axon"] - fit_params["param_min"]) / (
        fit_params["param_max"] - fit_params["param_min"]
    ) + fit_params["target_min"] * (df["mean_axon"] - fit_params["param_max"]) / (
        fit_params["param_min"] - fit_params["param_max"]
    )
    return df["target"]


def find_target_rho_axon(
    morphs_combos_df,
    emodel_db,
    emodel,
    morphology_path="morphology_path",
    resume=False,
    parallel_factory=None,
):
    """Find the target rho axons for an emodel.

    Args:
        morphs_combos_df (dataframe): data for me combos
        emodel_db (DataAccessPoint): object which contains API to access emodel data
        emodel (str): emodel to consider
        resume (bool): to ecrase previous AIS Rin computations

    Returns:
        (dataframe, dict): dataframe with results and dict target rhos for plots
    """
    rho_df = evaluate_rho_axon(
        morphs_combos_df,
        emodel_db,
        morphology_path=morphology_path,
        resume=resume,
        db_url=None,
        parallel_factory=parallel_factory,
    )

    return {emodel: {"all": float(rho_df.rho_axon.median())}}, rho_df
