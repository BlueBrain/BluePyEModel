"""Cli for emodel release with generalisation module."""
import json
import logging
import pickle
import shutil
from copy import copy
from functools import partial
from hashlib import sha256
from pathlib import Path

import click
import morphio
import numpy as np
import pandas as pd
import sh
import yaml
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from morphio.mut import Morphology
from tqdm import tqdm
from voxcell import CellCollection

from bluepyemodel.access_point import get_access_point
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.modifiers import synth_axon
from bluepyemodel.generalisation.ais_model import taper_function

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    """Cli to learn and generate diameters of neurons."""
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=loglevel, format=logformat)


def _get_database(api, emodel_path, final_path=None, legacy_dir_structure=True):
    if api == "local":
        return get_access_point(
            api,
            "cADpyr_L5TPC",  # assumes it exists to be able to load recipe
            emodel_dir=emodel_path,
            final_path=final_path,
            legacy_dir_structure=legacy_dir_structure,
        )
    raise NotImplementedError(f"api {api} is not implemented")


def get_release_paths(release_path):
    """Fix the folder structure of emodel release."""
    release_path = Path(release_path)
    return {
        "hoc_path": release_path / "hoc_files",
        "emodel_path": release_path / "emodels",
        "ais_model_path": release_path / "ais_models.yaml",
        "target_rho_path": release_path / "target_rho_factors.yaml",
        "etype_emodel_map": release_path / "etype_emodel_map.csv",
        "mechanisms": release_path / "mechanisms",
    }


def create_hoc_file(emodel, emodel_db, template_path, ais_models=None):
    """Create hoc files for a given emodel.
    Args:
        emodel (str): name of emodel
        emodel_db (DataAccessPoint): database for emodels
        template_path (str): path to jinja2 template file
        ais_models (dict): contains information about the AIS shape and eletrical properties
    """
    from bluepyemodel.model.model import create_cell_model

    if ais_models is None:
        from bluepyemodel.evaluation.modifiers import replace_axon_hoc as morph_modifier_hoc

    else:
        from bluepyemodel.evaluation.modifiers import get_synth_axon_hoc

        # TODO: this assumes that the AIS models are the same for all mtypes/etypes
        morph_modifier_hoc = get_synth_axon_hoc(ais_models["mtype"]["all"]["AIS"]["popt"])

    emodel_db.emodel = "_".join(emodel.split("_")[:2])
    model_configuration = emodel_db.get_model_configuration()
    cell_model = create_cell_model(
        emodel,
        model_configuration=model_configuration,
        morph_modifiers=[lambda: None],
        morph_modifiers_hoc=[morph_modifier_hoc],
    )
    emodel_db.emodel = emodel
    return cell_model.create_hoc(
        emodel_db.get_emodel()["parameters"],
        template=Path(template_path).name,
        template_dir=Path(template_path).parent,
    )


@cli.command("create_emodel_release")
@click.option("--ais-models", type=click.Path(exists=True), required=False)
@click.option("--target-rho-factors", type=click.Path(exists=True), required=False)
@click.option("--emodel-api", type=str, default="local")
@click.option("--emodel-path", type=click.Path(exists=True), required=True)
@click.option("--final-path", type=click.Path(exists=True), required=True)
@click.option("--template", type=click.Path(exists=True), required=True)
@click.option("--etype-emodel-map-path", type=click.Path(exists=True), required=True)
@click.option("--mechanisms", type=click.Path(exists=True), required=False)
@click.option("--emodel-release", default="emodel_release")
def create_emodel_release(
    ais_models,
    target_rho_factors,
    emodel_api,
    emodel_path,
    final_path,
    template,
    etype_emodel_map_path,
    mechanisms,
    emodel_release,
):
    """From ais_synthesis workflow, create emodel release files for circuit-building.

    Args:
        ais_models (yaml file): contains information about the AIS shape and eletrical properties
        target_rho_factors (yaml file): contains traget rhow factor for each emodel
        emodel_api (str): name of emodel api (only local for now)
        emodel_path (str): paths to the emodel config folders
        final_path (str): paths to the final.json file
        template_path (str): path to jinja2 template file
        etype_emodel_map_path (str): path to file with etype-emodel map
        mechanisms (str): path to mechanisms folder, if None, we will not try to copy them
        emodel_release (str): path to folder to store emodel release
    """
    # 1) create release folders
    L.info("Creating emodel release folder structure.")
    release_paths = get_release_paths(emodel_release)
    if not Path(emodel_release).exists():
        Path(emodel_release).mkdir(parents=True)
        release_paths["hoc_path"].mkdir()

    # 2) load required files
    L.info("Loading required files.")
    etype_emodel_map = pd.read_csv(etype_emodel_map_path)
    emodel_db = _get_database(emodel_api, emodel_path, final_path)
    if ais_models:
        with open(ais_models, "r") as ais_file:
            ais_models = yaml.full_load(ais_file)
        with open(target_rho_factors, "r") as rho_file:
            target_rho_factors_all = yaml.safe_load(rho_file)
            target_rho_factors = {
                emodel: target_rho_factors_all[emodel] for emodel in etype_emodel_map.emodel
            }
    else:
        ais_models = None
        target_rho_factors = None

    # 3) write hoc files for each emodel
    L.info("Writing hoc files for each emodel")
    for emodel in etype_emodel_map.emodel.unique():
        hoc = create_hoc_file(emodel, emodel_db, template, ais_models)
        with open(release_paths["hoc_path"] / f"{emodel}.hoc", "w") as hoc_file:
            hoc_file.write(hoc)

    # 4) write other release files
    L.info("Copy other release files.")
    if ais_models is not None:
        with open(release_paths["ais_model_path"], "w") as ais_file:
            yaml.dump(ais_models, ais_file)
        with open(release_paths["target_rho_path"], "w") as rho_file:
            yaml.dump(target_rho_factors, rho_file)

    shutil.copyfile(etype_emodel_map_path, release_paths["etype_emodel_map"])

    if Path(release_paths["emodel_path"]).exists():
        shutil.rmtree(release_paths["emodel_path"])
    shutil.copytree(emodel_path, release_paths["emodel_path"])
    shutil.copyfile(final_path, Path(release_paths["emodel_path"]) / "final.json")

    if mechanisms is not None:
        shutil.copytree(mechanisms, release_paths["mechanisms"])


def _load_cells(cells_path, mtype=None, n_cells=None, regions=None, emodel=None):
    """Load cells from a circuit into a dataframe."""
    cells = CellCollection.load(cells_path).as_dataframe()
    if regions is not None:
        cells = cells[cells.region.isin(regions)]
    if mtype is not None:
        cells = cells[cells.mtype == mtype]
    if emodel is not None:
        cells["emodel"] = cells["model_template"].str.rsplit(":", 1, expand=True)[1]
        cells = cells[cells.emodel == emodel]
    if n_cells is not None:
        cells = cells.sample(min(len(cells.index), n_cells), random_state=42)

    assert len(cells.index) > 0, "no cells selected!"

    return cells


def _create_emodel_column(cells, etype_emodel_map):
    """Assign emodel to each row, given there mtype and etype."""
    # TODO: the assertion are not complete, and this is slow for large number of rows
    assert set(cells.mtype.unique()).issubset(
        set(etype_emodel_map.mtype.unique())
    ), "There are missing mtypes in etype_emodel_map"
    assert set(cells.etype.unique()).issubset(
        set(etype_emodel_map.etype.unique())
    ), "There are missing etypes in etype_emodel_map"
    cells = pd.merge(cells, etype_emodel_map, on=["mtype", "etype"], how="left")

    unassigned_cells = cells[cells.emodel.isnull()]
    if len(unassigned_cells) > 0:
        raise Exception(f"{len(unassigned_cells)} have unassigned emodels!")

    return cells


def _add_combo_name(combos_df):
    if "combo_name" not in combos_df.columns:
        combos_df["combo_name"] = combos_df.apply(
            lambda x: "%s_%s_%s_%s" % (x["emodel"], x["fullmtype"], x["layer"], x["morph_name"]),
            axis=1,
        )
    return combos_df


def save_mecombos(combos_df, output, with_scaler=True):
    """Save dataframe to mecombo_emodel.tsv file"""
    if "AIS_scaler" not in combos_df.columns:
        combos_df = combos_df.rename(columns={"AIS_scaler": "AIS_scaler"})
    if "fullmtype" not in combos_df.columns:
        combos_df = combos_df.rename(columns={"mtype": "fullmtype"})
    if "morph_name" not in combos_df.columns:
        combos_df = combos_df.rename(columns={"morphology": "morph_name"})
    combos_df["layer"] = combos_df["layer"].astype(str)
    combos_df = _add_combo_name(combos_df)
    columns = [
        "morph_name",
        "layer",
        "fullmtype",
        "etype",
        "emodel",
        "combo_name",
    ]
    if "threshold_current" in combos_df.columns:
        columns.append("threshold_current")
        combos_df["threshold_current"] = combos_df["threshold_current"].astype(np.float64)

    if "holding_current" in combos_df.columns:
        columns.append("holding_current")
        combos_df["holding_current"] = combos_df["holding_current"].astype(np.float64)

    if with_scaler:
        columns.append("AIS_scaler")

    combos_df.to_csv(
        output,
        columns=columns,
        index=False,
        sep="\t",
    )


@cli.command("get_me_combos_scales")
@click.option("--cells-path", type=click.Path(exists=True), required=True)
@click.option("--release-path", type=click.Path(exists=True), required=True)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--output-sonata-path", default="circuit.AIS_scaler.h5", type=str)
@click.option("--combos-df-path", default=None, type=str)
@click.option("--emodel-api", default="local", type=str)
@click.option("--sql-tmp-path", default="tmp", type=str)
@click.option("--n-cells", default=None, type=int)
@click.option("--mtype", default=None, type=str)
@click.option("--parallel-lib", default="multiprocessing")
@click.option("--resume", is_flag=True)
@click.option("--with-db", is_flag=True)
@click.option("--drop-failed", is_flag=True)
def get_me_combos_scales(
    cells_path,
    release_path,
    morphology_path,
    output_sonata_path,
    combos_df_path,
    emodel_api,
    sql_tmp_path,
    n_cells,
    mtype,
    parallel_lib,
    resume,
    with_db,
    drop_failed,
):
    """For each me-combos, compute the AIS scale and thresholds currents.

    Args:
        cells_path (str): path to mvd3/sonata file with synthesized morphologies
        release_path (str): path to emodel release folder
        morphology_path (str): base path to morphologies, if none, same dir as mvd3 will be used
        output_sonata_path (str): path to output sonata file
        combos_df_path (str): path to .csv file to save all result dataframe
        emodel_api (str): name of emodel api, so far only 'local' is available
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
        n_cells (int): for testing, only use first n_cells in cells dataframe
        mtype (str): name of mtype to use (for testing only)
        resume (bool): resume computation is with_db
        with_db (bool): use sql backend for resume option
        drop_failed (bool): drop cells with failed computations
    """
    from bluepyemodel.generalisation.ais_synthesis import synthesize_ais

    if resume and not with_db:
        raise Exception("If --with-db is not used, --resume cannot work")

    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")
    cells = _load_cells(cells_path, mtype, n_cells)

    release_paths = get_release_paths(release_path)
    with open(release_paths["ais_model_path"], "r") as ais_file:
        ais_models = yaml.full_load(ais_file)
    with open(release_paths["target_rho_path"], "r") as rho_file:
        target_rhos = yaml.full_load(rho_file)
    etype_emodel_map = pd.read_csv(release_paths["etype_emodel_map"])

    if Path(release_paths["mechanisms"]).exists():
        sh.nrnivmodl(release_paths["mechanisms"])  # pylint: disable=no-member

    emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

    # 2) assign emodels to each row and only keep relevant columns
    L.info("Assign emodels to each rows and restructure dataframe.")
    cells = _create_emodel_column(cells, etype_emodel_map)

    if morphology_path is None:
        morphology_path = Path(cells_path).parent
    cells["morphology_path"] = str(morphology_path) + "/" + cells["morphology"] + ".asc"
    cells["gid"] = cells.index
    morphs_combos_df = cells[
        ["gid", "morphology_path", "emodel", "mtype", "etype", "layer", "morphology"]
    ]

    # 3) Compute AIS scales
    L.info("Compute AIS scales.")
    Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
    combos_df = synthesize_ais(
        morphs_combos_df,
        emodel_db,
        ais_models["mtype"],
        target_rhos,
        parallel_factory=parallel_factory,
        scales_params=ais_models["scales_params"],
        db_url=Path(sql_tmp_path) / "synth_db.sql" if with_db else None,
        resume=resume,
    )

    # 4) save all values in .csv
    if combos_df_path is not None:
        L.info("Save results in .csv")
        Path(combos_df_path).parent.mkdir(parents=True, exist_ok=True)
        combos_df.to_csv(combos_df_path, index=False)

    Path(output_sonata_path).parent.mkdir(parents=True, exist_ok=True)
    combos_df = combos_df[["morphology", "emodel", "AIS_scaler"]]
    failed_cells = combos_df[combos_df.isnull().any(axis=1)].index
    if len(failed_cells) > 0:
        L.info("%s failed cells:", len(failed_cells))
        L.info(combos_df.loc[failed_cells])
    else:
        L.info("No failed cells, hurray!")
    if drop_failed:
        # we remove combo_name here, so the remove_unassigned_cells() later will remove these cells
        combos_df.loc[failed_cells, "mtype"] = None
    else:
        combos_df.loc[failed_cells, "AIS_scaler"] = 1.0

    L.info("Save results in sonata")
    _cells = CellCollection.load(cells_path)
    combos_df = combos_df.set_index("morphology")
    valid_morphs = [morph for morph in _cells.properties["morphology"] if morph in combos_df.index]

    _cells.properties.loc[_cells.properties["morphology"].isin(valid_morphs), "model_template"] = (
        "hoc:" + combos_df.loc[valid_morphs, "emodel"]
    ).to_list()
    _cells.properties.loc[
        _cells.properties["morphology"].isin(valid_morphs), "@dynamics:AIS_scaler"
    ] = combos_df.loc[valid_morphs, "AIS_scaler"].to_list()

    _cells.remove_unassigned_cells()
    _cells.save(output_sonata_path)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


@cli.command("get_me_combos_currents")
@click.option("--input-sonata-path", type=click.Path(exists=True), required=True)
@click.option("--output-sonata-path", default="circuit.currents.h5", type=str)
@click.option("--release-path", type=click.Path(exists=True))
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--protocol-config-path", default=None, type=str)
@click.option("--combos-df-path", default=None, type=str)
@click.option("--emodel-api", default="local", type=str)
@click.option("--sql-tmp-path", default="tmp", type=str)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--bglibpy-template-format", default="v6_ais_scaler", type=str)
@click.option("--emodels-hoc-dir", type=str)
@click.option("--resume", is_flag=True)
@click.option("--with-db", is_flag=True)
@click.option("--drop_failed", is_flag=True)
def get_me_combos_currents(
    input_sonata_path,
    output_sonata_path,
    release_path,
    morphology_path,
    protocol_config_path,
    combos_df_path,
    emodel_api,
    sql_tmp_path,
    parallel_lib,
    bglibpy_template_format,
    emodels_hoc_dir,
    resume,
    with_db,
    drop_failed,
):
    """For each me-combos, compute the thresholds currents.

    Args:
        output_sonata_path (str): path to sonata file to save output data for all me-combos
        combos_df_path (str): path to dataframe with cells informations to run
        release_path (str): path to emodel release folder
        morphology_path (str): base path to morphologies, if none, same dir as mvd3 will be used
        output (str): .csv file to save output data for each me-combos
        emodel_api (str): name of emodel api, so far only 'local' is available
        protocol_config_path (str): path to yaml file with protocol config for bglibpy
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
        bglibpy_template_format (str): v6 for standar use v6_ais_scaler for custom hoc
        emodels_hoc_dir (str): paths to hoc files, if None hoc from release path will be used
        resume (bool): resume computation is with_db
        with_db (bool): use sql backend for resume option
        drop_failed (bool): drop cells with failed computations
    """
    if resume and not with_db:
        raise Exception("If --with-db is not used, --resume cannot work")
    if protocol_config_path is None:
        from bluepyemodel.generalisation.evaluators import evaluate_currents
    else:
        from bluepyemodel.generalisation.bglibpy_evaluators import evaluate_currents_bglibpy

    # Initialize the parallel library. If using dask, it must be called at the beginning,
    # but after nueuron imports.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load sonata file
    L.info("Load sonata file")
    cells = CellCollection.load(input_sonata_path)
    combos_df = cells.as_dataframe()
    combos_df["emodel"] = combos_df["model_template"].apply(lambda temp: temp[4:])
    if morphology_path is None:
        morphology_path = Path(input_sonata_path).parent
    combos_df["morphology_path"] = [f"{morphology_path}/{m}.asc" for m in combos_df["morphology"]]
    if "@dynamics:AIS_scaler" in combos_df.columns:
        combos_df["AIS_scaler"] = combos_df["@dynamics:AIS_scaler"]
    else:
        bglibpy_template_format = "v6"
        L.info("No @dynamics:AIS_scaler column, switching to v6")

    # 2) compute holding and threshold currents
    if protocol_config_path is None:
        L.info("Compute holding and threshold currents with opt protocols.")
        release_paths = get_release_paths(release_path)
        emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

        Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
        combos_df = evaluate_currents(
            combos_df,
            emodel_db,
            parallel_factory=parallel_factory,
            db_url=Path(sql_tmp_path) / "current_db.sql" if with_db else None,
        )
    else:
        L.info("Compute holding and threshold currents with bglibpy")
        with open(protocol_config_path, "r") as prot_file:
            protocol_config = yaml.safe_load(prot_file)

        if emodels_hoc_dir is None:
            release_paths = get_release_paths(release_path)
            emodels_hoc_dir = release_paths["hoc_path"]

        Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
        combos_df = evaluate_currents_bglibpy(
            combos_df,
            protocol_config,
            emodels_hoc_dir,
            parallel_factory=parallel_factory,
            db_url=Path(sql_tmp_path) / "current_db.sql" if with_db else None,
            template_format=bglibpy_template_format,
            resume=resume,
        )

    # 3) save in sonata and .csv
    if combos_df_path is not None:
        L.info("Save results in .csv")
        combos_df.to_csv(combos_df_path, index=False)

    Path(output_sonata_path).parent.mkdir(parents=True, exist_ok=True)
    combos_df = combos_df[["morphology", "emodel", "holding_current", "threshold_current"]]
    failed_cells = combos_df[combos_df.isnull().any(axis=1)].index
    if len(failed_cells) > 0:
        L.info("%s failed cells:", len(failed_cells))
        L.info(combos_df.loc[failed_cells])
    else:
        L.info("No failed cells, hurray!")

    if drop_failed:
        # we remove combo_name here, so the remove_unassigned_cells() later will remove these cells
        combos_df.loc[failed_cells, "mtype"] = None
    else:
        combos_df.loc[failed_cells, "holding_current"] = 0
        combos_df.loc[failed_cells, "threshold_current"] = 0

    L.info("Save results in sonata")
    combos_df = combos_df.set_index("morphology")
    valid_morphs = [morph for morph in cells.properties["morphology"] if morph in combos_df.index]

    cells.properties.loc[
        cells.properties["morphology"].isin(valid_morphs), "@dynamics:holding_current"
    ] = combos_df.loc[valid_morphs, "holding_current"].to_list()

    cells.properties.loc[
        cells.properties["morphology"].isin(valid_morphs), "@dynamics:threshold_current"
    ] = combos_df.loc[valid_morphs, "threshold_current"].to_list()

    cells.remove_unassigned_cells()
    cells.save(output_sonata_path)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


def _create_etype_column(cells, cell_composition):
    """Create etype column from cell_composition.yaml data."""
    dfs = []
    for data in cell_composition["neurons"]:
        for etype in data["traits"]["etype"]:
            _df = cells[cells.mtype == data["traits"]["mtype"]].copy()
            _df.loc[:, "etype"] = etype
            dfs.append(_df)
    return pd.concat(dfs).reset_index(drop=True)


def _single_evaluation(
    combo,
    emodel_api,
    final_path,
    emodel_path,
    morphology_path="morphology_path",
    stochasticity=False,
    timeout=1000,
    score_threshold=100.0,
    trace_data_path=None,
):
    """Evaluating single protocol and save traces."""
    emodel_db = get_access_point(
        emodel_api,
        combo["emodel"],
        emodel_dir=emodel_path,
        final_path=final_path,
        with_seeds=True,
        legacy_dir_structure=True,
    )
    if "path" in combo:
        emodel_db.morph_path = combo["path"]

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
    )
    responses = evaluator.run_protocols(
        evaluator.fitness_protocols.values(), emodel_db.get_emodel()["parameters"]
    )
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
        pickle.dump([stimuli, responses], open(trace_data_path, "wb"))

    return {
        "features": json.dumps(features),
        "scores": json.dumps(scores),
        "trace_data": trace_data_path,
    }


def _evaluate_exemplars(
    emodel_path,
    final_path,
    emodel_api,
    parallel_factory,
    emodel,
    trace_data_path,
    score_threshold=12.0,
):
    """Evaluate exemplars."""
    emodel_path = Path(emodel_path)
    emodel_db = get_access_point(
        emodel_api,
        "cADpyr_L5TPC",  # assumes it exists to be able to load recipe
        emodel_dir=emodel_path,
        final_path=final_path,
        with_seeds=True,
        legacy_dir_structure=True,
    )
    exemplar_df = pd.DataFrame()
    for i, _emodel in enumerate(emodel_db.get_final()):
        exemplar_df.loc[i, "emodel"] = _emodel

    if emodel is not None:
        exemplar_df = exemplar_df[exemplar_df.emodel == emodel]

    evaluation_function = partial(
        _single_evaluation,
        emodel_api=emodel_api,
        emodel_path=emodel_path,
        final_path=final_path,
        trace_data_path=trace_data_path,
        stochasticity=False,
        score_threshold=score_threshold,
    )

    return evaluate(
        exemplar_df,
        evaluation_function,
        new_columns=[["features", ""], ["scores", ""], ["trace_data", ""]],
        parallel_factory=parallel_factory,
    )


def _evaluate_emodels(
    sonata_path,
    morphology_path,
    emodel_path,
    final_path,
    emodel_api,
    ais_emodels_path,
    region,
    emodel,
    n_cells,
    seed,
    parallel_factory,
    trace_data_path,
    score_threshold=100.0,
):
    """Evaluate emodels."""

    combos_df = CellCollection.load(sonata_path).as_dataframe()
    combos_df["path"] = morphology_path + "/" + combos_df["morphology"] + ".asc"
    combos_df["emodel"] = combos_df["model_template"].str.rsplit(":", 1, expand=True)[1]
    if region is not None:
        combos_df = combos_df[combos_df.region == region]

    if emodel is not None:
        combos_df = combos_df[combos_df.emodel == emodel]

    combos_df = pd.concat(
        [
            combos_df[combos_df.model_template == model_template].sample(
                min(n_cells, len(combos_df[combos_df.model_template == model_template].index)),
                random_state=seed,
            )
            for model_template in combos_df.model_template.unique()
        ]
    )
    if ais_emodels_path is not None:
        ais_models = yaml.safe_load(open(ais_emodels_path, "r"))
        combos_df["AIS_params"] = len(combos_df) * [ais_models["mtype"]["all"]["AIS"]["popt"]]
        combos_df["AIS_scaler"] = combos_df["@dynamics:AIS_scaler"]
        combos_df = combos_df[["AIS_params", "AIS_scaler", "path", "mtype", "morphology", "emodel"]]
    else:
        combos_df = combos_df[["path", "mtype", "morphology", "emodel"]]
    combos_df["gid"] = combos_df.index
    combos_df = combos_df.reset_index(drop=True)

    evaluation_function = partial(
        _single_evaluation,
        emodel_api=emodel_api,
        emodel_path=Path(emodel_path),
        final_path=final_path,
        trace_data_path=trace_data_path,
        stochasticity=False,
        score_threshold=score_threshold,
    )

    return evaluate(
        combos_df,
        evaluation_function,
        new_columns=[["features", ""], ["scores", ""], ["trace_data", ""]],
        parallel_factory=parallel_factory,
    )


def _plot_reports(exemplar_df, result_df, folder, clip, feature_threshold, megate_thresholds_path):
    """Internal function to plot generalisation reports."""

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages

    from bluepyemodel.generalisation.select import apply_megating
    from bluepyemodel.generalisation.utils import get_scores

    matplotlib.use("Agg")

    Path(folder).mkdir(exist_ok=True, parents=True)
    result_df = result_df[["emodel", "mtype", "morphology", "scores"]]
    result_df["scores_raw"] = result_df["scores"]
    exemplar_df["scores_raw"] = exemplar_df["scores"]
    exemplar_df["for_optimisation"] = 1.0

    megate_thresholds = yaml.safe_load(open(megate_thresholds_path, "r"))
    megated_df = apply_megating(result_df, exemplar_df, megate_thresholds=megate_thresholds)
    result_df["pass"] = megated_df.all(axis=1).astype(int)

    pass_df = result_df.groupby("emodel").mean().sort_values(by="pass", ascending=False)

    plt.figure(figsize=(5, 10))
    pass_df["pass"].plot.barh(ax=plt.gca())
    plt.xlabel("fraction of pass combos")
    plt.savefig(f"{folder}/pass_fraction.pdf", bbox_inches="tight")

    scores = get_scores(result_df, clip=clip)

    _scores = scores[["emodel", "median_score", "max_score"]]
    plt.figure(figsize=(5, 10))
    ax = plt.gca()
    sns.violinplot(
        data=_scores,
        x="median_score",
        y="emodel",
        orient="h",
        linewidth=0.1,
        bw=0.1,
        scale="count",
        order=sorted(scores.emodel.unique()),
    )
    plt.savefig(f"{folder}/median_scores.pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 10))
    ax = plt.gca()
    sns.violinplot(
        data=_scores,
        x="max_score",
        y="emodel",
        orient="h",
        linewidth=0.1,
        bw=0.1,
        scale="count",
        order=sorted(scores.emodel.unique()),
    )
    plt.savefig(f"{folder}/max_scores.pdf", bbox_inches="tight")

    megated_df.index = result_df["emodel"]
    with PdfPages(f"{folder}/failed_features.pdf") as pdf:
        for emodel, scores in megated_df.groupby("emodel"):
            df = (scores).astype(int).mean(axis=0)
            df = df[df < feature_threshold].sort_values(ascending=False)

            if len(df) > 0:
                plt.figure(figsize=(5, 2 + len(df.index)))
                ax = plt.gca()
                df.plot.barh(ax=ax)
                plt.gca().set_xlim(0, 1)
                plt.suptitle(emodel)
                pdf.savefig(bbox_inches="tight")
                plt.close()


def _parse_regions(regions):
    """Parse region str as a list or single region."""
    if regions is None:
        return [None]
    if regions[0] == "[":
        return json.loads(regions)
    else:
        return [regions]


def _get_stuck(
    row,
    emodel_hoc_dir,
    morphology_dir,
    amplitude,
    step_length=2000,
    threshold=0.8,
    step_start=500,
    after_step=500,
    protocol_config_path="protocol_configs.yaml",
):
    """This computes stuck and depol stuck cells."""
    from bluepyemodel.tools.bglibpy_helper import get_cell
    from bluepyemodel.tools.bglibpy_helper import get_time_to_last_spike
    from bluepyemodel.tools.if_curve import run_step_sim

    cell = get_cell(
        morphology_name=row["morphology"],
        emodel=row["emodel"],
        emodels_hoc_dir=emodel_hoc_dir,
        morphology_dir=morphology_dir,
        calc_threshold=True,
        scale=row["@dynamics:AIS_scaler"],
        protocol_config_path=protocol_config_path,
    )
    result = run_step_sim(
        cell,
        amplitude * cell.threshold,
        step_start=step_start,
        step_stop=step_start + step_length,
        sim_end=step_start + step_length + 500,
        cvode=True,
    )

    t = result["time"]
    v = result["voltage_soma"]

    stuck = np.mean(v[t > step_start + step_length + 10]) > np.min(
        v[(t > step_start + 10) & (t < step_start + step_length - 10)]
    )

    time_to_last_spike = get_time_to_last_spike(result, step_start, step_start + step_length)
    depol_stuck = time_to_last_spike < threshold * step_length

    return {"stuck": float(stuck), "depol_stuck": float(depol_stuck)}


@cli.command("detect_stuck_cells")
@click.option("--sonata-path", type=click.Path(exists=True), required=True)
@click.option("--regions", type=str, default=None)
@click.option("--emodel", type=str, default=None)
@click.option("--mtype", type=str, default=None)
@click.option("--n-cells", type=int, default=10)
@click.option("--amplitude", type=float, default=3.0)
@click.option("--emodel-hoc-path", type=str, default=None)
@click.option("--morphology-path", type=str, default=None)
@click.option("--parallel-factory", type=str, default="multiprocessing")
@click.option("--result-path", type=str, default="stuck_df")
@click.option("--figure-path", type=str, default="stuck_figures")
@click.option("--seed", type=int, default=42)
@click.option("--step-length", type=float, default=2000)
@click.option("--threshold", type=float, default=0.8)
@click.option("--protocol-config-path", default=None, type=str)
def detect_stuck_cells(
    sonata_path,
    regions,
    emodel,
    mtype,
    n_cells,
    amplitude,
    emodel_hoc_path,
    morphology_path,
    parallel_factory,
    result_path,
    figure_path,
    seed,
    step_length,
    threshold,
    protocol_config_path,
):
    """Detect possible stuck cells.

    Depol stuck are obained by comparing the number of spikes from the first and last half of the
    step. They correspond to cells which don't fire anymore after some times.
    Stuck are obtained by comparing the average voltage value after the step and the min
    voltage in the step. They correspond to cells which are stuck to high voltages, even after the
    end of the step.
    """
    parallel_factory = init_parallel_factory(parallel_factory)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    cells = CellCollection.load(sonata_path).as_dataframe()
    cells["path"] = morphology_path + cells["morphology"] + ".asc"
    cells["emodel"] = cells["model_template"].apply(lambda m: m.split(":")[-1])
    cells["morphology"] = cells["morphology"] + ".asc"
    if emodel is not None:
        cells = cells[cells.emodel == emodel]

    for region in _parse_regions(regions):
        L.info("Evaluating region %s", region)
        if region is not None:
            _cells = cells[cells.region == region]
        else:
            _cells = cells

        _cells = pd.concat(
            [
                _cells[cells.model_template == model_template].sample(
                    min(n_cells, len(_cells[cells.model_template == model_template].index)),
                    random_state=seed,
                )
                for model_template in _cells.model_template.unique()
            ]
        )

        evaluation_function = partial(
            _get_stuck,
            emodel_hoc_dir=emodel_hoc_path,
            morphology_dir=morphology_path,
            amplitude=amplitude,
            step_length=step_length,
            threshold=threshold,
            protocol_config_path=protocol_config_path,
        )
        result = evaluate(
            _cells,
            evaluation_function,
            new_columns=[["stuck", ""], ["depol_stuck", ""]],
            parallel_factory=parallel_factory,
        )

        result[["morphology", "mtype", "etype", "emodel", "stuck", "depol_stuck"]].to_csv(
            f"{result_path}/stuck_df_{region}.csv"
        )

        import matplotlib.pyplot as plt

        summary = result[["emodel", "stuck"]].groupby("emodel").mean()
        plt.figure()
        summary.plot.barh(ax=plt.gca())
        plt.savefig(f"{figure_path}/stuck_cells_{region}.pdf", bbox_inches="tight")

        summary = result[["emodel", "depol_stuck"]].groupby("emodel").mean()
        plt.figure()
        summary.plot.barh(ax=plt.gca())
        plt.savefig(f"{figure_path}/depol_stuck_cells_{region}.pdf", bbox_inches="tight")


@cli.command("evaluate_emodels")
@click.option("--emodel-api", type=str, default="local")
@click.option("--emodel-path", type=click.Path(exists=True), required=True)
@click.option("--final-path", type=click.Path(exists=True), required=True)
@click.option("--exemplar-path", type=str, default="exemplar_evaluations.csv")
@click.option("--sonata-path", type=click.Path(exists=True), required=True)
@click.option("--morphology-path", type=click.Path(exists=True), required=True)
@click.option("--ais-emodels-path", type=click.Path(exists=True), default=None)
@click.option("--regions", type=str, default=None)
@click.option("--emodel", type=str, default=None)
@click.option("--n-cells", type=int, default=10)
@click.option("--seed", type=int, default=42)
@click.option("--result-path", type=str, default="results")
@click.option("--parallel-factory", type=str, default="multiprocessing")
@click.option("--clip", type=float, default=5.0)
@click.option("--feature-threshold", type=float, default=0.99)
@click.option(
    "--megate-thresholds-path", type=click.Path(exists=True), default="megate_thresholds.yaml"
)
@click.option("--report-folder", type=str, default="figures")
@click.option("--plot-only", is_flag=True)
@click.option("--trace-data-path", type=str, default=None)
@click.option("--score-threshold", type=float, default=100.0)
def evaluate_emodels(
    emodel_api,
    emodel_path,
    final_path,
    exemplar_path,
    sonata_path,
    morphology_path,
    ais_emodels_path,
    regions,
    emodel,
    n_cells,
    seed,
    result_path,
    parallel_factory,
    clip,
    feature_threshold,
    megate_thresholds_path,
    report_folder,
    plot_only,
    trace_data_path,
    score_threshold=100.0,
):
    """Evaluate exemplars and emodels."""
    parallel_factory = init_parallel_factory(parallel_factory)
    regions = _parse_regions(regions)

    if not plot_only:
        L.info("Evaluating exemplars")
        exemplar_df = _evaluate_exemplars(
            emodel_path,
            final_path,
            emodel_api,
            parallel_factory,
            emodel,
            trace_data_path,
            score_threshold=score_threshold,
        )
        exemplar_df.to_csv(exemplar_path, index=False)
    else:
        exemplar_df = pd.read_csv(exemplar_path)

    for region in regions:
        L.info("Evaluating region %s", region)

        region_folder = Path(f"region_{region}")
        _result_path = Path(result_path) / region_folder
        _result_path.mkdir(exist_ok=True, parents=True)
        _report_folder = Path(report_folder) / region_folder
        _report_folder.mkdir(exist_ok=True, parents=True)

        if not plot_only:
            result_df = _evaluate_emodels(
                sonata_path,
                morphology_path,
                emodel_path,
                final_path,
                emodel_api,
                ais_emodels_path,
                region,
                emodel,
                n_cells,
                seed,
                parallel_factory,
                trace_data_path,
                score_threshold=score_threshold,
            )
            result_df.to_csv(_result_path / "results.csv", index=False)
        else:
            result_df = pd.read_csv(_result_path / "results.csv")

        _plot_reports(
            exemplar_df, result_df, _report_folder, clip, feature_threshold, megate_thresholds_path
        )


# below not maintained, was used in sscx to fix some morphologies


def fix_ais(combo, out_path="fixed_cells_test", morph_name="combo_name"):
    """Modify morphology first axon section in place from combo data."""
    morphology = Morphology(combo.morphology_path)

    if combo.AIS_model is not None:
        ais_model = json.loads(combo.AIS_model)["popt"]
        _taper_func = partial(
            taper_function,
            strength=ais_model[1],
            taper_scale=ais_model[2],
            terminal_diameter=ais_model[3] * combo.AIS_scale,
        )
        modify_ais(morphology, _taper_func)
    else:
        raise Exception(combo)

    morphology = Morphology(morphology, morphio.Option.nrn_order)  # ensures NEURON order
    morphology.write(str(Path(out_path) / combo[morph_name]) + ".asc")


def modify_ais(morphology, taper_func, L_target=65):
    """Modify morphology first axon section in place using taper_func."""
    for root_section in morphology.root_sections:
        if root_section.type == morphio.SectionType.axon:
            dist = 0
            prev_point = root_section.points[0]
            for section in root_section.iter():
                diams = copy(section.diameters)  # this is needed because of morphio
                for i, point in enumerate(section.points):
                    dist += np.linalg.norm(point - prev_point)
                    prev_point = copy(point)
                    diams[i] = taper_func(dist)
                    if dist >= L_target:
                        break
                section.diameters = diams
                if dist >= L_target:
                    break


@cli.command("set_me_combos_scales_inplace")
@click.option("--mecombos-path", type=click.Path(exists=True), required=False)
@click.option("--to-fix-combos-path", type=click.Path(exists=True), required=False)
@click.option(
    "--release-path",
    type=click.Path(exists=True),
    default="emodel_release",
    required=True,
)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--fixed-morphology-path", default="fixed_morphologies", type=str)
@click.option("--output-sonata-path", default="mecombo_emodel.tsv", type=str)
@click.option("--combos-df-path", default="mecombo_emodel.tsv", type=str)
@click.option("--emodel-api", default="local", type=str)
@click.option("--cell-composition-path", type=click.Path(exists=True), required=False)
@click.option("--sql-tmp-path", default="tmp", type=str)
@click.option("--parallel-lib", default="multiprocessing")
@click.option("--resume", is_flag=True)
def set_me_combos_scales_inplace(  # pylint: disable-all
    mecombos_path,
    to_fix_combos_path,
    release_path,
    morphology_path,
    fixed_morphology_path,
    mecombo_emodel_tsv_path,
    combos_df_path,
    emodel_api,
    cell_composition_path,
    sql_tmp_path,
    parallel_lib,
    resume,
):
    """Similar function to get_me_combos_scales, but modifies the morphologies directly.

    Args:
        mecombos_path (str): csv with dataframe with cells informations
        to_fix_combos_path (str): file with list of combos to modify, if None, all will be used
        release_path (str): path to emodel release folder
        morphology_path (str): base path to morphologies, if none, same dir as mvd3 will be used
        mecombo_emodel_tsv_path (str): path to .tsv file to save output data for each me-combos
        fixed_morphology_path (str): path to folder to save modified morphologies
        combos_df_path (str): path to .csv file to save all result dataframe
        emodel_api (str): name of emodel api, so far only 'local' is available
        cell_composition_path (str): path to cell_composition.yaml
        protocol_config_path (str): path to yaml file with protocol config for bglibpy
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
    """
    from bluepyemodel.generalisation.ais_synthesis import synthesize_ais

    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")

    release_paths = get_release_paths(release_path)
    with open(release_paths["ais_model_path"], "r") as ais_file:
        ais_models = yaml.full_load(ais_file)
    with open(release_paths["target_rho_path"], "r") as rho_file:
        target_rhos = yaml.full_load(rho_file)

    if Path(release_paths["mechanisms"]).exists():
        raise Exception(release_paths["mechanisms"])
        sh.nrnivmodl(release_paths["mechanisms"])  # pylint: disable=no-member
    emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

    if mecombos_path:
        L.info("Loading cells from mecombo_emodel.tsv file")
        all_morphs_combos_df = pd.read_csv(mecombos_path, sep="\t")
        morph_name = "name"  # we will keep original morph names
    else:
        L.info("Loading cells from morphology folder")
        from synthesis_workflow.tools import load_neurondb_to_dataframe

        morph_name = "combo_name"  # we will write a morph per combo
        morphology_path = Path(morphology_path)
        morph_dirs = {"morphology_path": str(morphology_path.absolute())}
        neurondb_path = morphology_path / "neuronDB.xml"
        assert neurondb_path.exists(), "No neuronDB.xml file found in morphology_path"
        cells = load_neurondb_to_dataframe(neurondb_path, morphology_dirs=morph_dirs)
        cells["morphology_path"] = cells["morphology_path"].apply(lambda path: str(path))
        etype_emodel_map = pd.read_csv(release_paths["etype_emodel_map"])
        with open(cell_composition_path, "r") as cell_comp_file:
            cell_composition = yaml.safe_load(cell_comp_file)
        cells = _create_etype_column(cells, cell_composition)
        cells = _create_emodel_column(cells, etype_emodel_map)
        cells["gid"] = cells.index
        if "morphology" not in cells.columns:
            cells["morphology"] = cells["name"]
        if "name" not in cells.columns:
            cells["name"] = cells["morphology"]

        all_morphs_combos_df = cells[
            ["gid", "morphology_path", "emodel", "mtype", "etype", "layer", "morphology"]
        ]

        if "fullmtype" not in all_morphs_combos_df.columns:
            all_morphs_combos_df["fullmtype"] = all_morphs_combos_df["mtype"]
        if "name" not in all_morphs_combos_df.columns:
            all_morphs_combos_df["name"] = all_morphs_combos_df["morphology"]
        if "morph_name" not in all_morphs_combos_df.columns:
            all_morphs_combos_df["morph_name"] = all_morphs_combos_df["name"]
        all_morphs_combos_df = _add_combo_name(all_morphs_combos_df)

    if to_fix_combos_path is not None:
        to_fix_combos_df = pd.read_csv(to_fix_combos_path, header=None, names=["combo_name"])
        morphs_combos_df = all_morphs_combos_df[
            all_morphs_combos_df.combo_name.isin(to_fix_combos_df.combo_name)
        ].copy()
        assert len(morphs_combos_df.index) == len(to_fix_combos_df.index)
    else:
        morphs_combos_df = all_morphs_combos_df.copy()

    if "mtype" not in morphs_combos_df.columns:
        morphs_combos_df["mtype"] = morphs_combos_df["fullmtype"]
    if "morphology_path" not in morphs_combos_df.columns:
        morphs_combos_df["morphology_path"] = morphs_combos_df["name"].apply(
            lambda name: str(Path(morphology_path) / name) + ".asc"
        )

    # 2) Compute AIS scales
    L.info("Compute AIS scales.")
    Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
    combos_df = synthesize_ais(
        morphs_combos_df,
        emodel_db,
        ais_models["mtype"],
        target_rhos,
        parallel_factory=parallel_factory,
        scales_params=ais_models["scales_params"],
        db_url=Path(sql_tmp_path) / "synth_db.sql",
        resume=resume,
    )

    # 3) Modify morphologies
    L.info("Modify morphologies.")
    if Path(fixed_morphology_path).exists():
        shutil.rmtree(fixed_morphology_path)
    Path(fixed_morphology_path).mkdir(parents=True)

    for gid in tqdm(combos_df.index):
        fix_ais(combos_df.loc[gid], fixed_morphology_path, morph_name=morph_name)

    # 4) save new mecombos_emodel.tsv file with updated entries
    all_morphs_combos_df.update(combos_df)
    combos_df.to_csv(combos_df_path, index=False)
    Path(mecombo_emodel_tsv_path).parent.mkdir(parents=True, exist_ok=True)
    if morph_name == "combo_name":
        all_morphs_combos_df["morph_name"] = all_morphs_combos_df["combo_name"]
    save_mecombos(all_morphs_combos_df, mecombo_emodel_tsv_path, with_scaler=False)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()
