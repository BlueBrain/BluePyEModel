"""Cli for emodel release with ais_synthesis module."""
import json
import logging
import shutil
from copy import copy
from functools import partial
from pathlib import Path

import click
import morphio
import numpy as np
import pandas as pd
import sh
import yaml
from morphio.mut import Morphology
from tqdm import tqdm

from bluepy.v2 import Circuit
from bluepyemodel.ais_synthesis.ais_synthesis import synthesize_ais
from bluepyemodel.ais_synthesis.evaluators import evaluate_currents
from bluepyemodel.ais_synthesis.tools import InitParallelFactory
from bluepyemodel.api.singlecell import Singlecell_API
from bluepyemodel.evaluation.model import create_cell_model
from bluepyemodel.evaluation.modifiers import get_synth_axon_hoc

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    """Cli to learn and generate diameters of neurons."""
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=loglevel, format=logformat)


def _get_database(api, emodel_path):
    if api == "singlecell":
        return Singlecell_API(emodel_path)
    raise NotImplementedError(f"api {api} is not implemented")


def get_release_paths(release_path):
    """Fix the folder sturcture of emodel release."""
    release_path = Path(release_path)
    return {
        "hoc_path": release_path / "hoc_files",
        "emodel_path": release_path / "emodels",
        "ais_model_path": release_path / "ais_models.yaml",
        "target_rho_path": release_path / "target_rho_factors.yaml",
        "etype_emodel_map": release_path / "etype_emodel_map.csv",
        "mechanisms": release_path / "mechanisms",
    }


def create_hoc_file(emodel, emodel_db, template_path, ais_models):
    """Create hoc files for a given emodel.
    Args:
        emodel (str): name of emodel
        emodel_db (DatabaseAPI): database for emodels
        template_path (str): path to jinja2 template file
        ais_models (dict): contains information about the AIS shape and eletrical properties
    """
    # TODO: this assumes that the AIS models are the same for all mtypes/etypes
    morph_modifier_hoc = get_synth_axon_hoc(ais_models["mtype"]["all"]["AIS"]["popt"])

    parameters, mechanisms, _ = emodel_db.get_parameters(emodel)
    cell_model = create_cell_model(
        "cell_model",
        emodel_db.get_morphologies(emodel),
        mechanisms,
        parameters,
        morph_modifiers=[lambda: None],
        morph_modifiers_hoc=[morph_modifier_hoc],
    )
    return cell_model.create_hoc(
        emodel_db.get_emodel(emodel)["parameters"],
        template=Path(template_path).name,
        template_dir=Path(template_path).parent,
    )


@cli.command("create_emodel_release")
@click.option("--ais-models", type=click.Path(exists=True), required=True)
@click.option("--target-rho-factors", type=click.Path(exists=True), required=True)
@click.option("--emodel-api", type=str, default="singlecell")
@click.option("--emodel-path", type=click.Path(exists=True), required=True)
@click.option("--template", type=click.Path(exists=True), required=True)
@click.option("--etype-emodel-map", type=click.Path(exists=True), required=True)
@click.option("--mechanisms", type=click.Path(exists=True), required=True)
@click.option("--emodel-release", default="emodel_release")
def create_emodel_release(
    ais_models,
    target_rho_factors,
    emodel_api,
    emodel_path,
    template,
    etype_emodel_map,
    mechanisms,
    emodel_release,
):
    """From ais_synthesis workflow, create emodel release files for circuit-building.

    Args:
        ais_models (yaml file): contains information about the AIS shape and eletrical properties
        target_rho_factors (yaml file): contains traget rhow factor for each emodel
        emodel_api (str): name of emodel api (only singlecell for now)
        emodel_path (str): paths to the emodel config folders
        template_path (str): path to jinja2 template file
        mechanisms (str): path to mechanisms folder

    TODO:
        - update hoc with emodel hardcoded values (taper, length)
        - add option for final_path to load singlecell db
    """
    # 1) create release folders, the structure is fixed to be compatible with
    # the function get_me_combos_parameters below
    L.info("Creating emodel release folder structure.")
    release_paths = get_release_paths(emodel_release)
    if not Path(emodel_release).exists():
        Path(emodel_release).mkdir(parents=True)
        release_paths["hoc_path"].mkdir()

    # 2) load required files
    L.info("Loading required files.")
    with open(ais_models, "r") as ais_file:
        ais_models = yaml.full_load(ais_file)
    with open(target_rho_factors, "r") as rho_file:
        target_rho_factors = yaml.full_load(rho_file)

    emodel_db = _get_database(emodel_api, emodel_path)

    # 3) write hoc files for each emodel
    L.info("Writing hoc files for each emodel")
    for emodel in target_rho_factors:
        hoc = create_hoc_file(emodel, emodel_db, template, ais_models)
        with open(release_paths["hoc_path"] / f"{emodel}.hoc", "w") as hoc_file:
            hoc_file.write(hoc)

    # 4) write other release files
    L.info("Copy other release files.")
    with open(release_paths["ais_model_path"], "w") as ais_file:
        yaml.dump(ais_models, ais_file)
    with open(release_paths["target_rho_path"], "w") as rho_file:
        yaml.dump(target_rho_factors, rho_file)

    shutil.copyfile(etype_emodel_map, release_paths["etype_emodel_map"])
    shutil.copytree(emodel_path, release_paths["emodel_path"])
    shutil.copytree(mechanisms, release_paths["mechanisms"])


def _load_cells(circuit_config, mtype=None, n_cells=None):
    """Load cells from a circuit into a dataframe."""
    if mtype is not None:
        cells = Circuit(circuit_config).cells.get({"mtype": mtype})
    else:
        cells = Circuit(circuit_config).cells.get()
    if n_cells is not None:
        cells = cells[:n_cells]
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

    cells["emodel"] = cells[["mtype", "etype"]].apply(
        lambda row: etype_emodel_map.loc[
            (etype_emodel_map["mtype"] == row["mtype"])
            & (etype_emodel_map["etype"] == row["etype"]),
            "emodel",
        ],
        axis=1,
    )
    return cells


def _save_mecombos(results_df, output, with_scaler=True):
    """Save dataframe to mecombo_emodel.tsv file"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "morph_name",
        "layer",
        "fullmtype",
        "etype",
        "emodel",
        "combo_name",
        "threshold_current",
        "holding_current",
    ]
    if with_scaler:
        columns.append("AIS_scaler")
    results_df.to_csv(
        output,
        columns=columns,
        index=False,
        sep="\t",
    )


@cli.command("get_me_combos_parameters")
@click.option("--circuit-config", type=click.Path(exists=True), required=True)
@click.option(
    "--release-path",
    type=click.Path(exists=True),
    default="emodel_release",
    required=True,
)
@click.option("--morphology-path", type=click.Path(exists=True), required=True)
@click.option("--output", default="mecombo_emodel.tsv", type=str, show_default=True)
@click.option("--emodel-api", default="singlecell", type=str, show_default=True)
@click.option("--sql-tmp-path", default="tmp", type=str, show_default=True)
@click.option("--n-cells", default=None, type=int)
@click.option("--mtype", default=None, type=str)
@click.option("--parallel-lib", default="multiprocessing", show_default=True)
def get_me_combos_parameters(
    circuit_config,
    release_path,
    morphology_path,
    output,
    emodel_api,
    sql_tmp_path,
    n_cells,
    mtype,
    parallel_lib,
):
    """For each me-combos, compute the AIS scale and thresholds currents.

    Args:
        circuit_config (str): path to CircuitConfig
        release_path (str): path to emodel release folder
        morphology_path (str): base path to morphologies
        output (str): .csv file to save output data for each me-combos
        emodel_api (str): name of emodel api, so far only 'singlecell' is available
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
        n_cells (int): for testing, only use first n_cells in cells dataframe
        mtype (str): name of mtype to use (for testing only)
    """

    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = InitParallelFactory(parallel_lib).parallel_factory

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")
    cells = _load_cells(circuit_config, mtype, n_cells)

    release_paths = get_release_paths(release_path)
    with open(release_paths["ais_model_path"], "r") as ais_file:
        ais_models = yaml.full_load(ais_file)
    with open(release_paths["target_rho_path"], "r") as rho_file:
        target_rhos = yaml.full_load(rho_file)
    etype_emodel_map = pd.read_csv(release_paths["etype_emodel_map"])
    sh.nrnivmodl(release_paths["mechanisms"])  # pylint: disable=no-member
    emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

    # 2) assign emodels to each row and only keep relevant columns
    L.info("Assign emodels to each rows and restructure dataframe.")
    cells = _create_emodel_column(cells, etype_emodel_map)
    cells["morphology_path"] = cells["morphology"].apply(
        lambda morph: str((Path(morphology_path) / morph).with_suffix(".asc"))
    )
    cells["gid"] = cells.index
    morphs_combos_df = cells[
        ["gid", "morphology_path", "emodel", "mtype", "etype", "layer", "morphology"]
    ]

    # 3) Compute AIS scales
    L.info("Compute AIS scales.")
    Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
    results_df = synthesize_ais(
        morphs_combos_df,
        emodel_db,
        ais_models["mtype"],
        target_rhos,
        parallel_factory=parallel_factory,
        scales_params=ais_models["scales_params"],
        combos_db_filename=Path(sql_tmp_path) / "synth_db.sql",
    )

    # 4) compute holding and threshold currents
    L.info("Compute holding and threshold currents.")
    results_df = evaluate_currents(
        results_df,
        emodel_db,
        parallel_factory=parallel_factory,
        combos_db_filename=Path(sql_tmp_path) / "current_db.sql",
    )

    # 5) save all values in .tv
    L.info("Save results in .tsv")
    results_df = results_df.rename(
        columns={
            "AIS_scale": "AIS_scaler",
            "mtype": "fullmtype",
            "morphology": "morph_name",
        }
    )
    results_df["combo_name"] = results_df.apply(
        lambda x: "%s_%s_%s_%s" % (x["emodel"], x["fullmtype"], x["layer"], x["morph_name"]),
        axis=1,
    )

    _save_mecombos(results_df, output)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


def fix_ais(combo, out_path="fixed_cells_test"):
    """Modify morphology first axon section in place from combo data."""
    if combo.AIS_model is None:
        morphology = Morphology(combo.morphology_path)
        morphology.write(str(Path(out_path) / Path(combo.morphology_path).name))
        return combo["name"], 1.0

    ais_model = json.loads(combo.AIS_model)["popt"]
    _taper_func = partial(
        taper_function,
        strength=ais_model[1],
        taper_scale=ais_model[2],
        terminal_diameter=ais_model[3],
        scale=combo.AIS_scale,
    )

    morphology = Morphology(combo.morphology_path)
    modify_ais(morphology, _taper_func)
    morphology = Morphology(morphology, morphio.Option.nrn_order)  # ensure NEURON order
    morphology.write(str(Path(out_path) / Path(combo.morphology_path).name))
    return combo["name"], combo.AIS_scale


def taper_function(distance, strength, taper_scale, terminal_diameter, scale=1.0):
    """Function to model tappered AIS."""
    return strength * np.exp(-distance / taper_scale) + terminal_diameter * scale


def modify_ais(morphology, taper_func):
    """Modify morphology first axon section in place using taper_func."""
    L_target = 60
    for root_section in morphology.root_sections:
        if root_section.type == morphio.SectionType.axon:
            dist = 0
            prev_point = root_section.points[0]
            for section in root_section.iter():
                for i, point in enumerate(section.points):
                    dist += np.linalg.norm(point - prev_point)
                    prev_point = copy(point)
                    section.diameters[i] = taper_func(dist)
                    if dist >= L_target:
                        return


@cli.command("fix_ais_inplace")
@click.option("--mecombos-path", type=click.Path(exists=True), required=True)
@click.option("--to-fix-combos-path", type=click.Path(exists=True), required=False)
@click.option(
    "--release-path",
    type=click.Path(exists=True),
    default="emodel_release",
    required=True,
)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--applied-scales-path", default="scale_results.yaml", type=str, show_default=True)
@click.option("--fixed-morphology-path", default="fixed_morphologies", type=str, show_default=True)
@click.option("--output", default="mecombo_emodel.tsv", type=str, show_default=True)
@click.option("--emodel-api", default="singlecell", type=str, show_default=True)
@click.option("--sql-tmp-path", default="tmp", type=str, show_default=True)
@click.option("--parallel-lib", default="multiprocessing", show_default=True)
@click.option("--continu", is_flag=True)
def fix_ais_inplace(  # pylint: disable=too-many-locals
    mecombos_path,
    to_fix_combos_path,
    release_path,
    morphology_path,
    applied_scales_path,
    fixed_morphology_path,
    output,
    emodel_api,
    sql_tmp_path,
    parallel_lib,
    continu,
):
    """fix AIS size"""
    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = InitParallelFactory(parallel_lib).parallel_factory

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")
    all_morphs_combos_df = pd.read_csv(mecombos_path, sep="\t")
    to_fix_combos_df = pd.read_csv(to_fix_combos_path, header=None, names=["combo_name"])
    morphs_combos_df = all_morphs_combos_df[
        all_morphs_combos_df.combo_name.isin(to_fix_combos_df.combo_name)
    ].copy()

    assert len(morphs_combos_df.index) == len(to_fix_combos_df.index)

    morphs_combos_df["mtype"] = morphs_combos_df["fullmtype"]
    morphs_combos_df["name"] = morphs_combos_df["morph_name"]
    morphs_combos_df["morphology_path"] = morphs_combos_df["name"].apply(
        lambda name: str(Path(morphology_path) / name) + ".asc"
    )

    release_paths = get_release_paths(release_path)
    with open(release_paths["ais_model_path"], "r") as ais_file:
        ais_models = yaml.full_load(ais_file)
    with open(release_paths["target_rho_path"], "r") as rho_file:
        target_rhos = yaml.full_load(rho_file)
    sh.nrnivmodl(release_paths["mechanisms"])  # pylint: disable=no-member
    emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

    # 2) Compute AIS scales
    L.info("Compute AIS scales.")
    Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
    results_df = synthesize_ais(
        morphs_combos_df,
        emodel_db,
        ais_models["mtype"],
        target_rhos,
        parallel_factory=parallel_factory,
        scales_params=ais_models["scales_params"],
        combos_db_filename=Path(sql_tmp_path) / "synth_db.sql",
        continu=continu,
    )

    # 3) Modify morphologies
    if Path(fixed_morphology_path).exists():
        shutil.rmtree(fixed_morphology_path)
    Path(fixed_morphology_path).mkdir()

    results = {}
    for gid in tqdm(results_df.index):
        name, scale = fix_ais(results_df.loc[gid], fixed_morphology_path)
        results[str(name)] = float(scale)
    yaml.dump(results, open(applied_scales_path, "w"))

    # 4) save new mecombos_emodel.tsv file with updated entries
    all_morphs_combos_df.update(results_df)
    _save_mecombos(all_morphs_combos_df, output, with_scaler=False)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()
