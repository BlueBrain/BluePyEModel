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

from voxcell import CellCollection
from bluepyemodel.ais_synthesis.tools import init_parallel_factory
from bluepyemodel.api.singlecell import Singlecell_API

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    """Cli to learn and generate diameters of neurons."""
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=loglevel, format=logformat)


def _get_database(api, emodel_path, final_path=None, legacy_dir_structure=True):
    if api == "singlecell":
        return Singlecell_API(emodel_path, final_path, legacy_dir_structure=legacy_dir_structure)
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
    from bluepyemodel.evaluation.model import create_cell_model
    from bluepyemodel.evaluation.modifiers import get_synth_axon_hoc

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
        emodel_api (str): name of emodel api (only singlecell for now)
        emodel_path (str): paths to the emodel config folders
        final_path (str): paths to the final.json file
        template_path (str): path to jinja2 template file
        etype_emodel_map_path (str): path to file with etype-emodel map
        mechanisms (str): path to mechanisms folder, if None, we will not try to copy them
        emodel_release (str): path to folder to store emodel release
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
        target_rho_factors_all = yaml.full_load(rho_file)
    etype_emodel_map = pd.read_csv(etype_emodel_map_path)
    target_rho_factors = {
        emodel: target_rho_factors_all[emodel] for emodel in etype_emodel_map.emodel
    }

    emodel_db = _get_database(emodel_api, emodel_path, final_path)

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

    shutil.copyfile(etype_emodel_map_path, release_paths["etype_emodel_map"])

    if Path(release_paths["emodel_path"]).exists():
        shutil.rmtree(release_paths["emodel_path"])
    shutil.copytree(emodel_path, release_paths["emodel_path"])
    shutil.copyfile(final_path, Path(release_paths["emodel_path"]) / "final.json")

    if mechanisms is not None:
        shutil.copytree(mechanisms, release_paths["mechanisms"])


def _load_cells(circuit_morphologies_mvd3, mtype=None, n_cells=None):
    """Load cells from a circuit into a dataframe."""
    cells = CellCollection.load_mvd3(circuit_morphologies_mvd3).as_dataframe()
    if mtype is not None:
        cells = cells[cells.mtype == mtype]
    if n_cells is not None:
        cells = cells.sample(min(len(cells.index), n_cells))
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

    cells["emodel"] = cells[["mtype", "etype"]].apply(
        lambda row: etype_emodel_map.loc[
            (etype_emodel_map["mtype"] == row["mtype"])
            & (etype_emodel_map["etype"] == row["etype"]),
            "emodel",
        ].tolist()[0],
        axis=1,
    )
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
        combos_df = combos_df.rename(
            columns={
                "AIS_scale": "AIS_scaler",
                "mtype": "fullmtype",
                "morphology": "morph_name",
            }
        )
    combos_df["layer"] = combos_df["layer"].astype(int)
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
@click.option("--circuit-morphologies-mvd3", type=click.Path(exists=True), required=True)
@click.option(
    "--release-path",
    type=click.Path(exists=True),
    default="emodel_release",
    required=True,
)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--mecombo-emodel-tsv-path", default="mecombo_emodel.tsv", type=str)
@click.option("--combos-df-path", default="combos_df.csv", type=str)
@click.option("--emodel-api", default="singlecell", type=str)
@click.option("--sql-tmp-path", default="tmp", type=str)
@click.option("--n-cells", default=None, type=int)
@click.option("--mtype", default=None, type=str)
@click.option("--parallel-lib", default="multiprocessing")
def get_me_combos_scales(
    circuit_morphologies_mvd3,
    release_path,
    morphology_path,
    mecombo_emodel_tsv_path,
    combos_df_path,
    emodel_api,
    sql_tmp_path,
    n_cells,
    mtype,
    parallel_lib,
):
    """For each me-combos, compute the AIS scale and thresholds currents.

    Args:
        circuit_morphologies_mvd3 (str): path to mvd3 file with synthesized morphologies
        release_path (str): path to emodel release folder
        morphology_path (str): base path to morphologies, if none, same dir as mvd3 will be used
        mecombo_emodel_tsv_path (str): path to .tsv file to save output data for each me-combos
        combos_df_path (str): path to .csv file to save all result dataframe
        emodel_api (str): name of emodel api, so far only 'singlecell' is available
        protocol_config_path (str): path to yaml file with protocol config for bglibpy
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
        n_cells (int): for testing, only use first n_cells in cells dataframe
        mtype (str): name of mtype to use (for testing only)
    """
    from bluepyemodel.ais_synthesis.ais_synthesis import synthesize_ais

    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")
    cells = _load_cells(circuit_morphologies_mvd3, mtype, n_cells)

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
        morphology_path = Path(circuit_morphologies_mvd3).parent
    cells["morphology_path"] = cells["morphology"].apply(
        lambda morph: str(Path(morphology_path) / morph) + ".asc"
    )
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
        combos_db_filename=Path(sql_tmp_path) / "synth_db.sql",
    )

    # 4) save all values in .tsv
    L.info("Save results in .tsv")
    Path(mecombo_emodel_tsv_path).parent.mkdir(parents=True, exist_ok=True)
    combos_df.to_csv(combos_df_path, index=False)
    save_mecombos(combos_df, mecombo_emodel_tsv_path)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


@cli.command("get_me_combos_currents")
@click.option("--mecombos-path", type=click.Path(exists=True), required=True)
@click.option("--combos-df-path", default="combos_df.csv", type=str)
@click.option("--release-path", type=click.Path(exists=True))
@click.option("--mecombo-emodel-tsv-path", default="mecombo_emodel.tsv", type=str)
@click.option("--emodel-api", default="singlecell", type=str)
@click.option("--protocol-config-path", default=None, type=str)
@click.option("--sql-tmp-path", default="tmp", type=str)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--bglibpy-template-format", default="v6_ais_scaler", type=str)
@click.option("--emodels-hoc-dir", type=str)
@click.option("--continu", is_flag=True)
def get_me_combos_currents(
    mecombos_path,
    combos_df_path,
    release_path,
    mecombo_emodel_tsv_path,
    emodel_api,
    protocol_config_path,
    sql_tmp_path,
    parallel_lib,
    bglibpy_template_format,
    emodels_hoc_dir,
    continu,
):
    """For each me-combos, compute the thresholds currents.

    Args:
        mecombo_emodel_tsv_path (str): path to .tsv file to save output data for all me-combos
        combos_df_path (str): path to dataframe with cells informations to run
        release_path (str): path to emodel release folder
        morphology_path (str): base path to morphologies, if none, same dir as mvd3 will be used
        output (str): .csv file to save output data for each me-combos
        emodel_api (str): name of emodel api, so far only 'singlecell' is available
        protocol_config_path (str): path to yaml file with protocol config for bglibpy
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
        bglibpy_template_format (str): v6 for standar use v6_ais_scaler for custom hoc
        emodels_hoc_dir (str): paths to hoc files, if None hoc from release path will be used

    """
    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    combos_df = pd.read_csv(combos_df_path)

    # 2) compute holding and threshold currents
    L.info("Compute holding and threshold currents.")
    if protocol_config_path is None:
        from bluepyemodel.ais_synthesis.evaluators import evaluate_currents

        release_paths = get_release_paths(release_path)
        emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

        Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
        combos_df = evaluate_currents(
            combos_df,
            emodel_db,
            parallel_factory=parallel_factory,
            combos_db_filename=Path(sql_tmp_path) / "current_db.sql",
        )
    else:
        from bluepyemodel.ais_synthesis.bglibpy_evaluators import evaluate_currents_bglibpy

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
            combos_db_filename=Path(sql_tmp_path) / "current_db.sql",
            template_format=bglibpy_template_format,
            continu=continu,
        )

    # 3) save all values in .tsv
    L.info("Save results in .tsv")

    all_morphs_combos_df = pd.read_csv(mecombos_path, sep="\t")
    all_morphs_combos_df = _add_combo_name(all_morphs_combos_df).set_index("combo_name")
    if "fullmtype" not in combos_df.columns:
        combos_df["fullmtype"] = combos_df["mtype"]
    if "morph_name" not in combos_df.columns:
        combos_df["morph_name"] = combos_df["morphology"]

    if "threshold_current" not in all_morphs_combos_df.columns:
        all_morphs_combos_df["threshold_current"] = 0
        all_morphs_combos_df["holding_current"] = 0

    all_morphs_combos_df.update(_add_combo_name(combos_df).set_index("combo_name"))
    all_morphs_combos_df = all_morphs_combos_df.reset_index(drop=True)

    with_scaler = False
    if "AIS_scaler" in all_morphs_combos_df:
        with_scaler = True

    Path(mecombo_emodel_tsv_path).parent.mkdir(parents=True, exist_ok=True)
    save_mecombos(all_morphs_combos_df, mecombo_emodel_tsv_path, with_scaler=with_scaler)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


def fix_ais(combo, out_path="fixed_cells_test"):
    """Modify morphology first axon section in place from combo data."""
    morphology = Morphology(combo.morphology_path)

    if combo.AIS_model is not None:
        ais_model = json.loads(combo.AIS_model)["popt"]
        _taper_func = partial(
            taper_function,
            strength=ais_model[1],
            taper_scale=ais_model[2],
            terminal_diameter=ais_model[3],
            scale=combo.AIS_scale,
        )
        modify_ais(morphology, _taper_func)
    else:
        raise Exception(combo)

    morphology = Morphology(morphology, morphio.Option.nrn_order)  # ensures NEURON order
    morphology.write(str(Path(out_path) / Path(combo.morphology_path).name))


def taper_function(distance, strength, taper_scale, terminal_diameter, scale=1.0):
    """Function to model tappered AIS."""
    return strength * np.exp(-distance / taper_scale) + terminal_diameter * scale


def modify_ais(morphology, taper_func):
    """Modify morphology first axon section in place using taper_func."""
    L_target = 65  # we set it to a little longer thatn 60, to ensures the end points are good
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
@click.option("--mecombos-path", type=click.Path(exists=True), required=True)
@click.option("--to-fix-combos-path", type=click.Path(exists=True), required=False)
@click.option(
    "--release-path",
    type=click.Path(exists=True),
    default="emodel_release",
    required=True,
)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--fixed-morphology-path", default="fixed_morphologies", type=str)
@click.option("--mecombo-emodel-tsv-path", default="mecombo_emodel.tsv", type=str)
@click.option("--combos-df-path", default="mecombo_emodel.tsv", type=str)
@click.option("--emodel-api", default="singlecell", type=str)
@click.option("--sql-tmp-path", default="tmp", type=str)
@click.option("--parallel-lib", default="multiprocessing")
@click.option("--continu", is_flag=True)
def set_me_combos_scales_inplace(  # pylint: disable=too-many-locals
    mecombos_path,
    to_fix_combos_path,
    release_path,
    morphology_path,
    fixed_morphology_path,
    mecombo_emodel_tsv_path,
    combos_df_path,
    emodel_api,
    sql_tmp_path,
    parallel_lib,
    continu,
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
        emodel_api (str): name of emodel api, so far only 'singlecell' is available
        protocol_config_path (str): path to yaml file with protocol config for bglibpy
        sql_tmp_path (str): path to a folder to save sql files used during computations
        parallel_lib (str): parallel library
    """
    from bluepyemodel.ais_synthesis.ais_synthesis import synthesize_ais

    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")
    all_morphs_combos_df = pd.read_csv(mecombos_path, sep="\t")

    if to_fix_combos_path is not None:
        to_fix_combos_df = pd.read_csv(to_fix_combos_path, header=None, names=["combo_name"])
        morphs_combos_df = all_morphs_combos_df[
            all_morphs_combos_df.combo_name.isin(to_fix_combos_df.combo_name)
        ].copy()
    else:
        morphs_combos_df = all_morphs_combos_df.copy()

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

    if Path(release_paths["mechanisms"]).exists():
        sh.nrnivmodl(release_paths["mechanisms"])  # pylint: disable=no-member
    emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

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
        combos_db_filename=Path(sql_tmp_path) / "synth_db.sql",
        continu=continu,
    )

    # 3) Modify morphologies
    L.info("Modify morphologies.")
    if Path(fixed_morphology_path).exists():
        shutil.rmtree(fixed_morphology_path)
    Path(fixed_morphology_path).mkdir(parents=True)

    for gid in tqdm(combos_df.index):
        fix_ais(combos_df.loc[gid], fixed_morphology_path)

    # 4) save new mecombos_emodel.tsv file with updated entries
    all_morphs_combos_df.update(combos_df)
    combos_df.to_csv(combos_df_path, index=False)
    Path(mecombo_emodel_tsv_path).parent.mkdir(parents=True, exist_ok=True)
    save_mecombos(all_morphs_combos_df, mecombo_emodel_tsv_path, with_scaler=False)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()
