"""Cli for emodel release with generalisation module."""
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
from bluepyparallel import init_parallel_factory
from morphio.mut import Morphology
from tqdm import tqdm
from voxcell import CellCollection

from bluepyemodel.api.singlecell import SinglecellAPI
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
    if api == "singlecell":
        return SinglecellAPI(
            None, emodel_path, final_path, legacy_dir_structure=legacy_dir_structure
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
        emodel_db (DatabaseAPI): database for emodels
        template_path (str): path to jinja2 template file
        ais_models (dict): contains information about the AIS shape and eletrical properties
    """
    from bluepyemodel.evaluation.model import create_cell_model

    if ais_models is None:
        from bluepyemodel.evaluation.modifiers import replace_axon_hoc as morph_modifier_hoc

    else:
        from bluepyemodel.evaluation.modifiers import get_synth_axon_hoc

        # TODO: this assumes that the AIS models are the same for all mtypes/etypes
        morph_modifier_hoc = get_synth_axon_hoc(ais_models["mtype"]["all"]["AIS"]["popt"])

    emodel_db.emodel = "_".join(emodel.split("_")[:2])
    parameters, mechanisms, _ = emodel_db.get_parameters()
    morph = emodel_db.get_morphologies()
    cell_model = create_cell_model(
        emodel,
        morph["path"],
        mechanisms,
        parameters,
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


def _load_cells(circuit_morphologies, mtype=None, n_cells=None):
    """Load cells from a circuit into a dataframe."""
    if Path(circuit_morphologies).suffix == ".mvd3":
        cells = CellCollection.load_mvd3(circuit_morphologies).as_dataframe()
    if Path(circuit_morphologies).suffix == ".h5":
        cells = CellCollection.load_sonata(circuit_morphologies).as_dataframe()

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
    return pd.merge(cells, etype_emodel_map, on=["mtype", "etype"])


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
@click.option("--circuit-morphologies", type=click.Path(exists=True), required=True)
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
@click.option("--resume", is_flag=True)
def get_me_combos_scales(
    circuit_morphologies,
    release_path,
    morphology_path,
    mecombo_emodel_tsv_path,
    combos_df_path,
    emodel_api,
    sql_tmp_path,
    n_cells,
    mtype,
    parallel_lib,
    resume,
):
    """For each me-combos, compute the AIS scale and thresholds currents.

    Args:
        circuit_morphologies (str): path to mvd3/sonata file with synthesized morphologies
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
    from bluepyemodel.generalisation.ais_synthesis import synthesize_ais

    # Initialize the parallel library. If using dask, it must be called at the beginning.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    L.info("Load release data, cells and compile mechanisms.")
    cells = _load_cells(circuit_morphologies, mtype, n_cells)

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
        morphology_path = Path(circuit_morphologies).parent
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
        db_url=Path(sql_tmp_path) / "synth_db.sql",
        resume=resume,
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
@click.option("--resume", is_flag=True)
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
    resume,
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
    if protocol_config_path is None:
        from bluepyemodel.generalisation.evaluators import evaluate_currents
    else:
        from bluepyemodel.generalisation.bglibpy_evaluators import evaluate_currents_bglibpy

    # Initialize the parallel library. If using dask, it must be called at the beginning,
    # but after nueuron imports.
    parallel_factory = init_parallel_factory(parallel_lib)

    # 1) load release data, cells and compile mechanisms
    combos_df = pd.read_csv(combos_df_path)

    # 2) compute holding and threshold currents
    L.info("Compute holding and threshold currents.")
    if protocol_config_path is None:

        release_paths = get_release_paths(release_path)
        emodel_db = _get_database(emodel_api, release_paths["emodel_path"])

        Path(sql_tmp_path).mkdir(parents=True, exist_ok=True)
        combos_df = evaluate_currents(
            combos_df,
            emodel_db,
            parallel_factory=parallel_factory,
            db_url=Path(sql_tmp_path) / "current_db.sql",
        )
    else:
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
            db_url=Path(sql_tmp_path) / "current_db.sql",
            template_format=bglibpy_template_format,
            resume=resume,
        )

    # 3) save all values in .tsv
    L.info("Save results in .tsv")
    all_morphs_combos_df = pd.read_csv(mecombos_path, sep="\t")
    all_morphs_combos_df = _add_combo_name(all_morphs_combos_df)
    if "fullmtype" not in combos_df.columns:
        combos_df["fullmtype"] = combos_df["mtype"]
    if "morph_name" not in combos_df.columns:
        combos_df["fullmtype"] = combos_df["mtype"]
        combos_df["morph_name"] = combos_df["morphology"]
    combos_df = _add_combo_name(combos_df)

    if "threshold_current" not in all_morphs_combos_df.columns:
        all_morphs_combos_df["threshold_current"] = 0
        all_morphs_combos_df["holding_current"] = 0

    all_morphs_combos_df.loc[
        all_morphs_combos_df.combo_name.isin(combos_df.combo_name), "threshold_current"
    ] = combos_df["threshold_current"].to_list()
    all_morphs_combos_df.loc[
        all_morphs_combos_df.combo_name.isin(combos_df.combo_name), "holding_current"
    ] = combos_df["holding_current"].to_list()

    with_scaler = False
    if "AIS_scaler" in all_morphs_combos_df:
        with_scaler = True

    Path(mecombo_emodel_tsv_path).parent.mkdir(parents=True, exist_ok=True)
    save_mecombos(all_morphs_combos_df, mecombo_emodel_tsv_path, with_scaler=with_scaler)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


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


def _create_etype_column(cells, cell_composition):
    """Create etype column from cell_composition.yaml data."""
    dfs = []
    for data in cell_composition["neurons"]:
        for etype in data["traits"]["etype"]:
            _df = cells[cells.mtype == data["traits"]["mtype"]].copy()
            _df.loc[:, "etype"] = etype
            dfs.append(_df)
    return pd.concat(dfs).reset_index(drop=True)


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
@click.option("--mecombo-emodel-tsv-path", default="mecombo_emodel.tsv", type=str)
@click.option("--combos-df-path", default="mecombo_emodel.tsv", type=str)
@click.option("--emodel-api", default="singlecell", type=str)
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
        emodel_api (str): name of emodel api, so far only 'singlecell' is available
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


@cli.command("update_sonata")
@click.option("--input-sonata-path", type=click.Path(exists=True), required=True)
@click.option("--output-sonata-path", type=click.Path(exists=False), required=True)
@click.option("--mecombo-emodel-path", type=click.Path(exists=True), required=True)
@click.option("--remove-failed/--no-remove-failed", default=False)
def update_sonata(input_sonata_path, output_sonata_path, mecombo_emodel_path, remove_failed):
    """Update sonata file add threshols current, holding current and AIS_scaler.

    Args:
        input_sonata_path (str): path to sonata file to update
        output_sonata_path (str): path to new sonata file
        mecombo-emodel-path (str): path to mecombo_emodel.tsv file
    """
    mecombo_emodel = pd.read_csv(mecombo_emodel_path, sep=r"\s+")
    bad_cells = mecombo_emodel[mecombo_emodel.isnull().any(axis=1)].index
    L.info("Failed cells:")
    L.info(mecombo_emodel.loc[bad_cells])

    if remove_failed:
        L.info("We remove failed cells")
        # we remove combo_name here, so the remove_unassigned_cells() later will remove these cells
        mecombo_emodel.loc[bad_cells, "combo_name"] = None
    else:
        L.info("We do not remove failed cells")
        mecombo_emodel.loc[bad_cells, "AIS_scaler"] = 1
        mecombo_emodel.loc[bad_cells, "threshold_current"] = 0
        mecombo_emodel.loc[bad_cells, "holding_current"] = 0

    cells = CellCollection.load(input_sonata_path)
    mecombo_emodel = mecombo_emodel.set_index("morph_name")
    cells.properties["me_combo"] = mecombo_emodel.loc[
        cells.properties["morphology"], "combo_name"
    ].to_list()
    cells.properties["@dynamics:AIS_scaler"] = mecombo_emodel.loc[
        cells.properties["morphology"], "AIS_scaler"
    ].to_list()
    cells.properties["@dynamics:threshold_current"] = mecombo_emodel.loc[
        cells.properties["morphology"], "threshold_current"
    ].to_list()
    cells.properties["@dynamics:holding_current"] = mecombo_emodel.loc[
        cells.properties["morphology"], "holding_current"
    ].to_list()
    cells.remove_unassigned_cells()
    cells.save(output_sonata_path)
