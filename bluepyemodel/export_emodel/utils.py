"""Export the emodels in the SONATA format"""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import pathlib
import shutil

logger = logging.getLogger(__name__)


def get_output_path_from_metadata(output_base_dir, emodel_metadata, seed, use_allen_notation=True):
    """Get the output path from the emodel_metadata.

    Args:
        output_base_dir (str): output base directory
        emodel_metadata (EModelMetadata): emodel metadata
        seed (int): seed
    """
    return (
        f"./{output_base_dir}/"
        f"{emodel_metadata.as_string(seed=seed, use_allen_notation=use_allen_notation)}/"
    )


def get_output_path(
    emodel,
    output_dir=None,
    output_base_dir="export_emodels_hoc",
    use_allen_notation=True,
    create_dir=True,
):
    """Get the output path.

    Args:
        emodel (EModel): emodel
        output_dir (str): output directory
        output_base_dir (str): if output_dir is None, export to this directory instead,
            using also emodel metadata in the path
        use_allen_notation (bool): whether to replace brain region by its Allen notation
        create_dir (bool): whether to create the output folder if not existent
    """
    if output_dir is None:
        output_dir = get_output_path_from_metadata(
            output_base_dir, emodel.emodel_metadata, emodel.seed, use_allen_notation
        )
    output_path = pathlib.Path(output_dir)
    if create_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def get_hoc_file_path(output_path):
    """Get the hoc file path."""
    output_path = pathlib.Path(output_path)
    return str(output_path / "model.hoc")


def copy_hocs_to_new_output_path(emodel, output_base_dir):
    """Copy the hocs from the local output path to the new nexus output path."""
    old_output_path = get_output_path(
        emodel,
        output_dir=None,
        output_base_dir=output_base_dir,
        use_allen_notation=False,
        create_dir=False,
    )
    output_path_allen = get_output_path(
        emodel,
        output_dir=None,
        output_base_dir=output_base_dir,
        use_allen_notation=True,
        create_dir=False,
    )

    if (
        not pathlib.Path(get_hoc_file_path(output_path_allen)).is_file()
        and pathlib.Path(get_hoc_file_path(old_output_path)).is_file()
    ):
        shutil.copytree(old_output_path, output_path_allen)


def select_emodels(
    emodel_name,
    emodels,
    only_validated=False,
    only_best=True,
    seeds=None,
    iteration=None,
):
    if not emodels:
        logger.warning("In export_emodels_nexus, no emodel for %s", emodel_name)
        return []

    if iteration:
        emodels = [model for model in emodels if model.emodel_metadata.iteration == iteration]

    if only_best:
        emodels = [sorted(emodels, key=lambda x: x.fitness)[0]]

    if seeds:
        emodels = [e for e in emodels if e.seed in seeds]
        if not emodels:
            logger.warning(
                "In export_emodels_nexus, no emodel for %s and seeds %s",
                emodel_name,
                seeds,
            )
            return []

    if only_validated:
        emodels = [e for e in emodels if e.passed_validation]
        if not emodels:
            logger.warning(
                "In export_emodels_nexus, no emodel for %s that passed validation",
                emodel_name,
            )
            return []

    return emodels
