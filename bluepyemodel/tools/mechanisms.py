"""Mechanisms related functions"""

"""
Copyright 2023, EPFL/Blue Brain Project

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
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


NEURON_BUILTIN_MECHANISMS = ["hh", "pas", "fastpas", "extracellular", "capacitance"]


def copy_mechs(mechanism_paths, out_dir):
    """Copy mod files in the designated directory.

    Args:
        mechanism_paths (list): list of the paths to the mod files that
            have to be copied.
        out_dir (str): path to directory to which the mod files should
            be copied.
    """

    if mechanism_paths:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for m in mechanism_paths:
            p = Path(m["path"])

            if p.is_file():
                new_p = out_dir / p.name
                shutil.copy(str(p), str(new_p))

            else:
                raise FileNotFoundError(
                    "Cannot copy the .mod files locally because the "
                    f"'mechanism_paths' {p} does not exist."
                )


def delete_compiled_mechanisms():
    """Delete compiled mechanisms."""
    if Path("x86_64").is_dir():
        os.popen("rm -rf x86_64").read()


def compile_mechs(mechanisms_dir):
    """Compile the mechanisms.

    Args:
        mechanisms_dir (str): path to the directory containing the
            mod files to compile.
    """

    path_mechanisms_dir = Path(mechanisms_dir)

    if path_mechanisms_dir.is_dir():
        delete_compiled_mechanisms()
        os.popen(f"nrnivmodl {path_mechanisms_dir}").read()
    else:
        raise FileNotFoundError(
            "Cannot compile the mechanisms because 'mechanisms_dir':"
            f" {path_mechanisms_dir} does not exist."
        )


def compile_mechs_in_emodel_dir(mechanisms_directory):
    """Compile mechanisms in emodel directory.

    Args:
        mechanisms_dir (Path): path to the directory containing the
            mod files to compile.
    """
    # pylint: disable=broad-except
    cwd = os.getcwd()

    try:
        os.chdir(str(mechanisms_directory.parents[0]))
        compile_mechs("./mechanisms")
    except Exception as e:
        logger.exception(e)
    finally:
        os.chdir(cwd)


def copy_and_compile_mechanisms(access_point):
    """Copy mechs if asked, and compile them."""

    if access_point.__class__.__name__ == "NexusAccessPoint":
        # Mechanisms are automatically download by the Nexus API
        # when calling this function
        _ = access_point.get_model_configuration()
        compile_mechs("./mechanisms")


def get_mechanism_currents(mech_file):
    """Parse the mech mod file to get the mechanism ion and non-specific currents if any."""
    ion_currs = []
    nonspecific_currents = []
    ionic_concentrations = []
    with open(mech_file, "r") as f:
        mod_lines = f.readlines()
    for line in mod_lines:
        if "WRITE " in line:
            ion_var_name = line.split("WRITE ")[1].rstrip("\n").split(" ")[0]
            # ion current case
            if ion_var_name[0] == "i":
                ion_currs.append(ion_var_name)
                ionic_concentrations.append(f"{ion_var_name[1:]}i")
            # internal ionic concentration case
            elif ion_var_name[-1] == "i":
                ionic_concentrations.append(ion_var_name)
        elif "NONSPECIFIC_CURRENT" in line:
            var_name = line.split("NONSPECIFIC_CURRENT ")[1].rstrip("\n").split(" ")[0]
            if var_name[0] == "i":
                nonspecific_currents.append(var_name)

    return ion_currs, nonspecific_currents, ionic_concentrations


def get_mechanism_name(mech_file):
    """Parse the mech mod file to get the mechanism suffix or point process to use as name."""
    with open(mech_file, "r") as f:
        mod_lines = f.readlines()
    for line in mod_lines:
        if "SUFFIX " in line:
            suffix = line.split("SUFFIX ")[1].rstrip("\n").split(" ")[0]
            return suffix
        if "POINT_PROCESS" in line:
            point_process = line.split("POINT_PROCESS ")[1].rstrip("\n").split(" ")[0]
            return point_process
    raise RuntimeError(f"Could not find SUFFIX nor POINT_PROCESS in {mech_file}")


def discriminate_by_temp(resources, temperatures):
    """Select sublist of resources with given temperature."""
    if not temperatures:
        return resources
    new_temperatures = temperatures.copy()
    temp = new_temperatures.pop(0)
    tmp_resources = [r for r in resources if r.temperature.value == temp]
    if len(tmp_resources) > 0 and len(tmp_resources) < len(resources):
        logger.warning(
            "Discriminating resources based on temperature. "
            "Keeping only resource with temperature == %s C.",
            temp,
        )
        return tmp_resources
    if len(new_temperatures) > 0:
        return discriminate_by_temp(resources, new_temperatures)
    return resources
