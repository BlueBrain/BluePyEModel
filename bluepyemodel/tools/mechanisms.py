"""Mechanisms related functions"""
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
    # pylint: disable=broad-exception-caught
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


def to_current(name):
    """Turn current / ionic concentration name into current name."""
    # ion current case
    if name[0] == "i":
        return name
    # internal / external ionic concentration case
    if name[-1] == "i" or name[-1] == "o":
        return f"i{name[:-1]}"
    return None

def get_mechanism_currents(mech_file):
    """Parse the mech mod file to get the mechanism ion and non-specific currents if any."""
    ion_currs = []
    nonspecific_currents = []
    with open(mech_file, "r") as f:
        mod_lines = f.readlines()
    for line in mod_lines:
        if "WRITE " in line:
            ion_var_name = line.split("WRITE ")[1].rstrip("\n").split(" ")[0]
            current_name = to_current(ion_var_name)
            if current_name is not None:
                ion_currs.append(current_name)
        elif "NONSPECIFIC_CURRENT" in line:
            var_name = line.split("NONSPECIFIC_CURRENT ")[1].rstrip("\n").split(" ")[0]
            current_name = to_current(var_name)
            if current_name is not None:
                nonspecific_currents.append(current_name)

    return ion_currs, nonspecific_currents


def get_mechanism_suffix(mech_file):
    """Parse the mech mod file to get the mechanism suffix."""
    with open(mech_file, "r") as f:
        mod_lines = f.readlines()
    for line in mod_lines:
        if "SUFFIX " in line:
            suffix = line.split("SUFFIX ")[1].rstrip("\n").split(" ")[0]
            return suffix
    raise RuntimeError(f"Could not find SUFFIX in {mech_file}")
