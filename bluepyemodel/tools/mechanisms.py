"""Mechanisms related functions"""
import os
import shutil
from pathlib import Path


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
                raise Exception(
                    "Cannot copy the .mod files locally because the "
                    "'mechanism_paths' {} does not exist.".format(p)
                )


def compile_mechs(mechanisms_dir):
    """Compile the mechanisms.

    Args:
        mechanisms_dir (str): path to the directory containing the
            mod files to compile.
    """

    path_mechanisms_dir = Path(mechanisms_dir)

    if path_mechanisms_dir.is_dir():

        if Path("x86_64").is_dir():
            os.popen("rm -rf x86_64").read()
        os.popen("nrnivmodl {}".format(path_mechanisms_dir)).read()

    else:
        raise Exception(
            "Cannot compile the mechanisms because 'mechanisms_dir':"
            " {} does not exist.".format(str(path_mechanisms_dir))
        )


def copy_and_compile_mechanisms(db, emodel, species, copy_mechanisms, mechanisms_dir, githash=""):
    """Copy mechs if asked, and compile them."""
    if githash:
        raise Exception(
            "Compile mechanisms and the use of githash are not compatible "
            "yet. Please pre-compile the mechanisms and re-run with "
            "compile_mechanisms=False."
        )
    if copy_mechanisms:
        _, _, mechanism_names = db.get_parameters(emodel, species)
        mechanism_paths = db.get_mechanism_paths(mechanism_names)
        if not (mechanism_paths):
            raise Exception("No mechanisms paths for emodel %s" % emodel)
        copy_mechs(mechanism_paths, mechanisms_dir)

    compile_mechs(mechanisms_dir)
