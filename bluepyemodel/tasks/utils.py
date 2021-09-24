"""Utils functions."""
import datetime
import json
import logging
import multiprocessing
import os
import tarfile
import time
from multiprocessing import pool
from pathlib import Path

from git import Repo

logger = logging.getLogger(__name__)


def ipyparallel_map_function(ipython_profile="IPYTHON_PROFILE"):
    """Get the map function linked to the ipython profile

    Args:
       ipython_profile (str): name fo the environement variable containing
           the name of the name of the ipython profile

    Returns:
        map
    """
    if os.getenv(ipython_profile):
        from ipyparallel import Client

        rc = Client(profile=os.getenv(ipython_profile))
        lview = rc.load_balanced_view()

        def mapper(func, it):
            start_time = datetime.datetime.now()
            ret = lview.map_sync(func, it)
            logger.debug("Took %s", datetime.datetime.now() - start_time)
            return ret

    else:
        mapper = map

    return mapper


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):  # pylint: disable=no-self-use
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


def get_mapper(backend):
    """Get a mapper for parallel computations."""
    if backend == "ipyparallel":
        return ipyparallel_map_function()

    if backend == "multiprocessing":
        nested_pool = NestedPool()
        return nested_pool.map
    return map


#################
# UNSUSED BELOW #
#################


def json_load(obj, name):
    """Load json dict from json string OR path to json file."""
    var = getattr(obj, name)
    if var is not None and isinstance(var, str):
        try:
            if Path(var).is_file():

                with open(var, "r") as f:
                    var = json.load(f)
            else:
                var = json.loads(var)

            return var

        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            raise Exception(
                f"Expected a path to a json file or a json string in {name} in luigi.cfg."
            ) from e
    return var


def get_checkpoint_path(seed, path):
    """Returns the checkpoint path with the seed number in it."""
    path = Path(path)
    if "{}" in path.stem:
        return path.with_name(path.stem.format(seed)).with_suffix(path.suffix)
    if f"_{seed}" in path.stem:
        return path
    return path.with_name(path.stem + f"_{seed}").with_suffix(path.suffix)


def update_gitignore():
    """
    Adds the following lines to .gitignore: 'run/', 'checkpoints/', 'figures/',
    'logs/', '.ipython/', '.ipynb_checkpoints/'
    """

    path_gitignore = Path("./.gitignore")

    if not (path_gitignore.is_file()):
        raise Exception("Could not update .gitignore as it does not exist.")

    with open(str(path_gitignore), "r") as f:
        lines = f.readlines()

    to_add = [
        "run/",
        "checkpoints/",
        "figures/",
        "logs/",
        ".ipython/",
        ".ipynb_checkpoints/",
    ]
    not_to_add = []
    for d in to_add:
        for line in lines:
            if d in line:
                not_to_add.append(d)
                break

    for a in to_add:
        if a not in not_to_add:
            lines.append(f"{a}\n")

    with open(str(path_gitignore), "w") as f:
        f.writelines(lines)


def change_cwd(dir_path):
    """Changes the cwd for dir_path, creating it if it doesn't exist.

    Args:
        dir_path (str): path of the target directory
    """

    if str(Path(os.getcwd())) != str(Path(dir_path).absolute()):

        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        logger.warning("Moving to working_dir %s", dir_path)
        os.chdir(dir_path)


def generate_versions():
    """
    Save a list of the versions of the python packages in the current environnement.
    """

    path_versions = Path("./list_versions.log")

    if path_versions.is_file():
        logger.warning("%s already exists and will overwritten.", path_versions)

    os.popen(f"pip list > {str(path_versions)}").read()


def generate_githash(run_dir):
    """
    Generate a githash and create the associated run directory
    """

    path_run = Path(run_dir)

    if not (path_run.is_dir()):
        logger.warning("Directory %s does not exist and will be created.", run_dir)
        path_run.mkdir(parents=True, exist_ok=True)

    while Path("./.git/index.lock").is_file():
        time.sleep(5.0)
        logger.info("emodel_pipeline waiting for ./.git/index.lock.")

    repo = Repo("./")
    changedFiles = [item.a_path for item in repo.index.diff(None)]
    if changedFiles:
        os.popen('git add -A && git commit --allow-empty -a -m "Running pipeline"').read()

    githash = os.popen("git rev-parse --short HEAD").read()[:-1]

    tar_source = path_run / f"{githash}.tar"
    tar_target = path_run / githash

    if not (tar_target.is_dir()):

        logger.info("New githash directory: %s", githash)

        repo = Repo("./")
        with open(str(tar_source), "wb") as fp:
            repo.archive(fp)
        with tarfile.open(str(tar_source)) as tf:
            tf.extractall(str(tar_target))

        if tar_source.is_file():
            os.remove(str(tar_source))

    else:
        logger.info("Working from existing githash directory: %s", githash)

    return githash
