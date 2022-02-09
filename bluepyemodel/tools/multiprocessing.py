"""Function related to parallel computing"""
import datetime
import multiprocessing
import os
from multiprocessing import pool

from bluepyemodel.emodel_pipeline.utils import logger


def ipyparallel_map_function(os_env_profile="IPYTHON_PROFILE", profile=None):
    """Get the map function linked to the ipython profile

    Args:
        os_env_profile (str): name fo the environement variable containing
           the name of the name of the ipython profile
           Will be used is name of the ipython profile is not given
        profile (str): name of the name of the ipython profile

    Returns:
        map
    """
    logger.warning(f"ipython profile: {profile}")
    if not profile:
        profile = os.getenv(os_env_profile)
    if profile:
        from ipyparallel import Client

        rc = Client(profile=profile)
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


def get_mapper(backend, ipyparallel_profile=None):
    """Get a mapper for parallel computations."""
    if backend == "ipyparallel":
        return ipyparallel_map_function(profile=ipyparallel_profile)

    if backend == "multiprocessing":
        nested_pool = NestedPool()
        return nested_pool.map
    return map
