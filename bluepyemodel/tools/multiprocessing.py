"""Function related to parallel computing"""

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

import datetime
import multiprocessing
import os
from multiprocessing import pool

from bluepyemodel.tools.utils import logger


def ipyparallel_map_function(os_env_profile="IPYTHON_PROFILE", profile=None):
    """Get the map function linked to the ipython profile

    Args:
        os_env_profile (str): The name of the environment variable
            that holds the profile name for ipyparallel.
            If this environment variable exists and contains a valid profile name,
            it will be used to create an ipyparallel Client. Defaults to "IPYTHON_PROFILE"
        profile (str): name of the ipython profile

    Returns:
        map
    """
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
        logger.warning(
            "Environment variable 'os_env_profile' not set or invalid;"
            "falling back to the default map function."
        )
        mapper = map

    return mapper


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):
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

    # For any other backend, default to the built-in map function
    return map
