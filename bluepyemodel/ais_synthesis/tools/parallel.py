"""Parallel helper."""
import logging
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial

import dask_mpi
import dask.distributed
import ipyparallel

L = logging.getLogger(__name__)


class ParallelFactory(ABC):
    """Abstract class that should be subclassed to provide parallel functions."""

    @abstractmethod
    def get_mapper(self):
        """Return a mapper function that can be used to execute functions in parallel."""

    def shutdown(self):
        """Can be used to cleanup."""


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    def _get_daemon(self):  # pylint: disable=no-self-use
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


class MultiprocessingFactory(ParallelFactory):
    """Parallel helper class using multiprocessing."""

    def get_mapper(self):
        """Get a NestedPool."""
        return partial(NestedPool(maxtasksperchild=1).imap_unordered, chunksize=1)


class IPyParallelFactory(ParallelFactory):
    """Parallel helper class using ipyparallel."""

    _IPYTHON_PROFILE = "IPYTHON_PROFILE"

    def get_mapper(self):
        """Get an ipyparallel mapper using the profile name provided."""
        profile = os.getenv(self._IPYTHON_PROFILE, "DEFAULT_IPYTHON_PROFILE")
        L.debug("Using %s=%s", self._IPYTHON_PROFILE, profile)
        rc = ipyparallel.Client(profile=profile)
        lview = rc.load_balanced_view()
        return partial(lview.imap, ordered=False)


class DaskFactory(ParallelFactory):
    """Parallel helper class using dask."""

    _PARALLEL_BATCH_SIZE = "PARALLEL_BATCH_SIZE"

    def __init__(self):
        """Initialize the dask factory."""
        dask_mpi.initialize()
        self.client = dask.distributed.Client()

    def shutdown(self):
        """Retire the workers on the scheduler."""
        time.sleep(1)
        self.client.retire_workers()
        self.client = None

    def get_mapper(self):
        """Get a Dask mapper."""
        batch_size = int(os.getenv(self._PARALLEL_BATCH_SIZE, "0")) or None
        L.debug("Using %s=%s", self._PARALLEL_BATCH_SIZE, batch_size)

        def _mapper(func, iterable):
            if isinstance(iterable, Iterator):
                # Dask >= 2.0.0 no longer supports mapping over Iterators
                iterable = list(iterable)
            # map is less efficient than dask.bag, but yields results as soon as they are ready
            futures = self.client.map(func, iterable, batch_size=batch_size)
            for _future, result in dask.distributed.as_completed(
                futures, with_results=True
            ):
                yield result

        return _mapper


def init_parallel_factory(parallel_lib):
    """Return the desired instance of the parallel factory."""
    parallel_factory = {
        "dask": DaskFactory,
        "ipyparallel": IPyParallelFactory,
        "multiprocessing": MultiprocessingFactory,
    }[parallel_lib]()
    L.info("Initialized %s factory", parallel_lib)
    return parallel_factory
