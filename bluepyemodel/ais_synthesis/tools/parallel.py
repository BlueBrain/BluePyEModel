"""Parallel helper."""
import logging
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial


try:
    import dask_mpi
except ModuleNotFoundError:
    pass

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

    _CHUNKSIZE = "PARALLEL_CHUNKSIZE"

    def __init__(self):
        """Initialize multiprocessing factory."""
        self.pool = NestedPool()

    def get_mapper(self):
        """Get a NestedPool."""
        chunksize = int(os.getenv(self._CHUNKSIZE, "1"))
        L.debug("Using %s=%s", self._CHUNKSIZE, chunksize)
        return partial(self.pool.imap_unordered, chunksize=chunksize)

    def shutdown(self):
        """Close the pool."""
        self.pool.close()


class IPyParallelFactory(ParallelFactory):
    """Parallel helper class using ipyparallel."""

    _IPYTHON_PROFILE = "IPYTHON_PROFILE"

    def __init__(self):
        """Initialize the ipyparallel factory."""
        self.rc = None

    def get_mapper(self):
        """Get an ipyparallel mapper using the profile name provided."""
        profile = os.getenv(self._IPYTHON_PROFILE, "DEFAULT_IPYTHON_PROFILE")
        L.debug("Using %s=%s", self._IPYTHON_PROFILE, profile)
        self.rc = ipyparallel.Client(profile=profile)
        lview = self.rc.load_balanced_view()
        return partial(lview.imap, ordered=False)

    def shutdown(self):
        """Remove zmq."""
        if self.rc is not None:
            self.rc.close()


class DaskFactory(ParallelFactory):
    """Parallel helper class using dask."""

    _BATCH_SIZE = "PARALLEL_DASK_BATCH_SIZE"
    _SCHEDULER_PATH = "PARALLEL_DASK_SCHEDULER_PATH"

    def __init__(self):
        """Initialize the dask factory."""
        dask_scheduler_path = os.getenv(self._SCHEDULER_PATH)
        if dask_scheduler_path:
            self.interactive = True
            L.info("Connecting dask_mpi with scheduler %s", dask_scheduler_path)
            self.client = dask.distributed.Client(scheduler_file=dask_scheduler_path)
        else:
            self.interactive = False
            dask_mpi.initialize()
            L.info("Starting dask_mpi...")
            self.client = dask.distributed.Client()

    def shutdown(self):
        """Retire the workers on the scheduler."""
        if not self.interactive:
            time.sleep(1)
            self.client.retire_workers()
            self.client = None

    def get_mapper(self):
        """Get a Dask mapper."""
        batch_size = int(os.getenv(self._BATCH_SIZE, "0")) or None
        L.debug("Using %s=%s", self._BATCH_SIZE, batch_size)

        def _mapper(func, iterable):
            if isinstance(iterable, Iterator):
                # Dask >= 2.0.0 no longer supports mapping over Iterators
                iterable = list(iterable)
            # map is less efficient than dask.bag, but yields results as soon as they are ready
            futures = self.client.map(func, iterable, batch_size=batch_size)
            for _future, result in dask.distributed.as_completed(futures, with_results=True):
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
