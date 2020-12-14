"""Parallel helper."""
import logging
import multiprocessing
import os
import time
from abc import abstractmethod
from collections.abc import Iterator
from functools import partial
import numpy as np

try:
    import dask.distributed
except ModuleNotFoundError:
    pass

try:
    import ipyparallel
except ModuleNotFoundError:
    pass

try:
    import dask_mpi
except ModuleNotFoundError:
    pass


L = logging.getLogger(__name__)


class ParallelFactory:
    """Abstract class that should be subclassed to provide parallel functions."""

    _BATCH_SIZE = "PARALLEL_BATCH_SIZE"

    def __init__(self):
        self.batch_size = int(os.getenv(self._BATCH_SIZE, "0")) or None
        L.info("Using %s=%s", self._BATCH_SIZE, self.batch_size)

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


def _with_batches(mapper, func, iterable, batch_size=None):
    """Wrapper on mapper function creating batches of iterable to give to mapper.

    The batch_size is an int corresponding to the number of evaluation in each batch/
    """
    if isinstance(iterable, Iterator):
        iterable = list(iterable)
    if batch_size is not None:
        iterables = np.array_split(iterable, len(iterable) // batch_size)
    else:
        iterables = [iterable]

    for _iterable in iterables:
        yield from mapper(func, _iterable)


class MultiprocessingFactory(ParallelFactory):
    """Parallel helper class using multiprocessing."""

    _CHUNKSIZE = "PARALLEL_CHUNKSIZE"

    def __init__(self):
        """Initialize multiprocessing factory."""

        super().__init__()
        self.pool = NestedPool()

    def get_mapper(self):
        """Get a NestedPool."""

        def _mapper(func, iterable):
            return _with_batches(self.pool.imap_unordered, func, iterable, self.batch_size)

        return _mapper

    def shutdown(self):
        """Close the pool."""
        self.pool.close()


class IPyParallelFactory(ParallelFactory):
    """Parallel helper class using ipyparallel."""

    _IPYTHON_PROFILE = "IPYTHON_PROFILE"

    def __init__(self):
        """Initialize the ipyparallel factory."""

        super().__init__()
        self.rc = None

    def get_mapper(self):
        """Get an ipyparallel mapper using the profile name provided."""
        profile = os.getenv(self._IPYTHON_PROFILE, "DEFAULT_IPYTHON_PROFILE")
        L.debug("Using %s=%s", self._IPYTHON_PROFILE, profile)
        self.rc = ipyparallel.Client(profile=profile)
        lview = self.rc.load_balanced_view()

        def _mapper(func, iterable):
            return _with_batches(
                partial(lview.imap, ordered=False), func, iterable, self.batch_size
            )

        return _mapper

    def shutdown(self):
        """Remove zmq."""
        if self.rc is not None:
            self.rc.close()


class DaskFactory(ParallelFactory):
    """Parallel helper class using dask."""

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
        super().__init__()

    def shutdown(self):
        """Retire the workers on the scheduler."""
        if not self.interactive:
            time.sleep(1)
            self.client.retire_workers()
            self.client = None

    def get_mapper(self):
        """Get a Dask mapper."""

        def _mapper(func, iterable):
            def _dask_mapper(func, iterable):
                futures = self.client.map(func, iterable)
                for _future, result in dask.distributed.as_completed(futures, with_results=True):
                    yield result

            return _with_batches(_dask_mapper, func, iterable, self.batch_size)

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
