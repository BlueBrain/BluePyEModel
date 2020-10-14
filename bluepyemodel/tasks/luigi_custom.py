"""CustomFromFile luigi worker and custom luigi launcher."""

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

import signal

import luigi

from bluepyemodel.tasks.emodel_creation.optimisation import EModelCreation


class WorkerCustom(luigi.worker.Worker):
    """CustomFromFile Worker class."""

    def handle_interrupt(self, signum, _):
        """
        Trigger event to gracefully exit.
        Gracefully exit optimise deap loop, and allow to store models.
        All other pending tasks (optimise, validation, emodelcreation) are skipped.
        """
        if signum == signal.SIGUSR1:
            self._gracefully_exit()

    def _gracefully_exit(self):
        """
        Trigger event to gracefullt exit.
        Stop gracefully all Optimize tasks both running or pending.
        Pending Validation and EModelCreation are also skipped.
        """
        for task in self._scheduled_tasks.values():
            # trigger event all tasks
            if hasattr(task, "graceful_killer"):
                task.graceful_killer.set()

        # Force luigi not to check if deps are fulfilled
        # in case some Optimize tasks do not produce output
        # so that StoreBestModels still store best models of tasks that have run.
        self._config.check_unfulfilled_deps = False


class FactoryCustom(object):
    """CustomFromFile Worker Scheduler Factory class."""

    def create_local_scheduler(self):
        return luigi.scheduler.Scheduler(prune_on_get_work=True, record_task_history=False)

    def create_remote_scheduler(self, url):
        return luigi.rpc.RemoteScheduler(url)

    def create_worker(self, scheduler, worker_processes, assistant=False):
        # return your worker instance
        return WorkerCustom(
            scheduler=scheduler, worker_processes=worker_processes, assistant=assistant
        )


if __name__ == "__main__":
    luigi.build(
        [EModelCreation(emodel="L5PC", species="rat", seed=11)],
        worker_scheduler_factory=FactoryCustom(),
        local_scheduler=True,
    )
