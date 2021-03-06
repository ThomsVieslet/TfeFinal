import logging
import os

from balsam.core.models import ApplicationDefinition as AppDef
from balsam.core.models import BalsamJob
from balsam.launcher import dag
from balsam.launcher.futures import FutureTask
from balsam.launcher.futures import wait as balsam_wait

from deephyper.evaluator.evaluate import Evaluator
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction

logger = logging.getLogger(__name__)

# TODO(#30): "workers": should searcher be treated equivalently to evaluators?
LAUNCHER_NODES = int(os.environ.get("BALSAM_LAUNCHER_NODES", 1))
JOB_MODE = os.environ.get("BALSAM_JOB_MODE", "mpi")


class BalsamEvaluator(Evaluator):
    """Evaluator using Balsam software.

    Documentation to Balsam : https://balsam.readthedocs.io
    This class helps us to run task on HPC systems with more flexibility and ease of use.

    Args:
        run_function (func): takes one parameter of type dict and returns a scalar value.
        cache_key (func): takes one parameter of type dict and returns a hashable type,
            used as the key for caching evaluations. Multiple inputs that map to the same
            hashable key will only be evaluated once. If ``None``, then cache_key defaults
            to a lossless (identity) encoding of the input dict.
    """

    def __init__(self, run_function, cache_key=None, **kwargs):
        super().__init__(run_function, cache_key)
        self.id_key_map = {}
        # reserve 1 DeepHyper worker for searcher process
        if LAUNCHER_NODES == 1:
            # --job-mode=serial edge case where 2 ranks (Master, Worker) are placed on the node
            self.num_workers = self.WORKERS_PER_NODE - 1
            # 1 node case for --job-mode=mpi will result in search process occupying
            # entirety of the only node ---> no evaluator workers (also should have DEEPHYPER_WORKERS_PER_NODE=1)
        else:
            if JOB_MODE == "serial":
                # MPI ensemble Master rank0 occupies entirety of first node
                self.num_workers = (LAUNCHER_NODES - 1) * self.WORKERS_PER_NODE - 1
            if JOB_MODE == "mpi":
                # all nodes free, but restricted to 1 job=worker per node
                assert self.WORKERS_PER_NODE == 1
                self.num_workers = LAUNCHER_NODES * self.WORKERS_PER_NODE - 1
        logger.info("Balsam Evaluator instantiated")
        logger.debug(f"LAUNCHER_NODES = {LAUNCHER_NODES}")
        logger.debug(f"WORKERS_PER_NODE = {self.WORKERS_PER_NODE}")
        logger.debug(f"Total number of workers: {self.num_workers}")
        logger.info(f"Backend runs will use Python: {self.PYTHON_EXE}")
        self._init_app()
        if not self.run_returns_balsamjob:
            logger.info(f"Backend runs will execute function: {self.appName}")
        else:
            logger.info(
                f"Function: {self.appName} will directly create BalsamJob run tasks"
            )
        self.transaction_context = transaction.atomic

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        return balsam_wait(futures, timeout=timeout, return_when=return_when)

    def _init_app(self):
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        self.appName = ".".join((moduleName, funcName))

        if hasattr(self._run_function, "_balsamjob_spec"):
            self.run_returns_balsamjob = True
            return
        else:
            self.run_returns_balsamjob = False

        try:
            app = AppDef.objects.get(name=self.appName)
        except ObjectDoesNotExist:
            logger.info(
                f"ApplicationDefinition did not exist for {self.appName}; creating new app in BalsamDB"
            )
            app = AppDef(name=self.appName, executable=self._runner_executable)
            app.save()
        else:
            logger.info(
                f"BalsamEvaluator will use existing app {self.appName}: {app.executable}"
            )

    def _eval_exec(self, x):
        if self.run_returns_balsamjob:
            task = self._run_function(x)
        else:
            task = self._create_balsam_task(x)

        task.name = f"task{self.counter}"
        wf = dag.current_job.workflow
        task.workflow = wf if wf is not None else self.appName
        task.save()
        logger.debug(f"Created job {task.name}")
        logger.debug(f"Args: {task.args}")
        future = FutureTask(task, self._on_done, fail_callback=self._on_fail)
        future.task_args = task.args
        return future

    def _create_balsam_task(self, x):
        args = f"'{self.encode(x)}'"
        envs = f"KERAS_BACKEND={self.KERAS_BACKEND}"
        # envs = ":".join(f'KERAS_BACKEND={self.KERAS_BACKEND} OMP_NUM_THREADS=62 KMP_BLOCKTIME=0 KMP_AFFINITY=\"granularity=fine,compact,1,0\"'.split())
        resources = {
            "num_nodes": 1,
            "ranks_per_node": 1,
            "threads_per_rank": 64,
            "node_packing_count": self.WORKERS_PER_NODE,
        }
        for key in resources:
            if key in x:
                resources[key] = x[key]

        task = BalsamJob(
            application=self.appName, args=args, environ_vars=envs, **resources
        )
        return task

    @staticmethod
    def _on_done(job):
        if "dh_objective" in job.data:
            return job.data["dh_objective"]
        output = job.read_file_in_workdir(f"{job.name}.out")
        output = Evaluator._parse(output)
        return output

    @staticmethod
    def _on_fail(job):
        logger.info(f"Task {job.cute_id} failed; setting objective as float_max")
        return Evaluator.FAIL_RETURN_VALUE
