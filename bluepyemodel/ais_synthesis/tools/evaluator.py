"""Module to evaluate generic functions on rows of combos dataframe (similar to BluePyMMM)."""
import logging
import sqlite3
import sys
import traceback
from functools import partial
import multiprocessing
import multiprocessing.pool
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    def _get_daemon(self):  # pylint: disable=R0201
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


def _try_evaluation(task, evaluation_function=None):
    """Encapsulate the evaluation function into a try/except and isolate to record exceptions."""
    task_id = task[0]
    try:
        result = evaluation_function(task[1])
        exception = ""

    except Exception:  # pylint: disable=broad-except
        result = None
        exception = "".join(traceback.format_exception(*sys.exc_info()))
        logger.exception("Exception for combo %s", exception)
    return task_id, result, exception


def _create_database(combos, new_columns, combos_db_filename="combos_db.sql"):
    """Create a sqlite database from combos dataframe."""
    combos["exception"] = None
    for new_column in new_columns:
        combos[new_column[0]] = new_column[1]
        combos["to_run_" + new_column[0]] = 1
    with sqlite3.connect(combos_db_filename) as combos_db:
        combos.to_sql("combos", combos_db, if_exists="replace", index_label="index")
    return combos


def _load_database_to_dataframe(combos_db_filename="combos_db.sql"):
    """Load an sql database and construct the dataframe."""
    with sqlite3.connect(combos_db_filename) as combos_db:
        out = pd.read_sql("SELECT * FROM combos", combos_db, index_col="index")
        return out


def _write_to_sql(combos_db_filename, task_id, results, new_columns, exception):
    """Write row data to sql."""
    with sqlite3.connect(combos_db_filename) as combos_db:
        for new_column in new_columns:
            res = results[new_column[0]] if results is not None else None
            combos_db.execute(
                "UPDATE combos SET " + new_column[0] + "=?, "
                "exception=?, to_run_" + new_column[0] + "=? WHERE `index`=?",
                (res, exception, 0, task_id),
            )


def _get_mapper(ipyp_profile=None):
    """Get an ipyparallel map if profile name provided, else a NestedPool"""
    if ipyp_profile is not None:
        from ipyparallel import Client

        rc = Client(profile=ipyp_profile)
        lview = rc.load_balanced_view()
        return partial(lview.imap, ordered=False)

    return partial(NestedPool(maxtasksperchild=1).imap_unordered, chunksize=1)


def evaluate_combos(
    combos_df,
    evaluation_function,
    new_columns,
    task_ids=None,
    continu=False,
    ipyp_profile=None,
    combos_db_filename="combos_db.sql",
):
    """Evaluate combos and save results in a sqlite database on the fly and return combos dataframe.

    Args:
        combos_df (DataFrame): each row contains information for the computation
        evaluation_function (function): function used to evaluate each row,
            should have a single argument as row of combos, and return a dict with keys
            corresponding to the names in new_columns
        new_columns (list): list of names of new column and empty value to save evaluation results,
            i.e.: [['result', 0.0], ['valid', False]]
        task_ids (int): index of combos_original to compute, if None, all will be computed
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        ipyp_profile (str): name of ipyparallel profile
        combos_db_filename (str): filename for the combos sqlite database
    Return:
        pandas.DataFrame: combod_df dataframe with new columns containing computed results
    """
    if ipyp_profile == "None":
        ipyp_profile = None

    if task_ids is None:
        task_ids = combos_df.index
    else:
        combos_df = combos_df.loc[task_ids]

    if continu:
        combos_to_evaluate = _load_database_to_dataframe(
            combos_db_filename=combos_db_filename
        )
        for new_column in new_columns:
            task_ids = task_ids[
                combos_df.loc[task_ids, "to_run_" + new_column[0]].to_numpy() == 1
            ]
    else:
        combos_to_evaluate = _create_database(
            combos_df, new_columns, combos_db_filename=combos_db_filename
        )

        # this is a hack to make it work, otherwise it does not update the entries correctly
        combos_to_evaluate = _load_database_to_dataframe(combos_db_filename)
        combos_to_evaluate = _create_database(
            combos_to_evaluate, new_columns, combos_db_filename=combos_db_filename
        )

    if len(task_ids) > 0:
        logger.info("%s combos to compute.", str(len(task_ids)))
    else:
        logger.warning("WARNING: No combos to compute, something may be wrong")

    mapper = _get_mapper(ipyp_profile=ipyp_profile)
    eval_func = partial(_try_evaluation, evaluation_function=evaluation_function)
    try:
        for task_id, results, exception in tqdm(
            mapper(eval_func, combos_to_evaluate.iterrows()), total=len(task_ids)
        ):
            _write_to_sql(
                combos_db_filename,
                task_id,
                results,
                new_columns,
                exception,
            )
    # to save dataframe even if program is killed
    except (KeyboardInterrupt, SystemExit):
        pass
    return _load_database_to_dataframe(combos_db_filename)
