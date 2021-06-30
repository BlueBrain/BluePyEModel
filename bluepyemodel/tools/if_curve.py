"""Functions to compute and plot IF curves."""
import json
from functools import partial
from pathlib import Path

import bglibpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from tqdm import tqdm

from bluepyemodel.tools.bglibpy_helper import axon_loc
from bluepyemodel.tools.bglibpy_helper import get_cell
from bluepyemodel.tools.bglibpy_helper import get_spikefreq


def run_step_sim(
    cell,
    step_current,
    step_start=500,
    step_stop=1000,
    sim_end=1000,
    sim_dt=0.025,
    holding_current=None,
    cvode=False,
):
    """Run step protocol simulation."""

    if holding_current is None:
        holding_current = cell.hypamp

    cell.add_step(0, sim_end, holding_current)  # hypamp
    cell.add_step(step_start, step_stop, step_current)

    sim = bglibpy.Simulation()
    sim.run(
        sim_end,
        dt=sim_dt,
        celsius=34.0,
        v_init=-80.0,
        cvode=cvode,
    )

    results = pd.DataFrame()
    results["time"] = cell.get_time()
    results["voltage_axon"] = cell.get_recording(axon_loc(cell))
    results["voltage_soma"] = cell.get_soma_voltage()

    cell.persistent = []  # remove the step protocols for next run

    return results


def compute_if_curve(cell_kwargs, params=None, absolute=True, folder="traces"):
    """Compute if curve of a cell.

    With absolute=False, current is in firaction of threshold currentt"""
    cell = get_cell(**cell_kwargs)

    sim_params = dict(
        step_start=200,
        step_stop=600,
        sim_end=800,
        min_current=0.7,
        max_current=3,
        n_current=5,
    )
    if params is not None:
        sim_params.update(params)

    res_df = pd.DataFrame()
    res_df["current"] = np.linspace(
        sim_params["min_current"], sim_params["max_current"], sim_params["n_current"]
    )
    plt.figure()
    for i in res_df.index:
        results = run_step_sim(
            cell,
            res_df.loc[i, "current"] if absolute else res_df.loc[i, "current"] * cell.threshold,
            step_start=sim_params["step_start"],
            step_stop=sim_params["step_stop"],
            sim_end=sim_params["sim_end"],
        )
        # results.plot(x='time', y='voltage_axon', ax=plt.gca())
        results.plot(
            x="time", y="voltage_soma", ax=plt.gca(), label=f"current={res_df.loc[i, 'current']}"
        )
        res_df.loc[i, "freq"] = get_spikefreq(
            results, sim_params["step_start"], sim_params["step_stop"]
        )

    plt.legend()
    name = "cell_"
    if "morphology_name" in cell_kwargs:
        name += cell_kwargs["morphology_name"]
    if "gid" in cell_kwargs:
        name += str(cell_kwargs["gid"])
    Path(folder).mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(folder) / f"{name}.pdf")
    plt.close()
    return res_df


def _eval_if_curve(data, params=None, absolute=None, folder="traces"):
    """Wrapper for parallelisation."""
    res = compute_if_curve(
        json.loads(data["cell_kwargs"]), params=params, absolute=absolute, folder=folder
    )
    return {
        "current": json.dumps(res["current"].tolist()),
        "freq": json.dumps(res["freq"].tolist()),
    }


def compute_if_curves(
    cell_df, params=None, absolute=True, parallel="multiprocessing", folder="traces"
):
    """Compute if curve of several cell in parallel.

    With absolute=False, current is in fraction of threshold currentt.
    """
    new_columns = [["current", ""], ["freq", ""]]
    parallel_factory = init_parallel_factory(parallel)
    _eval = partial(_eval_if_curve, params=params, absolute=absolute, folder=folder)
    return evaluate(cell_df, _eval, new_columns, parallel_factory=parallel_factory)


def plot_if_curves(result, folder="if_curves"):
    """Plot if curve."""
    Path(folder).mkdir(exist_ok=True, parents=True)
    for gid in tqdm(result.index):
        df = pd.DataFrame()
        df["current"] = json.loads(result.loc[gid, "current"])
        df["freq"] = json.loads(result.loc[gid, "freq"])
        df.plot(x="current", y="freq")
        plt.savefig(Path(folder) / f"gid_{gid}.pdf")
        plt.close()


def plot_if_curve(spikecounts_df, filename="if_curve.pdf"):
    """Plot if curve."""
    spikecounts_df.plot(x="current", y="freq")
    plt.savefig(filename)
