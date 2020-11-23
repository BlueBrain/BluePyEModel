"""API to reproduce Singlecell repositories."""

import logging
from pathlib import Path
import numpy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager

from bluepyemodel.emodel_pipeline.utils import read_checkpoint, make_dir

# pylint: disable=W0612,W0102

matplotlib.rcParams["pdf.fonttype"] = 42

logger = logging.getLogger("__main__")


def save_fig(figures_dir, figure_name):
    """Save a matplotlib figure"""
    p = Path(figures_dir) / figure_name
    plt.savefig(str(p), dpi=100, bbox_inches="tight")
    plt.close("all")
    plt.clf()


def optimization(checkpoint_path="./checkpoint.pkl", figures_dir="./figures"):
    """Create plots related to a BluePyOpt optimization"""

    make_dir(figures_dir)
    run = read_checkpoint(checkpoint_path)

    nevals = numpy.cumsum(run["logbook"].select("nevals"))

    fig, axs = plt.subplots(1, figsize=(8, 8), squeeze=False)

    axs[0, 0].plot(nevals, run["logbook"].select("min"), label="Minimum", ls="--", c="gray")

    axs[0, 0].plot(nevals, run["logbook"].select("avg"), label="Average", c="gray")

    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("Number of evaluations")
    axs[0, 0].set_ylabel("Fitness")
    axs[0, 0].legend(loc="upper right", frameon=False)

    p = Path(checkpoint_path)
    figure_name = p.stem + ".pdf"
    plt.tight_layout()
    save_fig(figures_dir, figure_name)


def scores(model, figures_dir="./figures"):
    """Plot the scores of a model"""
    make_dir(figures_dir)

    score = list(model["scores"].values())
    scores_names = list(model["scores"].keys())
    pos = [*range(len(model["scores"]))]

    fig, axs = plt.subplots(1, figsize=(6, 0.8 + 0.150 * len(pos)), squeeze=False)

    axs[0, 0].barh(pos, score, height=0.7, align="center", color="gray")

    for p, s in zip(pos, score):
        if s > 5.0:
            axs[0, 0].text(5.15, p - 0.25, s="{:.1f}".format(s), color="red", fontsize=8)

    axs[0, 0].set_xlabel("z-score")

    axs[0, 0].set_yticks(pos)
    axs[0, 0].set_yticklabels(scores_names, size="small")

    axs[0, 0].set_xlim(0, 5)
    axs[0, 0].set_ylim(-0.5, len(pos) - 0.5)

    figure_name = "{}_{}_scores.pdf".format(
        model["emodel"],
        model["seed"],
    )
    plt.tight_layout()
    save_fig(figures_dir, figure_name)


def traces(model, responses, stimuli={}, figures_dir="./figures"):
    """Plot the traces of a model"""
    make_dir(figures_dir)

    traces_name = []
    threshold = None
    holding = None
    for resp_name, response in responses.items():

        if not (isinstance(response, float)):
            traces_name.append(resp_name)
        else:
            if "threshold" in resp_name:
                threshold = response
            elif "holding" in resp_name:
                holding = response

    fig, axs = plt.subplots(
        len(traces_name), 1, figsize=(10, 2 + (1.6 * len(traces_name))), squeeze=False
    )

    axs_c = []
    for idx, t in enumerate(sorted(traces_name)):

        axs[idx, 0].set_title(t)

        # Plot voltage
        axs[idx, 0].plot(responses[t]["time"], responses[t]["voltage"], color="black")
        axs[idx, 0].set_xlabel("Time (ms)")
        axs[idx, 0].set_ylabel("Voltage (mV)")

        # Plot current
        basename = t.split(".")[0]
        if basename in stimuli:

            if hasattr(stimuli[basename], "stimulus"):

                if hasattr(stimuli[basename], "thresh_perc") and threshold and holding:
                    stimuli[basename].stimulus.step_amplitude = threshold * (
                        float(stimuli[basename].thresh_perc) / 100.0
                    )
                    stimuli[basename].stimulus.holding_current = holding

                axs_c.append(axs[idx, 0].twinx())
                axs_c[-1].set_xlabel("Time (ms)")
                axs_c[-1].set_ylabel("Current (pA)")

                time, current = stimuli[basename].stimulus.generate()
                axs_c[-1].plot(time, current, color="gray", alpha=0.6)
                axs_c[-1].set_ylim(numpy.min(current) - 0.2, numpy.max(current) + 0.2)

        idx += 1

    title = str(model["emodel"])
    if threshold:
        title += " ; Threshold current = {:.4f} pA".format(threshold)
    if holding:
        title += " ; Holding current = {:.4f} pA".format(holding)
    fig.suptitle(title)

    figure_name = "{}_{}_traces.pdf".format(
        model["emodel"],
        model["seed"],
    )
    plt.tight_layout()
    save_fig(figures_dir, figure_name)


def parameters_distribution(models, lbounds, ubounds, figures_dir="./figures"):
    """Plot the distribution of the parameters across several models"""
    make_dir(figures_dir)

    if len({mo["emodel"] for mo in models}) > 1:
        logger.warning(
            "More than one e-type passed to the plotting.parameters_distribution function"
        )

    # Normalizes the parameters and makes sure they are listed in the same order
    data = []
    parameters = list(lbounds.keys())
    for mo in models:
        _ = []
        for param in parameters:
            bm = (ubounds[param] + lbounds[param]) / 2.0
            br = (ubounds[param] - lbounds[param]) / 2.0
            _.append((mo["parameters"][param] - bm) / br)
        data.append(_)
    data = numpy.array(data)

    fig, axs = plt.subplots(1, figsize=(0.8 + 0.21 * len(ubounds), 5), squeeze=False)

    v = axs[0, 0].violinplot(data)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in v:
            vp = v[partname]
            vp.set_edgecolor("black")
    for pc in v["bodies"]:
        pc.set_facecolor("black")

    axs[0, 0].set_xticks(ticks=range(1, 1 + len(ubounds)))
    axs[0, 0].set_xticklabels(labels=list(ubounds.keys()), rotation=90)

    axs[0, 0].plot([0, 1 + len(ubounds)], [-1, -1], c="black", ls="--", alpha=0.6, zorder=1)
    axs[0, 0].plot([0, 1 + len(ubounds)], [1, 1], c="black", ls="--", alpha=0.6, zorder=1)

    axs[0, 0].set_yticks(ticks=[-1, 1])
    axs[0, 0].set_yticklabels(labels=["Lower bounds", "Upper bounds"])

    axs[0, 0].set_xlim(0, 1 + len(ubounds))
    axs[0, 0].set_ylim(-1.05, 1.05)

    axs[0, 0].set_title(models[0]["emodel"])

    figure_name = "{}_parameters_distribution.pdf".format(models[0]["emodel"])
    plt.tight_layout()
    save_fig(figures_dir, figure_name)
