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

    axs[0, 0].plot(
        nevals, run["logbook"].select("min"), label="Minimum", ls="--", c="gray"
    )

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
            axs[0, 0].text(
                5.15, p - 0.25, s="{:.1f}".format(s), color="red", fontsize=8
            )

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
        len(traces_name), 1, figsize=(10, 2 + (1.5 * len(traces_name))), squeeze=False
    )

    axs_c = []
    for idx, t in enumerate(sorted(traces_name)):

        # Plot voltage
        axs[idx, 0].plot(responses[t]["time"], responses[t]["voltage"], color="black")

        axs[idx, 0].set_title(t)

        axs[idx, 0].set_xlabel("Time (ms)")
        axs[idx, 0].set_ylabel("Voltage (mV)")

        # Plot current
        basename = t.split(".")[0]
        if basename in stimuli:

            stim = stimuli[basename]

            axs_c.append(axs[idx, 0].twinx())
            axs_c[-1].set_xlabel("Time (ms)")
            axs_c[-1].set_ylabel("Current (pA)")

            if hasattr(stim, "thresh_perc") and threshold and holding:

                amp = holding + threshold * (float(stim.thresh_perc) / 100.0)

                ton = stim.step_stimulus.step_delay
                toff = stim.step_stimulus.step_delay + stim.step_stimulus.step_duration
                tend = stim.step_stimulus.total_duration

                axs_c[-1].plot([0.0, ton], [holding, holding], color="gray", alpha=0.6)
                axs_c[-1].plot([ton, ton], [holding, amp], color="gray", alpha=0.6)
                axs_c[-1].plot([ton, toff], [amp, amp], color="gray", alpha=0.6)
                axs_c[-1].plot([toff, toff], [amp, holding], color="gray", alpha=0.6)
                axs_c[-1].plot(
                    [toff, tend], [holding, holding], color="gray", alpha=0.6
                )

                axs_c[-1].set_ylim(min(holding, amp) - 0.2, max(holding, amp) + 0.2)

            else:
                pass

        idx += 1

    title = str(model["emodel"])
    if threshold:
        title += " ; Threshold current = {:.4f} pA".format(threshold)
    if holding:
        title += " ; Holding current = {:.4f} pA".format(holding)
    fig.suptitle(title)

    figure_name = "{}_{}_{}_traces.pdf".format(
        model["emodel"],
        model["seed"],
        model["optimizer"],
    )
    plt.tight_layout()
    save_fig(figures_dir, figure_name)
