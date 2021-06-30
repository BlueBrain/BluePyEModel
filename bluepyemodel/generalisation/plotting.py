"""Plotting functions."""
import pickle
from itertools import cycle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bluepyopt.ephys.responses import TimeVoltageResponse
from matplotlib.backends.backend_pdf import PdfPages
from neurom import load_neuron
from neurom import viewer
from tqdm import tqdm

from bluepyemodel.generalisation.ais_model import taper_function
from bluepyemodel.generalisation.evaluators import get_combo_hash
from bluepyemodel.generalisation.utils import get_emodels
from bluepyemodel.generalisation.utils import get_mtypes
from bluepyemodel.generalisation.utils import get_scores

matplotlib.use("Agg")


def plot_traces(trace_df, trace_path="traces", pdf_filename="traces.pdf"):
    """Plot traces from df, with highlighs on rows with trace_highlight = True.

    Args:
        trace_df (DataFrame): contains list of combos with traces to plot
        trace_path (str): path to folder with traces in .pkl
        pdf_filename (str): name of pdf to save
    """
    COLORS = cycle(["C{}".format(i) for i in range(10)])

    if "trace_highlight" not in trace_df.columns:
        trace_df["trace_highlight"] = True
    for index in trace_df.index:
        if trace_df.loc[index, "trace_highlight"]:
            c = next(COLORS)

        combo_hash = get_combo_hash(trace_df.loc[index])
        with open(Path(trace_path) / ("trace_id_" + str(combo_hash) + ".pkl"), "rb") as f:
            trace = pickle.load(f)
            for protocol, response in trace.items():
                if isinstance(response, TimeVoltageResponse):
                    if trace_df.loc[index, "trace_highlight"]:
                        label = trace_df.loc[index, "name"]
                        lw = 1
                        zorder = 1
                    else:
                        label = None
                        c = "0.5"
                        lw = 0.5
                        zorder = -1

                    plt.figure(protocol)
                    plt.plot(
                        response["time"],
                        response["voltage"],
                        label=label,
                        c=c,
                        lw=lw,
                        zorder=zorder,
                    )
                    # plt.gca().set_xlim(700, 800)

    with PdfPages(pdf_filename) as pdf:
        for fig_id in plt.get_fignums():
            fig = plt.figure(fig_id)
            plt.legend(loc="best")
            plt.suptitle(fig.get_label())
            pdf.savefig()
            plt.close()


def plot_ais_taper(data, model, ax=None):
    """Plot AIS taper."""
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    ax.plot(data["distances"], data["diameters"], ".", c="0.5", markersize=2)

    ax.plot(data["bins"], data["means"], "C0")
    ax.plot(data["bins"], np.array(data["means"]) - np.array(data["stds"]), "C0--")
    ax.plot(data["bins"], np.array(data["means"]) + np.array(data["stds"]), "C0--")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, np.max(data["diameters"]))
    ax.set_xlabel("Distance from soma")
    ax.set_ylabel("AIS diameter")

    ax.plot(
        data["bins"],
        taper_function(np.array(data["bins"]), *model["AIS"]["popt"][1:]),
        c="C1",
    )
    ax.set_title(
        "Taper strength: {:.3}, \n scale: {:.3},\n terminal diameter: {:.3}".format(
            *model["AIS"]["popt"][1:]
        ),
        fontsize=10,
        loc="left",
    )
    if ax is None:
        fig.savefig("ais_diameters.png", bbox_inches="tight")


def plot_ais_taper_models(models, pdf_filename="AIS_models.pdf"):
    """Create a pdf with all models of AIS and datapoints."""
    pdf = PdfPages(pdf_filename)
    for mtype in models:
        fig = plt.figure()
        fig.suptitle(mtype)
        plot_ais_taper(models[mtype]["data"], models[mtype], ax=plt.gca())
        pdf.savefig()
        plt.close()

    plt.figure()
    dists = np.linspace(0, 60, 100)
    plt.plot(
        models["all"]["data"]["distances"],
        models["all"]["data"]["diameters"],
        ".",
        c="0.5",
        markersize=0.5,
    )
    for mtype, model in models.items():
        plt.plot(dists, taper_function(dists, *model["AIS"]["popt"][1:]), lw=0.5, c="k")
    plt.plot(
        dists,
        taper_function(dists, *models["all"]["AIS"]["popt"][1:]),
        lw=2,
        c="r",
        label="model with all cells",
    )
    plt.legend()
    plt.axis([0, 60, 0, 3])
    plt.xlabel("distance from soma")
    plt.ylabel("diameter models per mtype")
    pdf.savefig()
    plt.close()
    pdf.close()


def plot_ais_resistance_models(fit_df, ais_models, pdf_filename="scan_scales.pdf"):
    """Plot the AIS resistance models."""
    emodels = fit_df.emodel.unique()
    with PdfPages(pdf_filename) as pdf:
        for emodel in emodels:
            for mtype in ais_models.keys():
                me_mask = fit_df.emodel == emodel
                if mtype != "all":
                    me_mask = me_mask & (fit_df.mtype == mtype)
                fit_df[me_mask].plot(x="AIS_scaler", y="rin_ais", marker="+")
                plt.plot(
                    fit_df[me_mask].AIS_scaler,
                    10
                    ** np.poly1d(ais_models[mtype]["resistance"][emodel]["polyfit_params"])(
                        np.log10(fit_df[me_mask].AIS_scaler)
                    ),
                    "-o",
                    ms=0.5,
                    label="fit",
                )
                plt.yscale("log")
                plt.xscale("log")
                plt.legend()
                plt.suptitle(mtype + "  " + emodel)
                plt.ylabel("AIS input resistance")

                pdf.savefig()
                plt.close()


def plot_target_rho_axon(
    rho_scan_df,
    target_rhos,
    pdf_filename="scan_rho.pdf",
    original_morphs_combos_path="../evaluate_original_cells/evaluation_results.csv",
):
    """Plot the results of the search for target rhow axon."""
    # pylint: disable=no-member
    mtypes = sorted(list(set(rho_scan_df.mtype.to_list())))
    emodels = sorted(list(set(rho_scan_df.emodel.to_list())))

    if original_morphs_combos_path is not None:
        all_morphs_combos_df = pd.read_csv(original_morphs_combos_path)
        all_morphs_combos_df = get_scores(all_morphs_combos_df)
    else:
        all_morphs_combos_df = None

    with PdfPages(pdf_filename) as pdf:
        for emodel in emodels:
            for mtype in mtypes:
                me_mask = (rho_scan_df.mtype == mtype) & (rho_scan_df.emodel == emodel)
                if len(rho_scan_df[me_mask]) > 0:
                    plt.figure()
                    ax = plt.gca()
                    scale_mask = rho_scan_df.AIS_scaler == 1
                    rho_scan_df[me_mask & ~scale_mask].plot(
                        x="rho_axon", y="median_score", ls="-", marker="+", ax=ax
                    )
                    rho_scan_df[me_mask & ~scale_mask].plot(
                        x="rho_axon", y="smooth_score", ls="-", ax=ax
                    )

                    ax.axvline(target_rhos[emodel][mtype], label="optimal rho", c="k")
                    ax.scatter(
                        rho_scan_df.loc[me_mask & scale_mask, "rho_axon"],
                        rho_scan_df.loc[me_mask & scale_mask, "median_score"],
                        c="r",
                        label="mean AIS model",
                    )
                    plt.xscale("log")
                    if all_morphs_combos_df is not None:
                        ax.scatter(
                            all_morphs_combos_df.loc[
                                (all_morphs_combos_df.emodel == emodel)
                                & all_morphs_combos_df.for_optimisation,
                                "rho_axon",
                            ],
                            all_morphs_combos_df.loc[
                                (all_morphs_combos_df.emodel == emodel)
                                & all_morphs_combos_df.for_optimisation,
                                "median_score",
                            ],
                            c="b",
                            label="original AIS",
                        )

                    plt.suptitle(emodel + "  " + mtype)
                    ax.set_ylim(0, 6)
                    plt.legend()
                    plt.ylabel("mean z-score of e-features")
                    pdf.savefig()
                    plt.close()


def plot_synth_ais_evaluations(
    morphs_combos_df,
    emodels="all",
    threshold=5,
    pdf_filename="evaluations.pdf",
):
    """Plot the results of ais synthesis evalutions."""
    mtypes = get_mtypes(morphs_combos_df, "all")
    emodels = get_emodels(morphs_combos_df, emodels)
    morphs_combos_df["median_score"] = morphs_combos_df["median_score"].clip(0.0, 2 * threshold)

    with PdfPages(pdf_filename) as pdf:
        for emodel in emodels:
            plt.figure()
            ax = plt.gca()

            mask = morphs_combos_df.emodel == emodel
            morphs_combos_df[mask & morphs_combos_df.for_optimisation].plot(
                x="rho_axon",
                y="median_score",
                c="r",
                ax=ax,
                label="optimized cell",
                ls="",
                marker=">",
                ms=10,
            )

            for mtype in mtypes:
                me_mask = (morphs_combos_df.emodel == emodel) & (morphs_combos_df.mtype == mtype)
                if len(morphs_combos_df[me_mask]) > 0:
                    if "ais_failed" in morphs_combos_df:
                        me_mask_no_failed = me_mask & (morphs_combos_df.ais_failed == 0)
                        me_mask_failed = me_mask & (morphs_combos_df.ais_failed == 1)
                    else:
                        me_mask_no_failed = me_mask
                        me_mask_failed = False * me_mask

                    morphs_combos_df[me_mask_no_failed].plot(
                        x="rho_axon",
                        y="median_score",
                        ls="",
                        marker="+",
                        ax=ax,
                        label=mtype,
                    )
                    if len(morphs_combos_df[me_mask_failed]) > 0:
                        morphs_combos_df[me_mask_failed].plot(
                            x="rho_axon",
                            y="median_score",
                            ls="",
                            marker=".",
                            ax=ax,
                            label=mtype + " (synth failed)",
                        )
            try:
                frac_pass_mean = len(
                    morphs_combos_df[mask][morphs_combos_df[mask].median_score < threshold].index
                ) / len(morphs_combos_df[mask].index)
            except ZeroDivisionError:
                frac_pass_mean = -1
            try:
                frac_pass_max = len(
                    morphs_combos_df[mask][morphs_combos_df[mask].max_score < threshold].index
                ) / len(morphs_combos_df[mask].index)
            except ZeroDivisionError:
                frac_pass_max = -1

            plt.axhline(threshold)
            plt.suptitle(
                emodel
                + ", fraction < "
                + str(threshold)
                + ": pass mean: "
                + str(np.round(frac_pass_mean, 3))
                + ", pass max: "
                + str(np.round(frac_pass_max, 3))
            )
            ax.set_ylim([0, 11.0])
            pdf.savefig()
            plt.close()


def _plot_neuron(selected_combos_df, cell_id, ax, color="k", morphology_path="morphology_path"):
    neuron = load_neuron(selected_combos_df.loc[cell_id, morphology_path])
    viewer.plot_neuron(ax, neuron, realistic_diameters=True)

    ax.set_title(
        selected_combos_df.loc[cell_id, "name"]
        + ",  mean score:"
        + str(np.round(selected_combos_df.loc[cell_id, "median_score"], 2)),
        color=color,
    )
    ax.set_aspect("equal", "box")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_rasterized(True)


def plot_non_selected_cells(
    selected_combos_df,
    emodel,
    pdf_filename="non_selected_cells.pdf",
    morphology_path="morphology_path",
):
    """Plot some selected cells and the non selectetd cells."""

    emodel_mask = selected_combos_df.emodel == emodel

    mtypes = sorted(list(set(selected_combos_df[emodel_mask].mtype)))

    with PdfPages(pdf_filename) as pdf:
        cell_ids = selected_combos_df[
            emodel_mask & (selected_combos_df.for_optimisation == 1)
        ].index
        if len(cell_ids) > 0:
            plt.figure()
            ax = plt.gca()
            _plot_neuron(
                selected_combos_df,
                cell_ids[0],
                ax,
                color="b",
                morphology_path=morphology_path,
            )
            plt.suptitle("mtype:" + selected_combos_df.loc[cell_ids[0], "mtype"])
            pdf.savefig()
            plt.close()

        for mtype in mtypes:
            mask = emodel_mask & (selected_combos_df.mtype == mtype)
            non_selected_cells = selected_combos_df[mask & ~selected_combos_df.selected].index
            selected_cells = selected_combos_df[mask & selected_combos_df.selected].index[
                : len(non_selected_cells)
            ]
            if len(non_selected_cells) > 0:

                _, axs = plt.subplots(
                    nrows=2,
                    ncols=len(non_selected_cells),
                    constrained_layout=True,
                    figsize=(max(len(non_selected_cells), len(selected_cells)) * 5, 5),
                )
                if len(non_selected_cells) > 1:
                    for cell_id, ax in tqdm(
                        zip(non_selected_cells, axs[0]), total=len(non_selected_cells)
                    ):
                        _plot_neuron(
                            selected_combos_df,
                            cell_id,
                            ax,
                            color="r",
                            morphology_path=morphology_path,
                        )
                elif len(non_selected_cells) == 1:
                    _plot_neuron(
                        selected_combos_df,
                        non_selected_cells[0],
                        axs[0],
                        color="r",
                        morphology_path=morphology_path,
                    )

                if len(selected_cells) > 1:
                    for cell_id, ax in tqdm(zip(selected_cells, axs[1]), total=len(selected_cells)):
                        _plot_neuron(
                            selected_combos_df,
                            cell_id,
                            ax,
                            color="k",
                            morphology_path=morphology_path,
                        )
                elif len(selected_cells) == 1:
                    _plot_neuron(
                        selected_combos_df,
                        selected_cells[0],
                        axs[1],
                        color="k",
                        morphology_path=morphology_path,
                    )

                plt.suptitle("mtype:" + mtype)
                pdf.savefig(dpi=500)
                plt.close()


def plot_summary_select(select_df, e_column="etype", select_column="selected"):
    """Plot fraction of selected cells."""
    select_df[select_column] = select_df[select_column].astype(int)
    select_plot_df = (
        select_df[["mtype", "emodel", "etype", select_column]]
        .groupby(["mtype", "etype", "emodel"])
        .mean()
    )

    if e_column == "etype":
        select_plot_df = select_plot_df.groupby(["mtype", "etype"]).max()

    select_plot_df = select_plot_df.reset_index()
    if e_column == "emodel":
        select_plot_df = select_plot_df.drop("etype", axis=1)

    select_plot_df = select_plot_df.pivot(index="mtype", columns=e_column, values=select_column)
    plt.figure(figsize=(20, 15))
    sns.heatmap(select_plot_df * 100, annot=True, fmt="3.0f")


def _create_plot_df(select_df, megate_df, e_column):
    """Create dataframe for failing features plots."""
    select_plot_df = megate_df.copy()
    select_plot_df[e_column] = select_df[e_column]
    return 1 - select_plot_df.fillna(0).set_index(e_column).groupby(e_column).mean()


def plot_feature_summary_select(select_df, megate_df, e_column="etype"):
    """Plot a summary of features scores."""
    plot_df = _create_plot_df(select_df, megate_df, e_column)
    plot_df.T.plot(kind="barh", stacked=True, figsize=(5, 40))


def plot_feature_select(select_df, megate_df, pdf, e_column="etype"):
    """Plot a summary of features scores, one figure for each e_col."""
    plot_df = _create_plot_df(select_df, megate_df, e_column)
    for e_col, df in plot_df.iterrows():
        plt.figure()
        ax = plt.gca()
        df.T.plot(kind="barh", figsize=(5, 40), ax=ax)
        plt.suptitle(e_column + ": " + e_col)
        plt.axvline(1.0, ls="-", c="r")
        pdf.savefig(bbox_inches="tight")
        plt.close()


def plot_frac_exceptions(select_df, e_column="emodel"):
    """Plot number of failed scores computations, when exceptions were raised."""
    select_plot_df = (
        select_df[[e_column, "scores_raw"]][select_df.scores_raw.isna()].groupby(e_column).size()
    )
    select_plot_df.T.plot(kind="barh", x=e_column)
