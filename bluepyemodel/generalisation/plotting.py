"""Plotting functions."""
import json
import pickle
from itertools import cycle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
import seaborn as sns
from bluepyopt.ephys.responses import TimeVoltageResponse
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from neurom.view import matplotlib_impl
from tqdm import tqdm

from bluepyemodel.generalisation.ais_model import taper_function
from bluepyemodel.generalisation.evaluators import get_combo_hash
from bluepyemodel.generalisation.utils import get_emodels
from bluepyemodel.generalisation.utils import get_mtypes

matplotlib.use("Agg")


def plot_traces(trace_df, trace_path="traces", pdf_filename="traces.pdf"):
    """Plot traces from df, with highlighs on rows with trace_highlight = True.

    Args:
        trace_df (DataFrame): contains list of combos with traces to plot
        trace_path (str): path to folder with traces in .pkl
        pdf_filename (str): name of pdf to save
    """
    COLORS = cycle([f"C{i}" for i in range(10)])

    if "trace_highlight" not in trace_df.columns:
        trace_df["trace_highlight"] = True
    for index in trace_df.index:
        if trace_df.loc[index, "trace_highlight"]:
            c = next(COLORS)

        if "trace_data" in trace_df.columns:
            trace_path = trace_df.loc[index, "trace_data"]
        else:
            combo_hash = get_combo_hash(trace_df.loc[index])
            trace_path = Path(trace_path) / ("trace_id_" + str(combo_hash) + ".pkl")

        with open(trace_path, "rb") as f:
            trace = pickle.load(f)
            if isinstance(trace, list):
                trace = trace[1]  # newer version the response are here
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
    # pylint: disable=consider-using-f-string

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
        taper_function(np.array(data["bins"]), *model["AIS_model"]["popt"][1:]),
        c="C1",
    )
    ax.set_title(
        "Taper strength: {:.3}, \n scale: {:.3},\n terminal diameter: {:.3}".format(
            *model["AIS_model"]["popt"][1:]
        ),
        fontsize=10,
        loc="left",
    )
    if ax is None:
        fig.savefig("ais_diameters.png", bbox_inches="tight")


def plot_soma_shape_models(models, pdf_filename="soma_shape_models.pdf"):
    """Plot soma shape models (surface area and radii)."""
    models = models["all"]
    with PdfPages(pdf_filename) as pdf:
        plt.figure()
        plt.hist(models["soma_data"]["soma_surfaces"], bins=20, label="data")
        plt.axvline(models["soma_model"]["soma_surface"], label="model", c="k")
        plt.xlabel("soma surface (NEURON)")
        plt.legend()
        pdf.savefig()

        plt.figure()
        plt.hist(models["soma_data"]["soma_radii"], bins=20, label="data")
        plt.axvline(models["soma_model"]["soma_radius"], label="model", c="k")
        plt.xlabel("soma radii (NeuroM)")
        plt.legend()
        pdf.savefig()


def plot_ais_taper_models(models, pdf_filename="AIS_models.pdf"):
    """Create a pdf with all models of AIS and datapoints."""
    pdf = PdfPages(pdf_filename)
    for mtype in models:
        fig = plt.figure()
        fig.suptitle(mtype)
        plot_ais_taper(models[mtype]["AIS_data"], models[mtype], ax=plt.gca())
        pdf.savefig()
        plt.close()

    plt.figure()
    dists = np.linspace(0, 60, 100)
    plt.plot(
        models["all"]["AIS_data"]["distances"],
        models["all"]["AIS_data"]["diameters"],
        ".",
        c="0.5",
        markersize=0.5,
    )
    for mtype, model in models.items():
        plt.plot(dists, taper_function(dists, *model["AIS_model"]["popt"][1:]), lw=0.5, c="k")
    plt.plot(
        dists,
        taper_function(dists, *models["all"]["AIS_model"]["popt"][1:]),
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


def plot_ais_resistance_models(fit_df, ais_models, pdf_filename="AIS_resistance_model.pdf"):
    """Plot the AIS resistance models."""
    plot_resistance_models(fit_df, ais_models, pdf_filename=pdf_filename, tpe="ais")


def plot_soma_resistance_models(fit_df, soma_models, pdf_filename="soma_resistance_model.pdf"):
    """Plot the AIS resistance models."""
    plot_resistance_models(fit_df, soma_models, pdf_filename=pdf_filename, tpe="soma")


def plot_resistance_models(fit_df, models, pdf_filename="resistance_model.pdf", tpe="ais"):
    """Plot the AIS resistance models."""
    if tpe == "ais":
        _tpe = "AIS"
    else:
        _tpe = tpe
    emodels = fit_df.emodel.unique()
    with PdfPages(pdf_filename) as pdf:
        for emodel in emodels:
            for mtype in models.keys():
                me_mask = fit_df.emodel == emodel
                if mtype != "all":
                    me_mask = me_mask & (fit_df.mtype == mtype)
                plt.figure(figsize=(5, 3))
                fit_df[me_mask].plot(x=f"{_tpe}_scaler", y=f"rin_{tpe}", marker="+", ax=plt.gca())
                plt.plot(
                    fit_df[me_mask][f"{_tpe}_scaler"],
                    10
                    ** np.poly1d(models[mtype]["resistance"][emodel]["polyfit_params"])(
                        np.log10(fit_df[me_mask][f"{_tpe}_scaler"])
                    ),
                    "-o",
                    ms=0.5,
                    label="fit",
                )
                plt.yscale("log")
                plt.xscale("log")
                plt.legend()
                plt.suptitle(mtype + "  " + emodel)
                plt.ylabel(f"{_tpe} input resistance")

                pdf.savefig(bbox_inches="tight")
                plt.close()


def plot_target_rhos(df, target_rhos, target_rho_axons, pdf_filename="scan_rho.pdf", clip=3):
    """Plot the results of the search for target rhow axon."""
    df["median_score"] = np.clip(df.median_score, 0, clip)
    with PdfPages(pdf_filename) as pdf:
        for emodel in df.emodel.unique():

            plt.figure(figsize=(5, 3))
            _df = df[df.for_optimisation]
            plt.scatter(df.rho, df.rho_axon, c=df.median_score, s=20)
            plt.scatter(_df.rho, _df.rho_axon, marker="+", c="r", s=30, label="exemplar")
            plt.scatter(
                target_rhos[emodel]["all"],
                target_rho_axons[emodel]["all"],
                marker="+",
                c="g",
                s=30,
                label="target",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(loc="best")

            plt.colorbar(label="median score")
            plt.xlabel("rho")
            plt.ylabel("rho axon")
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_synth_ais_evaluations(
    morphs_combos_df,
    emodels="all",
    threshold=5,
    pdf_filename="evaluations.pdf",
):
    """Plot the results of ais synthesis evaluations."""
    mtypes = get_mtypes(morphs_combos_df, "all")
    emodels = get_emodels(morphs_combos_df, emodels)
    morphs_combos_df["median_score"] = morphs_combos_df["median_score"].clip(0.0, 2 * threshold)

    with PdfPages(pdf_filename) as pdf:
        for emodel in emodels:
            plt.figure()
            ax = plt.gca()

            mask = morphs_combos_df.emodel == emodel
            score_df = pd.DataFrame()
            score_df["rho_axon"] = morphs_combos_df["rho_axon"]

            for score in json.loads(morphs_combos_df.loc[0, "scores_raw"]):
                score_df[score] = morphs_combos_df["scores_raw"].apply(
                    lambda s, score=score: json.loads(s)[score]
                )
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
            ax.set_ylim([0, 20.0])
            pdf.savefig()
            plt.close()


def _plot_neuron(selected_combos_df, cell_id, ax, color="k", morphology_path="morphology_path"):
    neuron = nm.load_morphology(selected_combos_df.loc[cell_id, morphology_path])
    matplotlib_impl.plot_morph(neuron, ax, realistic_diameters=True)

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


def plot_surface_comparison(surf_df, df, pdf_filename="surface_profile.pdf"):
    """Plot comparison of surface areas and median scores."""
    with PdfPages(pdf_filename) as pdf:
        mean = surf_df.mean(axis=0)
        if "median_score" in df.columns:
            cmappable = plt.cm.ScalarMappable(
                norm=Normalize(df.median_score.min(), df.median_score.max()), cmap="plasma"
            )

        plt.figure(figsize=(7, 4))
        for gid in surf_df.index:
            if "median_score" in df.columns:
                c = cmappable.to_rgba(df.loc[gid, "median_score"])
            else:
                c = "k"
            plt.plot(surf_df.columns, surf_df.loc[gid], c=c, lw=0.5)

        plt.plot(surf_df.columns, mean, c="r", lw=3, label="mean area")
        plt.plot(
            surf_df.columns,
            surf_df.loc[df.for_optimisation].to_numpy()[0],
            c="g",
            lw=3,
            label="exemplar",
        )
        if "median_score" in df.columns:
            plt.colorbar(cmappable, label="median score")
        plt.xlabel("path distance")
        plt.ylabel("surface area")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7, 4))
        for gid in surf_df.index:
            if "median_score" in df.columns:
                c = cmappable.to_rgba(df.loc[gid, "median_score"])
            else:
                c = "k"
            df.loc[gid, "diffs"] = np.linalg.norm(surf_df.loc[gid] - mean)
            plt.plot(surf_df.columns, surf_df.loc[gid] - mean, c=c, lw=0.5)
        plt.plot(
            surf_df.columns,
            surf_df.loc[df.for_optimisation].to_numpy()[0] - mean,
            c="g",
            lw=3,
            label="exemplar",
        )
        if "median_score" in df.columns:
            plt.colorbar(cmappable, label="median score")
        plt.xlabel("path distance")
        plt.ylabel("surface area - mean(surface area)")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        if "median_score" in df.columns:
            clip = 5
            plt.figure(figsize=(5, 3))
            plt.scatter(np.clip(df.median_score, 0, clip), df.diffs, label="adapted AIS/soma", s=10)
            plt.scatter(
                np.clip(df.generic_median_score, 0, clip),
                df.diffs,
                label="original AIS/SOMA",
                s=20,
                marker="+",
            )
            plt.legend(loc="best")
            plt.xlabel("median score")
            plt.ylabel("norm(surface area - mean)")
            plt.tight_layout()
            pdf.savefig()

            clip = 20
            plt.figure(figsize=(5, 3))
            plt.scatter(np.clip(df.max_score, 0, clip), df.diffs, label="adapted AIS/soma", s=10)
            plt.scatter(
                np.clip(df.generic_max_score, 0, clip),
                df.diffs,
                label="original AIS/SOMA",
                s=20,
                marker="+",
            )
            plt.legend(loc="best")
            plt.xlabel("max score")
            plt.ylabel("norm(surface area - mean)")
            plt.tight_layout()
            pdf.savefig()

            plt.figure(figsize=(5, 3))
            clip = 5
            plt.plot([0, clip], [0, clip], "k", lw=0.5)
            plt.scatter(
                np.clip(df.generic_median_score, 0, clip), np.clip(df.median_score, 0, clip), s=3
            )
            plt.xlabel("median original score")
            plt.ylabel("median adapted score")
            plt.axvline(2, ls="--", c="k", lw=0.5)
            plt.axhline(2, ls="--", c="k", lw=0.5)
            plt.axis([0, clip + 0.5, 0, clip + 0.5])
            plt.tight_layout()
            pdf.savefig()

            plt.figure(figsize=(5, 3))
            clip = 20
            plt.plot([0, clip], [0, clip], "k", lw=0.5)
            plt.scatter(np.clip(df.generic_max_score, 0, clip), np.clip(df.max_score, 0, clip), s=3)
            plt.xlabel("max original score")
            plt.ylabel("max adapted score")
            plt.axvline(5, ls="--", c="k", lw=0.5)
            plt.axhline(5, ls="--", c="k", lw=0.5)
            plt.axis([0, clip + 0.5, 0, clip + 0.5])
            plt.tight_layout()
            pdf.savefig()

        plt.close("all")
