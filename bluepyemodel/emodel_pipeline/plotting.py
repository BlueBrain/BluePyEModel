"""API to reproduce Singlecell repositories."""

import glob
import logging
from pathlib import Path

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy
from currentscape.currentscape import plot_currentscape as plot_currentscape_fct

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.protocols import BPEM_ThresholdProtocol
from bluepyemodel.optimisation.optimisation import read_checkpoint
from bluepyemodel.tools.utils import make_dir

# pylint: disable=W0612,W0102,C0209

matplotlib.rcParams["pdf.fonttype"] = 42

logger = logging.getLogger("__main__")
logging.getLogger("matplotlib").setLevel(level=logging.ERROR)


def save_fig(figures_dir, figure_name):
    """Save a matplotlib figure"""
    p = Path(figures_dir) / figure_name
    plt.savefig(str(p), dpi=100, bbox_inches="tight")
    plt.close("all")
    plt.clf()


def optimization(checkpoint_path="./checkpoint.pkl", figures_dir="./figures", write_fig=True):
    """Create plots related to a BluePyOpt optimization"""

    make_dir(figures_dir)
    run, _ = read_checkpoint(checkpoint_path)

    nevals = numpy.cumsum(run["logbook"].select("nevals"))

    fig, axs = plt.subplots(1, figsize=(8, 8), squeeze=False)

    axs[0, 0].plot(nevals, run["logbook"].select("min"), label="Minimum", ls="--", c="gray")

    axs[0, 0].plot(nevals, run["logbook"].select("avg"), label="Average", c="gray")

    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("Number of evaluations")
    axs[0, 0].set_ylabel("Fitness")
    axs[0, 0].legend(loc="upper right", frameon=False)

    p = Path(checkpoint_path)

    figure_name = p.stem
    figure_name += ".pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, figure_name)

    return fig, axs


def scores(model, figures_dir="./figures", write_fig=True):
    """Plot the scores of a model"""
    make_dir(figures_dir)

    score = list(model.scores.values()) + list(model.scores_validation.values())
    scores_names = list(model.scores.keys()) + list(model.scores_validation.keys())

    pos = [*range(len(score))]

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

    fname = model.emodel_metadata.as_string(model.seed) + "__scores.pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, fname)

    return fig, axs


def traces(model, responses, stimuli={}, figures_dir="./figures", write_fig=True):
    """Plot the traces of a model"""
    make_dir(figures_dir)

    traces_name = []
    threshold = None
    holding = None
    rmp = None
    rin = None

    for resp_name, response in responses.items():

        if not (isinstance(response, float)):
            if resp_name.split(".")[-1] != "v":
                continue
            traces_name.append(resp_name)

        else:

            if resp_name == "bpo_threshold_current":
                threshold = response
            elif resp_name == "bpo_holding_current":
                holding = response
            elif resp_name == "bpo_rmp":
                rmp = response
            elif resp_name == "bpo_rin":
                rin = response

    fig, axs = plt.subplots(
        len(traces_name), 1, figsize=(10, 2 + (1.6 * len(traces_name))), squeeze=False
    )

    axs_c = []
    for idx, t in enumerate(sorted(traces_name)):

        axs[idx, 0].set_title(t)

        if responses[t]:

            # Plot responses (voltage, current, etc.)
            axs[idx, 0].plot(responses[t]["time"], responses[t]["voltage"], color="black")
            axs[idx, 0].set_xlabel("Time (ms)")
            axs[idx, 0].set_ylabel("Voltage (mV)")

            # Plot current
            basename = t.split(".")[0]
            if basename in stimuli:

                if hasattr(stimuli[basename], "stimulus"):

                    if (
                        isinstance(stimuli[basename], BPEM_ThresholdProtocol)
                        and threshold
                        and holding
                    ):
                        stimuli[basename].stimulus.holding_current = holding
                        stimuli[basename].stimulus.threshold_current = threshold

                    axs_c.append(axs[idx, 0].twinx())
                    axs_c[-1].set_xlabel("Time (ms)")
                    axs_c[-1].set_ylabel("Stim Current (nA)")

                    time, current = stimuli[basename].stimulus.generate()
                    axs_c[-1].plot(time, current, color="gray", alpha=0.6)
                    axs_c[-1].set_ylim(numpy.min(current) - 0.2, numpy.max(current) + 0.2)

        idx += 1

    title = str(model.emodel_metadata.emodel)

    if threshold:
        title += "\n Threshold current = {:.4f} nA".format(threshold)
    if holding:
        title += " ; Holding current = {:.4f} nA".format(holding)
    if rmp:
        title += "\n Resting membrane potential = {:.2f} mV".format(rmp)
    if rin:
        title += " ; Input Resistance = {:.2f} MOhm".format(rin)

    fig.suptitle(title)

    fname = model.emodel_metadata.as_string(model.seed) + "__traces.pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, fname)

    return fig, axs


def parameters_distribution(models, lbounds, ubounds, figures_dir="./figures", write_fig=True):
    """Plot the distribution of the parameters across several models"""
    make_dir(figures_dir)

    if len({mo.emodel_metadata.as_string() for mo in models}) > 1:
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
            _.append((mo.parameters[param] - bm) / br)
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

    axs[0, 0].set_title(models[0].emodel_metadata.as_string())

    fname = models[0].emodel_metadata.as_string() + "__parameters_distribution.pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, fname)

    return fig, axs


def plot_models(
    access_point,
    mapper,
    seeds=None,
    figures_dir="./figures",
    plot_distributions=True,
    plot_scores=True,
    plot_traces=True,
    plot_currentscape=False,
    only_validated=False,
):
    """Plot the traces, scores and parameter distributions for all the models
        matching the emodels name.

    Args:
        access_point (DataAccessPoint): data access point.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        figures_dir (str): path of the directory in which the figures should be saved.
        plot_distributions (bool): True to plot the parameters distributions
        plot_scores (bool): True to plot the scores
        plot_traces (bool): True to plot the traces
        plot_currentscape (bool): True to plot the currentscapes
        only_validated (bool): True to only plot validated models

    Returns:
        emodels (list): list of emodels.
    """

    figures_dir = Path(figures_dir)

    cell_evaluator = get_evaluator_from_access_point(
        access_point,
        include_validation_protocols=True,
        use_fixed_dt_recordings=plot_currentscape,
        record_ions_and_currents=plot_currentscape,
    )

    if plot_traces or plot_currentscape:
        emodels = compute_responses(
            access_point, cell_evaluator, mapper, seeds, store_responses=plot_currentscape
        )
    else:
        emodels = access_point.get_emodels([access_point.emodel_metadata.emodel])
        if seeds:
            emodels = [model for model in emodels if model.seed in seeds]

    stimuli = cell_evaluator.fitness_protocols["main_protocol"].subprotocols()

    if only_validated:
        emodels = [model for model in emodels if model.passed_validation]
        dest_leaf = "validated"
    else:
        dest_leaf = "all"

    if not emodels:
        logger.warning("In plot_models, no emodel for %s", access_point.emodel_metadata.emodel)
        return []

    if plot_distributions:
        lbounds = {
            p.name: p.bounds[0]
            for p in cell_evaluator.cell_model.params.values()
            if p.bounds is not None
        }
        ubounds = {
            p.name: p.bounds[1]
            for p in cell_evaluator.cell_model.params.values()
            if p.bounds is not None
        }

        figures_dir_dist = figures_dir / "distributions" / dest_leaf

        parameters_distribution(
            models=emodels,
            lbounds=lbounds,
            ubounds=ubounds,
            figures_dir=figures_dir_dist,
        )

    for mo in emodels:
        if plot_scores:
            figures_dir_scores = figures_dir / "scores" / dest_leaf
            scores(mo, figures_dir_scores)
        if plot_traces:
            figures_dir_traces = figures_dir / "traces" / dest_leaf
            traces(mo, mo.responses, stimuli, figures_dir_traces)
        if plot_currentscape:
            config = access_point.pipeline_settings.currentscape_config
            figures_dir_currentscape = figures_dir / "currentscape" / dest_leaf
            currentscape(mo.responses, config=config, figures_dir=figures_dir_currentscape)

    return emodels


def get_ordered_currentscape_keys(keys):
    """Get responses keys (also filename strings) ordered by protocols and locations.

    Arguments:
        keys (list of str): list of responses keys (or filename stems).
            Each item should have the shape protocol.location.current

    Returns:
        dict: containing voltage key, ion current and ionic concentration keys,

        and ion current and ionic concentration names. Should have the shape:

        {
            "protocol_name": {
                "loc_name": {
                    "voltage_key": str, "current_keys": [], "current_names": [],
                    "ion_conc_keys": [], "ion_conc_names": [],
                }
            }
        }
    """
    # RMP and Rin only have voltage data, no currents, so they are skipped
    to_skip = [
        "RMPProtocol",
        "RinProtocol",
        "SearchHoldingCurrent",
        "bpo_rmp",
        "bpo_rin",
        "bpo_holding_current",
        "bpo_threshold_current",
    ]

    ordered_keys = {}
    for name in keys:
        n = name.split(".")
        prot_name = n[0]
        if prot_name not in to_skip:
            assert len(n) == 3
            loc_name = n[1]
            curr_name = n[2]

            if prot_name not in ordered_keys:
                ordered_keys[prot_name] = {}
            if loc_name not in ordered_keys[prot_name]:
                ordered_keys[prot_name][loc_name] = {
                    "voltage_key": None,
                    "current_keys": [],
                    "current_names": [],
                    "ion_conc_keys": [],
                    "ion_conc_names": [],
                }

            if curr_name == "v":
                ordered_keys[prot_name][loc_name]["voltage_key"] = name
            elif curr_name[0] == "i":
                ordered_keys[prot_name][loc_name]["current_keys"].append(name)
                ordered_keys[prot_name][loc_name]["current_names"].append(curr_name)
            elif curr_name[-1] == "i":
                ordered_keys[prot_name][loc_name]["ion_conc_keys"].append(name)
                ordered_keys[prot_name][loc_name]["ion_conc_names"].append(curr_name)

    return ordered_keys


def get_voltage_currents_from_files(key_dict, output_dir):
    """Get time, voltage, currents and ionic concentrations from output files"""
    v_path = Path(output_dir) / ".".join((key_dict["voltage_key"], "dat"))
    time = numpy.loadtxt(v_path)[:, 0]
    voltage = numpy.loadtxt(v_path)[:, 1]

    curr_paths = [
        Path(output_dir) / ".".join((curr_key, "dat")) for curr_key in key_dict["current_keys"]
    ]
    currents = [numpy.loadtxt(curr_path)[:, 1] for curr_path in curr_paths]

    ion_conc_paths = [
        Path(output_dir) / ".".join((ion_conc_key, "dat"))
        for ion_conc_key in key_dict["ion_conc_keys"]
    ]
    ionic_concentrations = [numpy.loadtxt(ion_conc_path)[:, 1] for ion_conc_path in ion_conc_paths]

    return time, voltage, currents, ionic_concentrations


def currentscape(responses=None, output_dir=None, config=None, figures_dir="./figures"):
    """Plot the currentscapes for all protocols.

    Arguments:
        responses (dict): dict containing the current and voltage responses.
        output_dur (str): path to the output dir containing the voltage and current responses.
            Will not be used if responses is set.
        config (dict): currentscape config. See currentscape package for more info.
        figures_dir (str): path to the directory where to put the figures.

    """
    if responses is None and output_dir is None:
        raise Exception("Responses or output directory must be set.")

    make_dir(figures_dir)

    if config is None:
        config = {}
    if "current" not in config:
        config["current"] = {}
    if "ions" not in config:
        config["ions"] = {}
    if "output" not in config:
        config["output"] = {}

    if responses is not None:
        ordered_keys = get_ordered_currentscape_keys(responses.keys())
    else:
        fnames = [
            str(Path(filepath).stem) for filepath in glob.glob(str(Path(output_dir) / "*.dat"))
        ]
        ordered_keys = get_ordered_currentscape_keys(fnames)

    for prot, locs in ordered_keys.items():
        for loc, key_dict in locs.items():
            if responses is not None:
                time = responses[key_dict["voltage_key"]]["time"]
                voltage = responses[key_dict["voltage_key"]]["voltage"]
                # current data has also voltage for a key
                currents = [responses[key]["voltage"] for key in key_dict["current_keys"]]
                ionic_concentrations = [
                    responses[key]["voltage"] for key in key_dict["ion_conc_keys"]
                ]
            else:
                time, voltage, currents, ionic_concentrations = get_voltage_currents_from_files(
                    key_dict, output_dir
                )

            name = ".".join((prot, loc))

            # adapt config
            config["current"]["names"] = key_dict["current_names"]
            config["ions"]["names"] = key_dict["ion_conc_names"]
            config["output"]["savefig"] = True
            config["output"]["fname"] = name
            if "dir" not in config["output"]:
                config["output"]["dir"] = figures_dir

            if len(voltage) == 0 or len(currents) == 0:
                logger.warning("Could not plot currentscape for %s: voltage or currents is empty.", name)
            else:
                logger.info("Plotting currentscape for %s", name)
                plot_currentscape_fct(
                    voltage, currents, config, ions_data=ionic_concentrations, time=time
                )
