"""Functions related to the plotting of the e-models."""

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

import copy
import glob
import logging
from pathlib import Path

import efel
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.protocols import SweepProtocol
from bluepyopt.ephys.recordings import CompRecording
from bluepyopt.ephys.stimuli import NrnSquarePulse
from matplotlib import cm
from matplotlib import colors

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.evaluator import add_recordings_to_evaluator
from bluepyemodel.evaluation.protocols import ThresholdBasedProtocol
from bluepyemodel.tools.utils import make_dir
from bluepyemodel.tools.utils import parse_checkpoint_path
from bluepyemodel.tools.utils import read_checkpoint

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


def get_traces_ylabel(var):
    """Get ylabel for traces subplot."""
    if var == "v":
        return "Voltage (mV)"
    if var[0] == "i":
        return "Current (pA)"
    if var[-1] == "i":
        return "Ionic concentration (mM)"
    return ""


def get_recording_names(protocol_config, stimuli):
    """Get recording names which traces are to be plotted.

    Does not return extra ion / current recordings.

    Args:
        protocol_config (list): list of ProtocolConfiguration from FitnessCalculatorConfiguration
        stimuli (list): list of all protocols (protocols from configuration + pre-protocols)
    """
    # recordings from fitness calculator
    recording_names = {
        recording["name"] for prot in protocol_config for recording in prot.recordings_from_config
    }

    # expects recording names to have prot_name.location_name.variable structure
    prot_names = {rec_name.split(".")[0] for rec_name in recording_names}

    # add pre-protocol recordings
    # expects pre-protocol to only have 1 recording
    pre_prot_rec_names = {
        protocol.recordings[0].name
        for protocol in stimuli.values()
        if protocol.name not in prot_names
    }
    recording_names.update(pre_prot_rec_names)

    return recording_names


def get_traces_names_and_float_responses(responses, recording_names):
    """Extract the names of the traces to be plotted, as well as the float responses values."""

    traces_names = []
    threshold = None
    holding = None
    rmp = None
    rin = None

    for resp_name, response in responses.items():
        if not (isinstance(response, float)):
            if resp_name in recording_names:
                traces_names.append(resp_name)

        else:
            if resp_name == "bpo_threshold_current":
                threshold = response
            elif resp_name == "bpo_holding_current":
                holding = response
            elif resp_name == "bpo_rmp":
                rmp = response
            elif resp_name == "bpo_rin":
                rin = response

    return traces_names, threshold, holding, rmp, rin


def get_title(emodel, iteration, seed):
    """Returns 'emodel ; iteration={iteration} ; seed={seed}'

    Args:
        emodel (str): emodel name
        iteration (str): githash
        seed (int): random number seed
    """
    title = str(emodel)
    if iteration is not None:
        title += f" ; iteration = {iteration}"
    if seed is not None:
        title += f" ; seed = {seed}"
    return title


def optimisation(
    optimiser,
    emodel,
    iteration,
    seed,
    checkpoint_path="./checkpoint.pkl",
    figures_dir="./figures",
    write_fig=True,
):
    """Create plots related to a BluePyOpt optimisation"""

    make_dir(figures_dir)
    run, _ = read_checkpoint(checkpoint_path)

    ngen = run["logbook"].select("gen")
    is_finished_msg = ""
    if "CMA" in optimiser:
        is_finished_msg = f"is finished: {not run['CMA_es'].active}"

    legend_text = "\n".join(
        (
            f"min score = {min(run['logbook'].select('min')):.3f}",
            f"# of generations = {run['generation']}",
            f"evolution algorithm: {optimiser}",
            is_finished_msg,
        )
    )

    fig, axs = plt.subplots(1, figsize=(8, 8), squeeze=False)

    title = get_title(emodel, iteration, seed)
    axs[0, 0].set_title(title)

    axs[0, 0].plot(ngen, run["logbook"].select("min"), label="Minimum", c="black")

    axs[0, 0].plot(ngen, run["logbook"].select("avg"), label="Average", c="grey")

    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("Number of generations")
    axs[0, 0].set_ylabel("Fitness")
    axs[0, 0].legend(title=legend_text, loc="upper right", frameon=False)

    p = Path(checkpoint_path)

    figure_name = p.stem
    figure_name += ".pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, figure_name)

    return fig, axs


def _create_figure_parameter_histograms(
    histograms, evaluator, checkpoint_path, max_n_gen, gen_per_bin, figures_dir, write_fig
):
    """Create figure and plot the data for the evolution of the density of parameters."""

    ncols = 5
    nrows = len(evaluator.params) // ncols + 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 2 * nrows))
    axs = axs.flat

    # Plot the histograms
    for param_index, param in enumerate(evaluator.params):
        axs[param_index].imshow(
            100.0 * numpy.flip(histograms[param_index].T, 0), aspect="auto", interpolation="none"
        )

        axs[param_index].set_title(list(evaluator.param_names)[param_index])

        x_ticks_pos = [0, int(max_n_gen / gen_per_bin) - 1]
        x_ticks_label = [0, int(max_n_gen / gen_per_bin) * gen_per_bin]
        axs[param_index].set_xticks(x_ticks_pos, x_ticks_label)
        axs[param_index].set_yticks([0, 19], [param.bounds[1], param.bounds[0]])
        axs[param_index].set_xlim(0, int(max_n_gen / gen_per_bin) - 1)

    for axs_index in range(len(evaluator.params), len(axs)):
        axs[axs_index].set_visible(False)

    # Add a colorbar common to all subplots
    norm = colors.Normalize(vmin=0, vmax=100, clip=False)
    fig.colorbar(
        mappable=cm.ScalarMappable(norm=norm, cmap="viridis"),
        orientation="vertical",
        ax=axs[-1],
        label="% of population",
    )

    fig.supxlabel("Generations", size="xx-large")
    fig.supylabel("Parameter value", size="xx-large")

    suptitle = "Parameter evolution\n"
    metadata = parse_checkpoint_path(checkpoint_path)
    if metadata.get("emodel", None) is not None:
        suptitle += f"e-model = {metadata['emodel']}"
    if metadata.get("iteration", None) is not None:
        suptitle += f" ; iteration = {metadata['iteration']}"
    if metadata.get("seed", None) is not None:
        suptitle += f" ; seed = {metadata['seed']}"
    fig.suptitle(suptitle, size="xx-large")

    p = Path(checkpoint_path)

    figure_name = p.stem
    figure_name += "_evo_parameter_density.pdf"

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if write_fig:
        save_fig(figures_dir, figure_name)

    return fig, axs


def evolution_parameters_density(
    evaluator, checkpoint_paths, figures_dir="./figures", write_fig=True
):
    """Create plots of the evolution of the density of parameters in the population as the
    optimisation progresses. Create one plot per checkpoint plus one plot with all checkpoints.

    WARNING: This plotting function assumes that all the checkpoint files come from the same run
    and have the same parameters with the same bounds, therefore, that they come from the same
    evaluator. Do not use otherwise.

    Args:
        evaluator (CellEvaluator): evaluator used to evaluate the individuals.
        checkpoint_paths (list of str): list of paths to the checkpoints .pkl.
        figures_dir (str): path to the directory where the figures will be saved.
        write_fig (bool): whether to write the figures to disk.
    """

    make_dir(figures_dir)

    max_n_gen = 0
    genealogies = {}
    for checkpoint_path in checkpoint_paths:
        run, _ = read_checkpoint(checkpoint_path)
        if run["generation"] < 4:
            continue

        max_n_gen = max(max_n_gen, run["generation"])
        genealogies[checkpoint_path] = run["history"].genealogy_history

    gen_per_bin = 4
    pop_size = len(run["population"])
    histo_bins = (int(max_n_gen / gen_per_bin), 20)
    normalization_factor = gen_per_bin * pop_size

    # Compute and plot the histograms for each checkpoint
    sum_histograms = {}
    for checkpoint_path, genealogy in genealogies.items():
        # Get the histograms for all parameters
        histograms = {}
        for param_index in range(len(genealogy[1])):
            x = [(ind_idx - 1) // pop_size for ind_idx in genealogy.keys()]
            y = [ind[param_index] for ind in genealogy.values()]

            histo_range = [
                [0, max_n_gen],
                [evaluator.params[param_index].bounds[0], evaluator.params[param_index].bounds[1]],
            ]

            h, _, _ = numpy.histogram2d(x, y, bins=histo_bins, range=histo_range)
            normalized_h = h / normalization_factor

            histograms[param_index] = normalized_h
            if param_index not in sum_histograms:
                sum_histograms[param_index] = normalized_h
            else:
                sum_histograms[param_index] = sum_histograms[param_index] + normalized_h

        # Create the figure
        _ = _create_figure_parameter_histograms(
            histograms, evaluator, checkpoint_path, max_n_gen, gen_per_bin, figures_dir, write_fig
        )

    # Plot the figure with the sums of all histograms
    fig, axs = None, None
    if sum_histograms:
        sum_histograms = {idx: h / len(checkpoint_path) for idx, h in sum_histograms.items()}
        dummy_path = checkpoint_paths[0].partition("__seed=")[0] + "__all_seeds.pkl"
        fig, axs = _create_figure_parameter_histograms(
            sum_histograms, evaluator, dummy_path, max_n_gen, gen_per_bin, figures_dir, write_fig
        )

    return fig, axs


def scores(model, figures_dir="./figures", write_fig=True):
    """Plot the scores of a model"""
    SCORE_THRESHOLD = 5.0

    make_dir(figures_dir)

    score = list(model.scores.values()) + list(model.scores_validation.values())
    scores_names = list(model.scores.keys()) + list(model.scores_validation.keys())

    pos = [*range(len(score))]

    fig, axs = plt.subplots(1, figsize=(6, 0.8 + 0.150 * len(pos)), squeeze=False)

    axs[0, 0].barh(pos, score, height=0.7, align="center", color="gray")

    for p, s in zip(pos, score):
        if s > SCORE_THRESHOLD:
            axs[0, 0].text(5.15, p - 0.25, s="{:.1f}".format(s), color="red", fontsize=8)

    axs[0, 0].set_xlabel("z-score")

    axs[0, 0].set_yticks(pos)
    axs[0, 0].set_yticklabels(scores_names, size="small")

    axs[0, 0].set_xlim(0, 5)
    axs[0, 0].set_ylim(-0.5, len(pos) - 0.5)

    title = get_title(model.emodel_metadata.emodel, model.emodel_metadata.iteration, model.seed)
    # tweak size and placement so that title does not overcross figure
    fig.suptitle(title, size="medium", y=0.99)

    fname = model.emodel_metadata.as_string(model.seed) + "__scores.pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, fname)

    return fig, axs


def plot_traces_current(ax, time, current):
    """Plot the current trace on top of the voltage trace"""
    ax.plot(time, current, color="gray", alpha=0.6)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Stim Current (nA)")

    min_lim = numpy.min(current) - 0.2
    max_lim = numpy.max(current) + 0.2
    if numpy.isfinite(min_lim) and numpy.isfinite(max_lim):
        ax.set_ylim(min_lim, max_lim)


def traces_title(model, threshold=None, holding=None, rmp=None, rin=None):
    """Return the title for the traces figure"""
    title = str(model.emodel_metadata.emodel)
    title += f"\n iteration = {model.emodel_metadata.iteration} ; seed = {model.seed}"

    if threshold:
        title += "\n Threshold current = {:.4f} nA".format(threshold)
    if holding:
        title += " ; Holding current = {:.4f} nA".format(holding)
    if rmp:
        title += "\n Resting membrane potential = {:.2f} mV".format(rmp)
    if rin:
        title += " ; Input Resistance = {:.2f} MOhm".format(rin)

    return title


def traces(model, responses, recording_names, stimuli={}, figures_dir="./figures", write_fig=True):
    """Plot the traces of a model"""
    make_dir(figures_dir)

    traces_names, threshold, holding, rmp, rin = get_traces_names_and_float_responses(
        responses, recording_names
    )

    if not traces_names:
        return None, None

    fig, axs = plt.subplots(
        len(traces_names),
        1,
        figsize=(10, 2 + (1.5 * len(traces_names))),
        squeeze=False,
        layout="constrained",
    )

    axs_c = []
    for idx, t in enumerate(sorted(traces_names)):
        axs[idx, 0].set_title(t)

        if responses[t]:
            ylabel = get_traces_ylabel(var=t.split(".")[-1])

            # Plot responses (voltage, current, etc.)
            axs[idx, 0].plot(responses[t]["time"], responses[t]["voltage"], color="black")
            axs[idx, 0].set_xlabel("Time (ms)")
            axs[idx, 0].set_ylabel(ylabel)

            # Plot current
            basename = t.split(".")[0]
            if basename in stimuli:
                if hasattr(stimuli[basename], "stimulus"):
                    if (
                        isinstance(stimuli[basename], ThresholdBasedProtocol)
                        and threshold
                        and holding
                    ):
                        stimuli[basename].stimulus.holding_current = holding
                        stimuli[basename].stimulus.threshold_current = threshold

                    time, current = stimuli[basename].stimulus.generate()
                    if len(time) > 0 and len(current) > 0:
                        axs_c.append(axs[idx, 0].twinx())
                        plot_traces_current(axs_c[-1], time, current)

        idx += 1

    title = traces_title(model, threshold, holding, rmp, rin)
    fig.suptitle(title)

    fname = model.emodel_metadata.as_string(model.seed) + "__traces.pdf"

    if write_fig:
        save_fig(figures_dir, fname)

    return fig, axs


def _get_if_curve_from_evaluator(
    holding, threshold, model, evaluator, delay, length_step, delta_current, max_offset_current
):
    total_duration = length_step + (2 * delay)
    stim_end = delay + length_step

    efel.reset()
    efel.setIntSetting("strict_stiminterval", True)

    soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
    rec = CompRecording(name="Step1.soma.v", location=soma_loc, variable="v")

    holding_pulse = NrnSquarePulse(
        step_amplitude=holding,
        step_delay=0.0,
        step_duration=total_duration,
        total_duration=total_duration,
        location=soma_loc,
    )

    spike_freq_equivalent = []
    frequencies = []
    amps = numpy.arange(0.0, threshold + max_offset_current, delta_current)
    for amp in amps:
        step_pulse = NrnSquarePulse(
            step_amplitude=amp,
            step_delay=delay,
            step_duration=length_step,
            total_duration=total_duration,
            location=soma_loc,
        )

        protocol = SweepProtocol("Step1", [holding_pulse, step_pulse], [rec])
        evaluator.fitness_protocols = {"Step1": protocol}

        responses = evaluator.run_protocols(
            protocols=evaluator.fitness_protocols.values(), param_values=model.parameters
        )

        efel_trace = {
            "T": responses["Step1.soma.v"].response["time"],
            "V": responses["Step1.soma.v"].response["voltage"],
            "stim_start": [delay],
            "stim_end": [stim_end],
        }
        features = efel.getFeatureValues(
            [efel_trace], ["Spikecount", "mean_frequency"], raise_warnings=False
        )[0]
        spike_freq_equivalent.append(1e3 * float(features["Spikecount"]) / length_step)
        frequencies.append(features.get("mean_frequency", None))

    return amps, frequencies, spike_freq_equivalent


def IF_curve(
    model,
    responses,
    evaluator,
    delay=100,
    length_step=1000,
    delta_current=0.01,
    max_offset_current=0.2,
    figures_dir="./figures",
    write_fig=True,
):
    """Plot the current / frequency curve for the model"""

    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax2 = ax.twinx()

    holding = responses.get("bpo_holding_current", None)
    threshold = responses.get("bpo_threshold_current", None)
    if holding is None or threshold is None:
        logger.warning("Not plotting IF curve, holding or threshold current is missing")
        return fig, [ax, ax2]

    amps, frequencies, spike_freq_equivalent = _get_if_curve_from_evaluator(
        holding, threshold, model, evaluator, delay, length_step, delta_current, max_offset_current
    )

    ax.scatter(amps, frequencies, c="C0", alpha=0.6)
    ax.set_xlabel("Step amplitude (nA)")
    ax.set_ylabel("Mean frequency (Hz)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2.scatter(amps, spike_freq_equivalent, c="C1", alpha=0.6)
    ax2.set_ylabel("Spikecount per s over the step", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    title = f"IF curve {model.emodel_metadata.emodel}, seed {model.seed}"

    fig.suptitle(title)

    fname = model.emodel_metadata.as_string(model.seed) + "__IF_curve.pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, fname)

    return fig, [ax, ax2]


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

    figure_width = 2.5 + 0.08 * max(len(pn) for pn in ubounds.keys())
    figure_height = 0.8 + 0.21 * len(ubounds)
    fig, axs = plt.subplots(1, figsize=(figure_width, figure_height), squeeze=False)

    v = axs[0, 0].violinplot(data, vert=False)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in v:
            vp = v[partname]
            vp.set_edgecolor("black")
    for pc in v["bodies"]:
        pc.set_facecolor("black")

    axs[0, 0].set_yticks(ticks=range(1, 1 + len(ubounds)))
    axs[0, 0].set_yticklabels(labels=list(ubounds.keys()))

    # Guides for the eye
    for p in range(len(ubounds)):
        axs[0, 0].plot([-1, 1], [p + 1, p + 1], c="black", ls="dotted", alpha=0.7, lw=0.7, zorder=1)
    axs[0, 0].plot([-1, -1], [0, 1 + len(ubounds)], c="black", ls="--", alpha=0.8, zorder=1)
    axs[0, 0].plot([0, 0], [0, 1 + len(ubounds)], c="black", ls="--", alpha=0.8, zorder=1)
    axs[0, 0].plot([1, 1], [0, 1 + len(ubounds)], c="black", ls="--", alpha=0.8, zorder=1)

    axs[0, 0].set_xticks(ticks=[-1, 0, 1])
    axs[0, 0].set_xticklabels(labels=["Lower bounds", "50%", "Upper bounds"], rotation=90)

    axs[0, 0].set_ylim(0, 1 + len(ubounds))
    axs[0, 0].set_xlim(-1.05, 1.05)

    for _, spine in axs[0, 0].spines.items():
        spine.set_visible(False)

    title = str(models[0].emodel_metadata.emodel)
    title += f"; iteration = {models[0].emodel_metadata.iteration}"
    axs[0, 0].set_title(title)

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
    plot_if_curve=False,
    only_validated=False,
    save_recordings=False,
    load_from_local=False,
    cell_evaluator=None,
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
        plot_if_curve (bool): True to plot the current / frequency curve
        only_validated (bool): True to only plot validated models
        save_recordings (bool): Whether to save the responses data under a folder
            named `recordings`. Responses can then be loaded using load_from_local
            instead of being re-run.
        load_from_local (bool): True to load responses from locally saved recordings.
            Responses are saved locally when save_recordings is True.
        cell_evaluator (CellEvaluator): cell evaluator used to compute the responses.

    Returns:
        emodels (list): list of emodels.
    """

    figures_dir = Path(figures_dir)

    if cell_evaluator is None:
        cell_evaluator = get_evaluator_from_access_point(
            access_point,
            include_validation_protocols=True,
            record_ions_and_currents=plot_currentscape,
        )

    if (
        plot_currentscape
        and access_point.pipeline_settings.currentscape_config is not None
        and "current" in access_point.pipeline_settings.currentscape_config
        and "names" in access_point.pipeline_settings.currentscape_config["current"]
    ):
        add_recordings_to_evaluator(
            cell_evaluator,
            access_point.pipeline_settings.currentscape_config["current"]["names"],
            use_fixed_dt_recordings=False,
        )

    if plot_traces or plot_currentscape:
        emodels = compute_responses(
            access_point,
            cell_evaluator,
            mapper,
            seeds,
            store_responses=save_recordings,
            load_from_local=load_from_local,
        )
    else:
        emodels = access_point.get_emodels([access_point.emodel_metadata.emodel])
        if seeds:
            emodels = [model for model in emodels if model.seed in seeds]

    stimuli = cell_evaluator.fitness_protocols["main_protocol"].protocols

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
            recording_names = get_recording_names(
                access_point.get_fitness_calculator_configuration().protocols,
                stimuli,
            )
            traces(mo, mo.responses, recording_names, stimuli, figures_dir_traces)

        if plot_currentscape:
            config = access_point.pipeline_settings.currentscape_config
            figures_dir_currentscape = figures_dir / "currentscape" / dest_leaf
            currentscape(
                mo.responses,
                config=config,
                metadata_str=mo.emodel_metadata.as_string(mo.seed),
                figures_dir=figures_dir_currentscape,
                emodel=mo.emodel_metadata.emodel,
                iteration_tag=mo.emodel_metadata.iteration,
                seed=mo.seed,
            )
            # reset rcParams which are modified by currentscape
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        if plot_if_curve:
            figures_dir_traces = figures_dir / "traces" / dest_leaf
            IF_curve(
                mo, mo.responses, copy.deepcopy(cell_evaluator), figures_dir=figures_dir_traces
            )

    return emodels


def get_ordered_currentscape_keys(keys):
    """Get responses keys (also filename strings) ordered by protocols and locations.

    Arguments:
        keys (list of str): list of responses keys (or filename stems).
            Each item should have the shape protocol.location.current

    Returns:
        dict: containing voltage key, ion current and ionic concentration keys,
                and ion current and ionic concentration names. Should have the shape:

            .. code-block::

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
        # case where protocol has '.' in its name, e.g. IV_-100.0
        if len(n) == 4 and n[1].isdigit():
            n = [".".join(n[:2]), n[2], n[3]]
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


def currentscape(
    responses=None,
    output_dir=None,
    config=None,
    metadata_str="",
    figures_dir="./figures",
    emodel="",
    seed=None,
    iteration_tag=None,
):
    """Plot the currentscapes for all protocols.

    Arguments:
        responses (dict): dict containing the current and voltage responses.
        output_dur (str): path to the output dir containing the voltage and current responses.
            Will not be used if responses is set.
        config (dict): currentscape config. See currentscape package for more info.
        metadata_str (str): Metadata of the model as a string. Used in the files naming.
        figures_dir (str): path to the directory where to put the figures.
        emodel (str): name of the emodel
        iteration_tag (str): githash
        seed (int): random seed number
    """
    # pylint: disable=too-many-branches
    if responses is None and output_dir is None:
        raise TypeError("Responses or output directory must be set.")

    make_dir(figures_dir)

    if config is None:
        config = {}
    # copy dict so that changes to dict won't affect next call to this function
    updated_config = copy.deepcopy(config)

    if "current" not in updated_config:
        updated_config["current"] = {}
    if "ions" not in updated_config:
        updated_config["ions"] = {}
    if "output" not in updated_config:
        updated_config["output"] = {}

    current_subset = None
    if "names" in updated_config["current"]:
        current_subset = updated_config["current"]["names"].copy()

    if responses is not None:
        ordered_keys = get_ordered_currentscape_keys(
            key for key, item in responses.items() if item is not None
        )
    else:
        fnames = [
            str(Path(filepath).stem) for filepath in glob.glob(str(Path(output_dir) / "*.dat"))
        ]
        ordered_keys = get_ordered_currentscape_keys(fnames)

    for prot, locs in ordered_keys.items():
        for loc, key_dict in locs.items():
            if key_dict["voltage_key"] is None:
                continue
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

            name = ".".join((metadata_str, prot, loc))

            # adapt config
            if current_subset and key_dict["current_names"]:
                try:
                    currents_indices = [
                        list(key_dict["current_names"]).index(c_name) for c_name in current_subset
                    ]
                    currents = numpy.array(currents)[currents_indices]
                    updated_config["current"]["names"] = current_subset
                except ValueError:
                    logger.warning(
                        "Recorded currents do not match current names given in config. "
                        "Skipping currentscape plotting."
                    )
                    # skip plotting by having an empty current list
                    currents = numpy.array([])
                    updated_config["current"]["names"] = []
            else:
                updated_config["current"]["names"] = key_dict["current_names"]
            updated_config["ions"]["names"] = key_dict["ion_conc_names"]
            updated_config["output"]["savefig"] = True
            updated_config["output"]["fname"] = name
            if "dir" not in updated_config["output"]:
                updated_config["output"]["dir"] = figures_dir
            if "title" not in updated_config and emodel:
                title = get_title(emodel, iteration_tag, seed)
                updated_config["title"] = title

            if len(voltage) == 0 or len(currents) == 0:
                logger.warning(
                    "Could not plot currentscape for %s: voltage or currents is empty.", name
                )
            else:
                try:
                    from currentscape.currentscape import plot_currentscape as plot_currentscape_fct

                    logger.info("Plotting currentscape for %s", name)
                    fig = plot_currentscape_fct(
                        voltage, currents, updated_config, ions_data=ionic_concentrations, time=time
                    )
                    plt.close(fig)
                except ModuleNotFoundError:
                    logger.warning("Currentscape module not found. Skipping currentscape plotting.")
