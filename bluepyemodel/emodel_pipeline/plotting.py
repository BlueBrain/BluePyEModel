"""Functions related to the plotting of the e-models."""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

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
import re
from pathlib import Path

import bluepyefe
import efel
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy
from bluepyopt.ephys.protocols import SweepProtocol
from bluepyopt.ephys.recordings import CompRecording
from bluepyopt.ephys.stimuli import NrnSquarePulse
from matplotlib import cm
from matplotlib import colors

from bluepyemodel.data.utils import read_dendritic_data
from bluepyemodel.efeatures_extraction.efeatures_extraction import read_extraction_output
from bluepyemodel.efeatures_extraction.efeatures_extraction import read_extraction_output_cells
from bluepyemodel.efeatures_extraction.efeatures_extraction import read_extraction_output_protocols
from bluepyemodel.emodel_pipeline.plotting_utils import extract_experimental_data_for_IV_curve
from bluepyemodel.emodel_pipeline.plotting_utils import fill_in_IV_curve_evaluator
from bluepyemodel.emodel_pipeline.plotting_utils import get_experimental_FI_curve_for_plotting
from bluepyemodel.emodel_pipeline.plotting_utils import get_impedance
from bluepyemodel.emodel_pipeline.plotting_utils import get_ordered_currentscape_keys
from bluepyemodel.emodel_pipeline.plotting_utils import get_original_protocol_name
from bluepyemodel.emodel_pipeline.plotting_utils import get_recording_names
from bluepyemodel.emodel_pipeline.plotting_utils import get_simulated_FI_curve_for_plotting
from bluepyemodel.emodel_pipeline.plotting_utils import get_sinespec_evaluator
from bluepyemodel.emodel_pipeline.plotting_utils import get_title
from bluepyemodel.emodel_pipeline.plotting_utils import get_traces_names_and_float_responses
from bluepyemodel.emodel_pipeline.plotting_utils import get_traces_ylabel
from bluepyemodel.emodel_pipeline.plotting_utils import get_voltage_currents_from_files
from bluepyemodel.emodel_pipeline.plotting_utils import plot_fi_curves
from bluepyemodel.emodel_pipeline.plotting_utils import rel_to_abs_amplitude
from bluepyemodel.emodel_pipeline.plotting_utils import save_fig
from bluepyemodel.emodel_pipeline.plotting_utils import update_evaluator
from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.evaluator import PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import add_recordings_to_evaluator
from bluepyemodel.evaluation.evaluator import soma_loc
from bluepyemodel.evaluation.protocols import ThresholdBasedProtocol
from bluepyemodel.evaluation.utils import define_bAP_feature
from bluepyemodel.evaluation.utils import define_bAP_protocol
from bluepyemodel.evaluation.utils import define_EPSP_feature
from bluepyemodel.evaluation.utils import define_EPSP_protocol
from bluepyemodel.model.morphology_utils import get_basal_and_apical_max_radial_distances
from bluepyemodel.tools.utils import existing_checkpoint_paths
from bluepyemodel.tools.utils import get_amplitude_from_feature_key
from bluepyemodel.tools.utils import make_dir
from bluepyemodel.tools.utils import read_checkpoint
from bluepyemodel.tools.utils import select_rec_for_thumbnail

# pylint: disable=W0612,W0102,C0209

matplotlib.rcParams["pdf.fonttype"] = 42

logger = logging.getLogger("__main__")
logging.getLogger("matplotlib").setLevel(level=logging.ERROR)

colours = {
    "datapoint": "orangered",
    "dataline": "red",
    "modelpoint_apical": "cornflowerblue",
    "modelline_apical": "darkblue",
    "modelpoint_basal": "mediumseagreen",
    "modelline_basal": "darkgreen",
}


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
    figure_name += "__optimisation.pdf"

    plt.tight_layout()

    if write_fig:
        save_fig(figures_dir, figure_name)

    return fig, axs


def _create_figure_parameter_histograms(
    histograms,
    evaluator,
    metadata,
    seed,
    max_n_gen,
    gen_per_bin,
    figures_dir,
    write_fig,
):
    """Create figure and plot the data for the evolution of the density of parameters."""

    ncols = 5
    nrows = len(evaluator.params) // ncols + 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 2 * nrows))
    axs = axs.flat

    # Plot the histograms
    for param_index, param in enumerate(evaluator.params):
        axs[param_index].imshow(
            100.0 * numpy.flip(histograms[param_index].T, 0),
            aspect="auto",
            interpolation="none",
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
    if metadata.emodel is not None:
        suptitle += f"e-model = {metadata.emodel}"
    if metadata.iteration is not None:
        suptitle += f" ; iteration = {metadata.iteration}"
    if seed is not None:
        suptitle += f" ; seed = {seed}"
    fig.suptitle(suptitle, size="xx-large")

    figure_name = metadata.as_string(seed=seed)
    figure_name += "__evo_parameter_density.pdf"

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if write_fig:
        save_fig(figures_dir, figure_name)

    return fig, axs


def evolution_parameters_density(
    evaluator, checkpoint_paths, metadata, figures_dir="./figures", write_fig=True
):
    """Create plots of the evolution of the density of parameters in the population as the
    optimisation progresses. Create one plot per checkpoint plus one plot with all checkpoints.

    WARNING: This plotting function assumes that all the checkpoint files come from the same run
    and have the same parameters with the same bounds, therefore, that they come from the same
    evaluator. Do not use otherwise.

    Args:
        evaluator (CellEvaluator): evaluator used to evaluate the individuals.
        checkpoint_paths (list of str): list of paths to the checkpoints .pkl.
        metadata (EModelMetadata): metadata of the emodel.
        figures_dir (str): path to the directory where the figures will be saved.
        write_fig (bool): whether to write the figures to disk.
    """

    make_dir(figures_dir)

    max_n_gen = 0
    genealogies = {}
    for checkpoint_path in checkpoint_paths:
        run, seed = read_checkpoint(checkpoint_path)
        if run["generation"] < 4:
            continue

        max_n_gen = max(max_n_gen, run["generation"])
        genealogies[checkpoint_path] = (run["history"].genealogy_history, seed)

    if max_n_gen >= 8:
        gen_per_bin = 4
    else:
        gen_per_bin = 1

    pop_size = len(run["population"])
    histo_bins = (int(max_n_gen / gen_per_bin), 20)
    normalization_factor = gen_per_bin * pop_size

    # Compute and plot the histograms for each checkpoint
    sum_histograms = {}
    for checkpoint_path, (genealogy, seed) in genealogies.items():
        # Get the histograms for all parameters
        histograms = {}
        for param_index in range(len(genealogy[1])):
            x = [(ind_idx - 1) // pop_size for ind_idx in genealogy.keys()]
            y = [ind[param_index] for ind in genealogy.values()]

            histo_range = [
                [0, max_n_gen],
                [
                    evaluator.params[param_index].bounds[0],
                    evaluator.params[param_index].bounds[1],
                ],
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
            histograms,
            evaluator,
            metadata,
            seed,
            max_n_gen,
            gen_per_bin,
            figures_dir,
            write_fig,
        )

    # Plot the figure with the sums of all histograms
    fig, axs = None, None
    if sum_histograms:
        sum_histograms = {idx: h / len(checkpoint_path) for idx, h in sum_histograms.items()}
        fig, axs = _create_figure_parameter_histograms(
            sum_histograms,
            evaluator,
            metadata,
            "all_seeds",
            max_n_gen,
            gen_per_bin,
            figures_dir,
            write_fig,
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


def thumbnail(
    model,
    responses,
    recording_names,
    figures_dir="./figures",
    write_fig=True,
    dpi=300,
    thumbnail_rec=None,
):
    """Plot the trace figure to use as thumbnail."""
    make_dir(figures_dir)

    trace_name = select_rec_for_thumbnail(recording_names, thumbnail_rec=thumbnail_rec)

    # in case e.g. the run fails during preprotocols
    try:
        time = responses[trace_name]["time"]
        voltage = responses[trace_name]["voltage"]
    except KeyError:
        logger.warning(
            "Could not find protocol %s in responses. Skipping thumbnail plotting.",
            trace_name,
        )
        return None, None

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ylabel = get_traces_ylabel(var=trace_name.split(".")[-1])
    ax.plot(time, voltage, color="black")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)

    fname = model.emodel_metadata.as_string(model.seed) + "__thumbnail.png"

    if write_fig:
        save_fig(figures_dir, fname, dpi=dpi)

    return fig, ax


def traces(
    model,
    responses,
    recording_names,
    stimuli={},
    figures_dir="./figures",
    write_fig=True,
):
    """Plot the traces of a model"""
    # pylint: disable=too-many-nested-blocks
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
            basename = re.split(r"\.(?=[a-zA-Z])", t, 1)[0]
            if basename in stimuli:
                prot = stimuli[basename]
                if hasattr(prot, "stimulus"):
                    if isinstance(prot, ThresholdBasedProtocol) and threshold and holding:
                        prot.stimulus.holding_current = holding
                        prot.stimulus.threshold_current = threshold
                        if (
                            hasattr(prot, "dependencies")
                            and prot.dependencies["stimulus.holding_current"][1]
                            != "bpo_holding_current"
                        ):
                            resp_name = prot.dependencies["stimulus.holding_current"][1]
                            if resp_name in responses:
                                prot.stimulus.holding_current = responses[resp_name]
                            else:
                                continue
                        if (
                            hasattr(prot, "dependencies")
                            and prot.dependencies["stimulus.threshold_current"][1]
                            != "bpo_threshold_current"
                        ):
                            resp_name = prot.dependencies["stimulus.threshold_current"][1]
                            if resp_name in responses:
                                prot.stimulus.threshold_current = responses[resp_name]
                            else:
                                continue

                    time, current = prot.stimulus.generate()
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


def dendritic_feature_plot(
    model, responses, feature, feature_name, figures_dir="./figures", write_fig=True
):
    """Plots a accross dendrites and compare it with experimental data.

    Args:
        model (bluepyopt.ephys.CellModel): cell model
        responses (dict): responses of the cell model
        feature (DendFitFeature): feature to plot
        feature_name (str): which feature to plot. Can be either 'ISI_CV' or 'rheobase'.
        figures_dir (str or Path): Where to save the figures.
        write_fig (bool): whether to save the figure

    Returns a figure and its single axe.
    """
    make_dir(figures_dir)

    if feature_name == "ISI_CV":
        exp_label = "exp. data (Shai et al. 2015)"
        y_label = "ISI CV"
        fig_title = "ISI CV along the apical dendrite main branch"
    elif feature_name == "rheobase":
        exp_label = "exp. data (Beaulieu-Laroche (2021))"
        y_label = "rheobase (nA)"
        fig_title = "rheobase along the apical dendrite main branch"
    else:
        raise ValueError(f"Expected 'ISI_CV' or 'rheobase' for feature_name. Got {feature_name}")

    distances, feat_values = feature.get_distances_feature_values(responses)
    exp_distances, exp_values = read_dendritic_data(feature_name)

    min_dist = min((distances[0], exp_distances[0]))
    max_dist = max((distances[-1], exp_distances[-1]))

    # model fit
    if 0 in distances:
        slope = numpy.array([feature.fit(distances, feat_values)])
        x_fit = numpy.linspace(min_dist, max_dist, num=20)
        y_fit = feature.linear_fit(x_fit, slope)

    # data fit
    data_fit = numpy.polyfit(exp_distances, exp_values, 1)
    x_data_fit = numpy.linspace(min_dist, max_dist, num=20)
    y_data_fit = numpy.poly1d(data_fit)(x_data_fit)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(exp_distances, exp_values, c=colours["datapoint"], label=exp_label)
    ax.plot(x_data_fit, y_data_fit, c=colours["dataline"], label="data fit")
    ax.scatter(distances, feat_values, c=colours["modelpoint_apical"], label="emodel")
    if 0 in distances:
        ax.plot(x_fit, y_fit, "--", c=colours["modelline_apical"], label="emodel fit")
    ax.set_xlabel(r"distance from soma ($\mu$m)")
    ax.set_ylabel(y_label)
    ax.legend(fontsize="x-small")

    fig.suptitle(fig_title)

    if write_fig:
        fname = model.emodel_metadata.as_string(model.seed) + f"__{feature.name}.pdf"
        # can have '[', ']' and ',' in feature name. Replace those characters
        fname = fname.replace("[", "_")
        fname = fname.replace("]", "_")
        fname = fname.replace(",", "_")
        save_fig(figures_dir, fname)

    return fig, ax


def dendritic_feature_plots(mo, feature_name, dest_leaf, figures_dir="./figures"):
    """Calls dendritic_feature_plot for all features corresponding to feature_name.

    Args:
        mo (bluepyopt.ephys.CellModel): cell model
        feature_name (str): which feature to plot. Can be either 'ISI_CV' or 'rheobase'.
        dest_leaf (str): name of repo to use in output path. Usually either 'validated' or 'all'.
        figures_dir (str or Path): base directory where to save the figures.
    """
    figures_dir = Path(figures_dir)
    # translate feature_name into whatever name we expect to be present in fitness_calculator
    if feature_name == "ISI_CV":
        efeature_name = "ISI_CV_linear"
    elif feature_name == "rheobase":
        efeature_name = "bpo_threshold_current_linear"
    else:
        raise ValueError(f"Expected 'ISI_CV' or 'rheobase' for feature_name. Got {feature_name}")

    figures_dir_dendritic = figures_dir / "dendritic" / dest_leaf
    dend_feat_list = [
        obj.features[0]
        for obj in mo.evaluator.fitness_calculator.objectives
        if efeature_name in obj.features[0].name
    ]
    if len(dend_feat_list) < 1:
        logger.debug(
            "Could not find any feature with %s feature name for emodel %s.",
            efeature_name,
            mo.emodel_metadata.emodel,
        )
    for feat in dend_feat_list:
        # prevent stimulus current to be None if load_from_local is True
        # it is not used in this feature computation, but can make the plot crash if None
        if (
            feat.stimulus_current is None
            or callable(feat.stimulus_current)
            and feat.stimulus_current() is None
        ):
            feat.stimulus_current = 0.0

        dendritic_feature_plot(mo, mo.responses, feat, feature_name, figures_dir_dendritic)


def _get_fi_curve_from_evaluator(
    holding,
    threshold,
    model,
    evaluator,
    delay,
    length_step,
    delta_current,
    max_offset_current,
):
    total_duration = length_step + (2 * delay)
    stim_end = delay + length_step

    efel.reset()
    efel.set_int_setting("strict_stiminterval", True)

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
            protocols=evaluator.fitness_protocols.values(),
            param_values=model.parameters,
        )

        efel_trace = {
            "T": responses["Step1.soma.v"].response["time"],
            "V": responses["Step1.soma.v"].response["voltage"],
            "stim_start": [delay],
            "stim_end": [stim_end],
        }
        features = efel.get_feature_values(
            [efel_trace], ["Spikecount", "mean_frequency"], raise_warnings=False
        )[0]
        spike_freq_equivalent.append(1e3 * float(features["Spikecount"]) / length_step)
        mean_freq = features.get("mean_frequency", None)
        frequencies.append(mean_freq[0] if mean_freq is not None else None)

    frequencies = numpy.array(frequencies, dtype=float)
    return amps, frequencies, spike_freq_equivalent


def FI_curve(
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
        logger.warning("Not plotting FI curve, holding or threshold current is missing")
        return fig, [ax, ax2]

    amps, frequencies, spike_freq_equivalent = _get_fi_curve_from_evaluator(
        holding,
        threshold,
        model,
        evaluator,
        delay,
        length_step,
        delta_current,
        max_offset_current,
    )

    ax.scatter(amps, frequencies, c="C0", alpha=0.6)
    ax.set_xlabel("Step amplitude (nA)")
    ax.set_ylabel("Mean frequency (Hz)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2.scatter(amps, spike_freq_equivalent, c="C1", alpha=0.6)
    ax2.set_ylabel("Spikecount per s over the step", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    title = f"FI curve {model.emodel_metadata.emodel}, seed {model.seed}"

    fig.suptitle(title)

    fname = model.emodel_metadata.as_string(model.seed) + "__FI_curve.pdf"

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


def bAP_fit(feature, distances, values, npoints=20):
    """Returns a x and y arrays, y being a exponential decay fit for bAP feature.

    Args:
        feature (DendFitFeature): bAP feature
        distances (list): distances of the recordings
        values (list): bAP values
        npoints (int): number of items in the returned ndarrays
    """
    slope = numpy.array([feature.fit(distances, values)])
    x_fit = numpy.linspace(distances[0], distances[-1], num=npoints)
    y_fit = feature.exp_decay(x_fit, slope)
    return x_fit, y_fit


def EPSP_fit(feature, distances, values, npoints=20):
    """Returns a x and y arrays, y being a exponential fit for EPSP feature.

    Args:
        feature (DendFitFeature): EPSP feature
        distances (list): distances of the recordings
        values (list): EPSP values
        npoints (int): number of items in the returned ndarrays
    """
    slope = numpy.array([feature.fit(distances, values)])
    x_fit = numpy.linspace(distances[0], distances[-1], num=npoints)
    y_fit = feature.exp(x_fit, slope)
    return x_fit, y_fit


def plot_bAP(
    model,
    responses,
    apical_feature,
    basal_feature,
    figures_dir="./figures",
    write_fig=True,
):
    """Plot back-propagating action potential.

    Args:
        model (bluepyopt.ephys.CellModel): cell model
        responses (dict): responses of the cell model
        apical_feature (DendFitFeature): bAP feature with apical recs,
        basal_feature (DendFitFeature): bAP feature with basal recs,
        figures_dir (str or Path): directory where to save the figures
        write_fig (bool): whether to save the figure
    """
    make_dir(figures_dir)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    apical_distances, apical_values = apical_feature.get_distances_feature_values(responses)
    basal_distances, basal_values = basal_feature.get_distances_feature_values(responses)

    if apical_values is None or basal_values is None:
        logger.warning(
            "Cannot plot figure for bAP because dendrite feature could not be computed. "
            "This can happen when the emodel is bad and cannot even compute curent threshold."
        )
        return fig, ax
    if len(apical_distances) != len(apical_values) or len(basal_distances) != len(basal_values):
        logger.warning(
            "Cannot plot figure for bAP because of mismatch between "
            "distances list and feature values list. This can happen when "
            "the emodel is bad and cannot even compute curent threshold."
        )
        return fig, ax

    if 0 in apical_distances:
        apical_x_fit, apical_y_fit = bAP_fit(apical_feature, apical_distances, apical_values)
    if 0 in basal_distances:
        basal_x_fit, basal_y_fit = bAP_fit(basal_feature, basal_distances, basal_values)

    ax.scatter(
        apical_distances,
        apical_values,
        c=colours["modelpoint_apical"],
        label="model apical",
    )
    ax.scatter(
        basal_distances,
        basal_values,
        c=colours["modelpoint_basal"],
        label="model basal",
    )
    if 0 in apical_distances:
        ax.plot(
            apical_x_fit,
            apical_y_fit,
            "--",
            c=colours["modelline_apical"],
            label="model apical fit",
        )
    if 0 in basal_distances:
        ax.plot(
            basal_x_fit,
            basal_y_fit,
            "--",
            c=colours["modelline_basal"],
            label="model basal fit",
        )
    ax.set_xlabel(r"Distance from soma ($\mu$m)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(fontsize="x-small")

    fig.suptitle("Dendrite back-propagating action potential")

    if write_fig:
        fname = (
            model.emodel_metadata.as_string(model.seed) + "__dendrite_backpropagation_fit_decay.pdf"
        )
        save_fig(figures_dir, fname)

    return fig, ax


def compute_attenuation(dendrec_feature, somarec_feature, responses):
    """Returns EPSP attenuation and corresponding distances.

    Args:
        dendrec_feature (DendFitFeature): feature with recordings in the dendrite
        somarec_feature (DendFitFeature): feature with recordings in the soma
        responses (dict): responses to feed to the features
    """
    distances, dend_values = dendrec_feature.get_distances_feature_values(responses)
    _, soma_values = somarec_feature.get_distances_feature_values(responses)
    attenuation = numpy.asarray(dend_values) / numpy.asarray(soma_values)

    return distances, attenuation


def plot_EPSP(
    model,
    responses,
    apical_apicrec_feat,
    apical_somarec_feat,
    basal_basalrec_feat,
    basal_somarec_feat,
    figures_dir="./figures",
    write_fig=True,
):
    """Plot EPSP attenuation across dendrites.

    Args:
        model (bluepyopt.ephys.CellModel): cell model
        responses (dict): responses of the cell model
        apical_apicrec_feat (DendFitFeature): EPSP feature with apical stim and apical recs,
        apical_somarec_feat (DendFitFeature): EPSP feature with apical stim and soma recs,
        basal_basalrec_feat (DendFitFeature): EPSP feature with basal stim and basal recs,
        basal_somarec_feat (DendFitFeature): EPSP feature with basal stim and soma recs,
        figures_dir (str or Path): directory where to save the figures
        write_fig (bool): whether to save the figure
    """
    make_dir(figures_dir)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    apical_distances, apical_attenuation = compute_attenuation(
        apical_apicrec_feat, apical_somarec_feat, responses
    )
    basal_distances, basal_attenuation = compute_attenuation(
        basal_basalrec_feat, basal_somarec_feat, responses
    )

    if 0 in apical_distances:
        apical_x_fit, apical_y_fit = EPSP_fit(
            apical_apicrec_feat, apical_distances, apical_attenuation
        )
    if 0 in basal_distances:
        basal_x_fit, basal_y_fit = EPSP_fit(basal_basalrec_feat, basal_distances, basal_attenuation)

    if len(apical_distances) != len(apical_attenuation) or len(basal_distances) != len(
        basal_attenuation
    ):
        logger.warning(
            "Cannot plot figure for bAP because of mismatch between "
            "distances list and feature values list. This can happen when "
            "the emodel is bad and cannot even compute curent threshold."
        )
        return fig, ax

    ax.scatter(
        apical_distances,
        apical_attenuation,
        c=colours["modelpoint_apical"],
        label="model apical",
    )
    ax.scatter(
        basal_distances,
        basal_attenuation,
        c=colours["modelpoint_basal"],
        label="model basal",
    )
    if 0 in apical_distances:
        ax.plot(
            apical_x_fit,
            apical_y_fit,
            "--",
            c=colours["modelline_apical"],
            label="model apical fit",
        )
    if 0 in basal_distances:
        ax.plot(
            basal_x_fit,
            basal_y_fit,
            "--",
            c=colours["modelline_basal"],
            label="model basal fit",
        )
    ax.set_xlabel(r"Distance from soma ($\mu$m)")
    ax.set_ylabel("Attenuation dendrite amplitude / soma amplitude")
    ax.legend(fontsize="x-small")

    fig.suptitle("EPSP attenuation")

    if write_fig:
        fname = model.emodel_metadata.as_string(model.seed) + "__dendrite_EPSP_attenuation_fit.pdf"
        save_fig(figures_dir, fname)

    return fig, ax


def run_and_plot_bAP(
    original_cell_evaluator,
    access_point,
    mapper,
    seeds,
    save_recordings,
    load_from_local,
    only_validated=False,
    figures_dir="./figures",
):
    """Runs and plots ready-to-use bAP protocol for apical and basal dendrites.

    Args:
        original_cell_evaluator (CellEvaluator): original cell evaluator.
            A copy will be modified and then used to compute the responses.
        access_point (DataAccessPoint): data access point.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        save_recordings (bool): Whether to save the responses data under a folder
            named `recordings`. Responses can then be loaded using load_from_local
            instead of being re-run.
        load_from_local (bool): True to load responses from locally saved recordings.
            Responses are saved locally when save_recordings is True.
        only_validated (bool): True to only plot validated models
        figures_dir (str): path of the directory in which the figures should be saved.
    """
    figures_dir = Path(figures_dir)
    cell_evaluator = copy.deepcopy(original_cell_evaluator)

    # get basal and apical lengths
    morph_path = cell_evaluator.cell_model.morphology.morphology_path
    max_basal_length, max_apical_length = get_basal_and_apical_max_radial_distances(morph_path)
    max_basal_length = int(max_basal_length)
    max_apical_length = int(max_apical_length)

    # remove protocols except for pre-protocols
    old_prots = cell_evaluator.fitness_protocols["main_protocol"].protocols
    new_prots = {}
    for k, v in old_prots.items():
        if k in PRE_PROTOCOLS:
            new_prots[k] = v

    bAP_prot = define_bAP_protocol(dist_end=max_apical_length, dist_end_basal=max_basal_length)
    new_prots[bAP_prot.name] = bAP_prot

    cell_evaluator.fitness_protocols["main_protocol"].protocols = new_prots
    cell_evaluator.fitness_protocols["main_protocol"].execution_order = (
        cell_evaluator.fitness_protocols["main_protocol"].compute_execution_order()
    )

    apical_feature = define_bAP_feature("apical", dist_end=max_apical_length)
    basal_feature = define_bAP_feature("basal", dist_end=max_basal_length)
    # run emodel(s)
    emodels = compute_responses(
        access_point,
        cell_evaluator,
        mapper,
        seeds,
        store_responses=save_recordings,
        load_from_local=load_from_local,
    )

    if only_validated:
        emodels = [model for model in emodels if model.passed_validation]
        dest_leaf = "validated"
    else:
        dest_leaf = "all"

    # plot
    for mo in emodels:
        figures_dir_bAP = figures_dir / "dendritic" / dest_leaf
        plot_bAP(mo, mo.responses, apical_feature, basal_feature, figures_dir_bAP)


def run_and_plot_EPSP(
    original_cell_evaluator,
    access_point,
    mapper,
    seeds,
    save_recordings,
    load_from_local,
    only_validated=False,
    figures_dir="./figures",
):
    """Runs and plots ready-to-use EPSP protocol for apical and basal dendrites.

    Args:
        original_cell_evaluator (CellEvaluator): original cell evaluator.
            A copy will be modified and then used to compute the responses.
        access_point (DataAccessPoint): data access point.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        save_recordings (bool): Whether to save the responses data under a folder
            named `recordings`. Responses can then be loaded using load_from_local
            instead of being re-run.
        load_from_local (bool): True to load responses from locally saved recordings.
            Responses are saved locally when save_recordings is True.
        only_validated (bool): True to only plot validated models
        figures_dir (str): path of the directory in which the figures should be saved.
    """
    figures_dir = Path(figures_dir)
    cell_evaluator = copy.deepcopy(original_cell_evaluator)

    # get basal and apical lengths
    morph_path = cell_evaluator.cell_model.morphology.morphology_path
    max_basal_length, max_apical_length = get_basal_and_apical_max_radial_distances(morph_path)
    max_basal_length = int(max_basal_length)
    max_apical_length = int(max_apical_length)

    # remove protocols except for pre-protocols
    apical_prots = define_EPSP_protocol(
        "apical", dist_end=max_apical_length, dist_start=100, dist_step=100
    )
    basal_prots = define_EPSP_protocol(
        "basal", dist_end=max_basal_length, dist_start=30, dist_step=30
    )
    EPSP_prots = {**apical_prots, **basal_prots}

    cell_evaluator.fitness_protocols["main_protocol"].protocols = EPSP_prots
    cell_evaluator.fitness_protocols["main_protocol"].execution_order = (
        cell_evaluator.fitness_protocols["main_protocol"].compute_execution_order()
    )

    apical_apicrec_feat = define_EPSP_feature(
        "apical", "dend", dist_end=max_apical_length, dist_start=100, dist_step=100
    )
    apical_somarec_feat = define_EPSP_feature(
        "apical", "soma", dist_end=max_apical_length, dist_start=100, dist_step=100
    )
    basal_basalrec_feat = define_EPSP_feature(
        "basal", "dend", dist_end=max_basal_length, dist_start=30, dist_step=30
    )
    basal_somarec_feat = define_EPSP_feature(
        "basal", "soma", dist_end=max_basal_length, dist_start=30, dist_step=30
    )
    # run emodel(s)
    emodels = compute_responses(
        access_point,
        cell_evaluator,
        mapper,
        seeds,
        store_responses=save_recordings,
        load_from_local=load_from_local,
    )

    if only_validated:
        emodels = [model for model in emodels if model.passed_validation]
        dest_leaf = "validated"
    else:
        dest_leaf = "all"

    # plot
    for mo in emodels:
        figures_dir_EPSP = figures_dir / "dendritic" / dest_leaf
        plot_EPSP(
            mo,
            mo.responses,
            apical_apicrec_feat,
            apical_somarec_feat,
            basal_basalrec_feat,
            basal_somarec_feat,
            figures_dir_EPSP,
        )


def plot_IV_curves(
    evaluator,
    emodels,
    access_point,
    figures_dir,
    efel_settings,
    mapper,
    seeds,
    prot_name="iv",
    custom_bluepyefe_cells_pklpath=None,
    write_fig=True,
    n_bin=5,
):
    """Plots IV curves of peak voltage and voltage_deflection
    with simulated and experimental values. Only works for threshold-based protocols.

    Args:
        evaluator (CellEvaluator): cell evaluator
        emodels (list): list of EModels
        access_point (DataAccessPoint): data access point
        figures_dir (str or Path): output directory for the figure to be saved on
        efel_settings (dict): eFEL settings in the form {setting_name: setting_value}.
        mapper (map): used to parallelize the evaluation of the individual in the population.
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        prot_name (str): Only recordings from this protocol will be used.
        custom_bluepyefe_cells_pklpath (str): file path to the cells.pkl output of BluePyEfe.
            If None, will use usual file path used in BluePyEfe,
            so this is to be set only to use a file at an unexpected path.
        write_fig (bool): whether to save the figure
        n_bin (int): number of bins to use
    """
    # pylint: disable=too-many-nested-blocks, possibly-used-before-assignment, disable=too-many-locals, disable=too-many-statements
    # note: should maybe also check location and recorded variable
    make_dir(figures_dir)
    if efel_settings is None:
        efel_settings = bluepyefe.tools.DEFAULT_EFEL_SETTINGS.copy()

    lower_bound = -100
    upper_bound = 100

    # Generate amplitude points
    sim_amp_points = list(map(int, numpy.linspace(lower_bound, upper_bound, n_bin + 1)))
    # add missing features (if any) to evaluator
    updated_evaluator = fill_in_IV_curve_evaluator(
        evaluator, efel_settings, prot_name, sim_amp_points
    )

    emodels = compute_responses(
        access_point,
        updated_evaluator,
        map_function=mapper,
        seeds=seeds,
    )

    emodel_name = None
    cells = None
    for emodel in emodels:
        # -- get extracted IV curve data -- #
        # do not re-extract data if the emodel is the same as previously
        if custom_bluepyefe_cells_pklpath is not None:
            if cells is None:
                cells = read_extraction_output(custom_bluepyefe_cells_pklpath)
            if cells is None:
                continue
            exp_peak, exp_vd = extract_experimental_data_for_IV_curve(
                cells, efel_settings, prot_name, n_bin
            )
        elif emodel_name != emodel.emodel_metadata.emodel or cells is None:
            # take extraction data from pickle file and rearange it for plotting
            cells = read_extraction_output_cells(emodel.emodel_metadata.emodel)
            emodel_name = emodel.emodel_metadata.emodel
            if cells is None:
                continue
            exp_peak, exp_vd = extract_experimental_data_for_IV_curve(
                cells, efel_settings, prot_name, n_bin
            )

            emodel_name = emodel.emodel_metadata.emodel

        # -- get simulated IV curve data -- #
        values = updated_evaluator.fitness_calculator.calculate_values(emodel.responses)

        simulated_peak_v = []
        simulated_peak_amp_rel = []
        simulated_peak_amp = []
        simulated_vd_v = []
        simulated_vd_amp_rel = []
        simulated_vd_amp = []
        for val in values:
            if prot_name.lower() in val.lower():
                amp_rel_temp = get_amplitude_from_feature_key(val)
                if "maximum_voltage_from_voltagebase" in val:
                    simulated_peak_v.append(values[val])
                    simulated_peak_amp_rel.append(amp_rel_temp)
                    simulated_peak_amp.append(rel_to_abs_amplitude(amp_rel_temp, emodel.responses))
                elif "voltage_deflection_vb_ssse" in val:
                    simulated_vd_v.append(values[val])
                    simulated_vd_amp_rel.append(amp_rel_temp)
                    simulated_vd_amp.append(rel_to_abs_amplitude(amp_rel_temp, emodel.responses))

        # plotting
        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        ax[0].errorbar(
            exp_peak["amp_rel"],
            exp_peak["feat_rel"],
            yerr=exp_peak["feat_rel_err"],
            marker="o",
            color="grey",
            label="expt max(V)",
        )
        ax[0].errorbar(
            exp_vd["amp_rel"],
            exp_vd["feat_rel"],
            yerr=exp_vd["feat_rel_err"],
            marker="o",
            color="lightgrey",
            label="expt V deflection",
        )
        ax[0].set_xlabel("Amplitude (% of rheobase)")
        ax[0].set_ylabel("Voltage (mV)")
        ax[0].set_title("IV curve (relative amplitude)")
        ax[0].plot(
            simulated_peak_amp_rel, simulated_peak_v, "o", color="blue", label="simulated max(V)"
        )
        ax[0].plot(
            simulated_vd_amp_rel,
            simulated_vd_v,
            "o",
            color="royalblue",
            label="simulated V deflection",
        )

        ax[1].errorbar(
            exp_peak["amp"],
            exp_peak["feat_abs"],
            yerr=exp_peak["feat_abs_err"],
            marker="o",
            color="grey",
            label="expt max(V)",
        )
        ax[1].errorbar(
            exp_vd["amp"],
            exp_vd["feat_abs"],
            yerr=exp_vd["feat_abs_err"],
            marker="o",
            color="lightgrey",
            label="expt V deflection",
        )
        ax[1].set_xlabel("Amplitude (nA)")
        ax[1].set_ylabel("Voltage (mV)")
        ax[1].set_title("IV curve (absolute amplitude)")
        ax[1].plot(
            simulated_peak_amp, simulated_peak_v, "o", color="blue", label="simulated max(V)"
        )
        ax[1].plot(
            simulated_vd_amp, simulated_vd_v, "o", color="royalblue", label="simulated V deflection"
        )

        ax[0].legend()
        ax[1].legend()
        if write_fig:
            save_fig(figures_dir, emodel.emodel_metadata.as_string(emodel.seed) + "__IV_curve.pdf")


def plot_FI_curves_comparison(
    evaluator,
    emodels,
    access_point,
    seeds,
    mapper,
    figures_dir,
    prot_name,
    custom_bluepyefe_cells_pklpath=None,
    write_fig=True,
    n_bin=5,
):
    """Plots FI (current vs frequency) curves with simulated and experimental values.
    Only works for threshold-based protocols.
    Expects mean_frequency to be available in extracted and simulated data.

    Args:
        evaluator (CellEvaluator): cell evaluator
        emodels (list): list of EModels
        access_point (DataAccessPoint): data access point
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        mapper (map): used to parallelize the evaluation of the individual in the population.
        figures_dir (str or Path): output directory for the figure to be saved on
        prot_name (str): name of the protocol to use for the FI curve
        custom_bluepyefe_cells_pklpath (str): file path to the cells.pkl output of BluePyEfe.
            If None, will use usual file path used in BluePyEfe,
            so this is to be set only to use a file at an unexpected path.
        write_fig (bool): whether to save the figure
        n_bin (int): number of bins to use
    """
    # pylint: disable=too-many-nested-blocks, possibly-used-before-assignment
    make_dir(figures_dir)

    emodel_name, cells = None, None
    updated_evaluator = copy.deepcopy(evaluator)
    for emodel in emodels:
        # do not re-extract data if the emodel is the same as previously
        if custom_bluepyefe_cells_pklpath is not None:
            if cells is None:
                cells = read_extraction_output(custom_bluepyefe_cells_pklpath)
            if cells is None:
                continue

            # experimental FI curve
            expt_data = get_experimental_FI_curve_for_plotting(cells, prot_name, n_bin=n_bin)
        elif emodel_name != emodel.emodel_metadata.emodel or cells is None:
            # take extraction data from pickle file and rearange it for plotting
            cells = read_extraction_output_cells(emodel.emodel_metadata.emodel)
            emodel_name = emodel.emodel_metadata.emodel
            if cells is None:
                continue

            # experimental FI curve
            expt_data = get_experimental_FI_curve_for_plotting(cells, prot_name, n_bin=n_bin)

            emodel_name = emodel.emodel_metadata.emodel

        expt_data_amp_rel = expt_data[0]
        prot_name_original = get_original_protocol_name(prot_name, evaluator)
        updated_evaluator = update_evaluator(
            expt_data_amp_rel, prot_name_original, updated_evaluator
        )

    updated_evaluator.fitness_protocols["main_protocol"].execution_order = (
        updated_evaluator.fitness_protocols["main_protocol"].compute_execution_order()
    )

    emodels = compute_responses(access_point, updated_evaluator, mapper, seeds)
    for emodel in emodels:
        # do not re-extract data if the emodel is the same as previously
        if custom_bluepyefe_cells_pklpath is not None:
            if cells is None:
                cells = read_extraction_output(custom_bluepyefe_cells_pklpath)
            if cells is None:
                continue

            # experimental FI curve
            expt_data = get_experimental_FI_curve_for_plotting(cells, prot_name, n_bin=n_bin)
        elif emodel_name != emodel.emodel_metadata.emodel or cells is None:
            # take extraction data from pickle file and rearange it for plotting
            cells = read_extraction_output_cells(emodel.emodel_metadata.emodel)
            emodel_name = emodel.emodel_metadata.emodel
            if cells is None:
                continue

            # experimental FI curve
            expt_data = get_experimental_FI_curve_for_plotting(cells, prot_name, n_bin=n_bin)

            emodel_name = emodel.emodel_metadata.emodel

        sim_data = get_simulated_FI_curve_for_plotting(
            updated_evaluator, emodel.responses, prot_name
        )
        plot_fi_curves(expt_data, sim_data, figures_dir, emodel, write_fig)


def phase_plot(
    emodels,
    figures_dir,
    prot_names,
    amplitude,
    amp_window,
    relative_amp=True,
    custom_bluepyefe_cells_pklpath=None,
    write_fig=True,
):
    """Plots recordings as voltage vs time and in phase space.

    Args:
        emodels (list): list of EModels
        figures_dir (str or Path): output directory for the figure to be saved on
        prot_names (list of str): the names of the protocols to select for plotting
        amplitude (float): amplitude of the protocol to select.
            Only exactly this amplitude will be selected for model.
            An amplitude window is used for experimental trace selection
        amp_window (float): amplitude window around amplitude for experimental recording selection
            Is not used for model trace selection
        relative_amp (bool): Are amplitde and amp_window in relative amplitude (True) or in
            absolute amplitude (False).
        custom_bluepyefe_cells_pklpath (str): file path to the cells.pkl output of BluePyEfe.
            If None, will use usual file path used in BluePyEfe,
            so this is to be set only to use a file at an unexpected path.
        write_fig (bool): whether to save the figure
    """
    make_dir(figures_dir)

    emodel_name = None
    cells = None
    for emodel in emodels:
        # do not re-extract data if the emodel is the same as previously
        if custom_bluepyefe_cells_pklpath is not None:
            if cells is None:
                cells = read_extraction_output(custom_bluepyefe_cells_pklpath)
        elif emodel_name != emodel.emodel_metadata.emodel or cells is None:
            # take extraction data from pickle file and rearange it for plotting
            cells = read_extraction_output_cells(emodel.emodel_metadata.emodel)
            emodel_name = emodel.emodel_metadata.emodel
        if cells is None:
            continue

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        for cell in cells:
            for rec in cell.recordings:
                if any(a.lower() in rec.protocol_name.lower() for a in prot_names):
                    if relative_amp:
                        rec_amp = rec.amp_rel
                    else:
                        rec_amp = rec.amp
                    if amplitude - amp_window < rec_amp < amplitude + amp_window:
                        ax[0].plot(rec.t, rec.voltage, color="grey")
                        # interpolate to reproduce what is being done in efel
                        interp_time = numpy.arange(rec.t[0], rec.t[-1] + 0.1, 0.1)
                        interp_voltage = numpy.interp(interp_time, rec.t, rec.voltage)
                        # plot the phase plot d(rec.voltage)/dt vs rec.voltage
                        ax[1].plot(
                            interp_voltage[1:],
                            numpy.diff(interp_voltage) / numpy.diff(interp_time),
                            color="grey",
                        )
                        ax[0].set_xlabel("Time (ms)")
                        ax[0].set_ylabel("Voltage (mV)")
                        ax[0].set_title("Traces")
                        ax[1].set_xlabel("Voltage (mV)")
                        ax[1].set_ylabel("dV/dt (V/s)")
                        ax[1].set_title("Traces in phase space")

        for resp_name, response in emodel.responses.items():
            if any(prot_name.lower() in resp_name.lower() for prot_name in prot_names):
                if str(amplitude) in resp_name and resp_name[-7:] == ".soma.v":
                    # have to turn reponse into numpy.array because they are pandas.Series
                    time = numpy.asarray(response["time"])
                    voltage = numpy.asarray(response["voltage"])
                    ax[0].plot(time, voltage, color="blue")
                    interp_time = numpy.arange(time[0], time[-1] + 0.1, 0.1)
                    interp_voltage = numpy.interp(interp_time, time, voltage)
                    # plot the phase plot d(rec.voltage)/dt vs rec.voltage
                    ax[1].plot(
                        interp_voltage[1:],
                        numpy.diff(interp_voltage) / numpy.diff(interp_time),
                        color="blue",
                    )

        # empty plots just for legend
        ax[0].plot([], [], label="experiment", color="grey")
        ax[0].plot([], [], label="model", color="blue")
        ax[1].plot([], [], label="experiment", color="grey")
        ax[1].plot([], [], label="model", color="blue")
        ax[0].legend()
        ax[1].legend()
        fig.suptitle(f"{', '.join(prot_names)} {amplitude} {'%' if relative_amp else 'nA'}")
        if write_fig:
            save_fig(
                figures_dir,
                figure_name=emodel.emodel_metadata.as_string(emodel.seed) + "__phase_plot.pdf",
            )


def plot_trace_comparison(
    emodels, figures_dir, custom_bluepyefe_protocols_pklpath=None, write_fig=True
):
    """Compare traces between experiments and models.

    Args:
        emodels (list): list of EModels
        figures_dir (str or Path): output directory for the figure to be saved on
        custom_bluepyefe_protocols_pklpath (str): file path to
            the protocols.pkl output of BluePyEfe.
            If None, will use usual file path used in BluePyEfe,
            so this is to be set only to use a file at an unexpected path.
        write_fig (bool): whether to save the figure
    """
    # pylint: disable=too-many-nested-blocks
    prots_to_skip = ["bpo", "SearchHoldingCurrent", "SearchThresholdCurrent", "bAP"]

    make_dir(figures_dir)

    emodel_name = None
    protocols = None
    for emodel in emodels:
        # do not re-extract data if the emodel is the same as previously
        if custom_bluepyefe_protocols_pklpath is not None:
            if protocols is None:
                protocols = read_extraction_output(custom_bluepyefe_protocols_pklpath)
        elif emodel_name != emodel.emodel_metadata.emodel or protocols is None:
            # take extraction data from pickle file and rearange it for plotting
            protocols = read_extraction_output_protocols(emodel.emodel_metadata.emodel)

            emodel_name = emodel.emodel_metadata.emodel
        if protocols is None:
            continue

        count = 0
        for key in emodel.responses.keys():
            # only voltage traces
            if any(s in key for s in prots_to_skip) or key[-2:] != ".v":
                continue
            count = count + 1

        # make subplots with count rows
        fig, axes = plt.subplots(count, figsize=(10, count * 2))

        i = 0
        for resp_name, response in sorted(emodel.responses.items()):
            if any(s in resp_name for s in prots_to_skip) or resp_name[-2:] != ".v":
                continue

            if "RMPProtocol.soma.v" in resp_name:
                # plot simulated trace
                axes[i].plot(response["time"], response["voltage"], label=resp_name, color="blue")
                axes[i].set_ylabel("Voltage (mV)")
                axes[i].set_xlabel("Time (ms)")
                axes[i].set_title(resp_name)
                i = i + 1
                continue

            if "RinProtocol.soma.v" in resp_name:
                # Expt Traces
                for p in protocols:
                    # amplitude in response name
                    if "-40" in f"{p.amplitude}":
                        # plot all recordings of the experimental protocol res. expt data
                        for rec in p.recordings:
                            axes[i].plot(
                                rec.time, rec.voltage, linestyle="-", color="grey", alpha=0.5
                            )

                # simulated Rin protocol
                axes[i].plot(response["time"], response["voltage"], label=resp_name, color="blue")
                axes[i].set_ylabel("Voltage (mV)")
                axes[i].set_xlabel("Time (ms)")
                axes[i].set_title(resp_name)
                i = i + 1
                continue

            # for each protocol in expt data
            for p in protocols:
                # protocol name and amplitude in response name
                if f"{p.name}_{p.amplitude}.soma" in resp_name:
                    if len(p.recordings) == 0:
                        continue

                    # plot all recordings of the experimental protocol % res. expt data
                    for rec in p.recordings:
                        axes[i].plot(rec.time, rec.voltage, linestyle="-", color="grey", alpha=0.5)

                    # plot simulated trace
                    axes[i].plot(
                        response["time"], response["voltage"], label=resp_name, color="blue"
                    )
                    axes[i].legend()
                    axes[i].set_ylabel("Voltage (mV)")
                    axes[i].set_xlabel("Time (ms)")
                    axes[i].set_title(f"{p.name} {p.amplitude}")
            i = i + 1

        fig.suptitle(f"emodel = {emodel_name}")
        fig.tight_layout()

        if write_fig:
            save_fig(
                figures_dir,
                figure_name=emodel.emodel_metadata.as_string(emodel.seed)
                + "__trace_comparison.pdf",
            )


def plot_sinespec(
    model, responses, sinespec_settings, efel_settings, figures_dir="./figures", write_fig=True
):
    """Plot SineSpec with current trace, voltage trace and impedance.

    Args:
        model (EModel): cell model
        responses (dict): responses of the cell model
        figures_dir (str or Path): output directory for the figure to be saved on
        write_fig (bool): whether to save the figure
    """
    make_dir(figures_dir)
    prot_name = f"SineSpec_{sinespec_settings['amp']}"
    current_key = f"{prot_name}.iclamp.i"
    voltage_key = f"{prot_name}.soma.v"

    fig, axs = plt.subplots(3, figsize=(6, 12))

    if voltage_key not in responses or current_key not in responses:
        logger.warning(
            "Could not find sinespec responses %s for emodel with seed %s. "
            "This is most probably due to a bad model unable to compute threshold. "
            "Skipping Sinespec plot.",
            model.emodel_metadata.emodel,
            model.seed,
        )
        return fig, axs

    freq, smooth_Z = get_impedance(
        responses[voltage_key]["time"],
        responses[voltage_key]["voltage"],
        responses[current_key]["voltage"],
        300.0,
        5300.0,
        efel_settings,
    )
    if freq is None or smooth_Z is None:
        return fig, axs

    # current trace
    axs[0].plot(responses[current_key]["time"], responses[current_key]["voltage"])
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Injected current (nA)")

    # voltage trace
    axs[1].plot(responses[voltage_key]["time"], responses[voltage_key]["voltage"])
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Voltage (mV)")

    # impedance trace
    axs[2].plot(freq, smooth_Z)
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("normalized Z")

    if write_fig:
        fname = model.emodel_metadata.as_string(model.seed) + "__sinespec.pdf"
        save_fig(figures_dir, fname)

    return fig, axs


def run_and_plot_custom_sinespec(
    evaluator,
    sinespec_settings,
    efel_settings,
    access_point,
    mapper,
    seeds,
    save_recordings,
    load_from_local,
    figures_dir="./figures",
):
    """Run a custom SineSpec protocol, and plot its trace and impedance feature.

    Args:
        evaluator (CellEvaluator): cell evaluator
        sinespec_settings (dict): contains amplitude settings for the SineSpec protocol,
            with keys 'amp' and 'threshold_based'.
            'amp' should be in percentage of threshold if 'threshold_based' is True, e.g. 150,
            or in nA if 'threshold_based' if false, e.g. 0.1.
        efel_settings (dict): eFEL settings in the form {setting_name: setting_value}
        access_point (DataAccessPoint): data access point
        mapper (map): used to parallelize the evaluation of the individual in the population
        seeds (list): if not None, filter emodels to keep only the ones with these seeds
        save_recordings (bool): Whether to save the responses data under a folder
            named `recordings`. Responses can then be loaded using load_from_local
            instead of being re-run.
        load_from_local (bool): True to load responses from locally saved recordings.
            Responses are saved locally when save_recordings is True.
        only_validated (bool): True to only plot validated models
        figures_dir (str): path of the directory in which the figures should be saved
    """
    new_eval = get_sinespec_evaluator(evaluator, sinespec_settings, efel_settings)

    # run evaluator
    emodels = compute_responses(
        access_point,
        new_eval,
        mapper,
        seeds,
        store_responses=save_recordings,
        load_from_local=load_from_local,
    )

    # plot current trace, voltage trace, impedance
    for mo in emodels:
        figures_dir_sinespec = Path(figures_dir) / "sinespec" / "all"
        mo.features = mo.evaluator.fitness_calculator.calculate_values(mo.responses)
        plot_sinespec(mo, mo.responses, sinespec_settings, efel_settings, figures_dir_sinespec)


def plot_models(
    access_point,
    mapper,
    seeds=None,
    figures_dir="./figures",
    plot_optimisation_progress=True,
    optimiser=None,
    plot_parameter_evolution=True,
    plot_distributions=True,
    plot_scores=True,
    plot_traces=True,
    plot_thumbnail=True,
    plot_currentscape=False,
    plot_fi_curve=False,
    plot_dendritic_ISI_CV=True,
    plot_dendritic_rheobase=True,
    plot_bAP_EPSP=False,
    plot_IV_curve=False,
    plot_FI_curve_comparison=False,
    plot_phase_plot=False,
    plot_traces_comparison=False,
    run_plot_custom_sinspec=False,
    IV_curve_prot_name="iv",
    FI_curve_prot_name="idrest",
    phase_plot_settings=None,
    sinespec_settings=None,
    custom_bluepyefe_cells_pklpath=None,
    custom_bluepyefe_protocols_pklpath=None,
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
        plot_optimisation_progress (bool): True to plot the optimisation progress from checkpoint
        optimiser (str): name of the algorithm used for optimisation, can be "IBEA", "SO-CMA"
            or "MO-CMA". Is used in optimisation progress plotting.
        plot_parameter_evolution (bool): True to plot parameter evolution
        plot_distributions (bool): True to plot the parameters distributions
        plot_scores (bool): True to plot the scores
        plot_traces (bool): True to plot the traces
        plot_thumbnail (bool): True to plot a trace used as thumbnail
        plot_currentscape (bool): True to plot the currentscapes
        plot_fi_curve (bool): True to plot the current / frequency curve
        plot_dendritic_ISI_CV (bool): True to plot dendritic ISI CV (if present)
        plot_dendritic_rheobase (bool): True to plot dendritic rheobase (if present)
        plot_bAP_EPSP (bool): True to plot bAP and EPSP protocol.
            Only use this on model having apical dendrites.
        plot_IV_curve (bool): True to plot IV curves for peak voltage and voltage_deflection.
            Expects threshold-based sub-threshold IV protocol.
        plot_FI_curve_comparison (bool): True to plot FI curve with experimental
            and simulated data. Expects threshold-based protocols.
        plot_phase_plot (bool): True to plot phase plot with experimental
            and simulated data. Can be threshold-based or non threshold-based.
        plot_traces_comparison (bool): True to plot a new figure with simulated traces
            on top of experimental traces.
        run_plot_custom_sinspec (bool): True to run a SineSpec protocol, and plot
            its voltage and current trace, along with its impedance.
        IV_curve_prot_name (str): which protocol to use to plot_IV_curves.
        FI_curve_prot_name (str): which protocol to use to plot FI_curve comparison.
            The protocol must have the mean_frequency feature associated to it.
        phase_plot_settings (dict): settings for the phase plot. Should contain the following keys:
            "prot_names" (list of str): the names of the protocols to select for phase plot
            "amplitude" (float): amplitude of the protocol to select.
                Only exactly this amplitude will be selected for model.
                An amplitude window is used for experimental trace selection
            "amp_window" (float): amplitude window around amplitude
                for experimental recording selection. Is not used for model trace selection
            "relative_amp" (bool): Are amplitde and amp_window in relative amplitude (True) or in
            absolute amplitude (False).
        sinespec_settings (dict): contains amplitude settings for the SineSpec protocol,
            with keys 'amp' and 'threshold_based'.
            'amp' should be in percentage of threshold if 'threshold_based' is True, e.g. 150,
            or in nA if 'threshold_based' if false, e.g. 0.1.
        custom_bluepyefe_cells_pklpath (str): file path to the cells.pkl output of BluePyEfe.
            If None, will use usual file path used in BluePyEfe,
            so this is to be set only to use a file at an unexpected path.
        custom_bluepyefe_protocols_pklpath (str): file path to
            the protocols.pkl output of BluePyEfe.
            If None, will use usual file path used in BluePyEfe,
            so this is to be set only to use a file at an unexpected path.
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
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

    figures_dir = Path(figures_dir)

    if cell_evaluator is None:
        cell_evaluator = get_evaluator_from_access_point(
            access_point,
            include_validation_protocols=True,
            record_ions_and_currents=plot_currentscape,
        )

    if (
        plot_currentscape
        # maybe we should give currentscape_config as an argument of this fct,
        # like the other pipeline_settings
        and access_point.pipeline_settings.currentscape_config is not None
        and "current" in access_point.pipeline_settings.currentscape_config
        and "names" in access_point.pipeline_settings.currentscape_config["current"]
    ):
        add_recordings_to_evaluator(
            cell_evaluator,
            access_point.pipeline_settings.currentscape_config["current"]["names"],
            use_fixed_dt_recordings=False,
        )

    if plot_optimisation_progress or plot_parameter_evolution:
        checkpoint_paths = existing_checkpoint_paths(access_point.emodel_metadata)

        if plot_optimisation_progress:
            if optimiser is None:
                logger.warning(
                    "Will not plot optimisation progress because optimiser was not given."
                )
            else:
                for chkp_path in checkpoint_paths:
                    stem = str(Path(chkp_path).stem)
                    seed = int(stem.rsplit("seed=", maxsplit=1)[-1])

                    optimisation(
                        optimiser=optimiser,
                        emodel=access_point.emodel_metadata.emodel,
                        iteration=access_point.emodel_metadata.iteration,
                        seed=seed,
                        checkpoint_path=chkp_path,
                        figures_dir=figures_dir / "optimisation",
                    )

        if plot_parameter_evolution:
            evolution_parameters_density(
                evaluator=cell_evaluator,
                checkpoint_paths=checkpoint_paths,
                metadata=access_point.emodel_metadata,
                figures_dir=figures_dir / "parameter_evolution",
            )

    if any(
        (
            plot_traces,
            plot_currentscape,
            plot_dendritic_ISI_CV,
            plot_dendritic_rheobase,
            plot_thumbnail,
            plot_IV_curve,
            plot_FI_curve_comparison,
            plot_phase_plot,
            plot_traces_comparison,
        )
    ):
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

        if plot_thumbnail:
            figures_dir_thumbnail = figures_dir / "thumbnail" / dest_leaf
            recording_names = get_recording_names(
                access_point.get_fitness_calculator_configuration().protocols,
                stimuli,
            )
            thumbnail(mo, mo.responses, recording_names, figures_dir_thumbnail)

        if plot_dendritic_ISI_CV:
            dendritic_feature_plots(mo, "ISI_CV", dest_leaf, figures_dir)

        if plot_dendritic_rheobase:
            dendritic_feature_plots(mo, "rheobase", dest_leaf, figures_dir)

        if plot_currentscape:
            logger.warning(
                "If an ion channel mod file lacks RANGE current variable (e.g. RANGE ik, ina), "
                "no associated current will be plotted for that SUFFIX in Currentscape."
            )
            config = access_point.pipeline_settings.currentscape_config
            figures_dir_currentscape = figures_dir / "currentscape" / dest_leaf
            currentscape(
                mo.responses,
                config=config,
                metadata_str=mo.emodel_metadata.as_string(mo.seed),
                figures_dir=figures_dir_currentscape,
                emodel=mo.emodel_metadata.emodel,
            )
            # reset rcParams which are modified by currentscape
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        if plot_fi_curve:
            figures_dir_FI_curves = figures_dir / "FI_curves" / dest_leaf
            FI_curve(
                mo,
                mo.responses,
                copy.deepcopy(cell_evaluator),
                figures_dir=figures_dir_FI_curves,
            )

    # outside of for loop because we want to load pickle file only once
    if plot_IV_curve:
        figures_dir_IV_curves = figures_dir / "IV_curves" / dest_leaf
        plot_IV_curves(
            cell_evaluator,
            emodels,
            access_point,
            figures_dir_IV_curves,
            # maybe we should give efel_settings as an argument of plot_models,
            # like the other pipeline_settings
            access_point.pipeline_settings.efel_settings,
            mapper,
            seeds,
            IV_curve_prot_name,
            custom_bluepyefe_cells_pklpath=custom_bluepyefe_cells_pklpath,
        )

    if plot_FI_curve_comparison:
        figures_dir_FI_curves = figures_dir / "FI_curves" / dest_leaf
        plot_FI_curves_comparison(
            cell_evaluator,
            emodels,
            access_point,
            seeds,
            mapper,
            figures_dir_FI_curves,
            FI_curve_prot_name,
            custom_bluepyefe_cells_pklpath=custom_bluepyefe_cells_pklpath,
        )

    if plot_phase_plot:
        figures_dir_phase_plot = figures_dir / "phase_plot" / dest_leaf
        if phase_plot_settings is None:
            phase_plot_settings = access_point.pipeline_settings.phase_plot_settings
        phase_plot(
            emodels,
            figures_dir_phase_plot,
            phase_plot_settings["prot_names"],
            phase_plot_settings["amplitude"],
            phase_plot_settings["amp_window"],
            phase_plot_settings["relative_amp"],
            custom_bluepyefe_cells_pklpath=custom_bluepyefe_cells_pklpath,
        )

    if plot_traces_comparison:
        figures_dir_traces = figures_dir / "traces" / dest_leaf
        plot_trace_comparison(
            emodels,
            figures_dir_traces,
            custom_bluepyefe_protocols_pklpath=custom_bluepyefe_protocols_pklpath,
        )

    if plot_bAP_EPSP:
        run_and_plot_bAP(
            cell_evaluator,
            access_point,
            mapper,
            seeds,
            save_recordings,
            load_from_local,
            only_validated,
            figures_dir,
        )
        run_and_plot_EPSP(
            cell_evaluator,
            access_point,
            mapper,
            seeds,
            save_recordings,
            load_from_local,
            only_validated,
            figures_dir,
        )

    if run_plot_custom_sinspec:
        if sinespec_settings is None:
            logger.warning(
                "sinspec_settings was not given. Will use 0.05 nA amplitude for SineSpec plotting."
            )
            sinespec_settings = {"amp": 0.05, "threshold_based": False}
        run_and_plot_custom_sinespec(
            cell_evaluator,
            sinespec_settings,
            access_point.pipeline_settings.efel_settings,
            access_point,
            mapper,
            seeds,
            save_recordings,
            load_from_local,
            figures_dir=figures_dir,
        )

    return emodels


def currentscape(
    responses=None,
    output_dir=None,
    config=None,
    metadata_str="",
    figures_dir="./figures",
    emodel="",
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
    # pylint: disable=too-many-branches, too-many-statements
    # TODO: refactor this a little
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
    if "legend" not in updated_config:
        updated_config["legend"] = {}
    if "show" not in updated_config:
        updated_config["show"] = {}

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

            name = ".".join((f"{metadata_str}__currentscape", prot, loc))

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
            if "title" not in config and emodel:
                # check config because we want to change this for each plot
                title = f"{emodel}\n{prot}"
                updated_config["title"] = title
            # resizing
            if "figsize" not in updated_config:
                updated_config["figsize"] = (4.5, 6)
            if "textsize" not in updated_config:
                updated_config["textsize"] = 8
            if "textsize" not in updated_config["legend"]:
                updated_config["legend"]["textsize"] = 5

            if len(voltage) == 0 or len(currents) == 0:
                logger.warning(
                    "Could not plot currentscape for %s: voltage or currents is empty.",
                    name,
                )
            else:
                try:
                    from currentscape.currentscape import plot_currentscape as plot_currentscape_fct

                    logger.info("Plotting currentscape for %s", name)
                    fig = plot_currentscape_fct(
                        voltage,
                        currents,
                        updated_config,
                        ions_data=ionic_concentrations,
                        time=time,
                    )
                    plt.close(fig)
                except ModuleNotFoundError:
                    logger.warning("Currentscape module not found. Skipping currentscape plotting.")
