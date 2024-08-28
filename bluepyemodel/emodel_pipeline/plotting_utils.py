"""Plotting utils functions."""

"""
Copyright 2024 Blue Brain Project / EPFL

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
import logging
from pathlib import Path

import efel
import numpy
from bluepyopt.ephys.objectives import SingletonWeightObjective

from bluepyemodel.ecode.sinespec import SineSpec
from bluepyemodel.evaluation.efel_feature_bpem import eFELFeatureBPEM
from bluepyemodel.evaluation.evaluator import PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import soma_loc
from bluepyemodel.evaluation.protocols import BPEMProtocol
from bluepyemodel.evaluation.protocols import ThresholdBasedProtocol
from bluepyemodel.evaluation.recordings import FixedDtRecordingCustom
from bluepyemodel.evaluation.recordings import FixedDtRecordingStimulus
from bluepyemodel.tools.utils import get_curr_name
from bluepyemodel.tools.utils import get_loc_name
from bluepyemodel.tools.utils import get_protocol_name

logger = logging.getLogger("__main__")


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
        if protocol.name not in prot_names and protocol.recordings
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


def rel_to_abs_amplitude(rel_amp, responses):
    """Converts relative amplitude to absolute amplitude.

    Args:
        rel_amp (float): relative amplitude in percentage of threshold current
        responses (dict): should contain 'bpo_threshold_current' and 'bpo_holding_current'
    """
    if "bpo_threshold_current" not in responses or "bpo_holding_current" not in responses:
        logger.warning(
            "Could not convert relative amplitude into absolute amplitude. "
            "Missing holding and threshold current in responses."
        )
        return numpy.nan
    return rel_amp * 0.01 * responses["bpo_threshold_current"] + responses["bpo_holding_current"]


def binning(x, y, n_bin=5):
    """Put x and y into bins. Returns the binned x, binned y and std of binned y.

    Args:
        x (list): x axis data points
        y (list): y axis data points corresponding to the x data points
        n_bin (int): number of bins to use
    """
    if len(x) <= n_bin:
        return x, y, list(numpy.zeros(len(x)))

    new_x = []
    new_y = []
    y_err = []
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    intervals = numpy.linspace(numpy.min(x), numpy.max(x), num=n_bin + 1)
    for i in range(n_bin):
        start = intervals[i]
        stop = intervals[i + 1]
        new_x.append((start + stop) / 2.0)
        y_select = y[numpy.all([x >= start, x <= stop], axis=0)]
        new_y.append(numpy.nanmean(y_select))
        y_err.append(numpy.nanstd(y_select))

    return new_x, new_y, y_err


def extract_experimental_data_for_IV_curve(cells, efel_settings, prot_name="iv", n_bin=5):
    """Get experimental data to be plotted for IV curve from bluepyefe cells.
    Use efel to extract missing features.

    Args:
        cells (list): list of bluepyefe.cell.Cell object to get recordings from
        efel_settings (dict): efel settings to use during feature extraction
        prot_name (str): Only recordings from this protocol will be used.
        n_bin (int): number of bins to use for plotting
    """
    # experimental IV curve
    expt_peak_amp_rel = []
    expt_peak_amp = []
    expt_peak_v = []
    expt_vd_amp_rel = []
    expt_vd_amp = []
    expt_vd_v = []
    for cell in cells:
        for rec in cell.recordings:
            if rec.protocol_name.lower() == prot_name.lower() and rec.amp_rel <= 100:
                if 0 <= rec.amp_rel:
                    feat_name = "maximum_voltage_from_voltagebase"
                    # if feature was not extracted with bpe during extraction phase, do it now
                    if feat_name not in rec.efeatures:
                        rec.compute_efeatures([feat_name], efel_settings=efel_settings)
                    if not numpy.isnan(rec.efeatures[feat_name]):
                        expt_peak_v.append(rec.efeatures[feat_name])
                        expt_peak_amp.append(float(rec.amp))
                        expt_peak_amp_rel.append(float(rec.amp_rel))
                else:
                    feat_name = "voltage_deflection_vb_ssse"
                    # if feature was not extracted with bpe during extraction phase, do it now
                    if feat_name not in rec.efeatures:
                        rec.compute_efeatures([feat_name], efel_settings=efel_settings)
                    if not numpy.isnan(rec.efeatures[feat_name]):
                        expt_vd_v.append(rec.efeatures[feat_name])
                        expt_vd_amp.append(float(rec.amp))
                        expt_vd_amp_rel.append(float(rec.amp_rel))

    # binning the experimental datapoints to avoid crowding the figure
    expt_peak_amp_rel, expt_peak_v_rel, expt_peak_v_rel_err = binning(
        expt_peak_amp_rel, expt_peak_v, n_bin
    )
    # has to re-do full bining for absolute amp (and not just re-use expt_peak_v_rel)
    # because threshold is different for each rec
    expt_peak_amp, expt_peak_v_abs, expt_peak_v_abs_err = binning(expt_peak_amp, expt_peak_v, n_bin)

    expt_vd_amp_rel, expt_vd_v_rel, expt_vd_v_rel_err = binning(expt_vd_amp_rel, expt_vd_v, n_bin)
    # same here: has to re-do full bining for absolute amp
    expt_vd_amp, expt_vd_v_abs, expt_vd_v_abs_err = binning(expt_vd_amp, expt_vd_v, n_bin)

    exp_peak = {
        "amp_rel": expt_peak_amp_rel,
        "amp": expt_peak_amp,
        "feat_rel": expt_peak_v_rel,
        "feat_rel_err": expt_peak_v_rel_err,
        "feat_abs": expt_peak_v_abs,
        "feat_abs_err": expt_peak_v_abs_err,
    }
    exp_vd = {
        "amp_rel": expt_vd_amp_rel,
        "amp": expt_vd_amp,
        "feat_rel": expt_vd_v_rel,
        "feat_rel_err": expt_vd_v_rel_err,
        "feat_abs": expt_vd_v_abs,
        "feat_abs_err": expt_vd_v_abs_err,
    }
    return exp_peak, exp_vd


def fill_in_IV_curve_evaluator(evaluator, efel_settings, prot_name="iv"):
    """Returns a copy of the evaluator, with missing features added for IV_curve computation.

    Args:
        evaluator (CellEvaluator): cell evaluator
        efel_settings (dict): eFEL settings in the form {setting_name: setting_value}.
        prot_name (str): Only recordings from this protocol will be used.
    """
    updated_evaluator = copy.deepcopy(evaluator)
    # find protocols we expect to have the features we want to plot
    prot_max_v = []
    prot_v_deflection = []
    for prot in evaluator.fitness_protocols["main_protocol"].protocols.values():
        if prot_name.lower() in prot.name.lower():
            if len(prot.stimuli) < 1 or not hasattr(prot.stimuli[0], "amp_rel"):
                continue
            if prot.stimuli[0].amp_rel < 100:
                if prot.stimuli[0].amp_rel >= 0:
                    prot_max_v.append(prot.name)
                else:
                    prot_v_deflection.append(prot.name)

    # maps protocols of interest with all its associated features
    # also get protocol data we need for feature registration
    prots_to_feats = {}
    prots_data = {}
    for objective in evaluator.fitness_calculator.objectives:
        feat = objective.features[0]
        for protocol_name in prot_v_deflection + prot_max_v:
            if protocol_name in feat.recording_names[""]:
                if protocol_name not in prots_to_feats:
                    prots_to_feats[protocol_name] = []
                if protocol_name not in prots_data:
                    prots_data[protocol_name] = {
                        "stimulus_current": feat.stimulus_current,
                        "stim_start": feat.stim_start,
                        "stim_end": feat.stim_end,
                    }
                prots_to_feats[protocol_name].append(feat.efel_feature_name)
                continue

    # add missing features
    for protocol_name, feat_list in prots_to_feats.items():
        if protocol_name in prot_v_deflection and "voltage_deflection_vb_ssse" not in feat_list:
            logger.debug("Adding voltage_deflection_vb_ssse to %s in plot_IV_curves", protocol_name)
            feat_name = f"{protocol_name}.soma.v.voltage_deflection_vb_ssse"
            updated_evaluator.fitness_calculator.objectives.append(
                SingletonWeightObjective(
                    feat_name,
                    eFELFeatureBPEM(
                        feat_name,
                        efel_feature_name="voltage_deflection_vb_ssse",
                        recording_names={"": f"{protocol_name}.soma.v"},
                        stim_start=prots_data[protocol_name]["stim_start"],
                        stim_end=prots_data[protocol_name]["stim_end"],
                        exp_mean=1.0,  # fodder: not used
                        exp_std=1.0,  # fodder: not used
                        stimulus_current=prots_data[protocol_name]["stimulus_current"],
                        threshold=efel_settings.get("Threshold", None),
                        interp_step=efel_settings.get("interp_step", None),
                        weight=1.0,
                    ),
                    1.0,
                )
            )
        if protocol_name in prot_max_v and "maximum_voltage_from_voltagebase" not in feat_list:
            logger.debug(
                "Adding maximum_voltage_from_voltagebase to %s in plot_IV_curves", protocol_name
            )
            feat_name = f"{protocol_name}.soma.v.maximum_voltage_from_voltagebase"
            updated_evaluator.fitness_calculator.objectives.append(
                SingletonWeightObjective(
                    feat_name,
                    eFELFeatureBPEM(
                        feat_name,
                        efel_feature_name="maximum_voltage_from_voltagebase",
                        recording_names={"": f"{protocol_name}.soma.v"},
                        stim_start=prots_data[protocol_name]["stim_start"],
                        stim_end=prots_data[protocol_name]["stim_end"],
                        exp_mean=1.0,  # fodder: not used
                        exp_std=1.0,  # fodder: not used
                        stimulus_current=prots_data[protocol_name]["stimulus_current"],
                        threshold=efel_settings.get("Threshold", None),
                        interp_step=efel_settings.get("interp_step", None),
                        weight=1.0,
                    ),
                    1.0,
                )
            )

    return updated_evaluator


def get_experimental_FI_curve_for_plotting(cells, prot_name, n_bin=5):
    """Get experimental FI curve data used in plotting.

    Args:
        cells (list): list of bluepyefe.cell.Cell object to get recordings from
        prot_name (str): name of the protocol to use for the FI curve
        n_bin (int): number of bins to use
    """
    # pylint: disable=too-many-nested-blocks
    # experimental FI curve
    expt_amp_rel = []
    expt_amp = []
    expt_freq = []
    for cell in cells:
        for rec in cell.recordings:
            if rec.protocol_name.lower() == prot_name.lower():
                for feat in rec.efeatures:
                    if feat == "mean_frequency":
                        expt_freq.append(rec.efeatures[feat])
                        expt_amp.append(float(rec.amp))
                        if rec.amp_rel is not None:
                            expt_amp_rel.append(float(rec.amp_rel))

    # binning the experimental datapoints to avoid crowding the figure
    expt_amp_rel, expt_freq_rel, expt_freq_rel_err = binning(expt_amp_rel, expt_freq, n_bin)
    # has to re-do full bining for absolute amp (and not just re-use expt_freq_rel)
    # because threshold is different for each rec
    expt_amp, expt_freq_abs, expt_freq_abs_err = binning(expt_amp, expt_freq, n_bin)

    return (
        expt_amp_rel,
        expt_freq_rel,
        expt_freq_rel_err,
        expt_amp,
        expt_freq_abs,
        expt_freq_abs_err,
    )


def get_simulated_FI_curve_for_plotting(evaluator, responses, prot_name):
    """Get FI curve data from model used in plotting.

    Args:
        evaluator (CellEvaluator): cell evaluator
        responses (dict): responses of the cell model
        prot_name (str): name of the protocol to use for the FI curve
    """
    values = evaluator.fitness_calculator.calculate_values(responses)
    simulated_freq = []
    simulated_amp_rel = []
    simulated_amp = []
    for val in values:
        if prot_name.lower() in val.lower():
            protocol_name = get_protocol_name(val)
            amp_temp = float(protocol_name.split("_")[-1])
            if "mean_frequency" in val:
                simulated_freq.append(values[val])
                if "bpo_threshold_current" in responses:
                    simulated_amp_rel.append(amp_temp)
                    simulated_amp.append(rel_to_abs_amplitude(amp_temp, responses))
                else:
                    simulated_amp_rel.append(numpy.nan)
                    simulated_amp.append(amp_temp)

    return simulated_amp_rel, simulated_amp, simulated_freq


def get_impedance(time, voltage, current, stim_start, stim_end, efel_settings):
    """Get impedance for plotting.

    Args:
        time (list): time series
        voltage (list): voltage series
        current (list): injected current series
        stim_start (float): stimulus start time
        stim_end (float): stimulus end time
        efel_settings (dict): eFEL settings in the form {setting_name: setting_value}
    """
    from scipy.ndimage.filters import gaussian_filter1d

    dt = 0.1
    Z_max_freq = 50.0
    if efel_settings is not None:
        dt = efel_settings.get("interp_step", dt)
        Z_max_freq = efel_settings.get("impedance_max_freq", Z_max_freq)

    trace = {
        "T": time,
        "V": voltage,
        "I": current,
        "stim_start": [stim_start],
        "stim_end": [stim_end],
    }

    efel.reset()
    for name, value in efel_settings.items():
        efel.set_setting(name, value)

    efel_vals = efel.get_feature_values(
        [trace],
        [
            "voltage_base",
            "steady_state_voltage_stimend",
            "current_base",
            "steady_state_current_stimend",
        ],
    )
    if efel_vals[0]["voltage_base"] is not None:
        holding_voltage = efel_vals[0]["voltage_base"][0]
    elif efel_vals[0]["steady_state_voltage_stimend"] is not None:
        holding_voltage = efel_vals[0]["steady_state_voltage_stimend"][0]
    else:
        logger.warning(
            "Could not get impedance because neither voltage_base "
            "nor steady_state_voltage_stimend could be retrieve from efel."
        )
        return None, None
    if efel_vals[0]["current_base"] is not None:
        holding_current = efel_vals[0]["current_base"][0]
    elif efel_vals[0]["steady_state_current_stimend"] is not None:
        holding_current = efel_vals[0]["steady_state_current_stimend"][0]
    else:
        logger.warning(
            "Could not get impedance because neither current_base "
            "nor steady_state_current_stimend could be retrieve from efel."
        )
        return None, None

    normalized_voltage = voltage - holding_voltage
    normalized_current = current - holding_current

    fft_volt = numpy.fft.fft(normalized_voltage)
    fft_cur = numpy.fft.fft(normalized_current)
    if any(fft_cur) == 0:
        return [], []
    # convert dt from ms to s to have freq in Hz
    freq = numpy.fft.fftfreq(len(normalized_voltage), d=dt / 1000.0)
    Z = fft_volt / fft_cur
    norm_Z = abs(Z) / max(abs(Z))
    select_idxs = numpy.swapaxes(numpy.argwhere((freq > 0) & (freq <= Z_max_freq)), 0, 1)[0]
    smooth_Z = gaussian_filter1d(norm_Z[select_idxs], 10)

    return freq[select_idxs], smooth_Z


def get_sinespec_evaluator(evaluator, sinespec_settings, efel_settings):
    """Returns evaluator with pre-protocols (if threshold based) and sinespec protocol
    and impedance feature.

    Args:
        evaluator (CellEvaluator): cell evaluator
        sinespec_settings (dict): contains amplitude settings for the SineSpec protocol,
            with keys 'amp' and 'threshold_based'.
            'amp' should be in percentage of threshold if 'threshold_based' is True, e.g. 150,
            or in nA if 'threshold_based' if false, e.g. 0.1.
        efel_settings (dict): eFEL settings in the form {setting_name: setting_value}
    """
    new_eval = copy.deepcopy(evaluator)

    # remove protocols except for pre-protocols
    old_prots = new_eval.fitness_protocols["main_protocol"].protocols
    new_prots = {}
    if sinespec_settings["threshold_based"]:
        for k, v in old_prots.items():
            if k in PRE_PROTOCOLS:
                new_prots[k] = v

    # add sinespec protocol
    prot_name = f"SineSpec_{sinespec_settings['amp']}"
    if sinespec_settings["threshold_based"]:
        prot_cls = ThresholdBasedProtocol
        kwargs = {"thresh_perc": sinespec_settings["amp"], "amp": None}
    else:
        prot_cls = BPEMProtocol
        kwargs = {"thresh_perc": None, "amp": sinespec_settings["amp"]}

    sinespec_prot = prot_cls(
        name=prot_name,
        stimulus=SineSpec(
            location=soma_loc, delay=300.0, duration=5000.0, totduration=5300.0, **kwargs
        ),
        recordings=[
            FixedDtRecordingCustom(f"{prot_name}.soma.v", location=soma_loc, variable="v"),
            FixedDtRecordingStimulus(f"{prot_name}.iclamp.i", location=None, variable="i"),
        ],
    )
    new_prots[prot_name] = sinespec_prot

    new_eval.fitness_protocols["main_protocol"].protocols = new_prots
    new_eval.fitness_protocols["main_protocol"].execution_order = new_eval.fitness_protocols[
        "main_protocol"
    ].compute_execution_order()

    # remove non-pre-protocol features
    new_objectives = []
    if sinespec_settings["threshold_based"]:
        new_objectives = [
            obj
            for obj in new_eval.fitness_calculator.objectives
            if any(a in obj.name for a in PRE_PROTOCOLS)
        ]

    # add impedance feature
    feat_name = f"{prot_name}.soma.v.impedance"
    new_objectives.append(
        SingletonWeightObjective(
            feat_name,
            eFELFeatureBPEM(
                feat_name,
                efel_feature_name="impedance",
                recording_names={"": f"{prot_name}.soma.v"},
                stim_start=300.0,
                stim_end=2300.0,
                exp_mean=1.0,  # fodder: not used
                exp_std=1.0,  # fodder: not used
                stimulus_current=sinespec_prot.amplitude,
                threshold=efel_settings.get("Threshold", None),
                interp_step=efel_settings.get("interp_step", None),
            ),
            1.0,
        )
    )
    new_eval.fitness_calculator.objectives = new_objectives

    return new_eval


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
        prot_name = get_protocol_name(name)
        # prot_name can be e.g. RMPProtocol, or RMPProtocol_apical055
        if not any(to_skip_ in prot_name for to_skip_ in to_skip):
            loc_name = get_loc_name(name)
            curr_name = get_curr_name(name)

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

            # check if we should skip curr_name == "i" case
            if curr_name == "v":
                ordered_keys[prot_name][loc_name]["voltage_key"] = name
            elif curr_name[-1] == "i":
                ordered_keys[prot_name][loc_name]["ion_conc_keys"].append(name)
                ordered_keys[prot_name][loc_name]["ion_conc_names"].append(curr_name)
            # assumes we don't have any extra-cellular concentrations (that ends with 'o')
            else:
                ordered_keys[prot_name][loc_name]["current_keys"].append(name)
                ordered_keys[prot_name][loc_name]["current_names"].append(curr_name)

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
