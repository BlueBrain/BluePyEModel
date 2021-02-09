"""Helper functions to use bglibpy."""
import json
from pathlib import Path

import bglibpy
import efel
import libsonata as sonata
import numpy as np
import yaml

from bluepyemodel.ais_synthesis.bglibpy_evaluators import calculate_holding_current
from bluepyemodel.ais_synthesis.bglibpy_evaluators import calculate_threshold_current


def _get_cell_kwargs_custom_template(
    morphology_name, morphology_dir, emodel, emodels_hoc_dir, scale=None
):

    cell_kwargs = {
        "template_filename": str(Path(emodels_hoc_dir) / f"{emodel}.hoc"),
        "morphology_name": morphology_name,
        "morph_dir": morphology_dir,
        "template_format": "v6_ais_scaler" if scale is not None else "v6",
        "extra_values": {
            "holding_current": None,
            "threshold_current": None,
            "AIS_scaler": scale,
        },
    }
    return cell_kwargs


def _get_cell_kwargs_from_sim(ssim, cell, scale=None):
    """Create bglibpy kwargs from ssim object."""
    emodels_hoc_dir = Path(ssim.bc.Run["METypePath"])
    morph_dir = ssim.bc.Run["MorphologyPath"]
    print(cell.template_name)
    emodel = "_".join(cell.template_name.split("_")[:-1])
    cell_kwargs = {
        "template_filename": str(emodels_hoc_dir / f"{emodel}.hoc"),
        "morphology_name": cell.morphology_name,
        "morph_dir": str(Path(morph_dir) / "ascii"),
        "template_format": "v6_ais_scaler" if scale is not None else "v6",
        "extra_values": {
            "holding_current": None,
            "threshold_current": None,
            "AIS_scaler": scale,
        },
    }

    return cell_kwargs


def _copy_cell_attrs(from_cell, to_cell):
    to_cell.rng_settings = from_cell.rng_settings
    to_cell.threshold = from_cell.threshold
    to_cell.hypamp = from_cell.hypamp


def _add_synapses(_cell, cell, ssim, out_h5):

    # copycat synapses and get pre_gids
    pre_gids = []
    for sid, syn in _cell.synapses.items():
        cell.add_replay_synapse(
            sid,
            syn.syn_description,
            syn.connection_parameters,
            popids=None,
            extracellular_calcium=syn.extracellular_calcium,
        )
        pre_gids.append(syn.syn_description[0])
    pre_gids = tuple(set(pre_gids))

    # load pre-synaptic spikes from SONATA
    spkrd = sonata.SpikeReader(out_h5)
    pop = spkrd.get_population_names()[0]

    sel = sonata.Selection(pre_gids)
    spikes = np.array(spkrd[pop].get(sel))
    if spikes.size:
        t_min = np.min(spikes[:, 1])  # first pre-spike
        spikes[:, 1] = spikes[:, 1] - t_min  # align to t = 0

        # put spike trains into dict
        trains = {}
        for gid in pre_gids:
            w = np.where(spikes[:, 0] == gid)
            if w:
                trains[gid] = spikes[w][:, 1]

        # add replay spike trains (taken from BGLibPy)
        replay_delay = t_min  # set replay start (default is original start)
        for syn_id, synapse in cell.synapses.items():
            syn_description = synapse.syn_description
            connection_parameters = synapse.connection_parameters
            pre_gid = syn_description[0]

            pre_spiketrain = trains.setdefault(pre_gid, None)
            if pre_spiketrain is not None:
                pre_spiketrain = pre_spiketrain + replay_delay
            connection = bglibpy.Connection(
                cell.synapses[syn_id], pre_spiketrain=pre_spiketrain, pre_cell=None, stim_dt=ssim.dt
            )

            if connection is not None:
                cell.connections[syn_id] = connection
                if "DelayWeights" in connection_parameters:
                    for delay, weight_scale in connection_parameters["DelayWeights"]:
                        cell.add_replay_delayed_weight(
                            syn_id, delay, weight_scale * connection.weight
                        )

    return cell


def axon_loc(cell):
    return "neuron.h." + [x.name() for x in cell.cell.getCell().axonal][1] + "(0.5)._ref_v"


def _add_recordings(cell):
    cell.add_recordings(["neuron.h._ref_t", axon_loc(cell)], dt=cell.record_dt)


def get_cell(
    circuit_config=None,
    gid=None,
    add_synapses=False,
    morphology_name=None,
    morphology_dir=None,
    emodel=None,
    emodels_hoc_dir=None,
    out_h5=None,
    calc_threshold=False,
):
    if morphology_name is not None:
        protocol_config = yaml.safe_load(open("protocol_config.yaml"))
        cell_kwargs = _get_cell_kwargs_custom_template(
            morphology_name, morphology_dir, emodel, emodels_hoc_dir
        )

        cell = bglibpy.Cell(**cell_kwargs)
        cell.hypamp = calculate_holding_current(cell, protocol_config)
        if calc_threshold:
            cell.threshold = calculate_threshold_current(cell, protocol_config, cell.hypamp)
    if circuit_config and gid is not None:
        ssim = bglibpy.SSim(circuit_config)
        ssim.instantiate_gids([gid], add_synapses=add_synapses, add_minis=False)
        _cell = ssim.cells[gid]
        cell_kwargs = _get_cell_kwargs_from_sim(ssim, _cell)
        cell = bglibpy.Cell(**cell_kwargs)
        _copy_cell_attrs(_cell, cell)

        if add_synapses and out_h5 is not None:
            print("adding synapses")
            cell = _add_synapses(_cell, cell, ssim, out_h5)

        _cell.delete()

    print("cell_kwargs: ", json.dumps(cell_kwargs, indent=4))
    print(f"threshold = {cell.threshold}, holding = {cell.hypamp}")
    _add_recordings(cell)
    return cell


def get_spikefreq(results, start, stop, location="voltage_soma"):
    """Compute spikefreq from a trace."""
    efel.reset()
    efel.setIntSetting("strict_stiminterval", True)
    return (
        efel.getFeatureValues(
            [
                {
                    "T": results["time"],
                    "V": results[location],
                    "stim_start": [start],
                    "stim_end": [stop],
                }
            ],
            ["Spikecount"],
        )[0]["Spikecount"][0]
        / (stop - start)
        * 1000.0
    )
