"""Compute the threshold and holding current using bglibpy, adapted from BluePyThresh."""
from functools import partial
from copy import copy
import logging
from pathlib import Path

import efel

from .tools.evaluator import evaluate_combos

logger = logging.getLogger(__name__)
AXON_LOC = "self.axonal[1](0.5)._ref_v"


def calculate_threshold_current(config, holding_current, cell_kwargs):
    """Calculate threshold current"""
    min_current_spike_count = run_spike_sim(
        config,
        cell_kwargs,
        holding_current,
        config["min_threshold_current"],
    )
    logger.info("min %s", min_current_spike_count)
    if min_current_spike_count > 0:
        logger.info("Cell is firing spontaneously at min current, we divide by 2")
        if config["min_threshold_current"] == 0:
            return None
        config["max_threshold_current"] = copy(config["min_threshold_current"])
        config["min_threshold_current"] /= 2.0
        return calculate_threshold_current(config, holding_current, cell_kwargs)

    max_current_spike_count = run_spike_sim(
        config,
        cell_kwargs,
        holding_current,
        config["max_threshold_current"],
    )

    logger.info("max %s", max_current_spike_count)
    if max_current_spike_count < 1:
        logger.info("Cell is not firing at max current, we multiply by 2")
        config["min_threshold_current"] = copy(config["max_threshold_current"])
        config["max_threshold_current"] *= 2.0
        return calculate_threshold_current(config, holding_current, cell_kwargs)

    return binsearch_threshold_current(
        config,
        cell_kwargs,
        holding_current,
        config["min_threshold_current"],
        config["max_threshold_current"],
        0,
    )


def binsearch_threshold_current(
    config,
    cell_kwargs,
    holding_current,
    min_current,
    max_current,
    depth,
):
    """Binary search for threshold currents"""
    logger.info("current %s, %s at depth %s", min_current, max_current, depth)
    med_current = min_current + abs(min_current - max_current) / 2

    if depth >= int(config["max_recursion_depth"]):
        logger.info("Reached maximal recursion depth, return latest value")
        return med_current

    spike_count = run_spike_sim(
        config,
        cell_kwargs,
        holding_current,
        med_current,
    )
    logger.info("Med spike count %d", spike_count)

    if spike_count == 0:
        logger.info("Searching upwards")
        return binsearch_threshold_current(
            config, cell_kwargs, holding_current, med_current, max_current, depth + 1
        )

    hs_spike_count = run_spike_sim(
        config,
        cell_kwargs,
        holding_current,
        float(config["highest_silent_perc"]) / 100 * med_current,
    )

    logger.info("Highest silent spike count %d", hs_spike_count)

    if hs_spike_count == 0 and spike_count <= int(config["max_spikes_at_threshold"]):
        logger.info("Found threshold %s", med_current)
        return med_current

    logger.info("Searching downwards")
    return binsearch_threshold_current(
        config, cell_kwargs, holding_current, min_current, med_current, depth + 1
    )


def run_spike_sim(config, cell_kwargs, holding_current, step_current):
    """Run simulation on a cell and compute number of spikes."""
    import bglibpy

    cell = bglibpy.Cell(**cell_kwargs)
    is_deterministic = set_cell_deterministic(cell, config["deterministic"])

    cell.add_step(0, config["step_stop"], holding_current)
    cell.add_step(config["step_start"], config["step_stop"], step_current)

    if config["spike_at_ais"]:
        cell.add_recordings(["neuron.h._ref_t", AXON_LOC], dt=cell.record_dt)

    sim = bglibpy.Simulation()
    sim.run(
        config["step_stop"],
        celsius=config["celsius"],
        v_init=config["v_init"],
        cvode=is_deterministic,
    )

    time = cell.get_time()
    if config["spike_at_ais"]:
        voltage = cell.get_recording(AXON_LOC)
    else:
        voltage = cell.get_soma_voltage()

    cell.delete()

    efel.reset()
    efel.setIntSetting("strict_stiminterval", True)
    spike_count_array = efel.getFeatureValues(
        [
            {
                "T": time,
                "V": voltage,
                "stim_start": [config["step_start"]],
                "stim_end": [config["step_stop"]],
            }
        ],
        ["Spikecount"],
    )[0]["Spikecount"]

    if spike_count_array is None or len(spike_count_array) != 1:
        raise Exception("Error during spike count calculation")
    return spike_count_array[0]


def set_cell_deterministic(cell, deterministic):
    """Disable stochasticity in ion channels"""
    is_deterministic = True
    for section in cell.cell.all:
        for compartment in section:
            for mech in compartment:
                mech_name = mech.name()
                if "Stoch" in mech_name:
                    if not deterministic:
                        is_deterministic = False
                    setattr(
                        section,
                        "deterministic_%s" % mech_name,
                        1 if deterministic else 0,
                    )
    return is_deterministic


def calculate_holding_current(holding_voltage, cell_kwargs, deterministic=True):
    """Calculate holding current.

    adapted from: bglibpy.tools.holding_current_subprocess,
    """
    import bglibpy

    cell = bglibpy.Cell(**cell_kwargs)
    is_deterministic = set_cell_deterministic(cell, deterministic)

    vclamp = bglibpy.neuron.h.SEClamp(0.5, sec=cell.soma)
    vclamp.rs = 0.01
    vclamp.dur1 = 2000
    vclamp.amp1 = holding_voltage

    simulation = bglibpy.Simulation()
    simulation.run(1000, cvode=is_deterministic)

    holding_current = vclamp.i
    cell.delete()

    return holding_current


def _current_bglibpy_evaluation(
    combo,
    protocol_config,
    emodels_hoc_dir,
    morphology_path="morphology_path",
    template_format="v6_ais_scaler",
):
    """Compute the threshold and holding currents using bglibpy."""
    AIS_scale = None
    if template_format == "v6_ais_scaler":
        AIS_scale = combo.AIS_scale

    cell_kwargs = {
        "template_filename": str(Path(emodels_hoc_dir) / f"{combo.emodel}.hoc"),
        "morphology_name": Path(combo[morphology_path]).name,
        "morph_dir": str(Path(combo[morphology_path]).parent),
        "template_format": template_format,
        "extra_values": {
            "holding_current": None,
            "threshold_current": None,
            "AIS_scaler": AIS_scale,
        },
    }

    holding_current = calculate_holding_current(protocol_config["holding_voltage"], cell_kwargs)
    threshold_current = calculate_threshold_current(protocol_config, holding_current, cell_kwargs)
    return {"holding_current": holding_current, "threshold_current": threshold_current}


def evaluate_currents_bglibpy(
    morphs_combos_df,
    protocol_config,
    emodels_hoc_dir,
    task_ids=None,
    morphology_path="morphology_path",
    continu=False,
    combos_db_filename="eval_db.sql",
    parallel_factory=None,
    template_format="v6_ais_scaler",
):
    """Compute the threshold and holding currents using bglibpy."""
    current_evaluation_bglibpy = partial(
        _current_bglibpy_evaluation,
        protocol_config=protocol_config,
        emodels_hoc_dir=emodels_hoc_dir,
        morphology_path=morphology_path,
        template_format=template_format,
    )
    return evaluate_combos(
        morphs_combos_df,
        current_evaluation_bglibpy,
        new_columns=[["holding_current", 0.0], ["threshold_current", 0.0]],
        task_ids=task_ids,
        continu=continu,
        parallel_factory=parallel_factory,
        combos_db_filename=combos_db_filename,
    )
