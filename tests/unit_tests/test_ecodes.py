"""ECodes tests."""

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

import numpy
import pytest
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.simulators import NrnSimulator

from bluepyemodel.ecode import *
from tests.test_models import dummycells

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)


def run_stim_on_dummy_cell(stimulus):
    """Run dummy cell with stimulus and return time and current of stimulus."""
    # instantiate
    nrn_sim = NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
    icell = dummy_cell.instantiate(sim=nrn_sim)
    stimulus.instantiate(sim=nrn_sim, icell=icell)

    # set up time, current vectors
    stim_i_vec = nrn_sim.neuron.h.Vector()
    stim_i_vec.record(stimulus.iclamp._ref_i)  # pylint: disable=W0212
    t_vec = nrn_sim.neuron.h.Vector()
    t_vec.record(nrn_sim.neuron.h._ref_t)  # pylint: disable=W0212

    # run stimulus
    nrn_sim.run(stimulus.total_duration)

    # get time, current
    current = numpy.array(stim_i_vec.to_python())
    time = numpy.array(t_vec.to_python())

    return time, current


def get_idrest_stimulus():
    """Return IDRest stimulus and stim properties."""
    # default values
    delay = 250.0
    duration = 1350.0
    total_duration = 1850.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["idrest"](location=soma_loc, **prot_def)

    return stimulus, delay, duration, total_duration, prot_def["amp"], prot_def["holding_current"]


def get_apwaveform_stimulus():
    """Return APWaveform stimulus and stim properties."""
    # default values
    delay = 250.0
    duration = 50.0
    total_duration = 550.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["apwaveform"](location=soma_loc, **prot_def)

    return stimulus, delay, duration, total_duration, prot_def["amp"], prot_def["holding_current"]


def get_firepattern_stimulus():
    """Return FirePattern stimulus and stim properties."""
    # default values
    delay = 250.0
    duration = 3600.0
    total_duration = 4100.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["firepattern"](location=soma_loc, **prot_def)

    return stimulus, delay, duration, total_duration, prot_def["amp"], prot_def["holding_current"]


def get_iv_stimulus():
    """Return IV stimulus and stim properties."""
    # default values
    delay = 250.0
    duration = 3000.0
    total_duration = 3500.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["iv"](location=soma_loc, **prot_def)

    return stimulus, delay, duration, total_duration, prot_def["amp"], prot_def["holding_current"]


def get_dehyperpol_stimulus():
    """Return DeHyperpol stimulus and stim properties."""
    # default values
    delay = 250.0
    tmid = 520.0
    toff = 970.0
    total_duration = 1220.0

    # generate stimulus
    prot_def = {"amp": 0.2, "amp2": -0.1, "holding_current": -0.001}
    stimulus = eCodes["dehyperpol"](location=soma_loc, **prot_def)

    return (
        stimulus,
        delay,
        tmid,
        toff,
        total_duration,
        prot_def["holding_current"],
        prot_def["amp"],
        prot_def["amp2"],
    )


def get_hyperdepol_stimulus():
    """Return HyperDepol stimulus and stim properties."""
    # default values
    delay = 250.0
    tmid = 700.0
    toff = 970.0
    total_duration = 1220.0

    # generate stimulus
    prot_def = {"depol_amp": 0.2, "hyper_amp": -0.1, "holding_current": -0.001}
    stimulus = eCodes["hyperdepol"](location=soma_loc, **prot_def)

    return (
        stimulus,
        delay,
        tmid,
        toff,
        total_duration,
        prot_def["holding_current"],
        prot_def["depol_amp"],
        prot_def["hyper_amp"],
    )


def get_ramp_stimulus():
    """Return Ramp stimulus and stim properties."""
    # default values
    delay = 250.0
    duration = 1350.0
    total_duration = 1850.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["ramp"](location=soma_loc, **prot_def)

    return stimulus, delay, duration, total_duration, prot_def["amp"], prot_def["holding_current"]


def get_sahp_stimulus():
    """Return SAHP stimulus and stim properties."""
    # default values
    delay = 250.0
    tmid = 500.0
    tmid2 = 725.0
    toff = 1175.0
    total_duration = 1425.0

    prot_def = {"holding_current": -0.001, "long_amp": 0.2, "amp": 0.3}
    stimulus = eCodes["sahp"](location=soma_loc, **prot_def)

    return (
        stimulus,
        delay,
        tmid,
        tmid2,
        toff,
        total_duration,
        prot_def["holding_current"],
        prot_def["long_amp"],
        prot_def["amp"],
    )


def get_poscheops_stimulus():
    """Return PosCheops stimulus and stim properties."""
    # default values
    delay = 250.0
    ramp1_duration = 4000.0
    ramp2_duration = 2000.0
    ramp3_duration = 1333.0
    inter_delay = 2000.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["poscheops"](location=soma_loc, **prot_def)

    return (
        stimulus,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        prot_def["amp"],
        prot_def["holding_current"],
    )

def get_negcheops_stimulus():
    """Return NegCheops stimulus and stim properties."""
    # default values
    delay = 1750.0
    ramp1_duration = 3333.0
    ramp2_duration = 1666.0
    ramp3_duration = 1111.0
    inter_delay = 2000.0
    totduration = 18222.0
    holding_current = 0.0

    # generate stimulus
    prot_def = {"amp": -0.2}
    stimulus = eCodes["negcheops"](location=soma_loc, **prot_def)

    return (
        stimulus,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        prot_def["amp"],
        holding_current,
    )

def get_sinespec_stimulus():
    """Return SineSpec stimulus and stim properties."""
    # default values
    duration = 5000.0

    # custom values, to check edge case (delay > 0)
    delay = 100.0
    total_duration = 5100.0 # delay + duration

    # generate stimulus
    prot_def = {
        "amp": 0.2, "holding_current": -0.001, "delay": delay, "totduration": total_duration
    }
    stimulus = eCodes["sinespec"](location=soma_loc, **prot_def)

    return stimulus, delay, duration, total_duration, prot_def["amp"], prot_def["holding_current"]

def get_spikerecmultispikes_stimulus():
    """Return SpikeRecMultiSpikes stimulus and stim properties."""
    # default values
    delay = 10.0
    n_spikes = 2
    spike_duration = 3.5
    delta = 3.5
    total_duration = 1500.0

    # generate stimulus
    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["spikerecmultispikes"](location=soma_loc, **prot_def)

    return stimulus, delay, n_spikes, spike_duration, delta, total_duration, prot_def["amp"], prot_def["holding_current"]


def check_ramp(time, current, ton, duration, holding_current, amp, ramp_up=True):
    """Assert the ramp part of a stimulus behaves as expected."""
    current_ramp = current[numpy.where((ton <= time) & (time < ton + duration))]
    # timesteps can be variable
    time_ramp = time[numpy.where((ton <= time) & (time < ton + duration))]
    # ramp going up case
    if ramp_up:
        theoretical_current_ramp = holding_current + amp * (time_ramp - ton) / duration
    # ramp going down case
    else:
        theoretical_current_ramp = holding_current + amp * (1.0 - (time_ramp - ton) / duration)
    assert numpy.all(current_ramp == pytest.approx(theoretical_current_ramp))


def test_subwhitenoise():

    prot_def = {"amp": 0.2}
    stimulus = eCodes["subwhitenoise"](location=soma_loc, **prot_def)
    t, i = stimulus.generate()

    assert stimulus.name == "SubWhiteNoise"
    assert stimulus.total_duration == 5099.8
    assert len(i) == 50999

    prot_def = {"amp": 0.2, "holding_current": -0.001}
    stimulus = eCodes["subwhitenoise"](location=soma_loc, **prot_def)
    t, i = stimulus.generate()

    assert stimulus.name == "SubWhiteNoise"
    assert stimulus.total_duration == 5099.8
    assert len(i) == 50999

def test_whitenoise():

    prot_def = {"amp": 0.2, "mu": 1}
    stimulus = eCodes["whitenoise"](location=soma_loc, **prot_def)
    t, i = stimulus.generate()

    assert stimulus.name == "WhiteNoise"
    assert stimulus.total_duration == 60099.9
    assert len(i) == 601000

    prot_def = {"amp": 0.2, "holding_current": -0.001, "mu": 1}
    stimulus = eCodes["whitenoise"](location=soma_loc, **prot_def)
    t, i = stimulus.generate()

    assert stimulus.name == "WhiteNoise"
    assert stimulus.total_duration == 60099.9
    assert len(i) == 601000

def test_noiseou3():

    prot_def = {"amp": 0.2, "mu": 1}
    stimulus = eCodes["noiseou3"](location=soma_loc, **prot_def)
    t, i = stimulus.generate()

    assert stimulus.name == "NoiseOU3"
    assert stimulus.total_duration == 60099.9
    assert len(i) == 601000

    prot_def = {"amp": 0.2, "holding_current": -0.001, "mu": 1}
    stimulus = eCodes["noiseou3"](location=soma_loc, **prot_def)
    t, i = stimulus.generate()

    assert stimulus.name == "NoiseOU3"
    assert stimulus.total_duration == 60099.9
    assert len(i) == 601000


def check_idrest_stim(time, current, delay, duration, total_duration, holding_current, amp):
    """Assert IDrest stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # after stimulus
    current_after = current[numpy.where((delay + duration < time) & (time <= total_duration))]
    assert numpy.all(current_after == holding_current)

    # during stimulus
    current_during = current[numpy.where((delay < time) & (time < delay + duration))]
    assert numpy.all(current_during == holding_current + amp)


def test_idrest():
    """Test IDRest generate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_idrest_stimulus()
    time, current = stimulus.generate()

    assert stimulus.name == "IDrest"
    assert stimulus.total_duration == total_duration
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_idrest_instantiate():
    """Test IDRest instantiate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_idrest_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_apwaveform():
    """Test APWaveform generate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_apwaveform_stimulus()
    time, current = stimulus.generate()

    # test
    assert stimulus.name == "APWaveform"
    assert stimulus.total_duration == total_duration
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_apwaveform_instantiate():
    """Test APWaveform instantiate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_apwaveform_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_firepattern():
    """Test FirePattern generate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_firepattern_stimulus()
    time, current = stimulus.generate()

    # test
    assert stimulus.name == "FirePattern"
    assert stimulus.total_duration == total_duration
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_firepattern_instantiate():
    """Test FirePattern instantiate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_firepattern_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_iv():
    """Test IV generate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_iv_stimulus()
    time, current = stimulus.generate()

    # test
    assert stimulus.name == "IV"
    assert stimulus.total_duration == total_duration
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_iv_instantiate():
    """Test IV instantiate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_iv_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_idrest_stim(time, current, delay, duration, total_duration, 0.0, amp)


def check_dehyperpol_stim(
    time, current, delay, tmid, toff, total_duration, holding_current, amp1, amp2
):
    """Assert DeHyperpol/HyperDepol stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # after stimulus
    current_after = current[numpy.where((toff < time) & (time <= total_duration))]
    assert numpy.all(current_after == holding_current)

    # between ton and tmid
    current_ton_tmid = current[numpy.where((delay < time) & (time < tmid))]
    assert numpy.all(current_ton_tmid == holding_current + amp1)

    # between tmid and toff
    current_tmid_toff = current[numpy.where((tmid < time) & (time < toff))]
    assert numpy.all(current_tmid_toff == holding_current + amp2)


def test_dehyperpol():
    """Test DeHyperpol generate."""
    (
        stimulus,
        delay,
        tmid,
        toff,
        total_duration,
        holding_curr,
        depol_amp,
        hyper_amp,
    ) = get_dehyperpol_stimulus()
    time, current = stimulus.generate()

    # test
    assert stimulus.name == "DeHyperpol"
    assert stimulus.total_duration == total_duration
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, holding_curr, depol_amp, hyper_amp
    )

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, 0.0, depol_amp, hyper_amp
    )


def test_dehyperpol_instantiate():
    """Test DeHyperpol instantiate."""
    (
        stimulus,
        delay,
        tmid,
        toff,
        total_duration,
        holding_curr,
        depol_amp,
        hyper_amp,
    ) = get_dehyperpol_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, holding_curr, depol_amp, hyper_amp
    )

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, 0.0, depol_amp, hyper_amp
    )


def test_hyperdepol():
    """Test HyperDepol generate."""
    (
        stimulus,
        delay,
        tmid,
        toff,
        total_duration,
        holding_curr,
        depol_amp,
        hyper_amp,
    ) = get_hyperdepol_stimulus()
    time, current = stimulus.generate()

    # test
    assert stimulus.name == "HyperDepol"
    assert stimulus.total_duration == total_duration
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, holding_curr, hyper_amp, depol_amp
    )

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, 0.0, hyper_amp, depol_amp
    )


def test_hyperdepol_instantiate():
    """Test HyperDepol instantiate."""
    (
        stimulus,
        delay,
        tmid,
        toff,
        total_duration,
        holding_curr,
        depol_amp,
        hyper_amp,
    ) = get_hyperdepol_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, holding_curr, hyper_amp, depol_amp
    )

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_dehyperpol_stim(
        time, current, delay, tmid, toff, total_duration, 0.0, hyper_amp, depol_amp
    )


def check_ramp_stim(time, current, delay, duration, total_duration, holding_current, amp):
    """Assert Ramp stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # after stimulus
    current_after = current[numpy.where((delay + duration < time) & (time <= total_duration))]
    assert numpy.all(current_after == holding_current)

    # during stimulus
    check_ramp(time, current, delay, duration, holding_current, amp, ramp_up=True)


def test_ramp():
    """Test Ramp generate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_ramp_stimulus()
    time, current = stimulus.generate()

    assert stimulus.name == "Ramp"
    assert stimulus.total_duration == total_duration
    check_ramp_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_ramp_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_ramp_instantiate():
    """Test Ramp instantiate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_ramp_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_ramp_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_ramp_stim(time, current, delay, duration, total_duration, 0.0, amp)


def check_sahp_stim(
    time, current, delay, tmid, tmid2, toff, total_duration, holding_current, long_amp, amp
):
    """Assert SAHP stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # after stimulus
    current_after = current[numpy.where((toff < time) & (time <= total_duration))]
    assert numpy.all(current_after == holding_current)

    # between ton and tmid
    current_ton_tmid = current[numpy.where((delay < time) & (time < tmid))]
    assert numpy.all(current_ton_tmid == holding_current + long_amp)

    # between tmid2 and toff
    current_tmid_toff = current[numpy.where((tmid2 < time) & (time < toff))]
    assert numpy.all(current_tmid_toff == holding_current + long_amp)

    # between tmid and tmid2
    current_tmid_tmid2 = current[numpy.where((tmid < time) & (time < tmid2))]
    assert numpy.all(current_tmid_tmid2 == holding_current + amp)


def test_sahp():
    """Test SAHP generate."""
    (
        stimulus,
        delay,
        tmid,
        tmid2,
        toff,
        total_duration,
        holding_curr,
        long_amp,
        amp,
    ) = get_sahp_stimulus()
    time, current = stimulus.generate()

    # test
    assert stimulus.name == "sAHP"
    assert stimulus.total_duration == total_duration
    check_sahp_stim(
        time, current, delay, tmid, tmid2, toff, total_duration, holding_curr, long_amp, amp
    )

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_sahp_stim(
        time, current, delay, tmid, tmid2, toff, total_duration, 0.0, long_amp, amp
    )


def test_sahp_instantiate():
    """Test SAHP instantiate."""
    (
        stimulus,
        delay,
        tmid,
        tmid2,
        toff,
        total_duration,
        holding_curr,
        long_amp,
        amp,
    ) = get_sahp_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_sahp_stim(
        time, current, delay, tmid, tmid2, toff, total_duration, holding_curr, long_amp, amp
    )

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_sahp_stim(
        time, current, delay, tmid, tmid2, toff, total_duration, 0.0, long_amp, amp
    )


def check_poscheops_stim(
    time,
    current,
    delay,
    ramp1_duration,
    ramp2_duration,
    ramp3_duration,
    inter_delay,
    holding_current,
    amp,
):
    """Assert Ramp stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # during ramp1 up
    check_ramp(time, current, delay, ramp1_duration, holding_current, amp, ramp_up=True)
    # during ramp1 down
    check_ramp(
        time, current, delay + ramp1_duration, ramp1_duration, holding_current, amp, ramp_up=False
    )

    # during ramp2 up
    ton2 = delay + 2 * ramp1_duration + inter_delay
    check_ramp(time, current, ton2, ramp2_duration, holding_current, amp, ramp_up=True)
    # during ramp2 down
    check_ramp(
        time, current, ton2 + ramp2_duration, ramp2_duration, holding_current, amp, ramp_up=False
    )

    # during ramp3 up
    ton3 = ton2 + 2 * ramp2_duration + inter_delay
    check_ramp(time, current, ton3, ramp3_duration, holding_current, amp, ramp_up=True)
    # during ramp3 down
    check_ramp(
        time, current, ton3 + ramp3_duration, ramp3_duration, holding_current, amp, ramp_up=False
    )

    # after stimulus
    toff = ton3 + 2 * ramp3_duration
    current_after = current[numpy.where((toff < time) & (time <= toff + delay))]
    assert numpy.all(current_after == holding_current)


def test_poscheops():
    """Test PosCheops generate."""
    (
        stimulus,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        amp,
        holding_curr,
    ) = get_poscheops_stimulus()
    time, current = stimulus.generate()

    assert stimulus.name == "PosCheops"
    assert stimulus.total_duration == 2 * (
        delay + ramp1_duration + ramp2_duration + ramp3_duration + inter_delay
    )
    check_poscheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        holding_curr,
        amp,
    )

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_poscheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        0.0,
        amp,
    )


def test_poscheops_instantiate():
    """Test PosCheops instantiate."""
    (
        stimulus,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        amp,
        holding_curr,
    ) = get_poscheops_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_poscheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        holding_curr,
        amp,
    )

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_poscheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        0.0,
        amp,
    )


def check_negcheops_stim(
    time,
    current,
    delay,
    ramp1_duration,
    ramp2_duration,
    ramp3_duration,
    inter_delay,
    totduration,
    holding_current,
    amp,
):
    """Assert Ramp stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # during ramp1 up
    check_ramp(time, current, delay, ramp1_duration, holding_current, amp, ramp_up=True)
    # during ramp1 down
    check_ramp(
        time, current, delay + ramp1_duration, ramp1_duration, holding_current, amp, ramp_up=False
    )

    # during ramp2 up
    ton2 = delay + 2 * ramp1_duration + inter_delay
    check_ramp(time, current, ton2, ramp2_duration, holding_current, amp, ramp_up=True)
    # during ramp2 down
    check_ramp(
        time, current, ton2 + ramp2_duration, ramp2_duration, holding_current, amp, ramp_up=False
    )

    # during ramp3 up
    ton3 = ton2 + 2 * ramp2_duration + inter_delay
    check_ramp(time, current, ton3, ramp3_duration, holding_current, amp, ramp_up=True)
    # during ramp3 down
    check_ramp(
        time, current, ton3 + ramp3_duration, ramp3_duration, holding_current, amp, ramp_up=False
    )

    # after stimulus
    toff = ton3 + 2 * ramp3_duration
    current_after = current[numpy.where((toff < time) & (time <= totduration))]
    assert numpy.all(current_after == holding_current)


def test_negcheops():
    """Test NegCheops generate."""
    (
        stimulus,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        amp,
        holding_curr,
    ) = get_negcheops_stimulus()
    time, current = stimulus.generate()

    assert stimulus.name == "NegCheops"
    assert stimulus.total_duration == totduration
    check_negcheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        holding_curr,
        amp,
    )

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_negcheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        0.0,
        amp,
    )


def test_negcheops_instantiate():
    """Test NegCheops instantiate."""
    (
        stimulus,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        amp,
        holding_curr,
    ) = get_negcheops_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_negcheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        holding_curr,
        amp,
    )

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_negcheops_stim(
        time,
        current,
        delay,
        ramp1_duration,
        ramp2_duration,
        ramp3_duration,
        inter_delay,
        totduration,
        0.0,
        amp,
    )


def check_sinespec_stim(time, current, delay, duration, total_duration, holding_current, amp):
    """Assert SineSpec stimulus behaves as expected."""
    if delay > 0:
        # before stimulus
        current_before = current[numpy.where((0 <= time) & (time < delay))]
        # remove last two values. They are probably affected by rounding error or something
        assert numpy.all(current_before[:-2] == holding_current)

        # after stimulus
        current_after = current[
            numpy.where((delay + duration < time) & (time <= total_duration))
        ]
        assert numpy.all(current_after == holding_current)

    # during stimulus
    current_during = current[numpy.where((delay < time) & (time < delay + duration))]
    # timesteps can be variable
    # also change time from ms to s
    time_during = (time[numpy.where((delay < time) & (time < delay + duration))] - delay) / 1e3
    theoretical_current_during = holding_current + amp * numpy.sin(
        2.0 * numpy.pi * (1.0 + (1.0 / (5.15 - (time_during - 0.1)))) * (time_during - 0.1)
    )
    # for some reasons, the last values of current after the run are 0, so I removed them.
    # also the run current needs a lower precision threshold than the generated current.
    # the precision loss happens at the end of the run, when the sine period becomes very small.
    assert numpy.all(
        current_during[:-2] == pytest.approx(theoretical_current_during[:-2], rel=1e-3)
    )


def test_sinespec():
    """Test SineSpec generate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_sinespec_stimulus()
    time, current = stimulus.generate()

    assert stimulus.name == "SineSpec"
    assert stimulus.total_duration == total_duration
    check_sinespec_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_sinespec_stim(time, current, delay, duration, total_duration, 0.0, amp)


def test_sinespec_instantiate():
    """Test SineSpec instantiate."""
    stimulus, delay, duration, total_duration, amp, holding_curr = get_sinespec_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_sinespec_stim(time, current, delay, duration, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_sinespec_stim(time, current, delay, duration, total_duration, 0.0, amp)


def check_spikerecmultispikes_stim(
    time, current, delay, n_spikes, spike_duration, delta, total_duration, holding_current, amp
):
    """Assert SpikeRecMultiSpikes stimulus behaves as expected."""
    # before stimulus
    current_before = current[numpy.where((0 <= time) & (time < delay))]
    assert numpy.all(current_before == holding_current)

    # during stimulus
    spike_start = delay
    spike_end = delay + spike_duration
    current_during = current[numpy.where((spike_start < time) & (time < spike_end))]
    assert numpy.all(current_during == holding_current + amp)

    for _ in range(1, n_spikes):
        # between two stimuli
        spike_start = spike_end + delta
        current_during = current[numpy.where((spike_end < time) & (time < spike_start))]
        assert numpy.all(current_during == holding_current)

        # during stimulus
        spike_end = spike_start + spike_duration
        current_during = current[numpy.where((spike_start < time) & (time < spike_end))]
        assert numpy.all(current_during == holding_current + amp)

    # after stimulus
    current_after = current[numpy.where((spike_end < time) & (time <= total_duration))]
    assert numpy.all(current_after == holding_current)


def test_spikerecmultispikes():
    """Test SpikeRecMultiSpikes generate."""
    stimulus, delay, n_spikes, spike_duration, delta, total_duration, amp, holding_curr = get_spikerecmultispikes_stimulus()
    time, current = stimulus.generate()

    assert stimulus.name == "SpikeRecMultiSpikes"
    assert stimulus.total_duration == total_duration
    check_spikerecmultispikes_stim(time, current, delay, n_spikes, spike_duration, delta, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = stimulus.generate()
    check_spikerecmultispikes_stim(time, current, delay, n_spikes, spike_duration, delta, total_duration, 0.0, amp)


def test_spikerecmultispikes_instantiate():
    """Test SpikeRecMultiSpikes instantiate."""
    stimulus, delay, n_spikes, spike_duration, delta, total_duration, amp, holding_curr = get_spikerecmultispikes_stimulus()
    time, current = run_stim_on_dummy_cell(stimulus)
    check_spikerecmultispikes_stim(time, current, delay, n_spikes, spike_duration, delta, total_duration, holding_curr, amp)

    stimulus.holding_current = None
    time, current = run_stim_on_dummy_cell(stimulus)
    check_spikerecmultispikes_stim(time, current, delay, n_spikes, spike_duration, delta, total_duration, 0.0, amp)
