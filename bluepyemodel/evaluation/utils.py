"""Utility module for evaluation."""

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

import logging

from ..ecode import eCodes
from .efel_feature_bpem import DendFitFeature
from .evaluator import define_location
from .evaluator import protocol_type_to_class
from .recordings import LooseDtRecordingCustom

logger = logging.getLogger(__name__)


def add_dendritic_recordings(
    recordings, prot_name, dend_type, dist_start=10, dist_end=600, dist_step=10
):
    """Add dendritic recordings to recordings list.

    Args:
        recordings (list): variable to be updated with new recordings
        prot_name (str): protocol name
        dend_type (str): On which dendrite should the recording be.
            Can be 'apical' or 'basal'
        dist_start (int): closest distance on the dendrite on which to record
        dist_end (int): furthest distance on the dendrite on which to record
        dist_step (int): record every dist_step distance on the dendrite
    """
    for dist in range(dist_start, dist_end, dist_step):
        rec_dict = {
            "type": "somadistanceapic",
            "somadistance": dist,
            "name": f"{prot_name}.{dend_type}{dist:03d}.v",
            "seclist_name": dend_type,
        }
        new_loc = define_location(rec_dict)
        rec = LooseDtRecordingCustom(name=rec_dict["name"], location=new_loc, variable="v")
        recordings.append(rec)


def define_bAP_protocol(dist_start=10, dist_end=600, dist_step=10, dist_end_basal=150):
    """Utility function to create a ready-to-use bAP protocol.

    Args:
        dist_start (int): closest distance on the dendrites on which to record
        dist_end (int): furthest distance on the apical dendrite on which to record
        dist_step (int): record every dist_step distance on the dendrites
        dist_end_basal (int): furthest distance on the basal dendrite on which to record
    """
    name = "bAP_1000"
    soma_loc = define_location("soma")
    stim = {
        "delay": 700,
        "amp": None,
        "thresh_perc": 1000.0,
        "duration": 4.0,
        "totduration": 1000,
        "holding_current": 0,
    }
    stimulus = eCodes["idrest"](location=soma_loc, **stim)

    recordings = [LooseDtRecordingCustom(name=f"{name}.soma.v", location=soma_loc, variable="v")]
    add_dendritic_recordings(recordings, name, "apical", dist_start, dist_end, dist_step)
    add_dendritic_recordings(recordings, name, "basal", dist_start, dist_end_basal, dist_step)

    return protocol_type_to_class["ThresholdBasedProtocol"](
        name=name,
        stimulus=stimulus,
        recordings=recordings,
        cvode_active=True,
        stochasticity=False,
    )


def define_bAP_feature(dend_type="apical", dist_start=10, dist_end=600, dist_step=10):
    """Utility function to create a ready-to-use dendrite backpropagation fit feature

    Args:
        dend_type (str): Which dendrite data should the feature compute.
            Can be 'apical' or 'basal'
        dist_start (int): closest distance on the dendrite on which to record
        dist_end (int): furthest distance on the dendrite on which to record
        dist_step (int): record every dist_step distance on the dendrite
    """
    name = "bAP_1000"

    recording_names = {"": f"{name}.soma.v"}
    for dist in range(dist_start, dist_end, dist_step):
        recording_names[dist] = f"{name}.{dend_type}{dist:03d}.v"

    feat = DendFitFeature(
        f"{dend_type}_dendrite_backpropagation_fit_decay",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names=recording_names,
        stim_start=700.0,
        stim_end=704.0,
        exp_mean=765.2223254692782,  # filler
        exp_std=37.98402943833677,  # filler
        stimulus_current=0.0,  # filler
        threshold=-30.0,
        interp_step=0.025,
        double_settings={},
        int_settings={"strict_stiminterval": 1},
        string_settings={},
        decay=True,
        linear=False,
    )
    return feat


def define_EPSP_protocol(dend_type, dist_start=100, dist_end=600, dist_step=100):
    """Returns ready-to-use EPSP protocols at multiple locations along the dendrite.

    Args:
        dend_type (str): Which dendrite data should the feature compute.
            Can be 'apical' or 'basal'
        dist_start (int): closest distance on the dendrite on which to record
        dist_end (int): furthest distance on the dendrite on which to record
        dist_step (int): record every dist_step distance on the dendrite
    """
    prots = {}
    # should we translate apical to apic here or not?
    soma_loc = define_location("soma")
    stim = {
        "syn_weight": 1.13,
        "syn_delay": 400.0,
        "totduration": 500.0,
    }

    # create a protocol injecting at soma location to have a datapoint at 0 distance
    name = "ProbAMPANMDA_EMS_0"
    stimulus = eCodes["probampanmda_ems"](location=soma_loc, **stim)
    recordings = [LooseDtRecordingCustom(name=f"{name}.soma.v", location=soma_loc, variable="v")]
    prot = protocol_type_to_class["Protocol"](
        name=name,
        stimulus=stimulus,
        recordings=recordings,
        cvode_active=False,  # cannot be used with cvode
        stochasticity=False,
    )
    prots[prot.name] = prot

    # create protocols injecting in the dendrites
    for dist in range(dist_start, dist_end, dist_step):
        name = f"ProbAMPANMDA_EMS{dend_type}{dist:03d}_0"
        loc_name = f"{dend_type}{dist:03d}"
        loc = define_location(
            {
                "type": "somadistanceapic",
                "somadistance": dist,
                "name": loc_name,
                "seclist_name": dend_type,
            }
        )
        stimulus = eCodes["probampanmda_ems"](location=loc, **stim)

        recordings = [
            LooseDtRecordingCustom(name=f"{name}.{loc_name}.v", location=loc, variable="v"),
            LooseDtRecordingCustom(name=f"{name}.soma.v", location=soma_loc, variable="v"),
        ]

        prot = protocol_type_to_class["Protocol"](
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=False,  # cannot be used with cvode
            stochasticity=False,
        )
        prots[prot.name] = prot

    return prots


def define_EPSP_feature(
    dend_type="apical", rec_loc="dend", dist_start=100, dist_end=600, dist_step=100
):
    """Utility function to create a ready-to-use dendrite EPSP fit feature

    Args:
        dend_type (str): Which dendrite data should the feature compute.
            Can be 'apical' or 'basal'
        rec_loc (str): where should data be recorded from.
            Can be either 'dend' or 'soma'
        dist_start (int): closest distance on the dendrite on which to record
        dist_end (int): furthest distance on the dendrite on which to record
        dist_step (int): record every dist_step distance on the dendrite

    Attention! This is just used for plotting function,
    BPEM cannot currently handle EPSP attenuation fit feature on its own
    """
    recording_names = {"": "ProbAMPANMDA_EMS_0.soma.v"}
    for dist in range(dist_start, dist_end, dist_step):
        loc_name = f"{dend_type}{dist:03d}"
        if rec_loc == "dend":
            rec_name = loc_name
        else:
            rec_name = "soma"
        recording_names[dist] = f"ProbAMPANMDA_EMS{loc_name}_0.{rec_name}.v"

    feat = DendFitFeature(
        f"EPSP_{dend_type}_{rec_loc}",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names=recording_names,
        stim_start=400.0,
        stim_end=500.0,
        exp_mean=1.0,  # filler
        exp_std=1.0,  # filler
        stimulus_current=0.0,  # filler
        threshold=-30.0,
        interp_step=0.025,
        double_settings={},
        int_settings={"strict_stiminterval": 1},
        string_settings={},
        decay=False,
        linear=False,
    )
    return feat
