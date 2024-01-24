"""Utility module for evaluation."""

"""
Copyright 2023-2024, EPFL/Blue Brain Project

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

def define_bAP_protocol(dist_start=10, dist_end=600, dist_step=10, dist_end_basal=150):
    """Utility function to create a ready-to-use bAP protocol."""
    soma_loc = define_location("soma")
    stim = {
        "delay": 700,
        "amp": None,
        "thresh_perc": 1000.0,
        "duration": 4.0,
        "totduration": 1000,
        "holding_current": 0
    }
    stimulus = eCodes["idrest"](location=soma_loc, **stim)

    recordings = []
    soma_rec = {
        "type": "CompRecording",
        "name": "bAP_1000.soma.v",
        "location": "soma",
        "variable": "v"
    }
    rec = LooseDtRecordingCustom(name=soma_rec["name"], location=soma_loc, variable="v")
    recordings.append(rec)
    # make a function called in those two loops
    for dist in range(dist_start, dist_end, dist_step):
        rec_dict = {
            "type": "somadistanceapic",
            "somadistance": dist,
            "name": f"bAP_1000.dend{dist:03d}.v",
            "seclist_name": "apical",
            "variable": "v"
        }
        new_loc = define_location(rec_dict)
        rec = LooseDtRecordingCustom(name=rec_dict["name"], location=new_loc, variable="v")
        recordings.append(rec)
    for dist in range(dist_start, dist_end_basal, dist_step):
        rec_dict = {
            "type": "somadistanceapic",
            "somadistance": dist,
            "name": f"bAP_1000.basal{dist:03d}.v",
            "seclist_name": "basal",
            "variable": "v"
        }
        new_loc = define_location(rec_dict)
        rec = LooseDtRecordingCustom(name=rec_dict["name"], location=new_loc, variable="v")
        recordings.append(rec)

    return protocol_type_to_class["ThresholdBasedProtocol"](
        name="bAP_1000",
        stimulus=stimulus,
        recordings=recordings,
        cvode_active=True,
        stochasticity=False,
    )


def define_bAP_feature(
    dend_type="apical", dist_start=10, dist_end=600, dist_step=10, dist_end_basal=150
):
    """Utility function to create a ready-to-use dendrite backpropagation fit feature
    
    dend_type can be 'apical' or 'basal'
    """
    if dend_type == "apical":
        rec_dend_type = "dend"
    elif dend_type == "basal":
        rec_dend_type = "basal"
        dist_end = dist_end_basal
    else:
        raise ValueError(f"Expected 'apical' or 'basal' for dend_type. Got {dend_type} instead")

    recording_names = {"": "bAP_1000.soma.v"}
    for dist in range(dist_start, dist_end, dist_step):
        recording_names[dist] = f"bAP_1000.{rec_dend_type}{dist:03d}.v"

    feat = DendFitFeature(
        f"{dend_type}_dendrite_backpropagation_fit_decay",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names=recording_names,
        stim_start=700.0,
        stim_end=704.0,
        exp_mean=765.2223254692782, # filler
        exp_std=37.98402943833677, # filler
        stimulus_current=0.0, # filler
        threshold=-30.0,
        interp_step=0.025,
        double_settings={},
        int_settings={"strict_stiminterval": 1},
        string_settings={},
        decay=True,
        linear=False,
    )
    return feat
