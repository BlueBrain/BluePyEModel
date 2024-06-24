"""MultiProtocol eFeature Utils"""

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

logger = logging.getLogger("__main__")


def get_distances_from_recording_name(rec_name):
    """Get apical distances from recording name."""
    isolate_list = rec_name.split("[")[1].split("]")[0]
    int_list = [int(num) for num in isolate_list.split(",")]
    int_list.insert(0, 0)
    return int_list


def get_soma_protocol_from_protocol_name(prot_name, amplitude):
    """Get recording name at soma from multi-locations protocols' name

    Args:
        prot_name (str): protocol name
        amplitude (int): amplitude
    """
    from bluepyemodel.ecode import eCodes

    # find ecode in protocol name
    for ecode in eCodes:
        idx = prot_name.lower().find(ecode)
        if idx > -1:
            # get ecode with capitalized letters
            prot_name = prot_name[idx : idx + len(ecode)]
            return f"{prot_name}_{amplitude}"

    raise ValueError(f"Could not find ecode in {prot_name}")


def get_soma_protocol_from_recording_name(rec_name, amplitude):
    """Get recording name at soma from multi-locations protocols' name"""
    return f"{get_soma_protocol_from_protocol_name(rec_name.split('.')[0], amplitude)}.soma.v"


def split_protocol_name_with_location_list(prot_name):
    """Split full protocol name into 'real' protocol name, location list and amplitude.

    Args:
        prot_name (str): full protocol name containing location list
            e.g. LocalInjectionIDrestapic[050,080,110,200,340]_100
    """
    prot_name, tmp = prot_name.split("[")
    isolate_list, amp = tmp.split("]")
    amplitude = int(amp.split("_")[1])
    return prot_name, isolate_list, amplitude


def get_protocol_list_from_protocol_name(prot_name):
    """Reconstruct protocol list from protocol name"""
    prot_name, isolate_list, amplitude = split_protocol_name_with_location_list(prot_name)
    prot_list = [f"{prot_name}{distance}_{amplitude}" for distance in isolate_list.split(",")]

    prot_list.insert(0, get_soma_protocol_from_protocol_name(prot_name, amplitude))

    return prot_list


def get_protocol_list_from_recording_name(rec_name):
    """Reconstruct recording list from recording name."""
    prot, loc, var = rec_name.split(".")
    prot_name, isolate_list, amplitude = split_protocol_name_with_location_list(prot)
    loc = loc.split("[")[0]
    prot_list = [
        (
            f"{prot_name}{distance}_{amplitude}.{loc}.{var}"
            if loc == "soma"
            else f"{prot_name}{distance}_{amplitude}.{loc}{distance}.{var}"
        )
        for distance in isolate_list.split(",")
    ]

    prot_list.insert(0, get_soma_protocol_from_recording_name(rec_name, amplitude))

    return prot_list


def get_locations_from_recording_name(rec_name):
    prot, loc, _ = rec_name.split(".")
    _, isolate_list, _ = split_protocol_name_with_location_list(prot)
    loc = loc.split("[")[0]
    loc_list = [f"{loc}{distance}" for distance in isolate_list.split(",")]
    loc_list.insert(0, "soma")
    return loc_list
