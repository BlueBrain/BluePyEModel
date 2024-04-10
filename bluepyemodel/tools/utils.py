"""Utils"""

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

import logging
import pickle
from pathlib import Path

import numpy

from bluepyemodel.ecode import IDrest
from bluepyemodel.ecode import eCodes

logger = logging.getLogger("__main__")


def get_checkpoint_path(metadata, seed=None):
    """"""

    base_path = f"./checkpoints/{metadata.emodel}/{metadata.iteration}/"
    filename = metadata.as_string(seed=seed, use_allen_notation=False)
    full_path = f"{base_path}{filename}.pkl"

    # Switch to Allen notation if no previous legacy checkpoint is found
    if not Path(full_path).is_file():
        filename = metadata.as_string(seed=seed, use_allen_notation=True)
        full_path = f"{base_path}{filename}.pkl"

    return full_path


def get_legacy_checkpoint_path(checkpoint_path):
    """Get legacy checkpoint path from checkpoint path"""

    filename = Path(checkpoint_path).stem
    return f"./checkpoints/{filename}.pkl"


def make_dir(path_dir):
    """Creates directory if it does not exist"""
    p = Path(path_dir)
    if not (p.is_dir()):
        logger.info("Creating directory %s.", p)
        p.mkdir(parents=True, exist_ok=True)


def yesno(question):
    """Ask a Yes/No question"""

    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()

    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)

    if ans == "y":
        return True

    return False


def parse_legacy_checkpoint_path(path):
    """"""

    filename = Path(path).stem.split("__")

    if len(filename) == 4:
        checkpoint_metadata = {
            "emodel": filename[1],
            "seed": filename[3],
            "iteration": filename[2],
            "ttype": None,
        }
    elif len(filename) == 3:
        checkpoint_metadata = {
            "emodel": filename[1],
            "seed": filename[2],
            "iteration": None,
            "ttype": None,
        }

    return checkpoint_metadata


def parse_checkpoint_path(path):
    """"""

    if "emodel" not in path and "checkpoint" in path:
        return parse_legacy_checkpoint_path(path)

    if path.endswith(".tmp"):
        path = path.replace(".tmp", "")

    filename = Path(path).stem.split("__")

    checkpoint_metadata = {}

    for field in [
        "emodel",
        "etype",
        "ttype",
        "mtype",
        "species",
        "brain_region",
        "seed",
        "iteration",
    ]:
        search_str = f"{field}="
        checkpoint_metadata[field] = next(
            (e.replace(search_str, "") for e in filename if search_str in e), None
        )

    return checkpoint_metadata


def read_checkpoint(checkpoint_path):
    """Reads a BluePyOpt checkpoint file"""

    p = Path(checkpoint_path)
    p_tmp = p.with_suffix(p.suffix + ".tmp")

    # legacy case
    if not p.is_file() and not p_tmp.is_file():
        legacy_checkpoint_path = get_legacy_checkpoint_path(checkpoint_path)
        p = Path(legacy_checkpoint_path)
        p_tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with open(str(p), "rb") as checkpoint_file:
            run = pickle.load(checkpoint_file, encoding="latin1")
            run_metadata = parse_checkpoint_path(str(p))
    except EOFError:
        try:
            with open(str(p_tmp), "rb") as checkpoint_tmp_file:
                run = pickle.load(checkpoint_tmp_file, encoding="latin1")
                run_metadata = parse_checkpoint_path(str(p_tmp))
        except EOFError:
            logger.error(
                "Cannot store model. Checkpoint file %s does not exist or is corrupted.",
                checkpoint_path,
            )

    return run, run_metadata


def format_protocol_name_to_list(protocol_name):
    """Make sure that the name of a protocol is a list [protocol_name, amplitude]"""

    if isinstance(protocol_name, str):
        try:
            name_parts = protocol_name.split("_")
            if name_parts[-1] == "hyp":
                amplitude = float(name_parts[-2])
                name_parts.pop(-2)
                name = "_".join(name_parts)
            else:
                name = "_".join(name_parts[:-1])
                amplitude = float(protocol_name.split("_")[-1])
        except ValueError:
            return protocol_name, None
        return name, amplitude

    if isinstance(protocol_name, list):
        return protocol_name

    raise TypeError("protocol_name should be a string or a list.")


def are_same_protocol(name_a, name_b):
    """Check if two protocol names or list are equal. Eg: is IV_0.0 the same as IV_0 and
    the same as ["IV", 0.0]."""

    if name_a is None or name_b is None:
        return False

    amps = []
    ecodes = []

    for name in [name_a, name_b]:
        tmp_p = format_protocol_name_to_list(name)
        ecodes.append(tmp_p[0])
        amps.append(tmp_p[1])

    if ecodes[0] == ecodes[1] and numpy.isclose(amps[0], amps[1]):
        return True
    return False


def get_mapped_protocol_name(protocol_name, protocols_mapping):
    """Returns the mapped protocol name from protocols_mapping if available;
    otherwise, returns the original name."""

    if protocols_mapping:
        for p in protocols_mapping:
            if protocol_name.lower() in p.lower():
                return protocols_mapping[p]
    return protocol_name


def select_rec_for_thumbnail(rec_names, additional_step_prots=None, thumbnail_rec=None):
    """Select a recording for thumbnail.

    Select the step protocol with lowest positive amplitude, so that delay is visible if present.

    Args:
        rec_names (set): the names of the recordings, following this naming convention:
            protocol_name_amplitude.location.variable
        additional_step_prots (list): step protocol names to look for (other than defaults ones)
        thumbnail_rec (str): recording name to use for thumbnail if present
    """
    if thumbnail_rec is not None:
        if thumbnail_rec in rec_names:
            return thumbnail_rec
        logger.warning(
            "Could not find %s in recording names. Will use another recording for thumbnail plot.",
            thumbnail_rec,
        )
    selected_rec = ""
    selected_amp = numpy.inf
    step_prots = [prot_name for prot_name, prot in eCodes.items() if prot is IDrest]
    if additional_step_prots:
        step_prots = step_prots + additional_step_prots

    for rec_name in rec_names:
        # TODO: have a more proper way to remove non somatic injections
        if "LocalInjection" not in rec_name and any(
            step_prot.lower() in rec_name.lower() for step_prot in step_prots
        ):
            prot_name = rec_name.split(".")[0]
            try:
                _, rec_amp = format_protocol_name_to_list(prot_name)
                if 0 < rec_amp < selected_amp:
                    selected_rec = rec_name
                    selected_amp = rec_amp
            except (TypeError, ValueError):
                logger.warning("Could not find amplitude in %s, skipping it.", prot_name)

    if selected_rec == "":
        if len(rec_names) < 1:
            raise ValueError("No recording in recording_names. Can not plot thumbnail.")
        logger.warning("Could not find any step protocol in recording. Will take the first one.")
        return next(iter(rec_names))

    logger.debug("Selected %s for thumbnail", selected_rec)

    return selected_rec
