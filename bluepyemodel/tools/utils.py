"""Utils"""
import logging
import pickle
from pathlib import Path

import numpy

logger = logging.getLogger("__main__")


def get_checkpoint_path(metadata, seed=None):
    """"""

    filename = metadata.as_string(seed=seed)

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
            name_parts = [e for e in protocol_name.split("_")]
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
