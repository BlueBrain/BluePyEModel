"""Utils"""

import logging
import pickle
from pathlib import Path

logger = logging.getLogger("__main__")


def make_dir(path_dir):
    """Creates directory if it does not exist"""
    p = Path(path_dir)
    if not (p.is_dir()):
        logger.info("Creating directory %s.", p)
        p.mkdir(parents=True, exist_ok=True)


def read_checkpoint(checkpoint_path):
    """Reads a BluePyOpt checkpoint file"""

    p = Path(checkpoint_path)
    p_tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with open(str(p), "rb") as checkpoint_file:
            run = pickle.load(checkpoint_file)
    except EOFError:
        try:
            with open(str(p_tmp), "rb") as checkpoint_tmp_file:
                run = pickle.load(checkpoint_tmp_file)
        except EOFError:
            logger.error(
                "Cannot store model. Checkpoint file %s does not exist or is corrupted.",
                checkpoint_path,
            )

    return run


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
