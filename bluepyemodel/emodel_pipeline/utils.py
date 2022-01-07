"""Utils"""
import logging
from pathlib import Path

logger = logging.getLogger("__main__")


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


def run_metadata_as_string(emodel, seed=None, ttype=None, iteration_tag=None):
    """"""

    s = f"emodel={emodel}"

    if seed is not None:
        s += f"__seed={seed}"
    if iteration_tag is not None:
        s += f"__iteration_tag={iteration_tag}"
    if ttype is not None:
        ttype_formatted = ttype.replace("/", "")
        s += f"__ttype={ttype_formatted}"

    return s
