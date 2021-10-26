"""Helper functions used to find the path to the pdfs generated during the different steps
of the emodel pipeline"""

import glob
import logging
from pathlib import Path

from bluepyemodel.emodel_pipeline.utils import run_metadata_as_string

logger = logging.getLogger(__name__)


def search_figure_path(pathname):
    """Search for a single pdf based on an expression"""

    matches = glob.glob(pathname)

    if not matches:
        logger.debug("No pdf for pathname %s", pathname)
        return None

    if len(matches) > 1:
        raise Exception(f"More than one pdf for pathname {pathname}")

    return matches[0]


def search_figure_efeatures(emodel, protocol_name, efeature):
    """Search for the pdf representing the efeature extracted from ephys recordings"""

    path = f"./figures/{emodel}/efeatures_extraction/*{protocol_name}_{efeature}"

    pdf_amp = search_figure_path(path + "_amp.pdf")
    pdf_amp_rel = search_figure_path(path + "_amp_rel.pdf")

    return pdf_amp, pdf_amp_rel


def search_figure_emodel_optimisation(emodel, seed, ttype=None, iteration_tag=None):
    """Search for the pdf representing the convergence of the optimisation"""

    fname = run_metadata_as_string(emodel, seed, ttype=ttype, iteration_tag=iteration_tag)
    fname += ".pdf"

    pathname = Path("./figures") / emodel / fname

    return search_figure_path(str(pathname))


def search_figure_emodel_traces(emodel, seed, ttype=None, iteration_tag=None):
    """Search for the pdf representing the traces of an emodel"""

    fname = run_metadata_as_string(emodel, seed, ttype=ttype, iteration_tag=iteration_tag)
    fname += "__traces.pdf"

    pathname = Path("./figures") / emodel / "traces" / "all" / fname
    pathname_val = Path("./figures") / emodel / "traces" / "validated" / fname

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def search_figure_emodel_score(emodel, seed, ttype=None, iteration_tag=None):
    """Search for the pdf representing the scores of an emodel"""

    fname = run_metadata_as_string(emodel, seed, ttype=ttype, iteration_tag=iteration_tag)
    fname += "__scores.pdf"

    pathname = Path("./figures") / emodel / "scores" / "all" / fname
    pathname_val = Path("./figures") / emodel / "scores" / "validated" / fname

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def search_figure_emodel_parameters(emodel, ttype=None, iteration_tag=None):
    """Search for the pdf representing the distribution of the parameters
    of an emodel"""

    fname = run_metadata_as_string(emodel, seed="", ttype=ttype, iteration_tag=iteration_tag)
    fname += "__parameters_distribution.pdf"

    pathname = Path("./figures") / emodel / "distributions" / "all" / fname
    pathname_val = Path("./figures") / emodel / "distributions" / "validated" / fname

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]
