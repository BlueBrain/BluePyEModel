"""Helper functions used to find the path to the pdfs generated during the different steps of the emodel pipeline"""

import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def search_figure_path(pathname):
    """Search for a single pdf based on an expression"""

    matches = glob.glob(pathname)

    if not matches:
        logger.debug("No pdf for pathname %s", pathname)
        return None

    if len(matches) > 1:
        raise Exception("More than one pdf for pathname %s" % pathname)

    return matches[0]


def search_figure_efeatures(emodel, protocol_name, efeature):
    """Search for the pdf representing the efeature extracted from ephys recordings"""

    path = f"./figures/{emodel}/efeatures_extraction/*{protocol_name}_{efeature}"

    pdf_amp = search_figure_path(path + "_amp.pdf")
    pdf_amp_rel = search_figure_path(path + "_amp_rel.pdf")

    return pdf_amp, pdf_amp_rel


def search_figure_emodel_optimisation(emodel, seed, githash=""):
    """Search for the pdf representing the convergence of the optimisation"""

    if githash:
        fname = f"checkpoint__{emodel}__{githash}__{seed}.pdf"
    else:
        fname = f"checkpoint__{emodel}__{seed}.pdf"

    pathname = Path("./figures") / emodel / fname

    return search_figure_path(str(pathname))


def search_figure_emodel_traces(emodel, seed, githash=""):
    """Search for the pdf representing the traces of an emodel"""

    fname = f"{emodel}_{githash}_{seed}_traces.pdf"
    pathname = Path("./figures") / emodel / "traces" / "all" / fname

    return search_figure_path(str(pathname))


def search_figure_emodel_score(emodel, seed, githash=None):
    """Search for the pdf representing the scores of an emodel"""

    if githash:
        fname = f"{emodel}_{githash}_{seed}_scores.pdf"
    else:
        fname = f"{emodel}_{seed}_scores.pdf"

    pathname = Path("./figures") / emodel / "scores" / "all" / fname

    return search_figure_path(str(pathname))


def search_figure_emodel_parameters(emodel):
    """Search for the pdf representing the distribution of the parameters
    of an emodel"""

    fname = f"{emodel}_parameters_distribution.pdf"
    pathname = Path("./figures") / emodel / "distributions" / "all" / fname

    return search_figure_path(str(pathname))
