"""Helper functions used to find the path to the pdfs generated during the different steps
of the emodel pipeline"""

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
        raise ValueError(f"More than one pdf for pathname {pathname}")

    return str(Path(matches[0]).resolve())


def search_figure_efeatures(emodel, protocol_name, efeature):
    """Search for the pdf representing the efeature extracted from ephys recordings"""

    path = f"./figures/{emodel}/efeatures_extraction/*{protocol_name}_{efeature}"

    pdf_amp = search_figure_path(path + "_amp.pdf")
    pdf_amp_rel = search_figure_path(path + "_amp_rel.pdf")

    return pdf_amp, pdf_amp_rel


def search_figure_emodel_optimisation(emodel_metadata, seed):
    """Search for the pdf representing the convergence of the optimisation"""

    fname = emodel_metadata.as_string(seed) + ".pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / fname

    return search_figure_path(str(pathname))


def search_figure_emodel_traces(emodel_metadata, seed):
    """Search for the pdf representing the traces of an emodel"""

    fname = emodel_metadata.as_string(seed) + "__traces.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "traces" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "traces" / "validated" / fname

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def search_figure_emodel_score(emodel_metadata, seed):
    """Search for the pdf representing the scores of an emodel"""

    fname = emodel_metadata.as_string(seed) + "__scores.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "scores" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "scores" / "validated" / fname

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def search_figure_emodel_parameters(emodel_metadata):
    """Search for the pdf representing the distribution of the parameters
    of an emodel"""

    fname = emodel_metadata.as_string() + "__parameters_distribution.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "distributions" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "distributions" / "validated"
    pathname_val = pathname_val / fname

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]
