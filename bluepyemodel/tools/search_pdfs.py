"""Helper functions used to find the path to the pdfs generated during the different steps
of the emodel pipeline"""

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

import glob
import logging
import shutil
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


def figure_efeatures(emodel, protocol_name, efeature):
    """Get path to pdf representing the efeature extracted from ephys recordings"""

    path = f"./figures/{emodel}/efeatures_extraction/*{protocol_name}_{efeature}"

    pdf_amp = f"{path}_amp.pdf"
    pdf_amp_rel = f"{path}_amp_rel.pdf"

    return pdf_amp, pdf_amp_rel


def search_figure_efeatures(emodel, protocol_name, efeature):
    """Search for the pdf representing the efeature extracted from ephys recordings"""

    pdf_amp, pdf_amp_rel = figure_efeatures(emodel, protocol_name, efeature)

    pdf_amp = search_figure_path(pdf_amp)
    pdf_amp_rel = search_figure_path(pdf_amp_rel)

    return pdf_amp, pdf_amp_rel


def figure_emodel_optimisation(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the convergence of the optimisation"""

    fname = f"{emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)}.pdf"

    return Path("./figures") / emodel_metadata.emodel / fname


def search_figure_emodel_optimisation(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the convergence of the optimisation"""

    pathname = figure_emodel_optimisation(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return search_figure_path(str(pathname))


def figure_emodel_traces(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the traces of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__traces.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "traces" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "traces" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_traces(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the traces of an emodel"""

    pathname, pathname_val = figure_emodel_traces(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def figure_emodel_score(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the scores of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__scores.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "scores" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "scores" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_score(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the scores of an emodel"""

    pathname, pathname_val = figure_emodel_score(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def figure_emodel_parameters(emodel_metadata, use_allen_notation=True):
    """Get path for the pdf representing the distribution of the parameters of an emodel"""

    metadata_str = emodel_metadata.as_string(use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__parameters_distribution.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "distributions" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "distributions" / "validated"
    pathname_val = pathname_val / fname

    return pathname, pathname_val


def search_figure_emodel_parameters(emodel_metadata, use_allen_notation=True):
    """Search for the pdf representing the distribution of the parameters
    of an emodel"""

    pathname, pathname_val = figure_emodel_parameters(
        emodel_metadata, use_allen_notation=use_allen_notation
    )

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def copy_emodel_pdf_dependency_to_new_path(old_path, new_path):
    """Copy a pdf dependency to new path using allen notation"""
    if old_path.is_file() and not new_path.is_file():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(old_path, new_path)


def copy_emodel_pdf_dependencies_to_new_path(emodel_metadata, seed):
    """Copy dependencies to new path using allen notation"""
    old_opt_path = figure_emodel_optimisation(emodel_metadata, seed, use_allen_notation=False)
    new_opt_path = figure_emodel_optimisation(emodel_metadata, seed, use_allen_notation=True)
    copy_emodel_pdf_dependency_to_new_path(old_opt_path, new_opt_path)

    old_traces_path, old_traces_path_val = figure_emodel_traces(
        emodel_metadata, seed, use_allen_notation=False
    )
    new_traces_path, new_traces_path_val = figure_emodel_traces(
        emodel_metadata, seed, use_allen_notation=True
    )
    copy_emodel_pdf_dependency_to_new_path(old_traces_path, new_traces_path)
    copy_emodel_pdf_dependency_to_new_path(old_traces_path_val, new_traces_path_val)

    old_score_path, old_score_path_val = figure_emodel_score(
        emodel_metadata, seed, use_allen_notation=False
    )
    new_score_path, new_score_path_val = figure_emodel_score(
        emodel_metadata, seed, use_allen_notation=True
    )
    copy_emodel_pdf_dependency_to_new_path(old_score_path, new_score_path)
    copy_emodel_pdf_dependency_to_new_path(old_score_path_val, new_score_path_val)

    old_params_path, old_params_path_val = figure_emodel_parameters(
        emodel_metadata, use_allen_notation=False
    )
    new_params_path, new_params_path_val = figure_emodel_parameters(
        emodel_metadata, use_allen_notation=True
    )
    copy_emodel_pdf_dependency_to_new_path(old_params_path, new_params_path)
    copy_emodel_pdf_dependency_to_new_path(old_params_path_val, new_params_path_val)
