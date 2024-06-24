"""Helper functions used to find the path to the pdfs generated during the different steps
of the emodel pipeline"""

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


def search_figure_paths(pathname):
    """Search for at least one pdf based on an expression"""

    matches = glob.glob(pathname)

    if not matches:
        logger.debug("No pdf for pathname %s", pathname)
        return []

    return [str(Path(match).resolve()) for match in matches]


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

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__optimisation.pdf"

    return Path("./figures") / emodel_metadata.emodel / "optimisation" / fname


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


def figure_emodel_thumbnail(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the thumbnail of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__thumbnail.png"

    pathname = Path("./figures") / emodel_metadata.emodel / "thumbnail" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "thumbnail" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_thumbnail(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the thumbnail of an emodel"""

    pathname, pathname_val = figure_emodel_thumbnail(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def figure_emodel_bAP(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the bAP of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__dendrite_backpropagation_fit_decay.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "dendritic" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "dendritic" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_bAP(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the bAP of an emodel"""

    pathname, pathname_val = figure_emodel_bAP(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def figure_emodel_EPSP(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the EPSP of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__dendrite_EPSP_attenuation_fit.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "dendritic" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "dendritic" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_EPSP(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the EPSP of an emodel"""

    pathname, pathname_val = figure_emodel_EPSP(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return [search_figure_path(str(pathname)), search_figure_path(str(pathname_val))]


def figure_emodel_ISI_CV(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the ISI_CV fit of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__*ISI_CV_linear.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "dendritic" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "dendritic" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_ISI_CV(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the ISI_CV fit of an emodel"""

    pathname, pathname_val = figure_emodel_ISI_CV(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return list(search_figure_paths(str(pathname))) + list(search_figure_paths(str(pathname_val)))


def figure_emodel_rheobase(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdf representing the rheobase fit of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__*bpo_threshold_current_linear.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "dendritic" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "dendritic" / "validated" / fname

    return pathname, pathname_val


def search_figure_emodel_rheobase(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdf representing the rheobase fit of an emodel"""

    pathname, pathname_val = figure_emodel_rheobase(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return list(search_figure_paths(str(pathname))) + list(search_figure_paths(str(pathname_val)))


def figure_emodel_parameters(emodel_metadata, seed=None, use_allen_notation=True):
    """Get path for the pdf representing the distribution of the parameters of an emodel"""
    # pylint: disable=unused-argument

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


def figure_emodel_parameters_evolution(emodel_metadata, seed=None, use_allen_notation=True):
    """Get path for the pdf representing the evolution of the parameters of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    if seed is None:
        fname = f"{metadata_str}__all_seeds__evo_parameter_density.pdf"
    else:
        fname = f"{metadata_str}__evo_parameter_density.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "parameter_evolution" / fname

    return pathname


def search_figure_emodel_parameters_evolution(emodel_metadata, seed=None, use_allen_notation=True):
    """Search for the pdf representing the evolution of the parameters of an emodel"""

    pathname = figure_emodel_parameters_evolution(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return search_figure_path(str(pathname))


def figure_emodel_currentscapes(emodel_metadata, seed, use_allen_notation=True):
    """Get path for the pdfs representing the currentscapes of an emodel"""

    metadata_str = emodel_metadata.as_string(seed, use_allen_notation=use_allen_notation)
    fname = f"{metadata_str}__currentscape*.pdf"

    pathname = Path("./figures") / emodel_metadata.emodel / "currentscape" / "all" / fname
    pathname_val = Path("./figures") / emodel_metadata.emodel / "currentscape" / "validated"
    pathname_val = pathname_val / fname

    return pathname, pathname_val


def search_figure_emodel_currentscapes(emodel_metadata, seed, use_allen_notation=True):
    """Search for the pdfs representing the currentscapes of an emodel"""

    pathname, pathname_val = figure_emodel_currentscapes(
        emodel_metadata, seed, use_allen_notation=use_allen_notation
    )

    return list(search_figure_paths(str(pathname))) + list(search_figure_paths(str(pathname_val)))


def copy_emodel_pdf_dependency_to_new_path(old_path, new_path, overwrite=False):
    """Copy a pdf dependency to new path using allen notation"""
    if old_path.is_file():
        if not new_path.is_file() or overwrite:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(old_path, new_path)


def copy_emodel_pdf_dependencies_to_new_path(
    old_metadata, new_metadata, old_allen_notation, new_allen_notation, seed, overwrite=False
):
    """Copy dependencies to new path using allen notation"""
    # pylint: disable=too-many-locals

    # do not have all and validated subfolders
    single_folder_fcts = [figure_emodel_optimisation, figure_emodel_parameters_evolution]
    # have all and validated subfolders
    two_folders_fcts = [
        figure_emodel_traces,
        figure_emodel_score,
        figure_emodel_parameters,
        figure_emodel_thumbnail,
    ]

    for fct in single_folder_fcts:
        old_path = fct(old_metadata, seed=seed, use_allen_notation=old_allen_notation)
        new_path = fct(new_metadata, seed=seed, use_allen_notation=new_allen_notation)
        copy_emodel_pdf_dependency_to_new_path(old_path, new_path, overwrite=overwrite)

    for fct in two_folders_fcts:
        old_path, old_path_val = fct(old_metadata, seed=seed, use_allen_notation=old_allen_notation)
        new_path, new_path_val = fct(new_metadata, seed=seed, use_allen_notation=new_allen_notation)
        copy_emodel_pdf_dependency_to_new_path(old_path, new_path, overwrite=overwrite)
        copy_emodel_pdf_dependency_to_new_path(old_path_val, new_path_val, overwrite=overwrite)

    # also check with seed = None for figure_emodel_parameters_evolution
    old_all_evo_path = figure_emodel_parameters_evolution(
        old_metadata, seed=None, use_allen_notation=old_allen_notation
    )
    new_all_evo_path = figure_emodel_parameters_evolution(
        new_metadata, seed=None, use_allen_notation=new_allen_notation
    )
    copy_emodel_pdf_dependency_to_new_path(old_all_evo_path, new_all_evo_path, overwrite=overwrite)

    # take into account that we have to search for currentscape plots
    # because we do not know a priori the protocols and locations
    old_currentscape_path = search_figure_emodel_currentscapes(
        old_metadata, seed, use_allen_notation=old_allen_notation
    )
    new_currentscape_path, new_currentscape_path_val = figure_emodel_currentscapes(
        new_metadata, seed, use_allen_notation=new_allen_notation
    )
    for old_path in old_currentscape_path:
        prot = str(Path(old_path).stem).rsplit("currentscape", maxsplit=1)[-1]
        if "/validated/" in str(old_path):
            new_path = str(new_currentscape_path_val).replace("*", prot)
        else:
            new_path = str(new_currentscape_path).replace("*", prot)
        copy_emodel_pdf_dependency_to_new_path(Path(old_path), Path(new_path), overwrite=overwrite)
