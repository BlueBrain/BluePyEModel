"""EModel class"""

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

import numpy

from bluepyemodel.tools import search_pdfs


def format_dict_for_resource(d):
    """Translates a dictionary to a list of the format used by resources"""

    out = []

    if d is None:
        return out

    for k, v in d.items():
        if v is None or numpy.isnan(v):
            v = None

        out.append({"name": k, "value": v, "unitCode": ""})

    return out


class EModel:
    """Contains all the information related to an optimized e-model, such as its parameters or
    its e-feature values and scores.

    This class is not meant to be instantiated by hand. It is used to transmit the information
    about optimized e-model information."""

    def __init__(
        self,
        fitness=None,
        parameter=None,
        score=None,
        features=None,
        scoreValidation=None,
        passedValidation=None,
        seed=None,
        emodel_metadata=None,
    ):
        """Init

        Args:
            fitness (float): fitness of the emodel as the sum of scores.
            parameter (dict or Resource): parameters of the emodel
            score (dict or Resource): scores of the emodel.
            features (dict or Resource): feature values of the emodel.
            scoreValidation (dict or Resource): scores obtained on the validation protocols.
            passedValidation (bool or None): did the model go through validation and if yes,
                did it pass it successfully (None: no validation, True: passed, False: didn't pass)
            seed (str): seed used during optimisation for this emodel.
            emodel_metadata (EModelMetadata): metadata of the model (emodel name, etype, ttype, ...)
        """

        self.emodel_metadata = emodel_metadata
        self.passed_validation = passedValidation
        self.fitness = fitness
        self.seed = seed

        self.workflow_id = None

        if isinstance(parameter, dict):
            self.parameters = parameter
        else:
            self.parameters = {p["name"]: p["value"] for p in parameter} if parameter else {}

        if isinstance(score, dict):
            self.scores = score
        else:
            self.scores = {p["name"]: p["value"] for p in score} if score else {}

        if isinstance(features, dict):
            self.features = features
        else:
            self.features = (
                {p["name"]: (p["value"] if "value" in p else numpy.nan) for p in features}
                if features
                else {}
            )

        if isinstance(scoreValidation, dict):
            self.scores_validation = scoreValidation
        else:
            self.scores_validation = (
                {p["name"]: p["value"] for p in scoreValidation} if scoreValidation else {}
            )

        self.responses = {}
        self.evaluator = None

    def copy_pdf_dependencies_to_new_path(self, seed, overwrite=False):
        """Copy pdf dependencies to new path using allen notation"""
        search_pdfs.copy_emodel_pdf_dependencies_to_new_path(
            self.emodel_metadata, seed, overwrite=overwrite
        )

    def build_pdf_dependencies(self, seed):
        """Find all the pdfs associated to an emodel"""

        pdfs = []

        opt_pdf = search_pdfs.search_figure_emodel_optimisation(self.emodel_metadata, seed)
        if opt_pdf:
            pdfs.append(opt_pdf)

        traces_pdf = search_pdfs.search_figure_emodel_traces(self.emodel_metadata, seed)
        if traces_pdf:
            pdfs += [p for p in traces_pdf if p]

        scores_pdf = search_pdfs.search_figure_emodel_score(self.emodel_metadata, seed)
        if scores_pdf:
            pdfs += [p for p in scores_pdf if p]

        thumbnail_pdf = search_pdfs.search_figure_emodel_thumbnail(self.emodel_metadata, seed)
        if thumbnail_pdf:
            pdfs += [p for p in thumbnail_pdf if p]

        parameters_pdf = search_pdfs.search_figure_emodel_parameters(self.emodel_metadata)
        if parameters_pdf:
            pdfs += [p for p in parameters_pdf if p]

        parameters_evo_pdf = search_pdfs.search_figure_emodel_parameters_evolution(
            self.emodel_metadata, seed
        )
        if parameters_evo_pdf:
            pdfs.append(parameters_evo_pdf)

        all_parameters_evo_pdf = search_pdfs.search_figure_emodel_parameters_evolution(
            self.emodel_metadata, seed=None
        )
        if all_parameters_evo_pdf:
            pdfs.append(all_parameters_evo_pdf)

        currentscape_pdfs = search_pdfs.search_figure_emodel_currentscapes(
            self.emodel_metadata, seed
        )
        if currentscape_pdfs:
            pdfs += [p for p in currentscape_pdfs if p]

        bAP_pdf = search_pdfs.search_figure_emodel_bAP(self.emodel_metadata, seed)
        if bAP_pdf:
            pdfs += [p for p in bAP_pdf if p]

        EPSP_pdf = search_pdfs.search_figure_emodel_EPSP(self.emodel_metadata, seed)
        if EPSP_pdf:
            pdfs += [p for p in EPSP_pdf if p]

        ISI_CV_pdf = search_pdfs.search_figure_emodel_ISI_CV(self.emodel_metadata, seed)
        if ISI_CV_pdf:
            pdfs += [p for p in ISI_CV_pdf if p]

        rheobase_pdf = search_pdfs.search_figure_emodel_rheobase(self.emodel_metadata, seed)
        if rheobase_pdf:
            pdfs += [p for p in rheobase_pdf if p]

        return pdfs

    def get_related_nexus_ids(self):
        return {
            "generation": {
                "type": "Generation",
                "activity": {
                    "type": "Activity",
                    "followedWorkflow": {"type": "EModelWorkflow", "id": self.workflow_id},
                },
            }
        }

    def as_dict(self):
        scores_validation_resource = format_dict_for_resource(self.scores_validation)
        scores_resource = format_dict_for_resource(self.scores)
        features_resource = format_dict_for_resource(self.features)
        parameters_resource = format_dict_for_resource(self.parameters)
        pdf_dependencies = self.build_pdf_dependencies(self.seed)

        return {
            "fitness": sum(list(self.scores.values())),
            "parameter": parameters_resource,
            "score": scores_resource,
            "features": features_resource,
            "scoreValidation": scores_validation_resource,
            "passedValidation": self.passed_validation,
            "nexus_images": pdf_dependencies,
            "seed": self.seed,
        }
