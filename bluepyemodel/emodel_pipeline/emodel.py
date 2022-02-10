"""EModel class"""
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

    """Contains a set of parameters for the EModel and its matching scores and efeatures"""

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
            emodel_metadata (str): metadata of the model (emodel name, etype, ttype, ...)
        """

        self.emodel_metadata = emodel_metadata
        self.passed_validation = passedValidation
        self.fitness = fitness
        self.seed = seed

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
            self.features = {p["name"]: (p["value"] if "value" in p else 0) for p in features} if features else {}

        if isinstance(scoreValidation, dict):
            self.scores_validation = scoreValidation
        else:
            self.scores_validation = (
                {p["name"]: p["value"] for p in scoreValidation} if scoreValidation else {}
            )

        self.responses = {}
        self.evaluator = None

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

        parameters_pdf = search_pdfs.search_figure_emodel_parameters(self.emodel_metadata)
        if parameters_pdf:
            pdfs += [p for p in parameters_pdf if p]

        return pdfs

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
            "nexus_distributions": pdf_dependencies,
        }
