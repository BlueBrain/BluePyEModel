"""MEModel class"""

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

from bluepyemodel.emodel_pipeline.emodel import EModelMixin


class MEModel(EModelMixin):
    """Combination of an EModel and a Morphology. Should contain ids of these resources,
    as well as analysis plotting figure paths."""

    def __init__(
        self,
        seed=None,
        emodel_metadata=None,
        emodel_id=None,
        morphology_id=None,
        validated=False,
        status="initialized",
    ):
        """Init

        Args:
            seed (int): seed used during optimisation for this emodel.
            emodel_metadata (EModelMetadata): metadata of the model (emodel name, etype, ttype, ...)
            emodel_id (str): nexus if of the e-model used in this me-model
            morphology_id (str): nexus id of the morphology used in this me-model
            validated (bool): whether the MEModel has been validated by user
            status (str): whether the analysis has run or not. Can be "initialized" or "done".
        """

        self.emodel_metadata = emodel_metadata
        self.seed = seed

        self.emodel_id = emodel_id
        self.morphology_id = morphology_id

        self.validated = validated
        self.status = status

    def get_related_nexus_ids(self):
        uses = []
        if self.emodel_id:
            uses.append({"id": self.emodel_id, "type": "EModel"})
        if self.morphology_id:
            uses.append({"id": self.morphology_id, "type": "NeuronMorphology"})
        return {"uses": uses}

    def as_dict(self):
        pdf_dependencies = self.build_pdf_dependencies(self.seed)

        return {
            "nexus_images": pdf_dependencies,
            "seed": self.seed,
            "emodel_id": self.emodel_id,
            "morphology_id": self.morphology_id,
            "validated": self.validated,
            "status": self.status,
        }
