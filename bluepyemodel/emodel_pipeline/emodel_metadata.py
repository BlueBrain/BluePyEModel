"""EModelMetadata class"""

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


class EModelMetadata:

    """Contains the metadata of an emodel such as its e-model name or its brain region. These
    metadata can be understood as a unique identifier of an e-model.

    This class is a backend class and is not meant to be used directly by the user."""

    def __init__(
        self,
        emodel=None,
        etype=None,
        ttype=None,
        mtype=None,
        species=None,
        brain_region=None,
        iteration_tag=None,
        synapse_class=None,
    ):
        """Constructor

        Args:
            emodel (str): name of the e-model
            etype (str): name of the electric type of the e-model.
            ttype (str): name of the transcriptomic type of the e-model.
            mtype (str): name of the morphology type of the e-model.
            species (str): name of the species of the e-model.
            brain_region (str): name of the brain location of the e-model.
            iteration_tag (str): tag associated to the current run.
            synapse_class (str): synapse class (neurotransmitter)  of the e-model.
        """

        if emodel is None and etype is None:
            raise ValueError("At least emodel or etype should be informed")

        self.emodel = emodel
        self.etype = None if etype == "None" else etype
        self.ttype = None if ttype == "None" else ttype
        self.mtype = None if mtype == "None" else mtype
        self.species = None if species == "None" else species
        self.brain_region = None if brain_region == "None" else brain_region
        self.iteration = None if iteration_tag == "None" else iteration_tag
        self.synapse_class = None if synapse_class == "None" else synapse_class

    def etype_annotation_dict(self):
        """Returns an etype annotation dict to be added to annotations list."""
        # TODO also add id from Cell Type Ontology in WebProtege (optional)
        return {
            "type": [
                "ETypeAnnotation",
                "Annotation",
            ],
            "hasBody": {
                "type": [
                    "EType",
                    "AnnotationBody",
                ],
                "label": self.etype,
            },
            "name": "E-type annotation",
        }

    def mtype_annotation_dict(self):
        """Returns an mtype annotation dict to be added to annotations list."""
        # TODO also add id from Cell Type Ontology in WebProtege (optional)
        return {
            "type": [
                "MTypeAnnotation",
                "Annotation",
            ],
            "hasBody": {
                "type": [
                    "MType",
                    "AnnotationBody",
                ],
                "label": self.mtype,
            },
            "name": "M-type annotation",
        }

    def ttype_annotation_dict(self):
        """Returns an ttype annotation dict to be added to annotations list."""
        # TODO also add id from Cell Type Ontology in WebProtege (optional)
        return {
            "type": [
                "TTypeAnnotation",
                "Annotation",
            ],
            "hasBody": {
                "type": [
                    "TType",
                    "AnnotationBody",
                ],
                "label": self.ttype,
            },
            "name": "T-type annotation",
        }

    def annotation_list(self):
        """Returns an annotation list containing mtype, etype and ttype annotations"""
        annotation_list = []
        if self.etype:
            annotation_list.append(self.etype_annotation_dict())
        if self.mtype:
            annotation_list.append(self.mtype_annotation_dict())
        if self.ttype:
            annotation_list.append(self.ttype_annotation_dict())

        return annotation_list

    def get_metadata_dict(self):
        """Metadata as a dict, with keys consistent with nexus."""

        metadata = {}

        for k, v in vars(self).items():
            if v and v != "None":
                # rename species into subject and brain_region into brainLocation
                if k == "species":
                    metadata["subject"] = v
                elif k == "brain_region":
                    metadata["brainLocation"] = v
                else:
                    metadata[k] = v

        return metadata

    def filters_for_resource(self):
        """Metadata used for filtering, without the annotation list"""
        return self.get_metadata_dict()

    def for_resource(self):
        """Metadata to add to a resource to register.

        DO NOT use for filtering. For filtering, use self.filters_for_resource() instead.
        """

        metadata = self.get_metadata_dict()

        metadata["annotation"] = self.annotation_list()

        return metadata

    def as_string(self, seed=None):
        s = ""

        for k in ["emodel", "etype", "ttype", "mtype", "species", "brain_region", "iteration"]:
            v = getattr(self, k)
            if v:
                if isinstance(v, int):
                    v = str(v)
                s += f"{k}={v.replace('/', '')}__"

        if seed not in [None, "None"]:
            s += f"seed={seed}__"

        return s[:-2]
