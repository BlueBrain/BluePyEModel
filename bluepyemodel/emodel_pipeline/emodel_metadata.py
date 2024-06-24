"""EModelMetadata class"""

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

import string


def update_metadata_dict(metadata_dict, var, new_key):
    """Update metadata dict with new key and variable if not None."""
    if var is not None and var != "None":
        metadata_dict[new_key] = var


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
        allen_notation=None,
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
            allen_notation (str): Allen acronym for the brain region (Optional).
                Can be used to replace brain region in as_string().
        """

        if emodel is None and etype is None:
            raise ValueError("At least emodel or etype should be informed")

        self.emodel = emodel
        self.check_emodel_name()  # check if name complies with requirements
        self.etype = None if etype == "None" else etype
        self.ttype = None if ttype == "None" else ttype
        self.mtype = None if mtype == "None" else mtype
        self.species = None if species == "None" else species
        self.brain_region = None if brain_region == "None" else brain_region
        self.iteration = None if iteration_tag == "None" else iteration_tag
        self.synapse_class = None if synapse_class == "None" else synapse_class
        self.allen_notation = allen_notation

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

    def as_dict_for_resource(self):
        """Metadata as a dict, with keys consistent with nexus."""

        metadata_dict = {}

        update_metadata_dict(metadata_dict, self.emodel, "eModel")
        update_metadata_dict(metadata_dict, self.etype, "eType")
        update_metadata_dict(metadata_dict, self.ttype, "tType")
        update_metadata_dict(metadata_dict, self.mtype, "mType")
        update_metadata_dict(metadata_dict, self.iteration, "iteration")
        update_metadata_dict(metadata_dict, self.synapse_class, "synapseClass")
        # rename species into subject and brain_region into brainLocation
        update_metadata_dict(metadata_dict, self.species, "subject")
        update_metadata_dict(metadata_dict, self.brain_region, "brainLocation")
        # we do not want allen_notation in resource metadata

        return metadata_dict

    def as_dict_for_resource_legacy(self):
        """Metadata as a dict, with keys consistent with legacy nexus."""

        metadata_dict = {}

        for k, v in vars(self).items():
            # we do not want allen_notation in resource metadata
            if v and v != "None" and k != "allen_notation":
                # rename species into subject and brain_region into brainLocation
                if k == "species":
                    metadata_dict["subject"] = v
                elif k == "brain_region":
                    metadata_dict["brainLocation"] = v
                else:
                    metadata_dict[k] = v

        return metadata_dict

    def filters_for_resource(self):
        """Metadata used for filtering, without the annotation list"""
        return self.as_dict_for_resource()

    def filters_for_resource_legacy(self):
        """Legacy metadata used for filtering, without the annotation list"""
        return self.as_dict_for_resource_legacy()

    def for_resource(self):
        """Metadata to add to a resource to register.

        DO NOT use for filtering. For filtering, use self.filters_for_resource() instead.
        """

        metadata = self.as_dict_for_resource()

        metadata["annotation"] = self.annotation_list()

        return metadata

    def as_dict(self):
        """Metadata as dict."""
        return vars(self)

    def as_string(
        self, seed=None, use_allen_notation=True, replace_semicolons=True, replace_spaces=True
    ):
        s = ""

        for k in [
            "emodel",
            "etype",
            "ttype",
            "mtype",
            "species",
            "brain_region",
            "iteration",
        ]:
            v = getattr(self, k)
            if use_allen_notation and k == "brain_region" and self.allen_notation:
                v = self.allen_notation
            if v:
                if isinstance(v, int):
                    v = str(v)
                s += f"{k}={v.replace('/', '')}__"

        if seed not in [None, "None"]:
            s += f"seed={seed}__"

        # can have ':' in mtype. Replace this character.
        if replace_semicolons:
            s = s.replace(":", "_")

        # also replace spaces if any
        if replace_spaces:
            s = s.replace(" ", "_")

        return s[:-2]

    def check_emodel_name(self):
        """Check if name complies with requirements:
        https://nrn.readthedocs.io/en/8.2.3/guide/hoc_chapter_11_old_reference.html#names
        """

        allowed_chars = string.ascii_letters + string.digits + "_"
        translate_args = str.maketrans("", "", allowed_chars)

        if (
            self.emodel == ""
            or self.emodel[0] not in string.ascii_letters
            or not str(self.emodel).translate(translate_args) == ""
        ):
            raise TypeError(
                f"Emodel: name {self.emodel} provided to constructor does not comply "
                "with the rules for Neuron template name: name should be "
                "alphanumeric "
                "non-empty string, underscores are allowed, "
                "first char should be letter"
            )
