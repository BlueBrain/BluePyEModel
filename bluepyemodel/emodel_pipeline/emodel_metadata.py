"""EModelMetadata class"""


class EModelMetadata:

    """Contains the metadata of an emodel such as its emodel name or its brain region"""

    def __init__(
        self,
        emodel=None,
        etype=None,
        ttype=None,
        mtype=None,
        species=None,
        brain_region=None,
        iteration_tag=None,
        morph_class=None,
        synapse_class=None,
        layer=None,
    ):

        if emodel is None and etype is None:
            raise Exception("At least emodel or etype should be informed")
        if morph_class not in ["PYR", "INT", None]:
            raise Exception("morph_class should be 'PYR' or 'INT'")
        if synapse_class not in ["EXC", "INH", None]:
            raise Exception("synapse_class should be 'EXC' or 'INH'")

        self.emodel = emodel
        self.etype = None if etype == "None" else etype
        self.ttype = None if ttype == "None" else ttype
        self.mtype = None if mtype == "None" else mtype
        self.species = None if species == "None" else species
        self.brain_region = None if brain_region == "None" else brain_region
        self.iteration = None if iteration_tag == "None" else iteration_tag
        self.morph_class = morph_class
        self.synapse_class = synapse_class
        self.layer = layer

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
        metadata = self.get_metadata_dict()
        return metadata

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
