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
    ):

        if emodel is None and etype is None:
            raise Exception("At least emodel or etype should be informed")

        self.emodel = emodel
        self.etype = None if etype == "None" else etype
        self.ttype = None if ttype == "None" else ttype
        self.mtype = None if mtype == "None" else mtype
        self.species = None if species == "None" else species
        self.brain_region = None if brain_region == "None" else brain_region
        self.iteration = None if iteration_tag == "None" else iteration_tag

    def for_resource(self):

        metadata = {}

        for k, v in vars(self).items():
            if v and v != "None":
                # rename species into subject
                if k == "species":
                    metadata["subject"] = v
                else:
                    metadata[k] = v

        return metadata

    def as_string(self, seed=None):

        s = ""

        for k, v in vars(self).items():
            if v:
                if isinstance(v, int):
                    v = str(v)
                s += f"{k}={v.replace('/', '')}__"

        if seed not in [None, "None"]:
            s += f"seed={seed}__"

        return s[:-2]
