"""TraceFile"""
import logging

logger = logging.getLogger(__name__)


class TraceFile:

    """Contains the metadata of a trace file"""

    def __init__(
        self,
        cell_name,
        filename=None,
        filepath=None,
        resource_id=None,
        etype=None,
        species=None,
        brain_region=None,
        ecodes=None,
        other_metadata=None,
    ):

        self.cell_name = cell_name
        self.filename = filename if filename else cell_name
        self.filepath = filepath
        self.resource_id = resource_id

        self.etype = etype
        self.species = species
        self.brain_region = brain_region

        self.ecodes = ecodes

        self.other_metadata = other_metadata if other_metadata is not None else {}

    def matching_score(self, ecode, etype=None, species=None, brain_region=None):

        if self.ecodes is not None and ecode in self.ecodes:

            criteria = [etype, species, brain_region]
            self_criteria = [self.etype, self.species, self.brain_region]

            score = sum(
                c == sc
                for c, sc in zip(criteria, self_criteria)
                if c is not None and sc is not None
            )

            return score / len(criteria)

        return 0

    def as_dict(self):

        return vars(self)
