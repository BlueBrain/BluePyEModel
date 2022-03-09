"""TraceFile"""
import logging

logger = logging.getLogger(__name__)


def list_ecodes_per_traces(traces, threshold_count=0):
    """Utility method. Return the list of ecodes available for each trace. Only
    the ecodes present more than threshold_count times are considered."""

    ecodes_per_traces = {}
    all_ = []
    for t in traces:
        if t.ecodes:
            ecodes = [str(e) for e in t.ecodes]
            ecodes_per_traces[t.cell_name] = ecodes
            all_ += ecodes
        else:
            ecodes_per_traces[t.cell_name] = []

    ecodes_per_traces["all"] = sorted(list(set(all_)))

    count_ecodes = {e: all_.count(e) for e in ecodes_per_traces["all"]}

    for t in ecodes_per_traces:
        ecodes_per_traces[t] = list(
            set(e for e in ecodes_per_traces[t] if e and count_ecodes[e] > threshold_count)
        )

    return ecodes_per_traces


class TraceFile:

    """Contains the metadata of a trace file"""

    def __init__(
        self,
        cell_name,
        filename=None,
        filepath=None,
        resource_id=None,
        ecodes=None,
        other_metadata=None,
        species=None,
        brain_region=None,
        etype=None,
    ):

        self.cell_name = cell_name
        self.filename = filename if filename else cell_name
        self.filepath = filepath
        self.resource_id = resource_id

        self.ecodes = ecodes

        self.other_metadata = other_metadata if other_metadata is not None else {}

        self.species = species
        self.brain_region = brain_region
        self.etype = etype

    def as_dict(self):

        return vars(self)

    def __eq__(self, other):

        if self.cell_name == other.cell_name:

            if self.filename and other.filename:
                if self.filename == other.filename:
                    return True
                return False

            return True

        return False
