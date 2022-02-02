"""TargetsConfiguration"""
import logging

from bluepyemodel.efeatures_extraction.target import Target
from bluepyemodel.efeatures_extraction.trace_file import TraceFile

logger = logging.getLogger(__name__)


class TargetsConfiguration:

    """The goal of this class is to configure the targets and files metadata that will be
    used during efeature extraction"""

    def __init__(self, files=None, targets=None, protocols_rheobase=None):
        """Init

        Args:
            files (list of dict): File names with their metadata in the format:
                [file1_metadata, file2_metadata, ...]
                file1_metadata = {
                    cell_name=XXX,
                    filename=XXX,
                    resource_id=XXX,
                    etype=XXX,
                    species=XXX,
                    brain_region=XXX,
                    ecodes={},
                    other_metadata={},
                }
            targets (list): define the efeatures to extract as well as which
                protocols and current amplitude they should be extracted for. Of
                the form:
                [{
                    "efeature": "AP_amplitude",
                    "protocol": "IDRest",
                    "amplitude": 150.,
                    "tolerance": 10.,
                    "efel_settings": {
                        'stim_start': 200.,
                        'stim_end': 500.,
                        'Threshold': -10.
                    }
                }]
            protocols_rheobase (list): names of the protocols that will be
                used to compute the rheobase of the cells. E.g: ['IDthresh']."""

        if files is not None:
            self.files = [TraceFile(**f) for f in files]
        else:
            self.files = []

        if targets is not None:
            self.targets = [Target(**t) for t in targets]
        else:
            self.targets = []

        if protocols_rheobase is None:
            self.protocols_rheobase = []
        elif isinstance(protocols_rheobase, str):
            self.protocols_rheobase = [protocols_rheobase]
        else:
            self.protocols_rheobase = protocols_rheobase

    @property
    def files_metadata_BPE(self):
        """In BPE2 input format"""

        files_metadata = {}

        used_protocols = set([t.protocol for t in self.targets] + self.protocols_rheobase)

        for f in self.files:

            if f.cell_name not in files_metadata:
                files_metadata[f.cell_name] = {}

            for p in used_protocols:

                if p not in files_metadata[f.cell_name]:
                    files_metadata[f.cell_name][p] = []

                ecodes_metadata = {**f.ecodes.get(p, {}), **f.other_metadata}
                ecodes_metadata["filepath"] = f.filepath

                files_metadata[f.cell_name][p].append(ecodes_metadata)

            for protocol in self.protocols_rheobase:
                if protocol not in files_metadata[f.cell_name]:
                    raise Exception(
                        f"{protocol} is part of the protocols_rheobase but it has"
                        f" no associated ephys data for cell {f.cell_name}"
                    )

        return files_metadata

    @property
    def targets_BPE(self):
        """In BPE2 input format"""

        return [t.as_dict() for t in self.targets]

    def as_dict(self):

        return {
            "files": [f.as_dict() for f in self.files],
            "targets": [t.as_dict() for t in self.targets],
            "protocols_rheobase": self.protocols_rheobase,
        }
