"""TargetsConfiguration"""
import itertools
import logging

from bluepyemodel.efeatures_extraction.target import Target
from bluepyemodel.efeatures_extraction.trace_file import TraceFile

logger = logging.getLogger(__name__)


class TargetsConfiguration:

    """The goal of this class is to configure the targets and files metadata that will be
    used during efeature extraction"""

    def __init__(
        self,
        files=None,
        targets=None,
        protocols_rheobase=None,
        available_traces=None,
        available_efeatures=None,
    ):
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
                    "efeature_name": "AP_amplitude_1",
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
                used to compute the rheobase of the cells. E.g: ['IDthresh'].
            available_traces (list of TraceFile)
            available_efeatures (llist of strings)
        """

        self.available_traces = available_traces
        self.available_efeatures = available_efeatures

        self.files = []
        if files is not None:
            for f in files:
                tmp_trace = TraceFile(**f)
                if not self.is_trace_available(tmp_trace):
                    raise Exception(f"File named {tmp_trace.cell_name} is not present on Nexus")
                self.files.append(tmp_trace)

        self.targets = []
        if targets is not None:
            for t in targets:
                tmp_target = Target(**t)
                if not self.is_target_available(tmp_target):
                    raise Exception(f"Efeature name {tmp_target.efeature} does not exist")
                self.targets.append(tmp_target)

        if protocols_rheobase is None:
            self.protocols_rheobase = []
        elif isinstance(protocols_rheobase, str):
            self.protocols_rheobase = [protocols_rheobase]
        else:
            self.protocols_rheobase = protocols_rheobase

    def is_trace_available(self, trace):
        if self.available_traces:
            available = next((a for a in self.available_traces if a == trace), False)
            return bool(available)
        return True

    def is_target_available(self, target):
        if self.available_efeatures and target.efeature not in self.available_efeatures:
            return False
        return True

    @property
    def files_metadata_BPE(self):
        """In BPE2 input format"""

        files_metadata = {}

        used_protocols = set([t.protocol for t in self.targets] + self.protocols_rheobase)

        for f in self.files:
            for protocol in f.ecodes:
                if protocol in used_protocols:

                    if f.cell_name not in files_metadata:
                        files_metadata[f.cell_name] = {}
                    if protocol not in files_metadata[f.cell_name]:
                        files_metadata[f.cell_name][protocol] = []

                    ecodes_metadata = {
                        **f.ecodes.get(protocol, {}),
                        **f.other_metadata,
                        "filepath": f.filepath,
                    }

                    if "protocol_name" not in ecodes_metadata:
                        ecodes_metadata["protocol_name"] = protocol

                    files_metadata[f.cell_name][protocol].append(ecodes_metadata)

        for cell_name, protocols in files_metadata.items():
            for protocol in self.protocols_rheobase:
                if protocol in protocols:
                    break
            else:
                raise Exception(
                    f"{protocol} is part of the protocols_rheobase but it has"
                    f" no associated ephys data for cell {cell_name}"
                )

        return files_metadata

    @property
    def targets_BPE(self):
        """In BPE2 input format"""

        return [t.as_dict() for t in self.targets]

    @property
    def is_configuration_valid(self):
        """Checks that the configuration has targets, traces and that the targets can
        be found in the traces. This check can only be performed if the ecodes present
        in each files are known."""

        if not self.targets or not self.files:
            return False

        ecodes = set(
            itertools.chain(*[file.ecodes for file in self.files if file.ecodes is not None])
        )

        for target in self.targets:
            if ecodes and target.protocol not in ecodes:
                return False

        return True

    def check_presence_RMP_Rin_efeatures(self, name_rmp_protocol, name_Rin_protocol):
        """Check that the protocols supposed to be used for RMP and Rin are present in the target
        and that they have the correct efeatures. If some features are missing, add them."""

        name_rmp, amplitude_rmp = name_rmp_protocol.efeature.split("_")
        name_rin, amplitude_rin = name_Rin_protocol.efeature.split("_")

        efeatures_rmp = [
            t.efeature
            for t in self.targets
            if t.protocol == name_rmp and t.amplitude == int(amplitude_rmp)
        ]
        efeatures_rin = [
            t.efeature
            for t in self.targets
            if t.protocol == name_rin and t.amplitude == int(amplitude_rin)
        ]

        error_message = (
            "Target for feature {} is missing for RMP protocol {}. Please add "
            "it if you wish to do a threshold-based optimization."
        )

        if "voltage_base" not in efeatures_rmp:
            raise Exception(error_message.format("voltage_base", name_rmp_protocol))
        if "voltage_base" not in efeatures_rin:
            raise Exception(error_message.format("voltage_base", name_rmp_protocol))
        if "ohmic_input_resistance_vb_ssse" not in efeatures_rin:
            raise Exception(
                error_message.format("ohmic_input_resistance_vb_ssse", name_Rin_protocol)
            )

    def as_dict(self):

        return {
            "files": [f.as_dict() for f in self.files],
            "targets": [t.as_dict() for t in self.targets],
            "protocols_rheobase": self.protocols_rheobase,
        }
