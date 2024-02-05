"""TargetsConfiguration"""

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

import itertools
import logging

from bluepyefe.auto_targets import AutoTarget

from bluepyemodel.efeatures_extraction.target import Target
from bluepyemodel.efeatures_extraction.trace_file import TraceFile
from bluepyemodel.tools.utils import are_same_protocol
from bluepyemodel.tools.utils import format_protocol_name_to_list

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
        auto_targets=None,
        additional_fitness_efeatures=None,
        additional_fitness_protocols=None,
        protocols_mapping=None,
    ):
        """Init

        Args:
            files (list of dict): File names with their metadata in the format:

                .. code-block::

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

                .. code-block::

                    [{
                        "efeature_name": "AP_amplitude_1",
                        "efeature": "AP_amplitude",
                        "protocol": "IDRest",
                        "amplitude": 150.,
                        "tolerance": 10.,
                        "weight": 1.0,  # optional
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
            auto_targets (list): if targets is not given, auto_targets
                define the efeatures to extract as well as which
                protocols and current amplitude they should be extracted for,
                given a list of possible protocols and amplitudes.
                min_recordings_per_amplitude, preferred_number_protocols and
                tolerance are optional.
                Of the form:

                .. code-block::

                    [{
                        "protocols": ["IDRest", "IV"],
                        "amplitudes": [150, 250],
                        "efeatures": ["AP_amplitude", "mean_frequency"],
                        "min_recordings_per_amplitude": 10,
                        "preferred_number_protocols": 1,
                        "tolerance": 10.,
                    }]
            additional_fitness_efeatures (list of dicts): efeatures to add to
                the output of the extraction, i.e. to the FitnessCalculatorConfiguration (FCC).
                These efeatures will not be extracted from the targets,
                but will be used during optimisation and / or validation.
                They should have the same format as the efeatures of the FCC, e.g.

                .. code-block::

                    [
                        {
                            "efel_feature_name": "Spikecount",
                            "protocol_name": "IDrest_130",
                            "recording_name": "soma.v",
                            "mean": 6.026,
                            "original_std": 4.016,
                            "efeature_name": "Spikecount",
                            "efel_settings": {
                                "strict_stiminterval": true,
                                "Threshold": -30.0,
                                "interp_step": 0.025,
                                "stim_start": 700.0,
                                "stim_end": 2700.0
                            }
                        }
                    ]
            additional_fitness_protocols (list of dicts): protocols to add to
                the output of the extraction, i.e. to the FitnessCalculatorConfiguration (FCC).
                These protocols will not be used during extraction,
                but will be used during optimisation and / or validation.
                They should have the same format as the protocols of the FCC, e.g.

                .. code-block::

                    [
                        {
                            "name": "IDrest_130",
                            "stimuli": [
                                {
                                "delay": 663.3473451327434,
                                "amp": 0.3491827776582547,
                                "thresh_perc": 130.56103157894464,
                                "duration": 2053.050884955752,
                                "totduration": 3000.0,
                                "holding_current": -0.13307330411636328
                                }
                            ],
                            "recordings_from_config": [
                                {
                                "type": "CompRecording",
                                "name": "IDrest_130.soma.v",
                                "location": "soma",
                                "variable": "v"
                                }
                            ],
                            "validation": false,
                            "protocol_type": "ThresholdBasedProtocol",
                            "stochasticity": false
                        }
                    ]
             protocols_mapping (dict, optional): maps original protocol
                identifiers to renamed versions for standardization.
                Defaults to None if not provided.

                Example:

                .. code-block::

                    {
                        "IDRest_150": 'Step_150_hyp',
                        "IDRest_200": 'Step_200_hyp',
                        "IDRest_250": 'Step_250_hyp',
                        "IV_-120": 'IV_-120_hyp',
                    }
        """

        self.available_traces = available_traces
        self.available_efeatures = available_efeatures

        self.files = []
        if files is not None:
            for f in files:
                tmp_trace = TraceFile(**f)
                if not self.is_trace_available(tmp_trace):
                    raise ValueError(f"File named {tmp_trace.cell_name} is not present on Nexus")
                self.files.append(tmp_trace)

        self.targets = []
        self.auto_targets = None
        if targets is not None:
            for t in targets:
                tmp_target = Target(**t)
                if not self.is_target_available(tmp_target):
                    raise ValueError(f"Efeature name {tmp_target.efeature} does not exist")
                self.targets.append(tmp_target)
        if auto_targets is not None:
            self.auto_targets = auto_targets

        if protocols_rheobase is None:
            self.protocols_rheobase = []
        elif isinstance(protocols_rheobase, str):
            self.protocols_rheobase = [protocols_rheobase]
        else:
            self.protocols_rheobase = protocols_rheobase

        self.additional_fitness_efeatures = additional_fitness_efeatures
        self.additional_fitness_protocols = additional_fitness_protocols

        self.protocols_mapping = protocols_mapping

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

        if self.files:
            files = self.files
        else:
            logger.info("No files given. Will use all available traces instead.")
            files = self.available_traces
        files_metadata = {}

        if self.targets:
            used_protocols = set([t.protocol for t in self.targets] + self.protocols_rheobase)
        elif self.auto_targets:
            used_protocols = set(
                [p for t in self.auto_targets for p in t["protocols"]] + self.protocols_rheobase
            )
        else:
            raise TypeError("either targets or autotargets should be set.")

        for f in files:
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
            if self.protocols_rheobase:
                for protocol in self.protocols_rheobase:
                    if protocol in protocols:
                        break
                else:
                    raise ValueError(
                        f"{protocol} is part of the protocols_rheobase but it has"
                        f" no associated ephys data for cell {cell_name}"
                    )
        return files_metadata

    @property
    def targets_BPE(self):
        """In BPE2 input format"""
        if not self.targets:
            return None
        return [t.as_dict() for t in self.targets]

    @property
    def auto_targets_BPE(self):
        """In BPE2 input format"""
        if not self.auto_targets:
            return None
        return [AutoTarget(**at) for at in self.auto_targets]

    @property
    def protocols_rheobase_BPE(self):
        """Returns None if empty"""
        if not self.protocols_rheobase:
            return None
        return self.protocols_rheobase

    @property
    def is_configuration_valid(self):
        """Checks that the configuration has targets, traces and that the targets can
        be found in the traces. This check can only be performed if the ecodes present
        in each files are known."""

        if not self.auto_targets:
            if not self.targets or not self.files:
                return False

        if self.targets and self.auto_targets:
            return False

        ecodes = set(
            itertools.chain(*[file.ecodes for file in self.files if file.ecodes is not None])
        )

        for target in self.targets:
            if ecodes and target.protocol not in ecodes:
                return False

        if self.auto_targets:
            for at in self.auto_targets:
                if not (
                    "protocols" in at.keys()
                    and "amplitudes" in at.keys()
                    and "efeatures" in at.keys()
                ):
                    return False

        return True

    def check_presence_RMP_Rin_efeatures(self, name_rmp_protocol, name_Rin_protocol):
        """Check that the protocols supposed to be used for RMP and Rin are present in the target
        and that they have the correct efeatures. If some features are missing, add them."""

        if self.targets:
            efeatures_rmp = [
                t.efeature
                for t in self.targets
                if are_same_protocol([t.protocol, t.amplitude], name_rmp_protocol)
            ]
            efeatures_rin = [
                t.efeature
                for t in self.targets
                if are_same_protocol([t.protocol, t.amplitude], name_Rin_protocol)
            ]
        elif self.auto_targets:
            efeatures_rmp = [
                efeat
                for t in self.auto_targets
                for efeat in t["efeatures"]
                if format_protocol_name_to_list(name_rmp_protocol)[0] in t["protocols"]
            ]
            efeatures_rin = [
                efeat
                for t in self.auto_targets
                for efeat in t["efeatures"]
                if format_protocol_name_to_list(name_Rin_protocol)[0] in t["protocols"]
            ]
        else:
            raise TypeError("either targets or autotargets should be set.")

        error_message = (
            "Target for feature {} is missing for protocol {}. Please add "
            "it if you wish to do a threshold-based optimisation."
        )

        if "voltage_base" not in efeatures_rmp:
            raise ValueError(error_message.format("voltage_base", name_rmp_protocol))
        if "voltage_base" not in efeatures_rin:
            raise ValueError(error_message.format("voltage_base", name_Rin_protocol))
        if "ohmic_input_resistance_vb_ssse" not in efeatures_rin:
            raise ValueError(
                error_message.format("ohmic_input_resistance_vb_ssse", name_Rin_protocol)
            )

    def get_related_nexus_ids(self):
        uses = []
        for f in self.files:
            if f.id:
                f_dict = {"id": f.id, "type": "Trace"}
                if f_dict not in uses:
                    uses.append(f_dict)

        return {"uses": uses}

    def as_dict(self):
        return {
            "files": [f.as_dict() for f in self.files],
            "targets": [t.as_dict() for t in self.targets],
            "protocols_rheobase": self.protocols_rheobase,
            "auto_targets": self.auto_targets,
            "additional_fitness_efeatures": self.additional_fitness_efeatures,
            "additional_fitness_protocols": self.additional_fitness_protocols,
            "protocols_mapping": self.protocols_mapping,
        }
