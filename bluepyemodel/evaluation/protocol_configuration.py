"""ProtocolConfiguration"""

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

import copy


class ProtocolConfiguration:
    """Container for the definition of a protocol"""

    def __init__(
        self,
        name,
        stimuli,
        recordings_from_config=None,
        recordings=None,
        validation=False,
        ion_variables=None,
        protocol_type="ThresholdBasedProtocol",
        stochasticity=False,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            name (str): name of the protocol
            stimuli (list of dict): contains the description of the stimuli. The exact format has
                to match what is expected by the related eCode class (see the classes defined
                in bluepyemodel.ecodes for more information). For example, for a Step protocol,
                the format will be:

                .. code-block::

                    [{
                        'amp': float, 'thresh_perc': float, 'holding_current': float,
                        'delay': float, 'duration': float, 'totduration': float, 'location': dict
                    }]
                    The location should have a format that can be read by
                    evaluation.evaluator.define_location()

            recordings_from_config (list of dict): contains the description of the recordings.
                For a recording at a given compartment, the format is for example:

                .. code-block::

                    [{
                        "type": "CompRecording",
                        "name": f"{protocol_name}.soma.v",
                        "location": "soma",
                        "variable": "v",
                    }]
            recordings (list of dict): same as recordings_from_config. Is here for backward
                compatibility only.
            ion_variables (list of str): ion current names and ionic concentration names
                for all available mechanisms.
            protocol_type (str): type of the protocol. Can be "ThresholdBasedProtocol" or
                "Protocol". When using "ThresholdBasedProtocol", the current amplitude and step
                amplitude of the stimulus will be ignored and replaced by values obtained from
                the holding current and rheobase of the cell model respectively. When using
                "Protocol", the current amplitude and step amplitude of the stimulus will be
                used directly, in this case, if a "thresh_perc" was informed, it will be ignored.
            stochasticity (bool): whether the mechanisms should be on stochastic mode
                when the protocol runs, or not.
        """

        self.name = name

        self.stimuli = stimuli
        if isinstance(self.stimuli, dict):
            self.stimuli = [self.stimuli]

        if recordings_from_config is None:
            if recordings is None:
                raise ValueError("Expected recordings_from_config to be not None")
            recordings_from_config = recordings
        if isinstance(recordings_from_config, dict):
            recordings_from_config = [recordings_from_config]

        self.recordings = []
        self.recordings_from_config = []
        for recording in recordings_from_config:
            self.recordings.append(recording)
            self.recordings_from_config.append(recording)

            if ion_variables is not None:
                for ion in ion_variables:
                    new_rec = recording.copy()

                    if "variable" in recording:
                        new_rec["variable"] = ion
                    elif "var" in recording:
                        new_rec["var"] = ion
                    else:
                        raise KeyError("Expected 'var' or 'variable' in recording list.")

                    new_rec["name"] = ".".join(new_rec["name"].split(".")[:-1] + [ion])

                    self.recordings.append(new_rec)

        self.validation = validation
        self.protocol_type = protocol_type

        self.stochasticity = stochasticity

    def as_dict(self):
        """Dictionary form"""

        prot_as_dict = copy.deepcopy(vars(self))
        prot_as_dict.pop("recordings")
        return prot_as_dict
