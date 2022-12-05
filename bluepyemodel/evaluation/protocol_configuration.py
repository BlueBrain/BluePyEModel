"""ProtocolConfiguration"""


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
                [{
                    'amp': float, 'thresh_perc': float, 'holding_current': float, 'delay': float,
                    'duration': float, 'totduration': float
                }]
            recordings_from_config (list of dict): contains the description of the recordings.
                For a recording at a given compartment, the format is for example:
                [{
                    "type": "CompRecording",
                    "name": f"{protocol_name}.soma.v",
                    "location": "soma",
                    "variable": "v",
                }]
            recordings (list of dict): same as recordings_from_config.
                Is here for backward compatibility
            ion_variables (list of str): ion current names and ionic concentration names
                for all available mechanisms
            protocol_type (str): type of the protocol. Can be "ThresholdBasedProtocol" or
                "Protocol".
            stochasticity (bool): whether the mechanisms should be on stochastic mode
                when the protocol runs, or not
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

        prot_as_dict = vars(self)
        prot_as_dict.pop("recordings")
        return prot_as_dict
