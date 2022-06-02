"""ProtocolConfiguration"""


class ProtocolConfiguration:

    """Container for the definition of a protocol"""

    def __init__(
        self,
        name,
        stimuli,
        recordings,
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
            recordings (list of dict): contains the description of the recordings. For a recording
                at a given compartment, the format is for example:
                [{
                    "type": "CompRecording",
                    "name": f"{protocol_name}.soma.v",
                    "location": "soma",
                    "variable": "v",
                }]
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

        if isinstance(recordings, dict):
            recordings = [recordings]

        self.recordings = []
        for recording in recordings:
            self.recordings.append(recording)

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

        return vars(self)
