"""ProtocolConfiguration"""


class ProtocolConfiguration:

    """Container for the definition of a protocol"""

    def __init__(
        self,
        name,
        stimuli,
        recordings,
        validation=False,
        protocol_type="ThresholdBasedProtocol",
        ion_currents=None,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            name (str): name of the protocol
            stimuli (list of dict):
            recordings (list of dict):
            ion_currents (list of str): ion current names for all available mechanisms
            protocol_type (str):type of the protocol. Can be "ThresholdBasedProtocol" or "Protocol".
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

            if ion_currents is not None:
                for ion in ion_currents:
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

    def as_dict(self):
        """Dictionary form"""

        return vars(self)
