"""ProtocolConfiguration"""


class ProtocolConfiguration:

    """Container for the definition of a protocol"""

    def __init__(self, name, stimuli, recordings, validation=False):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            name (str): name of the protocol
            stimuli (list of dict):
            recordings (list of dict):
        """

        self.name = name

        self.stimuli = stimuli
        if isinstance(self.stimuli, dict):
            self.stimuli = [self.stimuli]

        self.recordings = recordings
        if isinstance(self.recordings, dict):
            self.recordings = [self.recordings]

        self.validation = validation

    def as_dict(self):
        """Dictionary form"""

        out = {}
        for a in ["name", "stimuli", "recordings", "validation"]:
            out[a] = getattr(self, a)

        return out
