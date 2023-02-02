"""Target"""
import logging

logger = logging.getLogger(__name__)


class Target:

    """Describes an extraction (or optimisation) target"""

    def __init__(
        self,
        efeature,
        protocol,
        amplitude,
        tolerance,
        efeature_name=None,
        efel_settings=None,
    ):
        """Constructor

        Args:
            efeature (str): name of the eFeature in the eFEL library
                (ex: 'AP1_peak')
            protocol (str): name of the recording from which the efeature
                should be computed
            amplitude (float): amplitude of the current stimuli for which the
                efeature should be computed (expressed as a percentage of the
                threshold amplitude (rheobase))
            tolerance (float): tolerance around the target amplitude in which
                an experimental recording will be seen as a hit during
                efeatures extraction (expressed as a percentage of the
                threshold amplitude (rheobase))
            efeature_name (str): given name for this specific target. Can be different
                from the efel efeature name.
            efel_settings (dict): target specific efel settings.
        """

        self.efeature = efeature
        self.protocol = protocol

        self.amplitude = amplitude
        self.tolerance = tolerance

        self.efeature_name = efeature_name

        if efel_settings is None:
            self.efel_settings = {"strict_stiminterval": True}
        else:
            self.efel_settings = efel_settings

    def as_dict(self):
        return vars(self)
