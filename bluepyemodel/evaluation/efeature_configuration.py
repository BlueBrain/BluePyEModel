"""EFeatureConfiguration"""


def _limit_std(mean, std, threshold_efeature_std):
    """Limit the standard deviation with a lower bound equal to a percentage of the mean."""

    if threshold_efeature_std is None:
        return std

    limit = abs(threshold_efeature_std * mean)

    if std < limit:
        return limit

    return std


class EFeatureConfiguration:

    """Container for the definition of an EFeature"""

    def __init__(
        self,
        efel_feature_name,
        protocol_name,
        recording_name,
        mean,
        std,
        efel_settings=None,
        threshold_efeature_std=None,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efel_feature_name (str): name of the efel feature.
            protocol_name (str): name of the protocol to which the efeature is associated.
            recording_name (str): name of the recording of the procol
            mean (float): mean of the efeature.
            std (float): standard deviation of the efeature.
            efel_settings (dict): eFEl settings.
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional).
        """

        self.efel_feature_name = efel_feature_name
        self.protocol_name = protocol_name
        self.recording_name = recording_name

        self.mean = mean
        self.std = _limit_std(mean, std, threshold_efeature_std)

        if efel_settings is None:
            self.efel_settings = {"strict_stim": True}
        else:
            self.efel_settings = efel_settings

    @property
    def name(self):
        return f"{self.protocol_name}.{self.recording_name}.{self.efel_feature_name}"

    def as_dict(self):
        """Dictionary form"""

        return {
            "efel_feature_name": self.efel_feature_name,
            "protocol_name": self.protocol_name,
            "recording_name": self.recording_name,
            "efel_settings": self.efel_settings,
            "mean": self.mean,
            "std": self.std,
        }