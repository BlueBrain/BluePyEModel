"""EFeatureConfiguration"""


class EFeatureConfiguration:

    """Container for the definition of an EFeature"""

    def __init__(
        self,
        efel_feature_name,
        protocol_name,
        recording_name,
        mean,
        efeature_name=None,
        efel_settings=None,
        threshold_efeature_std=None,
        original_std=None,
        std=None,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efel_feature_name (str): name of the eFEl feature.
            protocol_name (str): name of the protocol to which the efeature is associated. For
                example "Step_200".
            recording_name (str): name of the recording of the protocol. For example: "soma.v"
            mean (float): mean of the efeature.
            original_std (float): unmodified standard deviation of the efeature
            std (float): kept for legacy purposes.
            efeature_name (str):given name for this specific feature. Can be different
                from the efel efeature name.
            efel_settings (dict): eFEl settings.
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional).
        """

        self.efel_feature_name = efel_feature_name
        self.protocol_name = protocol_name
        self.recording_name = recording_name
        self.threshold_efeature_std = threshold_efeature_std

        self.mean = mean
        self.original_std = original_std if original_std is not None else std

        self.efeature_name = efeature_name

        if efel_settings is None:
            self.efel_settings = {"strict_stiminterval": True}
        else:
            self.efel_settings = efel_settings

    @property
    def name(self):
        n = self.efeature_name if self.efeature_name else self.efel_feature_name
        return f"{self.protocol_name}.{self.recording_name}.{n}"

    @property
    def std(self):
        """Limit the standard deviation with a lower bound equal to a percentage of the mean."""

        if self.threshold_efeature_std is None:
            return self.original_std

        if self.mean == 0.0:
            return self.threshold_efeature_std

        limit = abs(self.threshold_efeature_std * self.mean)
        if self.original_std < limit:
            return limit

        return self.original_std

    def as_dict(self):
        """Dictionary form"""

        return vars(self)
