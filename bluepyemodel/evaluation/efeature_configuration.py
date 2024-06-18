"""EFeatureConfiguration"""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

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
        sample_size=None,
        default_std_value=1e-3,
        weight=1.0,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efel_feature_name (str): name of the eFEl feature.
            protocol_name (str): name of the protocol to which the efeature is associated. For
                example "Step_200".
            recording_name (str or dict): name of the recording(s) of the protocol. For
                example: "soma.v" or if and only if the feature depends on several recordings:
                {"": "soma.v", "location_AIS": "axon.v"}.
            mean (float): mean of the efeature.
            original_std (float): unmodified standard deviation of the efeature
            std (float): kept for legacy purposes.
            efeature_name (str):given name for this specific feature. Can be different
                from the efel efeature name.
            efel_settings (dict): eFEl settings.
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional).
            sample_size (float): number of data point that were used to compute the present
                average and standard deviation.
            weight (float): weight of the efeature.
                Basically multiplies the score of the efeature by this value.
        """

        self.efel_feature_name = efel_feature_name
        self.protocol_name = protocol_name
        self.recording_name = recording_name
        self.threshold_efeature_std = threshold_efeature_std
        self.default_std_value = default_std_value

        self.mean = mean
        self.original_std = original_std if original_std is not None else std
        self.sample_size = sample_size

        self.efeature_name = efeature_name
        self.weight = weight

        if efel_settings is None:
            self.efel_settings = {"strict_stiminterval": True}
        else:
            self.efel_settings = efel_settings

    @property
    def name(self):
        n = self.efeature_name if self.efeature_name else self.efel_feature_name
        if isinstance(self.recording_name, dict):
            return f"{self.protocol_name}.{self.recording_name['']}.{n}"
        return f"{self.protocol_name}.{self.recording_name}.{n}"

    @property
    def recording_name_for_instantiation(self):
        if isinstance(self.recording_name, dict):
            return {k: f"{self.protocol_name}.{v}" for k, v in self.recording_name.items()}
        return {"": f"{self.protocol_name}.{self.recording_name}"}

    @property
    def std(self):
        """Limit the standard deviation with a lower bound equal to a percentage of the mean."""

        if self.threshold_efeature_std is None:
            return self.original_std

        if self.mean == 0.0:
            if self.threshold_efeature_std:
                return self.threshold_efeature_std
            return self.default_std_value

        limit = abs(self.threshold_efeature_std * self.mean)
        if self.original_std < limit:
            return limit

        return self.original_std

    def as_dict(self):
        """Dictionary form"""

        return vars(self)
