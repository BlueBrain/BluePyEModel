"""Target"""

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
        weight=1.0,
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
            weight (float): weight of the efeature.
                Basically multiplies the score of the efeature by this value.
        """

        self.efeature = efeature
        self.protocol = protocol

        self.amplitude = amplitude
        self.tolerance = tolerance

        self.efeature_name = efeature_name

        self.weight = weight

        if efel_settings is None:
            self.efel_settings = {"strict_stiminterval": True}
        else:
            self.efel_settings = efel_settings

    def as_dict(self):
        return vars(self)
