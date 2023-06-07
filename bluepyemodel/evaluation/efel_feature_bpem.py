"""Class eFELFeatureBPEM"""

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

import logging
import math

import numpy
from bluepyopt.ephys.efeatures import eFELFeature

logger = logging.getLogger(__name__)


class eFELFeatureBPEM(eFELFeature):

    """eFEL feature extra"""

    SERIALIZED_FIELDS = (
        "name",
        "efel_feature_name",
        "recording_names",
        "stim_start",
        "stim_end",
        "exp_mean",
        "exp_std",
        "threshold",
        "comment",
    )

    def __init__(
        self,
        name,
        efel_feature_name=None,
        recording_names=None,
        stim_start=None,
        stim_end=None,
        exp_mean=None,
        exp_std=None,
        threshold=None,
        stimulus_current=None,
        comment="",
        interp_step=None,
        double_settings=None,
        int_settings=None,
        string_settings=None,
    ):
        """Constructor

        Args:
            name (str): name of the eFELFeature object
            efel_feature_name (str): name of the eFeature in the eFEL library
                (ex: 'AP1_peak')
            recording_names (dict): eFEL features can accept several recordings
                as input
            stim_start (float): stimulation start time (ms)
            stim_end (float): stimulation end time (ms)
            exp_mean (float): experimental mean of this eFeature
            exp_std(float): experimental standard deviation of this eFeature
            threshold(float): spike detection threshold (mV)
            comment (str): comment
        """

        super().__init__(
            name,
            efel_feature_name,
            recording_names,
            stim_start,
            stim_end,
            exp_mean,
            exp_std,
            threshold,
            stimulus_current,
            comment,
            interp_step,
            double_settings,
            int_settings,
            string_settings,
            max_score=250.0,
        )

    def calculate_bpo_feature(self, responses):
        """Return internal feature which is directly passed as a response"""

        if self.efel_feature_name not in responses:
            return None

        return responses[self.efel_feature_name]

    def calculate_bpo_score(self, responses):
        """Return score for bpo feature"""

        feature_value = self.calculate_bpo_feature(responses)

        if feature_value is None:
            return self.max_score

        return abs(feature_value - self.exp_mean) / self.exp_std

    def _construct_efel_trace(self, responses):
        """Construct trace that can be passed to eFEL"""

        trace = {}
        if "" not in self.recording_names:
            raise ValueError("eFELFeature: '' needs to be in recording_names")
        for location_name, recording_name in self.recording_names.items():
            if location_name == "":
                postfix = ""
            else:
                postfix = f";{location_name}"

            if recording_name not in responses:
                logger.debug(
                    "Recording named %s not found in responses %s", recording_name, str(responses)
                )
                return None

            if responses[self.recording_names[""]] is None or responses[recording_name] is None:
                return None
            trace[f"T{postfix}"] = responses[self.recording_names[""]]["time"]
            trace[f"V{postfix}"] = responses[recording_name]["voltage"]

            if callable(self.stim_start):
                trace[f"stim_start{postfix}"] = [self.stim_start()]
            else:
                trace[f"stim_start{postfix}"] = [self.stim_start]

            if callable(self.stim_end):
                trace[f"stim_end{postfix}"] = [self.stim_end()]
            else:
                trace[f"stim_end{postfix}"] = [self.stim_end]

        return trace

    def _setup_efel(self):
        """Set up efel before extracting the feature"""

        import efel

        efel.reset()

        if self.threshold is not None:
            efel.setThreshold(self.threshold)

        if self.stimulus_current is not None:
            if callable(self.stimulus_current):
                efel.setDoubleSetting("stimulus_current", self.stimulus_current())
            else:
                efel.setDoubleSetting("stimulus_current", self.stimulus_current)

        if self.interp_step is not None:
            efel.setDoubleSetting("interp_step", self.interp_step)

        if self.double_settings is not None:
            for setting_name, setting_value in self.double_settings.items():
                efel.setDoubleSetting(setting_name, setting_value)

        if self.int_settings is not None:
            for setting_name, setting_value in self.int_settings.items():
                efel.setIntSetting(setting_name, setting_value)

        if self.string_settings is not None:
            for setting_name, setting_value in self.string_settings.items():
                efel.setStrSetting(setting_name, setting_value)

    def calculate_feature(self, responses, raise_warnings=False):
        """Calculate feature value"""
        if self.efel_feature_name.startswith("bpo_"):
            feature_values = numpy.array([self.calculate_bpo_feature(responses)])

        else:
            efel_trace = self._construct_efel_trace(responses)

            if efel_trace is None:
                feature_values = None
            else:
                self._setup_efel()
                logger.debug("Amplitude for %s: %s", self.name, self.stimulus_current)
                import efel

                values = efel.getFeatureValues(
                    [efel_trace], [self.efel_feature_name], raise_warnings=raise_warnings
                )

                feature_values = values[0][self.efel_feature_name]

                efel.reset()

        logger.debug("Calculated values for %s: %s", self.name, str(feature_values))
        return feature_values

    def calculate_score(self, responses, trace_check=False):
        """Calculate the score"""

        if self.efel_feature_name.startswith("bpo_"):
            score = self.calculate_bpo_score(responses)

        elif self.exp_mean is None:
            score = 0

        else:
            feature_values = self.calculate_feature(responses)
            if (feature_values is None) or (len(feature_values) == 0):
                score = self.max_score
            else:
                score = (
                    numpy.sum(numpy.fabs(feature_values - self.exp_mean))
                    / self.exp_std
                    / len(feature_values)
                )
                logger.debug("Calculated score for %s: %f", self.name, score)

            score = numpy.min([score, self.max_score])

        if score is None or math.isnan(score):
            return self.max_score

        return score
