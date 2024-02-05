"""Class eFELFeatureBPEM"""

"""
Copyright 2023-2024, EPFL/Blue Brain Project

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
from scipy import optimize as opt

from bluepyemodel.tools.multiprotocols_efeatures_utils import get_distances_from_recording_name
from bluepyemodel.tools.multiprotocols_efeatures_utils import get_locations_from_recording_name
from bluepyemodel.tools.multiprotocols_efeatures_utils import get_protocol_list_from_recording_name

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
    # pylint: disable=too-many-arguments

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
        weight=1.0,
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
            weight (float): weight of the efeature.
                Basically multiplies the score of the efeature by this value.
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
        self.weight = weight  # used in objective

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

    def calulate_score_(self, responses):
        """Calculate the score for non-bpo feature"""
        if self.exp_mean is None:
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
        return score

    def calculate_score(self, responses, trace_check=False):
        """Calculate the score"""

        if self.efel_feature_name.startswith("bpo_"):
            score = self.calculate_bpo_score(responses)

        else:
            score = self.calulate_score_(responses)

        if score is None or math.isnan(score):
            return self.max_score

        return score


class DendFitFeature(eFELFeatureBPEM):
    """Fit to back propagation feature

    To use this class:
        - have "dendrite_backpropagation_fit" as the efeature name
        - have "maximum_voltage_from_voltagebase" as the efel_feature_name
        - have keys in recording names matching the distance from soma, and "" for soma, e.g.
            {"": "soma.v", "50": "dend50.v", "100": "dend100.v", "150": "dend150.v"}
        - have appropriate recordings in protocols
    """

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
        decay=None,
        linear=None,
        weight=1.0,
    ):
        """Constructor"""
        # pylint: disable=too-many-arguments
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
            weight,
        )
        self.decay = decay
        self.linear = linear
        self.ymult = None  # set on the fly before the fitting

    @property
    def recording_names_list(self):
        return self.recording_names.values()

    @property
    def distances(self):
        # expects keys in recordings names to be distances from soma (e.g. "50") or "" if at soma
        return [int(rec_name) if rec_name != "" else 0 for rec_name in self.recording_names.keys()]

    @property
    def locations(self):
        # locations used in bpo features (holding current, threshold current)
        # as such, they are implemented for MultiProtocols, but not for simple protocol
        raise NotImplementedError

    def _construct_efel_trace(self, responses):
        """Construct trace that can be passed to eFEL"""

        traces = []
        if "" not in self.recording_names:
            raise ValueError("eFELFeature: '' needs to be in recording_names")

        for recording_name in self.recording_names_list:
            if recording_name not in responses:
                logger.debug(
                    "Recording named %s not found in responses %s", recording_name, str(responses)
                )
                return None

            if responses[recording_name] is None:
                logger.debug("resp of %s is None", responses[recording_name])
                return None

            trace = {}
            trace["T"] = responses[recording_name]["time"]
            trace["V"] = responses[recording_name]["voltage"]
            if callable(self.stim_start):
                trace["stim_start"] = [self.stim_start()]
            else:
                trace["stim_start"] = [self.stim_start]

            if callable(self.stim_end):
                trace["stim_end"] = [self.stim_end()]
            else:
                trace["stim_end"] = [self.stim_end]
            traces.append(trace)

        return traces

    def exp_decay(self, x, p):
        return numpy.exp(-x / p) * self.ymult

    def exp(self, x, p):
        return numpy.exp(x / p) * self.ymult

    def linear_fit(self, x, p):
        return self.ymult + p * x

    def fit(self, distances, values):
        """Fit back propagation"""
        guess = [50]
        self.ymult = values[distances.index(0)]
        if self.linear:
            params, _ = opt.curve_fit(self.linear_fit, distances, values, p0=guess)
        elif self.decay:
            params, _ = opt.curve_fit(self.exp_decay, distances, values, p0=guess)
        else:
            params, _ = opt.curve_fit(self.exp, distances, values, p0=guess)

        return params[0]

    def get_distances_feature_values(self, responses, raise_warnings=False):
        """Compute feature at each distance, and return distances and feature values."""
        distances = []
        feature_values_ = []
        if self.efel_feature_name.startswith("bpo_"):
            feature_names = [
                f"{self.efel_feature_name}_{loc}" if loc != "soma" else self.efel_feature_name
                for loc in self.locations
            ]
            feature_values_ = [
                responses[fname]
                for fname in feature_names
                if fname in responses and responses[fname] is not None
            ]
            distances = [
                d
                for d, fname in zip(self.distances, feature_names)
                if fname in responses and responses[fname] is not None
            ]

            # adjust soma value with holding current
            if (
                self.efel_feature_name == "bpo_threshold_current"
                and distances[0] == 0
                and "bpo_holding_current" in responses
                and responses["bpo_holding_current"] is not None
            ):
                feature_values_[0] += responses["bpo_holding_current"]

        else:
            efel_traces = self._construct_efel_trace(responses)

            if efel_traces is None:
                feature_values_ = None
            else:
                self._setup_efel()
                import efel

                values = efel.getFeatureValues(
                    efel_traces, [self.efel_feature_name], raise_warnings=raise_warnings
                )
                feature_values_ = [
                    val[self.efel_feature_name][0]
                    for val in values
                    if val[self.efel_feature_name] is not None
                ]
                distances = [
                    d
                    for d, v in zip(self.distances, values)
                    if v[self.efel_feature_name] is not None
                ]

                efel.reset()

        return distances, feature_values_

    def calculate_feature(self, responses, raise_warnings=False):
        """Calculate feature value"""
        distances, feature_values_ = self.get_distances_feature_values(responses, raise_warnings)

        if distances and feature_values_:
            if 0 in distances:
                feature_values = numpy.array([self.fit(distances, feature_values_)])
            else:
                feature_values = None
        else:
            feature_values = None

        logger.debug("Calculated values for %s: %s", self.name, str(feature_values))
        return feature_values

    def calculate_score(self, responses, trace_check=False):
        """Calculate score. bpo and non-bpo feature should use calulate_feature"""
        score = self.calulate_score_(responses)

        if score is None or math.isnan(score):
            return self.max_score

        return score


class DendFitMultiProtocolsFeature(DendFitFeature):
    """Fit across apical dendrite using multiple protocols.

    Attention! Since this feature depends on multiple protocols,
    the stimulus_current passed can be wrong for some of them, and in such a case,
    efel features depending on stimulus_current should not be used.

    To use this class:
        - have a protocol_name with distances in brackets in it
            e.g. LocalInjectionIDrestapic[050,100,150,200]_100
        - same with recording_name
            e.g. apical[055,080,110,200,340].v
        - have corresponding protocols with normal names under "protocols" in config
            e.g. LocalInjectionIDrestapic050_100, LocalInjectionIDrestapic100_100, etc.
        - have a soma protocol under "protocols" with same ecode in config
            e.g. IDrest_100
    """

    @property
    def recording_names_list(self):
        return get_protocol_list_from_recording_name(self.recording_names[""])

    @property
    def distances(self):
        return get_distances_from_recording_name(self.recording_names[""])

    @property
    def locations(self):
        return get_locations_from_recording_name(self.recording_names[""])
