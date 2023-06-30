"""Class eFELFeatureBPEM"""
import logging
import math

import numpy
from scipy import optimize as opt
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
                print(f"Recording named {recording_name} not found in responses {str(responses)}")
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
                print(f"efel trace is None for {self.name}")
                feature_values = None
            else:
                self._setup_efel()
                logger.debug("Amplitude for %s: %s", self.name, self.stimulus_current)
                import efel
                print(f"computing feature for {self.name}")
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


class DendFitFeature(eFELFeatureBPEM):

    """Fit to back propagation feature
    
    To use this class:
        - have "dendrite_backpropagation_fit" as the efeature name
        - have "maximum_voltage_from_voltagebase" as the efel_feature_name
        - have keys in recording names matching the distance from soma, and "" for soma, e.g.
            {"": "soma.v", "50": "dend50.v", "100": "dend100.v", "150": "dend150.v", "200": "dend200.v"}
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
    ):
        """Constructor"""
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
        )
        self.decay = decay


    def _construct_efel_trace(self, responses):
        """Construct trace that can be passed to eFEL"""

        traces = []
        if "" not in self.recording_names:
            raise ValueError("eFELFeature: '' needs to be in recording_names")

        for recording_name in self.recording_names.values():
            if recording_name not in responses:
                logger.debug(
                    "Recording named %s not found in responses %s", recording_name, str(responses)
                )
                logger.warning(f"{recording_name} not found in responses") # to remove before merging
                return None

            if responses[self.recording_names[""]] is None or responses[recording_name] is None:
                logger.warning("responses is None") # to ermove before merging
                return None
            
            trace = {}
            trace["T"] = responses[self.recording_names[""]]["time"]
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

    def fit(self, distances, values):
        """Fit back propagation"""
        guess = [50]
        self.ymult = values[distances.index(0)]
        if self.decay:
            params, _ = opt.minpack.curve_fit(
                self.exp_decay, distances, values, p0=guess
            )
        else:
            params, _ = opt.minpack.curve_fit(
                self.exp, distances, values, p0=guess
            )

        return params[0]

    def calculate_feature(self, responses, raise_warnings=False):
        """Calculate feature value"""
        if self.efel_feature_name.startswith("bpo_"):
            feature_values = numpy.array([self.calculate_bpo_feature(responses)])

        else:
            efel_traces = self._construct_efel_trace(responses)

            if efel_traces is None:
                logger.warning("efel traces is None in dendritic feature") # remove this before merging
                feature_values = None
            else:
                self._setup_efel()
                logger.debug("Amplitude for %s: %s", self.name, self.stimulus_current)
                import efel

                values = efel.getFeatureValues(
                    efel_traces, [self.efel_feature_name], raise_warnings=raise_warnings
                )

                feature_values_ = [val[self.efel_feature_name][0] for val in values]
                # expects keys in recordings names to be distances from soma (e.g. "50") or "" if at soma
                distances = [int(rec_name) if rec_name != "" else 0 for rec_name in self.recording_names.keys()]
                logger.warning(self.name)
                logger.warning(f"distances: {distances}")
                logger.warning(f"values: {feature_values_}")

                feature_values = numpy.array([self.fit(distances, feature_values_)])
                logger.warning(f"feature: {feature_values}")

                efel.reset()

        logger.debug("Calculated values for %s: %s", self.name, str(feature_values))
        return feature_values
