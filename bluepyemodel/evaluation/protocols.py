"""Module with protocol classes."""
import logging
import time
from collections import OrderedDict

from bluepyopt import ephys

from ..ecode import eCodes
from .recordings import LooseDtRecordingCustom
from .recordings import check_recordings

# pylint: disable=W0613

logger = logging.getLogger(__name__)


class BPEM_Protocol(ephys.protocols.SweepProtocol):

    """Protocol with stochasticity capabilities"""

    def __init__(self, name=None, stimulus=None, recordings=None, stochasticity=False):
        """Constructor

        Args:
            name (str): name of this object
            stimulus (Stimulus): stimulus objects
            recordings (list of Recordings): Recording objects used in the
                protocol
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name,
            stimuli=[stimulus],
            recordings=recordings,
        )

        self.stimulus = stimulus
        self.stochasticity = stochasticity

        self.features = []

    def instantiate(self, sim=None, icell=None):
        """Check recordings, then instantiate."""
        if not all(rec.checked for rec in self.recordings):
            self.recordings = check_recordings(self.recordings, icell, sim)

        super().instantiate(sim, icell)

    def stim_start(self):
        """Time stimulus starts"""
        return self.stimulus.stim_start

    def stim_end(self):
        return self.stimulus.stim_end

    def amplitude(self):
        return self.stimulus.amplitude

    def run(  # pylint: disable=arguments-differ, arguments-renamed
        self,
        cell_model,
        responses,
        sim=None,
        isolate=None,
        timeout=None,
    ):
        """Run protocol"""

        # Set the stochasticity
        if not self.stochasticity:
            for mechanism in cell_model.mechanisms:
                mechanism.deterministic = True

        # param_values is {} because BPEM_protocols should always be used inside of MainProtocol
        return super().run(cell_model, param_values={}, sim=sim, isolate=isolate, timeout=timeout)


class BPEM_ThresholdProtocol(BPEM_Protocol):

    """Protocol having rheobase-rescaling and stochasticity capabilities"""

    def __init__(self, name=None, stimulus=None, recordings=None, stochasticity=False, suffix=""):
        """Constructor

        Args:
            name (str): name of this object
            stimulus (Stimulus): stimulus objects
            recordings (list of Recordings): Recording objects used in the
                protocol
            stochasticity (bool): turns on or off the channels that can be
                stochastic
            suffix (str): suffix used in case they are several holding or threshold currents.
        """

        super().__init__(name, stimulus, recordings, stochasticity)

        self.suffix = suffix

    def run(self, cell_model, responses, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        holding_current = responses[f"bpo_holding_current{self.suffix}"]
        threshold_current = responses[f"bpo_threshold_current{self.suffix}"]

        if holding_current is None or threshold_current is None:
            raise Exception("StepProtocol: holding or threshold current is None")

        self.stimulus.holding_current = holding_current
        self.stimulus.threshold_current = threshold_current

        return super().run(cell_model, responses, sim, isolate, timeout)


class RMPProtocol:
    """Protocol consisting of step of amplitude zero"""

    def __init__(self, name, location, target_voltage, stimulus_duration=500.0):
        """Constructor

        Args:
            name (str): name of this object
        """
        self.name = name
        self.location = location
        self.recording_name = f"{self.name}.{self.location.name}.v"

        self.stimulus_duration = stimulus_duration
        self.target_voltage = target_voltage

        self.target_voltage.stim_start = self.stimulus_duration - 100.0
        self.target_voltage.stim_end = self.stimulus_duration
        self.target_voltage.stimulus_current = 0.0

    def create_protocol(self):
        """Create a one-time use protocol"""
        stimulus_definition = {
            "delay": 0,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": self.stimulus_duration,
            "totduration": self.stimulus_duration,
            "holding_current": 0.0,
        }
        stimulus = eCodes["step"](location=self.location, **stimulus_definition)

        recordings = [
            LooseDtRecordingCustom(
                name=self.recording_name,
                location=self.location,
                variable="v",
            )
        ]
        return BPEM_Protocol(name="RMPProtocol", stimulus=stimulus, recordings=recordings)

    def run(self, cell_model, responses, sim, isolate, timeout):
        """Compute the RMP"""

        rmp_protocol = self.create_protocol()

        response = rmp_protocol.run(
            cell_model, responses, sim=sim, isolate=isolate, timeout=timeout
        )

        bpo_rmp = self.target_voltage.calculate_feature(response)
        response["bpo_rmp"] = bpo_rmp if bpo_rmp is None else bpo_rmp[0]

        score = self.target_voltage.calculate_score(response)

        return response, score


class RinProtocol:
    """Protocol used to find the input resistance of a model"""

    def __init__(
        self,
        name,
        location,
        target_rin,
        amp=-0.02,
        stimulus_delay=500.0,
        stimulus_duration=500.0,
        suffix=""
    ):

        self.name = name
        self.location = location
        self.recording_name = f"{self.name}.{self.location.name}.v"

        self.stimulus_delay = stimulus_delay
        self.stimulus_duration = stimulus_duration
        self.amp = amp
        self.target_rin = target_rin

        self.target_rin.stim_start = self.stimulus_delay
        self.target_rin.stim_end = self.stimulus_delay + self.stimulus_duration
        self.target_rin.stimulus_current = self.amp

        self.suffix = suffix

    def create_protocol(self, holding_current):
        """Create a one-time use protocol to compute Rin"""
        stimulus_definition = {
            "delay": self.stimulus_delay,
            "amp": self.amp,
            "thresh_perc": None,
            "duration": self.stimulus_duration,
            "totduration": self.stimulus_delay + self.stimulus_duration,
            "holding_current": holding_current,
        }
        stimulus = eCodes["step"](location=self.location, **stimulus_definition)

        recordings = [
            LooseDtRecordingCustom(
                name=self.recording_name,
                location=self.location,
                variable="v",
            )
        ]

        return BPEM_Protocol(
            name=f"RinProtocol{self.suffix}", stimulus=stimulus, recordings=recordings
        )

    def run(self, cell_model, responses, sim, isolate, timeout):
        """Compute input resistance"""

        rin_protocol = self.create_protocol(responses.get(f"bpo_holding_current{self.suffix}", 0))

        response = rin_protocol.run(
            cell_model, responses, sim=sim, isolate=isolate, timeout=timeout
        )

        rin_feature = self.target_rin.calculate_feature(response)
        score = self.target_rin.calculate_score(response)
        response[f"bpo_rin{self.suffix}"] = rin_feature[0] if rin_feature else None

        return response, score


class SearchHoldingCurrent:
    """Protocol used to find the holding current of a model"""

    def __init__(
        self,
        name,
        location,
        target_voltage=None,
        target_holding=None,
        max_depth=7,
        stimulus_duration=500.0,
        upper_bound=0.2,
        lower_bound=-0.5,
        strict_bounds=True,
        suffix=""
    ):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_voltage (EFeature): target for the voltage at holding_current
            target_holding (EFeature): target for the holding_current
            max_depth (int): maxium depth for the bisection search
            stimulus_duration (float): length of the protocol
            upper_bound (float): upper bound for the holding current, in pA
            lower_bound (float): lower bound for the holding current, in pA
            strict_bounds (bool): to adaptively enlarge bounds is current is outside
            suffix (str):
        """

        self.name = name
        self.location = location

        self.target_voltage = target_voltage
        self.target_holding = target_holding

        self.max_depth = max_depth
        self.stimulus_duration = stimulus_duration
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.strict_bounds = strict_bounds

        self.target_voltage.stim_start = self.stimulus_duration - 100.0
        self.target_voltage.stim_end = self.stimulus_duration
        self.target_voltage.stimulus_current = 0.0

        self.suffix = suffix

    def create_protocol(self, holding_current):
        """Create a one-time use protocol made of a holding and step current"""
        # Create the stimuli and recording
        stimulus_definition = {
            "delay": 0,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": self.stimulus_duration,
            "totduration": self.stimulus_duration,
            "holding_current": holding_current,
        }
        stimulus = eCodes["step"](location=self.location, **stimulus_definition)

        recordings = [
            LooseDtRecordingCustom(
                name=self.target_voltage.recording_names[""],
                location=self.location,
                variable="v",
            )
        ]
        return BPEM_Protocol(
            name=f"SearchHoldingCurrent{self.suffix}",
            stimulus=stimulus,
            recordings=recordings
        )

    def get_voltage_base(self, holding_current, cell_model, responses, sim, isolate, timeout=None):
        """Calculate voltage base for a certain holding current"""
        protocol = self.create_protocol(holding_current=holding_current)
        response = protocol.run(cell_model, responses, sim=sim, isolate=isolate, timeout=timeout)

        return self.target_voltage.calculate_feature(response)

    def run(
        self,
        cell_model,
        responses,
        sim,
        isolate=None,
        timeout=None,
    ):
        """Run protocol"""
        if not self.strict_bounds:
            # first readjust the bounds if needed
            voltage_min = 1e10
            while voltage_min > self.target_voltage.exp_mean:
                voltage_min = self.get_voltage_base(
                    holding_current=self.lower_bound,
                    cell_model=cell_model,
                    responses=responses,
                    sim=sim,
                    isolate=isolate,
                    timeout=timeout,
                )
                if voltage_min > self.target_voltage.exp_mean:
                    self.lower_bound -= 0.2
                    self.max_depth += 1

            voltage_max = -1e10
            while voltage_max < self.target_voltage.exp_mean:
                voltage_max = self.get_voltage_base(
                    holding_current=self.upper_bound,
                    cell_model=cell_model,
                    responses=responses,
                    sim=sim,
                    isolate=isolate,
                    timeout=timeout,
                )
                if voltage_max < self.target_voltage.exp_mean:
                    self.upper_bound += 0.2
                    self.max_depth += 1

        response = {
            f"bpo_holding_current{self.suffix}": self.bisection_search(
                cell_model,
                responses,
                sim=sim,
                isolate=isolate,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                timeout=timeout,
            )
        }

        score = self.target_holding.calculate_score(response)

        return response, score

    def bisection_search(
        self,
        cell_model,
        responses,
        sim,
        isolate,
        upper_bound,
        lower_bound,
        depth=1,
        timeout=None,
    ):
        """Do bisection search to find holding current"""
        logger.debug(
            "Bisection search for Holding current. Depth = %s / %s",
            depth,
            self.max_depth,
        )

        mid_bound = (upper_bound + lower_bound) * 0.5
        if depth >= self.max_depth:
            return mid_bound

        voltage = self.get_voltage_base(
            holding_current=mid_bound,
            cell_model=cell_model,
            responses=responses,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )

        if voltage is None:
            return None

        if voltage > self.target_voltage.exp_mean:
            return self.bisection_search(
                cell_model,
                responses,
                sim=sim,
                isolate=isolate,
                lower_bound=lower_bound,
                upper_bound=mid_bound,
                depth=depth + 1,
            )

        return self.bisection_search(
            cell_model,
            responses,
            sim=sim,
            isolate=isolate,
            lower_bound=mid_bound,
            upper_bound=upper_bound,
            depth=depth + 1,
        )


class SearchThresholdCurrent:
    """Protocol used to find the threshold current (rheobase) of a model"""

    def __init__(
        self,
        name,
        location,
        target_threshold=None,
        max_depth=7,
        stimulus_duration=1000.0,
        max_threshold_voltage=-30,
        suffix=""
    ):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_threshold (Efeature): target for the threshold_current
            max_depth (int): maxium depth for the bisection search
            stimulus_duration (float): duration of the step used to create the
                protocol
            max_threshold_voltage (float): maximum voltage used as upper
                bound in the threshold current search
            suffix (str):
        """

        self.name = name
        self.target_threshold = target_threshold
        self.location = location

        self.max_threshold_voltage = max_threshold_voltage
        self.max_depth = max_depth
        self.stimulus_duration = stimulus_duration

        self.spike_feature = ephys.efeatures.eFELFeature(
            name="SearchThresholdCurrent.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"SearchThresholdCurrent.{self.location.name}.v"},
            stim_start=0.0,
            stim_end=self.stimulus_duration,
            exp_mean=1,
            exp_std=0.1,
        )
        self.flag_spike_detected = False

        self.suffix = suffix

    def create_protocol(self, holding_current, step_current):
        """Create a one-time use protocol made of a holding and step current"""
        # Create the stimuli and recording
        stimulus_definition = {
            "delay": 0,
            "amp": step_current,
            "thresh_perc": None,
            "duration": self.stimulus_duration,
            "totduration": self.stimulus_duration,
            "holding_current": holding_current,
        }

        stimulus = eCodes["step"](location=self.location, **stimulus_definition)

        recordings = [
            LooseDtRecordingCustom(
                name=f"SearchThresholdCurrent{self.suffix}.{self.location.name}.v",
                location=self.location,
                variable="v",
            )
        ]

        return BPEM_Protocol(
            name=f"SearchThresholdCurrent{self.suffix}",
            stimulus=stimulus,
            recordings=recordings,
        )

    def run(self, cell_model, responses, sim, isolate=None, timeout=None):
        """Run protocol"""

        # Calculate max threshold current
        print(responses)
        max_threshold_current = self.max_threshold_current(
            rin=responses[f"bpo_rin{self.suffix}"], rmp=responses["bpo_rmp"]
        )

        threshold = self.bisection_search(
            cell_model,
            responses,
            responses[f"bpo_holding_current{self.suffix}"],
            sim,
            isolate,
            upper_bound=max_threshold_current,
            lower_bound=responses[f"bpo_holding_current{self.suffix}"],
            timeout=timeout,
        )

        response = {
            f"bpo_threshold_current{self.suffix}": threshold if self.flag_spike_detected else None
        }
        score = self.target_threshold.calculate_score(response)

        return response, score

    def max_threshold_current(self, rin, rmp):
        """Find the current necessary to get to max_threshold_voltage"""
        max_threshold_current = (self.max_threshold_voltage - rmp) / rin
        logger.debug("Max threshold current: %.6g", max_threshold_current)
        return max_threshold_current

    def bisection_search(
        self,
        cell_model,
        responses,
        holding_current,
        sim,
        isolate,
        upper_bound,
        lower_bound,
        depth=1,
        timeout=None,
    ):
        """Do bisection search to find threshold current"""

        logger.debug(
            "Bisection search for Threshold current. Depth = %s / %s",
            depth,
            self.max_depth,
        )

        mid_bound = (upper_bound + lower_bound) * 0.5
        if depth >= self.max_depth:
            return mid_bound

        protocol = self.create_protocol(holding_current, mid_bound)
        response = protocol.run(cell_model, responses, sim=sim, isolate=isolate, timeout=timeout)
        spikecount = self.spike_feature.calculate_feature(response)

        if spikecount is None:
            return None

        if spikecount == 1:
            self.flag_spike_detected = True
            return mid_bound

        if spikecount == 0:
            return self.bisection_search(
                cell_model,
                responses,
                holding_current,
                sim=sim,
                isolate=isolate,
                lower_bound=mid_bound,
                upper_bound=upper_bound,
                depth=depth + 1,
                timeout=timeout,
            )

        if spikecount > 1:
            self.flag_spike_detected = True
            return self.bisection_search(
                cell_model,
                responses,
                holding_current,
                sim=sim,
                isolate=isolate,
                lower_bound=lower_bound,
                upper_bound=mid_bound,
                depth=depth + 1,
                timeout=timeout,
            )

        return None


class MainProtocol(ephys.protocols.Protocol):

    """Holding and threshold current search protocol."""

    def __init__(
        self,
        name,
        pre_protocols=None,
        protocols=None,
        score_threshold=12.0,
    ):
        """Constructor

        Args:
            name (str): name of this object
            pre_protocols (dict): pre-protocols such as RMP or ThresholdCurrent computation.
            protocols (dict): protocols.
            score_threshold (float): threshold for the pre-protocol scores above which the
                evaluation will stop.
        """

        super().__init__(name=name)

        self.name = name

        self.pre_protocols = {} if pre_protocols is None else pre_protocols
        self.protocols = {} if protocols is None else protocols

        self.score_threshold = score_threshold

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        responses = OrderedDict()
        cell_model.freeze(param_values)

        logger.debug("Computing pre-protocols ...")
        for pre_protocol in self.pre_protocols.values():

            t1 = time.time()
            response, score = pre_protocol.run(
                cell_model, responses, sim=sim, isolate=isolate, timeout=timeout
            )
            dt = time.time() - t1

            responses.update(response)

            if score is None:
                logger.debug("Score is None. Stopping MainProtocol run.")
                cell_model.unfreeze(param_values.keys())
                return responses

            logger.debug(
                "Computed %s in %s s. Value = %s, Score = %s",
                pre_protocol.name,
                dt,
                list(responses.values())[-1],
                score,
            )

            if score > self.score_threshold:
                logger.debug("Score is higher than score_threshold. Stopping MainProtocol run.")
                cell_model.unfreeze(param_values.keys())
                return responses

        logger.debug("Computing other protocols ...")
        for protocol in self.protocols.values():
            responses.update(
                protocol.run(cell_model, responses, sim=sim, isolate=isolate, timeout=timeout)
            )

        cell_model.unfreeze(param_values.keys())
        return responses
