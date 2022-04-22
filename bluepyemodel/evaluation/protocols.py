"""Module with protocol classes."""
import logging
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
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""

        if param_values is None:
            param_values = {}

        # Set the stochasticity
        if not self.stochasticity:
            for mechanism in cell_model.mechanisms:
                mechanism.deterministic = True

        return super().run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)


class BPEM_ThresholdProtocol(BPEM_Protocol):

    """Protocol having rheobase-rescaling and stochasticity capabilities"""

    def return_none_responses(self):
        return {k.name: None for k in self.recordings}

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""

        param_values = {} if param_values is None else param_values
        responses = {} if responses is None else responses

        if (
            responses.get("bpo_holding_current", None) is None
            or responses.get("bpo_threshold_current", None) is None
        ):
            logger.warning("BPEM_ThresholdProtocol: holding or threshold current is None")
            return self.return_none_responses()

        self.stimulus.holding_current = responses["bpo_holding_current"]
        self.stimulus.threshold_current = responses["bpo_threshold_current"]

        return super().run(cell_model, param_values, sim, isolate, timeout)


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

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Compute the RMP"""

        rmp_protocol = self.create_protocol()

        response = rmp_protocol.run(
            cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )

        bpo_rmp = self.target_voltage.calculate_feature(response)
        response["bpo_rmp"] = bpo_rmp if bpo_rmp is None else bpo_rmp[0]

        return response


class RinProtocol:
    """Protocol used to find the input resistance of a model"""

    def __init__(
        self, name, location, target_rin, amp=-0.02, stimulus_delay=500.0, stimulus_duration=500.0
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

        return BPEM_Protocol(name="RinProtocol", stimulus=stimulus, recordings=recordings)

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Compute input resistance"""

        holding_current = responses.get("bpo_holding_current", 0)

        if holding_current is None:
            logger.warning("RinProtocol: holding current is None")
            return {self.recording_name: None, "bpo_rin": None}

        rin_protocol = self.create_protocol(holding_current)

        response = rin_protocol.run(
            cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )
        response["bpo_rin"] = self.target_rin.calculate_feature(response)[0]

        return response


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
        return BPEM_Protocol(name="SearchHoldingCurrent", stimulus=stimulus, recordings=recordings)

    def get_voltage_base(
        self, holding_current, cell_model, param_values, sim, isolate, timeout=None
    ):
        """Calculate voltage base for a certain holding current"""
        protocol = self.create_protocol(holding_current=holding_current)
        response = protocol.run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)

        return self.target_voltage.calculate_feature(response)

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Run protocol"""

        if not self.strict_bounds:
            # first readjust the bounds if needed
            voltage_min = 1e10
            while voltage_min > self.target_voltage.exp_mean:
                voltage_min = self.get_voltage_base(
                    holding_current=self.lower_bound,
                    cell_model=cell_model,
                    param_values=param_values,
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
                    param_values=param_values,
                    sim=sim,
                    isolate=isolate,
                    timeout=timeout,
                )
                if voltage_max < self.target_voltage.exp_mean:
                    self.upper_bound += 0.2
                    self.max_depth += 1

        response = {
            "bpo_holding_current": self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                timeout=timeout,
            )
        }

        if response["bpo_holding_current"] is None:
            return response

        protocol = self.create_protocol(holding_current=response["bpo_holding_current"])
        response.update(
            protocol.run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)
        )

        return response

    def bisection_search(
        self,
        cell_model,
        param_values,
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
            param_values=param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )

        if voltage is None:
            return None

        if voltage > self.target_voltage.exp_mean:
            return self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                lower_bound=lower_bound,
                upper_bound=mid_bound,
                depth=depth + 1,
            )

        return self.bisection_search(
            cell_model,
            param_values,
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
                name=f"SearchThresholdCurrent.{self.location.name}.v",
                location=self.location,
                variable="v",
            )
        ]

        return BPEM_Protocol(
            name="SearchThresholdCurrent",
            stimulus=stimulus,
            recordings=recordings,
        )

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Run protocol"""

        holding_current = responses.get("bpo_holding_current", None)
        rin = responses.get("bpo_rin", None)
        rmp = responses.get("bpo_rmp", None)

        if holding_current is None or rin is None or rmp is None:
            logger.warning("SearchThresholdCurrent: rmp, rin or holding current is None")
            return {
                "bpo_threshold_current": None,
                f"SearchThresholdCurrent.{self.location.name}.v": None,
            }

        # Calculate max threshold current
        max_threshold_current = self.max_threshold_current(rin=rin, rmp=rmp)
        threshold = self.bisection_search(
            cell_model,
            param_values,
            holding_current,
            sim,
            isolate,
            upper_bound=max_threshold_current,
            lower_bound=holding_current,
            timeout=timeout,
        )

        response = {"bpo_threshold_current": threshold if self.flag_spike_detected else None}

        if response["bpo_threshold_current"] is None:
            return response

        protocol = self.create_protocol(holding_current, response["bpo_threshold_current"])
        response.update(
            protocol.run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)
        )

        return response

    def max_threshold_current(self, rin, rmp):
        """Find the current necessary to get to max_threshold_voltage"""
        max_threshold_current = (self.max_threshold_voltage - rmp) / rin
        logger.debug("Max threshold current: %.6g", max_threshold_current)
        return max_threshold_current

    def bisection_search(
        self,
        cell_model,
        param_values,
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
        response = protocol.run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)
        spikecount = self.spike_feature.calculate_feature(response)

        if spikecount is None:
            return None

        if spikecount == 1:
            self.flag_spike_detected = True
            return mid_bound

        if spikecount == 0:
            return self.bisection_search(
                cell_model,
                param_values,
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
                param_values,
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
    """Umbrella protocol that can run normal protocols but also Threshold protocols whose holding
    current and step current can be computed on the go.

    Pseudo code:
        Run the none-threshold based protocols
        Find resting membrane potential
        Find input resistance
        Find holding current
        Find threshold current (lowest current inducing an AP)
        Run the other protocols
    """

    def __init__(
        self,
        name="Main",
        pre_protocols=None,
        threshold_protocols=None,
        other_protocols=None,
    ):
        """Constructor

        Args:
            name (str): name of this object
            pre_protocols (dict) special protocols used to compute the RMP, Rin, holding
                and threshold(rheobase current), to be ran before the threshold protocols
            threshold_protocols (dict): protocols that will use the automatic
                computation of the RMP, holding_current and threshold_current
            other_protocols (dict): additional regular protocols
        """

        super().__init__(name=name)

        self.name = name

        self.pre_protocols = pre_protocols if pre_protocols else {}
        self.threshold_protocols = threshold_protocols if threshold_protocols else {}
        self.other_protocols = other_protocols if other_protocols else {}

    def all_subprotocols(self):
        """Return all the protocols contained in MainProtocol"""
        return {**self.pre_protocols, **self.threshold_protocols, **self.other_protocols}

    def subprotocols(self):
        """Return all the protocols contained in MainProtocol except the pre_protocols"""
        return {**self.threshold_protocols, **self.other_protocols}

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""
        responses = OrderedDict()
        cell_model.freeze(param_values)

        for protocol in self.all_subprotocols().values():
            logger.debug("Computing protocol %s", protocol.name)
            responses.update(protocol.run(cell_model, {}, sim, isolate, timeout, responses))

        cell_model.unfreeze(param_values.keys())
        return responses
