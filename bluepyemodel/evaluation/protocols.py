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

        super().__init__(name=name, stimuli=[stimulus], recordings=recordings)

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

        param_values = {} if param_values is None else param_values
        responses = {} if responses is None else responses

        # Set the stochasticity
        if not self.stochasticity:
            for mechanism in cell_model.mechanisms:
                mechanism.deterministic = True

        return super().run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)


class ResponseDependencies:

    """To add to a protocol to specify that it depends on the responses of other protocols"""

    def __init__(self, dependencies):
        """Constructor

        Args:
            dependencies (dict): of the form: {"response_name": "self attribute name"}
        """
        self.dependencies = dependencies

    def return_none_responses(self):
        # pylint: disable=inconsistent-return-statements
        raise NotImplementedError()

    def set_attribute(self, attribute, value):
        """Set an attribute of the class based on the name of the attribute. Also handles
        the case where the name is of the form: attribute.sub_attribute"""

        if "." in attribute:
            obj2 = getattr(self, attribute.split(".")[0])
            setattr(obj2, attribute.split(".")[1], value)
        else:
            setattr(self, attribute, value)

    def set_dependencies(self, responses=None):
        for dep in self.dependencies:
            if responses.get(dep, None) is None:
                logger.debug("Dependency %s missing", dep)
                return False
            self.set_attribute(self.dependencies[dep], responses[dep])
        return True


class BPEM_ThresholdProtocol(BPEM_Protocol, ResponseDependencies):

    """Protocol having rheobase-rescaling and stochasticity capabilities"""

    def __init__(self, name=None, stimulus=None, recordings=None, stochasticity=False):

        BPEM_Protocol.__init__(self, name, stimulus, recordings, stochasticity)
        ResponseDependencies.__init__(
            self,
            dependencies={
                "bpo_holding_current": "stimulus.holding_current",
                "bpo_threshold_current": "stimulus.threshold_current",
            },
        )

    def return_none_responses(self):
        return {k.name: None for k in self.recordings}

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):

        if not self.set_dependencies(responses):
            logger.debug("Not computing %s, some response dependencies are missing", self.name)
            return self.return_none_responses()

        return super().run(cell_model, param_values, sim, isolate, timeout, responses)


class BPEM_DynamicStepProtocol(BPEM_Protocol, ResponseDependencies):

    """Protocol whose holding and step currents are based on 2 specific pre-protocols"""

    def __init__(self, name=None, stimulus=None, recordings=None, stochasticity=False):

        BPEM_Protocol.__init__(self, name, stimulus, recordings, stochasticity)

        suffix = "_noburst" if "noburst" in name else "_burst"

        ResponseDependencies.__init__( 
            self,
            dependencies={
                f"TRNSearchHolding_current{suffix}": "stimulus.holding_current",
                f"TRNSearchCurrentStep_current{suffix}": "stimulus_amplitude",
            },
        )

        self._stimulus_amplitude = None

    @property
    def stimulus_amplitude(self):
        return self._stimulus_amplitude

    @stimulus_amplitude.setter
    def stimulus_amplitude(self, v):
        """In step eCodes, the step amplitue is defined as holding_current + amplitude. But when
        searching for the step current in the pre-protocols, we did not use a holding_current,
        therefore, it need to be substracted"""
        self._stimulus_amplitude = v - self.stimulus.holding_current
        self.stimulus.amp = v - self.stimulus.holding_current

    def return_none_responses(self):
        return {k.name: None for k in self.recordings}

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):

        if not self.set_dependencies(responses):
            logger.debug("Not computing %s, some response dependencies are missing", self.name)
            return self.return_none_responses()

        return super().run(cell_model, param_values, sim, isolate, timeout, responses)


class PreProtocol:

    """Abstract pre-protocol class. Pre-protocols are ran before the other protocols that
    might have them as dependencies. Note that pre-protocol can also have dependencies,
    therefore they are ran in a specific order: RMP -> SearchHolding -> Rin -> SearchThreshold."""

    def __init__(self, name, location, recording_name):

        self.name = name
        self.location = location
        self.recording_name = recording_name

    def create_one_use_step(
        self, delay=0.0, holding_current=0.0, amp=0.0, duration=100.0, totduration=100.0
    ):

        stimulus_definition = {
            "delay": delay,
            "amp": amp,
            "thresh_perc": None,
            "duration": duration,
            "totduration": totduration,
            "holding_current": holding_current,
        }

        stimulus = eCodes["step"](location=self.location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=self.location, variable="v")
        ]

        return BPEM_Protocol(name=self.name, stimulus=stimulus, recordings=recordings)


class RMPProtocol(PreProtocol):
    """Protocol consisting of step of amplitude zero"""

    def __init__(self, name, location, target_voltage, stimulus_duration=500.0):
        """Constructor

        Args:
            name (str): name of this object
        """

        super().__init__(name, location, f"{name}.{location.name}.v")

        self.stimulus_duration = stimulus_duration
        self.target_voltage = target_voltage

        self.target_voltage.stim_start = self.stimulus_duration - 100.0
        self.target_voltage.stim_end = self.stimulus_duration
        self.target_voltage.stimulus_current = 0.0

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Compute the RMP"""

        rmp_protocol = self.create_one_use_step(
            duration=self.stimulus_duration, totduration=self.stimulus_duration
        )

        response = rmp_protocol.run(
            cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )

        bpo_rmp = self.target_voltage.calculate_feature(response)
        response["bpo_rmp"] = bpo_rmp if bpo_rmp is None else bpo_rmp[0]

        return response


class RinProtocol(PreProtocol, ResponseDependencies):
    """Protocol used to find the input resistance of a model"""

    def __init__(
        self, name, location, target_rin, amp=-0.02, stimulus_delay=500.0, stimulus_duration=500.0
    ):

        PreProtocol.__init__(
            self, name=name, location=location, recording_name=f"{name}.{location.name}.v"
        )
        ResponseDependencies.__init__(self, dependencies={"bpo_holding_current": "holding_current"})

        self.stimulus_delay = stimulus_delay
        self.stimulus_duration = stimulus_duration
        self.amp = amp
        self.target_rin = target_rin

        self.target_rin.stim_start = self.stimulus_delay
        self.target_rin.stim_end = self.stimulus_delay + self.stimulus_duration
        self.target_rin.stimulus_current = self.amp

        self.holding_current = None

    def return_none_responses(self):
        return {self.recording_name: None, "bpo_rin": None}

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Compute input resistance"""

        if not self.set_dependencies(responses):
            logger.debug("Not computing %s, some response dependencies are missing", self.name)
            return self.return_none_responses()

        rin_protocol = self.create_one_use_step(
            amp=self.amp,
            delay=self.stimulus_delay,
            duration=self.stimulus_duration,
            totduration=self.stimulus_delay + self.stimulus_duration,
            holding_current=self.holding_current,
        )

        response = rin_protocol.run(
            cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )
        response["bpo_rin"] = self.target_rin.calculate_feature(response)[0]

        return response


class SearchCurrentForVoltage(PreProtocol):
    """Protocol used to find the holding current of a model"""

    def __init__(
        self,
        name,
        location,
        target_voltage=None,
        max_depth=7,
        stimulus_duration=1000.0,
        upper_bound=0.2,
        lower_bound=-0.3,
        strict_bounds=True,
        target_current_name="bpo_holding_current",
    ):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_voltage (EFeature): target for the voltage at holding_current
            max_depth (int): maxium depth for the bisection search
            stimulus_duration (float): length of the protocol
            upper_bound (float): upper bound for the holding current, in pA
            lower_bound (float): lower bound for the holding current, in pA
            strict_bounds (bool): to adaptively enlarge bounds is current is outside
        """

        super().__init__(
            name=name, location=location, recording_name=target_voltage.recording_names[""]
        )

        self.target_voltage = target_voltage

        self.max_depth = max_depth
        self.stimulus_duration = stimulus_duration
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.strict_bounds = strict_bounds

        self.target_voltage.stim_start = self.stimulus_duration - 100.0
        self.target_voltage.stim_end = self.stimulus_duration
        self.target_voltage.stimulus_current = 0.0

        self.target_current_name = target_current_name

    def get_voltage_base(
        self, holding_current, cell_model, param_values, sim, isolate, timeout=None
    ):
        """Calculate voltage base for a certain holding current"""

        protocol = self.create_one_use_step(
            duration=self.stimulus_duration,
            totduration=self.stimulus_duration,
            holding_current=holding_current,
        )

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
            self.target_current_name: self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                timeout=timeout,
            )
        }

        if response[self.target_current_name] is None:
            return response

        protocol = self.create_one_use_step(
            duration=self.stimulus_duration,
            totduration=self.stimulus_duration,
            holding_current=response[self.target_current_name],
        )

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
            "Bisection search for %s. Depth = %s / %s",
            self.target_current_name,
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


class SearchThresholdCurrent(PreProtocol, ResponseDependencies):
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

        PreProtocol.__init__(self, name, location, f"{name}.{location.name}.v")
        ResponseDependencies.__init__(
            self,
            dependencies={
                "bpo_holding_current": "holding_current",
                "bpo_rin": "rin",
                "bpo_rmp": "rmp",
            },
        )

        self.target_threshold = target_threshold
        self.max_threshold_voltage = max_threshold_voltage
        self.max_depth = max_depth
        self.stimulus_duration = stimulus_duration

        self.spike_feature = ephys.efeatures.eFELFeature(
            name="f{name}.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"{self.name}.{self.location.name}.v"},
            stim_start=0.0,
            stim_end=self.stimulus_duration,
            exp_mean=1,
            exp_std=0.1,
        )
        self.flag_spike_detected = False

        self.holding_current = None
        self.rin = None
        self.rmp = None

    def return_none_responses(self):
        return {
            "bpo_threshold_current": None,
            f"{self.name}.{self.location.name}.v": None,
        }

    def run(self, cell_model, param_values, sim, isolate=None, timeout=None, responses=None):
        """Run protocol"""

        if not self.set_dependencies(responses):
            logger.debug("Not computing %s, some response dependencies are missing", self.name)
            return self.return_none_responses()

        max_threshold_current = self.max_threshold_current()
        threshold = self.bisection_search(
            cell_model,
            param_values,
            sim,
            isolate,
            upper_bound=max_threshold_current,
            lower_bound=self.holding_current,
            timeout=timeout,
        )

        response = {"bpo_threshold_current": threshold if self.flag_spike_detected else None}

        if response["bpo_threshold_current"] is None:
            return response

        protocol = self.create_one_use_step(
            amp=response["bpo_threshold_current"],
            duration=self.stimulus_duration,
            totduration=self.stimulus_duration,
            holding_current=self.holding_current,
        )

        response.update(
            protocol.run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)
        )

        return response

    def max_threshold_current(self):
        """Find the current necessary to get to max_threshold_voltage"""
        max_threshold_current = (self.max_threshold_voltage - self.rmp) / self.rin
        logger.debug("Max threshold current: %.6g", max_threshold_current)
        return max_threshold_current

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
        """Do bisection search to find threshold current"""
        logger.debug(
            "Bisection search for Threshold current. Depth = %s / %s",
            depth,
            self.max_depth,
        )

        mid_bound = (upper_bound + lower_bound) * 0.5
        if depth >= self.max_depth:
            return mid_bound

        protocol = self.create_one_use_step(
            amp=mid_bound,
            duration=self.stimulus_duration,
            totduration=self.stimulus_duration,
            holding_current=self.holding_current,
        )

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
        dynamic_protocols=None,
        other_protocols=None,
    ):
        """Constructor

        Args:
            name (str): name of this object
            pre_protocols (dict) special protocols used to compute the RMP, Rin, holding
                and threshold(rheobase current), to be ran before the threshold protocols
            dynamic_protocols (dict): protocols whose amplitude or holding depends on
                the responses from the pre_protocols
            other_protocols (dict): additional regular protocols
        """
        super().__init__(name=name)

        self.pre_protocols = pre_protocols if pre_protocols else {}
        self.dynamic_protocols = dynamic_protocols if dynamic_protocols else {}
        self.other_protocols = other_protocols if other_protocols else {}

    def all_subprotocols(self):
        """Return all the protocols contained in MainProtocol"""
        return {**self.pre_protocols, **self.dynamic_protocols, **self.other_protocols}

    def subprotocols(self):
        """Return all the protocols contained in MainProtocol except the pre_protocols"""
        return {**self.dynamic_protocols, **self.other_protocols}

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""
        responses = OrderedDict()
        cell_model.freeze(param_values)

        for protocol in self.all_subprotocols().values():
            logger.debug("Computing protocol %s", protocol.name)
            responses.update(protocol.run(cell_model, {}, sim, isolate, timeout, responses))

        cell_model.unfreeze(param_values.keys())
        return responses
