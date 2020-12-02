"""Module with protocoal classes."""
from collections import OrderedDict
import logging
import time

import bluepyopt.ephys as ephys

from ..ecode import eCodes

# pylint: disable=W0613

logger = logging.getLogger(__name__)

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


class StepProtocol(ephys.protocols.SweepProtocol):
    """Protocol consisting of step and holding current with an additional
    stochasticity option"""

    def __init__(
        self,
        name=None,
        stimulus=None,
        recordings=None,
        stochasticity=True,
    ):
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

    @property
    def stim_start(self):
        """Time stimulus starts"""
        return self.stimulus.step_delay

    @property
    def stim_end(self):
        return self.stimulus.step_delay + self.stimulus.step_duration

    @property
    def step_amplitude(self):
        return self.stimulus.step_amplitude

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""
        if self.stochasticity:
            for mechanism in cell_model.mechanisms:
                if not mechanism.deterministic:
                    if "Stoch" not in mechanism.suffix:
                        logger.warning(
                            """You are trying to set a mechanism to stochastic mode
                                       without 'Stoch' in the mechanism prefix, this may not work
                                       with current version of BluePyOpt."""
                        )

                    self.cvode_active = False
        else:
            for mechanism in cell_model.mechanisms:
                if not mechanism.deterministic:
                    mechanism.deterministic = True

        return super().run(cell_model, param_values, sim=sim, timeout=timeout)


class StepThresholdProtocol(StepProtocol):
    """Protocol consisting of step and holding current with an additional
    stochasticity option. The step amplitude and holding current are at first
    undefined and copied from the cell model before evaluation"""

    def __init__(
        self,
        name,
        thresh_perc=None,
        stimulus=None,
        recordings=None,
        stochasticity=True,
    ):
        """Constructor

        Args:
            name (str): name of this object
            thresh_perc (float): amplitude in percentage of the rheobase of
                the step current
            stimulus (Stimulus): Stimulus object
            recordings (list of Recordings): Recording objects used in the
                protocol
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name,
            stimulus=stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )

        self.thresh_perc = thresh_perc

    def run(  # pylint: disable=arguments-differ
        self,
        cell_model,
        param_values,
        holding_current=None,
        threshold_current=None,
        sim=None,
        isolate=None,
        timeout=None,
    ):
        """Run protocol"""
        if holding_current is None or threshold_current is None:
            raise Exception("StepThresholdProtocol: missing holding or threshold current " "value")

        self.stimulus.step_amplitude = threshold_current * (float(self.thresh_perc) / 100.0)
        self.stimulus.holding_current = holding_current

        return super().run(cell_model, param_values, sim=sim, timeout=timeout)


class RMPProtocol:
    """Protocol consisting of step of amplitude zero"""

    def __init__(self, name, location, target_voltage, stimulus_duration=500.0):
        """Constructor

        Args:
            name (str): name of this object
        """
        self.name = name
        self.location = location
        self.recording_name = "%s.%s.%s" % (self.name, self.location.name, "v")

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
            ephys.recordings.CompRecording(
                name=self.recording_name,
                location=self.location,
                variable="v",
            )
        ]
        return StepProtocol(name="RMPProtocol", stimulus=stimulus, recordings=recordings)

    def run(self, cell_model, param_values, sim, isolate, timeout):
        """Compute the RMP"""
        rmp_protocol = self.create_protocol()

        response = rmp_protocol.run(
            cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )
        response["bpo_rmp"] = self.target_voltage.calculate_feature(response)

        return response


class RinProtocol:
    """Protocol used to find the input resistance of a model"""

    def __init__(
        self, name, location, target_rin, amp=-0.02, stimulus_delay=500.0, stimulus_duration=500.0
    ):

        self.name = name
        self.location = location
        self.recording_name = "%s.%s.%s" % (self.name, self.location.name, "v")

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
            ephys.recordings.CompRecording(
                name=self.recording_name,
                location=self.location,
                variable="v",
            )
        ]

        return StepProtocol(name="RinProtocol", stimulus=stimulus, recordings=recordings)

    def run(self, cell_model, param_values, holding_current, sim, isolate, timeout):
        """Compute input resistance"""
        rin_protocol = self.create_protocol(holding_current)

        response = rin_protocol.run(cell_model, param_values, sim, isolate, timeout)
        response["bpo_rin"] = self.target_rin.calculate_feature(response)

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
        """
        self.name = name
        self.location = location

        self.target_voltage = target_voltage
        self.target_holding = target_holding

        self.max_depth = max_depth
        self.stimulus_duration = stimulus_duration
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

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
            ephys.recordings.CompRecording(
                name="%s.%s.%s" % ("SearchHoldingCurrent", self.location.name, "v"),
                location=self.location,
                variable="v",
            )
        ]
        return StepProtocol(name="SearchHoldingCurrent", stimulus=stimulus, recordings=recordings)

    def get_voltage_base(
        self, holding_current, cell_model, param_values, sim, isolate, timeout=None
    ):
        """ Calculate voltage base for a certain holding current """
        protocol = self.create_protocol(holding_current=holding_current)
        response = protocol.run(cell_model, param_values, sim, isolate, timeout=timeout)

        return self.target_voltage.calculate_feature(response)

    def run(
        self,
        cell_model,
        param_values,
        sim,
        isolate=None,
        timeout=None,
    ):
        """Run protocol"""
        step_current = self.bisection_search(
            cell_model,
            param_values,
            sim,
            isolate,
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound,
            timeout=timeout,
        )

        return {"bpo_holding_current": step_current}

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
            recording_names={"": "SearchThresholdCurrent.{}.v".format(self.location.name)},
            stim_start=0.0,
            stim_end=self.stimulus_duration,
            exp_mean=1,
            exp_std=0.1,
        )

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
            ephys.recordings.CompRecording(
                name="%s.%s.%s" % ("SearchThresholdCurrent", self.location.name, "v"),
                location=self.location,
                variable="v",
            )
        ]

        return StepProtocol(
            name="SearchThresholdCurrent",
            stimulus=stimulus,
            recordings=recordings,
        )

    def run(
        self,
        cell_model,
        param_values,
        holding_current,
        rin,
        rmp,
        sim,
        isolate=None,
        timeout=None,
    ):
        """Run protocol"""

        # Calculate max threshold current
        max_threshold_current = self.max_threshold_current(rin=rin, rmp=rmp)

        # Calculate spike threshold
        step_current = self.bisection_search(
            cell_model,
            param_values,
            holding_current,
            sim,
            isolate,
            upper_bound=max_threshold_current,
            lower_bound=holding_current,
            timeout=timeout,
        )

        return {"bpo_threshold_current": step_current}

    def max_threshold_current(self, rin, rmp):
        """Find the current necessary to get to max_threshold_voltage"""
        max_threshold_current = (self.max_threshold_voltage - rmp) / rin
        logger.debug("Max threshold current: %.6g", max_threshold_current)
        return max_threshold_current

    def is_spike(
        self,
        cell_model,
        param_values,
        holding_current,
        step_current,
        sim,
        isolate,
        timeout=None,
    ):
        """Returns True if step_current makes the model produce an AP,
        False otherwise"""
        protocol = self.create_protocol(holding_current, step_current)
        response = protocol.run(cell_model, param_values, sim, isolate, timeout)
        spikecount = self.spike_feature.calculate_feature(response)

        return spikecount

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

        spikecount = self.is_spike(
            cell_model, param_values, holding_current, mid_bound, sim, isolate, timeout
        )

        if spikecount == 1:
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
    """Holding and threshold current search protocol.

    Pseudo code:
        Find resting membrane potential
        Find input resistance
        If both of these scores are within bounds, run other protocols:
            Find holding current
            Find lowest current inducing an AP
            Run the other protocols
            Return all the responses
        Run the none-threshold based protocols
        Otherwise return return Rin and RMP protocol responses
    """

    def __init__(
        self,
        name,
        rmp_protocol,
        rin_protocol,
        search_holding_protocol,
        search_threshold_protocol,
        threshold_protocols=None,
        other_protocols=None,
        score_threshold=10.0,
    ):
        """Constructor

        Args:
            name (str): name of this object
            rmp_protocol (Protocol): protocol that will be used to
                compute the RMP
            rin_protocol (Protocol): protocol used to compute Rin
            search_holding_protocol (Protocol): protocol that will be used
                to find the holding current of the cell.
            search_threshold_protocol (Protocol): protocol that will be used
                to find the firing threshold (rheobase) of the cell.
            threshold_protocols (dict): protocols that will use the automatic
                computation of the RMP, holding_current and threshold_current
            other_protocols (dict): additional regular protocols
            score_threshold (int): threshold above which the computation of
                the RMP, holding_current and threshold_current has failed,
                if None, the computations will proceed without this check
        """

        super().__init__(name=name)

        self.name = name

        self.threshold_protocols = threshold_protocols
        self.other_protocols = other_protocols
        self.score_threshold = score_threshold

        self.RMP_protocol = rmp_protocol
        self.rin_protocol = rin_protocol
        self.search_holding_protocol = search_holding_protocol
        self.search_threshold_protocol = search_threshold_protocol

    def subprotocols(self):
        """ Return all the subprotocols contained in MainProtocol """
        subprotocols = {}

        if self.threshold_protocols is not None:
            subprotocols.update(self.threshold_protocols)

        if self.other_protocols is not None:
            subprotocols.update(self.other_protocols)

        return subprotocols

    def run_RMP(self, cell_model, responses, sim=None, isolate=None, timeout=None):
        """Compute resting membrane potential """
        response = self.RMP_protocol.run(cell_model, {}, sim=sim, isolate=isolate, timeout=timeout)
        score = self.RMP_protocol.target_voltage.calculate_score(response)

        return response, score

    def run_holding(self, cell_model, responses, sim=None, isolate=None, timeout=None):
        """Compute the holding current"""
        response = self.search_holding_protocol.run(
            cell_model, {}, sim=sim, isolate=isolate, timeout=timeout
        )
        score = self.search_holding_protocol.target_holding.calculate_score(response)

        return response, score

    def run_rin(self, cell_model, responses, sim=None, isolate=None, timeout=None):
        """Compute the input resistance"""
        response = self.rin_protocol.run(
            cell_model,
            {},
            responses.get("bpo_holding_current", 0),
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )
        score = self.rin_protocol.target_rin.calculate_score(response)

        return response, score

    def run_threshold(self, cell_model, responses, sim=None, isolate=None, timeout=None):
        """Compute the current at rheobase"""
        response = self.search_threshold_protocol.run(
            cell_model,
            {},
            responses["bpo_holding_current"],
            responses["bpo_rin"],
            responses["bpo_rmp"],
            sim,
            isolate,
            timeout,
        )
        score = self.search_threshold_protocol.target_threshold.calculate_score(response)

        return response, score

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        responses = OrderedDict()
        cell_model.freeze(param_values)

        logger.debug("Computing StepProtocols ...")
        for protocol in self.other_protocols.values():
            responses.update(protocol.run(cell_model, {}, sim, isolate, timeout))

        if (
            self.RMP_protocol
            and self.rin_protocol
            and self.search_holding_protocol
            and self.search_threshold_protocol
        ):

            for pre_run in [self.run_RMP, self.run_holding, self.run_rin, self.run_threshold]:

                t1 = time.time()
                response, score = pre_run(cell_model, responses, sim, isolate, timeout)
                dt = time.time() - t1

                responses.update(response)

                if score is None:
                    logger.debug("Score is None. Stopping MainProtocol run.")
                    cell_model.unfreeze(param_values.keys())
                    return responses

                logger.debug(
                    "Computed {} in {:.2f}s. Value = {:.2f}, Score = {:.2f}".format(
                        pre_run.__name__,
                        dt,
                        list(responses.values())[-1],
                        score,
                    )
                )

                if score > self.score_threshold:
                    logger.debug("Score is higher than score_threshold. Stopping MainProtocol run.")
                    cell_model.unfreeze(param_values.keys())
                    return responses

            logger.debug("Computing StepThresholdProtocols ...")
            for protocol in self.threshold_protocols.values():
                responses.update(
                    protocol.run(
                        cell_model,
                        {},
                        holding_current=responses["bpo_holding_current"],
                        threshold_current=responses["bpo_threshold_current"],
                        sim=sim,
                        isolate=isolate,
                        timeout=timeout,
                    )
                )

        cell_model.unfreeze(param_values.keys())
        return responses
