"""Module with protocoal classes."""
import copy
import logging
import time

import numpy

import bluepyopt.ephys as ephys

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
        step_stimulus=None,
        holding_stimulus=None,
        recordings=None,
        stochasticity=True,
    ):
        """Constructor

        Args:
            name (str): name of this object
            step_stimulus (lstimuli): List of Stimulus objects used in
            recordings (list of Recordings): Recording objects used in the
                protocol
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name,
            stimuli=[step_stimulus, holding_stimulus]
            if holding_stimulus is not None
            else [step_stimulus],
            recordings=recordings,
        )

        self.step_stimulus = step_stimulus
        self.holding_stimulus = holding_stimulus
        self.stochasticity = stochasticity

        self.features = []

    @property
    def stim_start(self):
        """Time stimulus starts"""
        return self.step_stimulus.step_delay

    @property
    def stim_end(self):
        return self.step_stimulus.step_delay + self.step_stimulus.step_duration

    @property
    def step_amplitude(self):
        return self.step_stimulus.step_amplitude

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
        step_stimulus=None,
        holding_stimulus=None,
        recordings=None,
        stochasticity=True,
    ):
        """Constructor

        Args:
            name (str): name of this object
            thresh_perc (float): amplitude in percentage of the rheobase of
                the step current
            step_stimulus (lstimuli): List of Stimulus objects used in
                protocol
            recordings (list of Recordings): Recording objects used in the
                protocol
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name,
            step_stimulus=step_stimulus,
            holding_stimulus=holding_stimulus,
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

        self.step_stimulus.step_amplitude = threshold_current * (float(self.thresh_perc) / 100.0)
        self.holding_stimulus.step_amplitude = holding_current

        return super().run(cell_model, param_values, sim=sim, timeout=timeout)


class RMPProtocol(StepProtocol):

    """Protocol consisting of step of amplitude zero"""

    def __init__(
        self,
        name=None,
        step_stimulus=None,
        holding_stimulus=None,
        recordings=None,
        stochasticity=True,
        target_voltage=None,
    ):
        """Constructor

        Args:
            name (str): name of this object
            step_stimulus (lstimuli): List of Stimulus objects used in
            recordings (list of Recordings): Recording objects used in the
                protocol
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name=name,
            step_stimulus=step_stimulus,
            holding_stimulus=holding_stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )

        self.target_voltage = target_voltage


class SearchRinHoldingCurrent:

    """Protocol used to find the holding current and input resistance of a
    model"""

    def __init__(
        self,
        name,
        protocol,
        target_voltage=None,
        target_Rin=None,
        target_current=None,
        lbound_factor=2.0,
        atol=0.1,
        max_depth=7,
    ):
        """Constructor

        Args:
            name (str): name of this object
            protocol (Protocol): protocol used to perform the search, usually
                an IV-like protocol.
            target_voltage (eFELFeature): efeature corresponding to the
                target base voltage.
            target_Rin (eFELFeature): efeature corresponding to the
                target input resistance.
            target_current (eFELFeature): efeature corresponding to the holding
                current of the experimental cells
            lbound_factor (float): multiplicative factor used to set the
                lower bound of the holding current search
            atol (float): The absolute tolerance parameter used as a
                threshold to stop the search
            max_depth (int): maximum recursion depth of the holding current
                search
        """

        self.name = name
        self.protocol = protocol

        self.target_voltage = target_voltage
        self.target_Rin = target_Rin
        self.target_current = target_current

        self.lbound_factor = lbound_factor
        self.atol = atol
        self.max_depth = max_depth

    @property
    def stim_start(self):
        return self.protocol.stim_start

    @property
    def stim_end(self):
        return self.protocol.stim_end

    @property
    def step_amplitude(self):
        return self.protocol.step_amplitude

    def subprotocols(self):
        """Return subprotocols"""
        subprotocols = {self.name: self, self.protocol.name: self.protocol}
        return subprotocols

    def create_protocol(self, holding_current):
        """Create a one-time use protocol made of the original protocol with
        a different holding current"""
        protocol = copy.deepcopy(self.protocol)
        protocol.name = "SearchRinHoldingCurrent"
        for recording in protocol.recordings:
            recording.name = recording.name.replace(self.protocol.name, protocol.name)
        protocol.holding_stimulus.step_amplitude = holding_current
        return protocol

    def get_voltage_base(
        self, holding_current, cell_model, param_values, sim, isolate, timeout=None
    ):
        """ Calculate voltage base for a certain holding current """
        protocol = self.create_protocol(holding_current=holding_current)
        response = protocol.run(cell_model, param_values, sim, isolate, timeout=timeout)
        self.target_voltage.stim_start = protocol.stim_start
        self.target_voltage.stim_end = protocol.stim_end
        self.target_voltage.stim_amp = protocol.step_amplitude
        return self.target_voltage.calculate_feature(response)

    def run(self, cell_model, param_values, rmp, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        # Calculate Rin without holding current
        protocol = self.create_protocol(holding_current=0.0)
        response = protocol.run(cell_model, param_values, sim, isolate, timeout)
        rin = self.target_Rin.calculate_feature(response)

        holding_current = self.search_holding_current(
            cell_model, param_values, rmp, rin, sim, isolate
        )
        if holding_current is None:
            return None

        # Return the response of the final estimate of the holding current
        protocol = self.create_protocol(holding_current=holding_current)
        response = protocol.run(cell_model, param_values, sim, isolate, timeout)
        response["bpo_holding_current"] = holding_current

        return response

    def search_holding_current(self, cell_model, param_values, rmp, Rin, sim, isolate):
        """Search the holding current to hold cell at the target base voltage"""

        # Estimate the holding current
        estimate = float(self.target_voltage.exp_mean - rmp) / Rin

        # Perform a bisection search until atol or max_depth is reached
        return self.bisection_search(
            cell_model,
            param_values,
            sim,
            isolate,
            upper_bound=0.0,
            lower_bound=self.lbound_factor * estimate,
        )

    def bisection_search(
        self, cell_model, param_values, sim, isolate, upper_bound, lower_bound, depth=1
    ):
        """Do bisection search to find holding current"""

        logger.debug(
            "Bisection search for SearchRinHoldingCurrent. Depth = %s / %s ",
            depth,
            self.max_depth,
        )

        mid_bound = upper_bound - abs(upper_bound - lower_bound) / 2

        if depth >= self.max_depth:
            return mid_bound

        middle_voltage = self.get_voltage_base(mid_bound, cell_model, param_values, sim, isolate)
        if abs(middle_voltage - self.target_voltage.exp_mean) < self.atol:
            return mid_bound
        if middle_voltage > self.target_voltage.exp_mean:
            return self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                lower_bound=lower_bound,
                upper_bound=mid_bound,
                depth=depth + 1,
            )
        if middle_voltage < self.target_voltage.exp_mean:
            return self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                lower_bound=mid_bound,
                upper_bound=upper_bound,
                depth=depth + 1,
            )
        return None


class SearchThresholdCurrent:

    """Protocol used to find the threshold current (rheobase) of a model"""

    def __init__(
        self,
        name,
        location,
        target_threshold=None,
        nsteps=20,
        max_depth=3,
        step_duration=700.0,
        max_threshold_voltage=-40,
    ):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            nsteps (int): number of steps in which to divide the voltage
                range in which we look for a spike
            max_depth (int): maxium depth for the bisection search
            step_duration (float): duration of the step used to create the
                protocol
            max_threshold_voltage (float): maximum voltage used as upper
                bound in the threshold current search
        """
        self.name = name
        self.target_threshold = target_threshold
        self.location = location

        self.max_threshold_voltage = max_threshold_voltage
        self.nsteps = nsteps
        self.max_depth = max_depth
        self.step_duration = step_duration

        self.spike_feature = ephys.efeatures.eFELFeature(
            name="ThresholdCurrentSearch.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": "ThresholdCurrentSearch.{}.v".format(self.location.name)},
            stim_start=200.0,
            stim_end=200.0 + self.step_duration,
            exp_mean=1,
            exp_std=0.1,
        )

    def create_protocol(self, holding_current, step_current):
        """Create a one-time use protocol made of a holding and step current"""
        # Create the stimuli and recording
        tot_duration = self.step_duration + 400.0
        stimuli = [
            ephys.stimuli.NrnSquarePulse(
                step_amplitude=holding_current,
                step_delay=0.0,
                step_duration=tot_duration,
                location=soma_loc,
                total_duration=tot_duration,
            ),
            ephys.stimuli.NrnSquarePulse(
                step_amplitude=step_current,
                step_delay=200.0,
                step_duration=self.step_duration,
                location=soma_loc,
                total_duration=tot_duration,
            ),
        ]
        recordings = [
            ephys.recordings.CompRecording(
                name="%s.%s.%s" % ("ThresholdCurrentSearch", self.location.name, "v"),
                location=self.location,
                variable="v",
            )
        ]
        return ephys.protocols.SweepProtocol(
            name="ThresholdCurrentSearch",
            stimuli=stimuli,
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
        step_current = self.search_spike_threshold(
            cell_model,
            param_values,
            holding_current,
            holding_current,
            max_threshold_current,
            sim,
            isolate,
            timeout,
        )

        responses = {"bpo_threshold_current": step_current}
        return responses

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

    def search_spike_threshold(
        self,
        cell_model,
        param_values,
        holding_current,
        lower_bound,
        upper_bound,
        sim,
        isolate,
        timeout=None,
    ):
        step_currents = numpy.linspace(lower_bound, upper_bound, num=self.nsteps)
        if len(step_currents) == 0:
            return None
        for i, step_current in enumerate(step_currents):
            spikecount = self.is_spike(
                cell_model,
                param_values,
                holding_current,
                step_current,
                sim,
                isolate,
                timeout,
            )
            if spikecount:
                return self.bisection_search(
                    cell_model,
                    param_values,
                    holding_current,
                    sim,
                    isolate,
                    upper_bound=step_current,
                    lower_bound=step_currents[i - 1],
                    timeout=timeout,
                )
        return None

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
        holding_rin_protocol,
        search_threshold_protocol,
        threshold_protocols=None,
        other_protocols=None,
        score_threshold=None,
    ):
        """Constructor

        Args:
            name (str): name of this object
            rmp_protocol (Protocol): protocol that will be used to
                compute the RMP
            holding_rin_protocol (Protocol): protocol that will
                be used to compute the input resistance and the holding current
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
        self.Rin_protocol = holding_rin_protocol
        self.search_threshold_protocol = search_threshold_protocol

    def subprotocols(self):
        """ Return all the subprotocols contained in the main protocol """
        subprotocols = {}

        if self.RMP_protocol is not None:
            subprotocols[self.RMP_protocol.name] = self.RMP_protocol
        if self.Rin_protocol is not None:
            subprotocols.update(self.Rin_protocol.subprotocols())

        if self.threshold_protocols is not None:
            for name, protocol in self.threshold_protocols.items():
                subprotocols.update({name: protocol})

        if self.other_protocols is not None:
            for name, protocol in self.other_protocols.items():
                subprotocols.update({name: protocol})

        return subprotocols

    def run_threshold(self, cell_model, sim=None, isolate=None, timeout=None):

        responses = {}

        # Find the RMP and check if it is close to the exp RMP
        t1 = time.time()
        rmp_response = self.RMP_protocol.run(cell_model, {}, sim=sim, timeout=timeout)
        responses.update(rmp_response)
        rmp = self.RMP_protocol.target_voltage.calculate_feature(rmp_response)
        rmp_score = self.RMP_protocol.target_voltage.calculate_score(rmp_response)
        logger.debug(
            "Computed RMP in {:.2f}s. Final value = {}; score = "
            "{}".format(time.time() - t1, rmp, rmp_score)
        )

        if self.score_threshold is None or rmp_score <= self.score_threshold:

            # Find the holding current and input resistance
            t1 = time.time()
            rin_response = self.Rin_protocol.run(cell_model, {}, sim=sim, rmp=rmp, timeout=timeout)
            responses.update(rin_response)
            rin = self.Rin_protocol.target_Rin.calculate_feature(rin_response)
            rin_score = self.Rin_protocol.target_Rin.calculate_score(rin_response)
            logger.debug(
                "Computed SearchRinHoldingCurrent in {:.2f}s. Final value = {}; score = {}".format(
                    time.time() - t1, rin, rin_score
                )
            )
            if responses["SearchRinHoldingCurrent.soma.v"] is not None and (
                self.score_threshold is None or rin_score < self.score_threshold
            ):

                # Search for the spiking current.
                t1 = time.time()
                threshold_response = self.search_threshold_protocol.run(
                    cell_model,
                    {},
                    responses["bpo_holding_current"],
                    rin,
                    rmp,
                    sim,
                    isolate,
                    timeout,
                )
                responses.update(threshold_response)

                if responses["bpo_threshold_current"] is not None:

                    threshold_score = (
                        self.search_threshold_protocol.target_threshold.calculate_score(responses)
                    )

                    logger.debug(
                        "Computed threshold current in {:.2f}s. Value = {}; score = {}"
                        "".format(
                            time.time() - t1,
                            responses["bpo_threshold_current"],
                            threshold_score,
                        )
                    )

                    # Compute StepThresholdProtocols
                    logger.debug("Computing StepThresholdProtocols.")
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

                else:
                    logger.debug("Threshold_current is None. Stopping " "MainProtocol run.")
            else:
                logger.debug(
                    "SearchRinHoldingCurrent score higher than score_threshold. Stopping "
                    "MainProtocol run."
                )
        else:
            logger.debug("RMP score lower than score_threshold. Stopping " "MainProtocol run.")

        return responses

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        responses = {}
        cell_model.freeze(param_values)

        # Run the RMP detection, Rin, threshold current detection. Followed
        # by the protocols relying on these computations.
        if self.RMP_protocol and self.Rin_protocol:
            responses.update(self.run_threshold(cell_model, sim, isolate, timeout))

        # Run protocols that are not based on holding/threshold detection
        for protocol in self.other_protocols.values():
            responses.update(protocol.run(cell_model, {}, sim, isolate, timeout))

        cell_model.unfreeze(param_values.keys())
        return responses
