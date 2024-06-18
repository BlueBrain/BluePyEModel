"""Module with protocol classes."""

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
from collections import OrderedDict

import numpy
from bluepyopt import ephys

from ..ecode import eCodes
from .recordings import LooseDtRecordingCustom
from .recordings import check_recordings

# pylint: disable=abstract-method

logger = logging.getLogger(__name__)


class BPEMProtocol(ephys.protocols.SweepProtocol):
    """Base protocol"""

    def __init__(
        self,
        name=None,
        stimulus=None,
        recordings=None,
        cvode_active=None,
        stochasticity=False,
    ):
        """Constructor

        Args:
            name (str): name of this object
            stimulus (Stimulus): stimulus objects
            recordings (list of Recordings): Recording objects used in the
                protocol
            cvode_active (bool): whether to use variable time step
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name=name,
            stimuli=[stimulus],
            recordings=recordings,
            cvode_active=cvode_active,
            deterministic=not stochasticity,
        )

        self.stimulus = stimulus

        self.features = []

    def instantiate(self, sim=None, cell_model=None):
        """Check recordings, then instantiate."""
        if not all(rec.checked for rec in self.recordings):
            self.recordings = check_recordings(self.recordings, cell_model.icell, sim)

        super().instantiate(sim, cell_model)

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
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        """Run protocol"""

        param_values = {} if param_values is None else param_values
        responses = {} if responses is None else responses

        return super().run(
            cell_model=cell_model,
            param_values=param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )


class ResponseDependencies:
    """To add to a protocol to specify that it depends on the responses of other protocols"""

    def __init__(self, dependencies=None):
        """Constructor

        Args:
            dependencies (dict): dictionary of dependencies of the form
                {self_attribute_name: [protocol_name, response_name]}.
        """

        self.dependencies = {} if dependencies is None else dependencies

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

    def set_dependencies(self, responses):
        for attribute_name, dep in self.dependencies.items():
            if responses.get(dep[1], None) is None:
                logger.debug("Dependency %s missing", dep[1])
                return False
            self.set_attribute(attribute_name, responses[dep[1]])
        return True

    def _run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        raise NotImplementedError("The run code of the sub-classes goes here!")

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        """Run protocol"""

        param_values = {} if param_values is None else param_values
        responses = {} if responses is None else responses

        if not self.set_dependencies(responses):
            return self.return_none_responses()

        return self._run(cell_model, param_values, sim, isolate, timeout)


class ProtocolWithDependencies(BPEMProtocol, ResponseDependencies):
    """To add to a protocol to specify that it depends on the responses of other protocols"""

    def __init__(
        self,
        dependencies=None,
        name=None,
        stimulus=None,
        recordings=None,
        cvode_active=None,
        stochasticity=False,
    ):
        """Constructor

        Args:
            dependencies (dict): dictionary of dependencies of the form
                {self_attribute_name: [protocol_name, response_name]}.
        """

        ResponseDependencies.__init__(self, dependencies)
        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochasticity=stochasticity,
        )

    def _run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        return BPEMProtocol.run(self, cell_model, param_values, sim, isolate, timeout, responses)


class ThresholdBasedProtocol(ProtocolWithDependencies):
    """Protocol having rheobase-rescaling capabilities. When using ThresholdBasedProtocol,
    the holding current amplitude and step amplitude of the stimulus will be ignored and
    replaced by values obtained from the holding current and rheobase of the cell model
    respectively."""

    def __init__(
        self,
        name=None,
        stimulus=None,
        recordings=None,
        cvode_active=None,
        stochasticity=False,
        hold_key="bpo_holding_current",
        thres_key="bpo_threshold_current",
        hold_prot_name="SearchHoldingCurrent",
        thres_prot_name="SearchThresholdCurrent",
    ):
        """Constructor"""

        dependencies = {
            "stimulus.holding_current": [hold_prot_name, hold_key],
            "stimulus.threshold_current": [thres_prot_name, thres_key],
        }

        ProtocolWithDependencies.__init__(
            self,
            dependencies=dependencies,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochasticity=stochasticity,
        )

    def return_none_responses(self):
        return {k.name: None for k in self.recordings}

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        return ResponseDependencies.run(
            self, cell_model, param_values, sim, isolate, timeout, responses
        )


class RMPProtocol(BPEMProtocol):
    """Protocol consisting of a step of amplitude zero"""

    def __init__(
        self,
        name,
        location,
        target_voltage,
        stimulus_duration=500.0,
        output_key="bpo_rmp",
    ):
        """Constructor"""

        stimulus_definition = {
            "delay": 0.0,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": stimulus_duration,
            "holding_current": 0.0,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]
        self.output_key = output_key

        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.stimulus_duration = stimulus_duration

        self.target_voltage = target_voltage
        self.target_voltage.stim_start = stimulus_duration - 100.0
        self.target_voltage.stim_end = stimulus_duration
        self.target_voltage.stimulus_current = 0.0

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        """Compute the RMP"""

        response = BPEMProtocol.run(
            self,
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
            responses=responses,
        )
        if responses is None or response[self.recording_name] is None:
            return {self.recording_name: None, self.output_key: None}

        bpo_rmp = self.target_voltage.calculate_feature(response)
        response[self.output_key] = bpo_rmp if bpo_rmp is None else bpo_rmp[0]

        return response


class RinProtocol(ProtocolWithDependencies):
    """Protocol used to find the input resistance of a model"""

    def __init__(
        self,
        name,
        location,
        target_rin,
        amp=-0.02,
        stimulus_delay=500.0,
        stimulus_duration=500.0,
        totduration=1000.0,
        output_key="bpo_rin",
        hold_key="bpo_holding_current",
        hold_prot_name="SearchHoldingCurrent",
    ):
        """Constructor"""

        stimulus_definition = {
            "delay": stimulus_delay,
            "amp": amp,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": totduration,
            "holding_current": None,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]
        self.output_key = output_key

        dependencies = {"stimulus.holding_current": [hold_prot_name, hold_key]}

        ProtocolWithDependencies.__init__(
            self,
            dependencies=dependencies,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.target_rin = target_rin
        self.target_rin.stim_start = stimulus_delay
        self.target_rin.stim_end = stimulus_delay + stimulus_duration
        self.target_rin.stimulus_current = amp

    def return_none_responses(self):
        return {self.recording_name: None, self.output_key: None}

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        """Compute the Rin"""

        response = ResponseDependencies.run(
            self,
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
            responses=responses,
        )

        bpo_rin = self.target_rin.calculate_feature(response)
        response[self.output_key] = bpo_rin if bpo_rin is None else bpo_rin[0]

        return response


class NoHoldingCurrent(ephys.protocols.Protocol):
    """Empty class returning a holding current of zero."""

    def __init__(self, name, output_key="bpo_holding_current"):
        """Constructor."""
        super().__init__(
            name=name,
        )
        self.output_key = output_key
        self.recordings = {}

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        # pylint: disable=unused-argument
        return {self.output_key: 0}


class SearchHoldingCurrent(BPEMProtocol):
    """Protocol used to find the holding current of a model"""

    def __init__(
        self,
        name,
        location,
        target_voltage=None,
        voltage_precision=0.1,
        stimulus_duration=500.0,
        upper_bound=0.2,
        lower_bound=-0.2,
        strict_bounds=True,
        max_depth=7,
        no_spikes=True,
        output_key="bpo_holding_current",
    ):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_voltage (EFeature): target for the voltage at holding_current
            voltage_precision (float): accuracy for holding voltage, in mV, to stop the search
            stimulus_duration (float): length of the protocol
            upper_bound (float): upper bound for the holding current, in pA
            lower_bound (float): lower bound for the holding current, in pA
            strict_bounds (bool): to adaptively enlarge bounds if current is outside
            max_depth (int): maximum depth for the binary search
            no_spikes (bool): if True, the holding current will only be considered valid if there
                are no spikes at holding.
        """

        stimulus_definition = {
            "delay": 0.0,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": stimulus_duration,
            "holding_current": 0.0,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]
        self.output_key = output_key

        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.voltage_precision = voltage_precision
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.strict_bounds = strict_bounds
        self.no_spikes = no_spikes

        self.target_voltage = target_voltage
        self.holding_voltage = self.target_voltage.exp_mean

        self.target_voltage.stim_start = stimulus_duration - 100.0
        self.target_voltage.stim_end = stimulus_duration
        self.target_voltage.stimulus_current = 0.0

        self.max_depth = max_depth

        self.spike_feature = ephys.efeatures.eFELFeature(
            name="SearchHoldingCurrent.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"SearchHoldingCurrent.{location.name}.v"},
            stim_start=0.0,
            stim_end=stimulus_duration,
            exp_mean=1,
            exp_std=0.1,
        )

    def get_voltage_base(
        self, holding_current, cell_model, param_values, sim, isolate, timeout=None
    ):
        """Calculate voltage base for a certain holding current"""

        self.stimuli[0].amp = holding_current
        response = BPEMProtocol.run(
            self, cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )
        if response is None or response[self.recording_name] is None:
            return None

        if self.no_spikes:
            n_spikes = self.spike_feature.calculate_feature(response)
            if n_spikes is None or n_spikes > 0:
                return None

        voltage_base = self.target_voltage.calculate_feature(response)
        if voltage_base is None:
            return None
        return voltage_base[0]

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
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
                if voltage_min is None:
                    voltage_min = 1e10

                if voltage_min > self.target_voltage.exp_mean:
                    self.lower_bound -= 0.2

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
                if voltage_max is None:
                    # if we spike, we let it pass to the search
                    voltage_max = 1e10

                elif voltage_max < self.target_voltage.exp_mean:
                    self.upper_bound += 0.2

        response = {
            self.output_key: self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                timeout=timeout,
            )
        }

        if response[self.output_key] is None:
            return response

        response.update(
            BPEMProtocol.run(
                self,
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                timeout=timeout,
            )
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
        timeout=None,
        depth=0,
    ):
        """Do bisection search to find holding current"""
        mid_bound = (upper_bound + lower_bound) * 0.5
        voltage = self.get_voltage_base(
            holding_current=mid_bound,
            cell_model=cell_model,
            param_values=param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )
        # if we don't converge fast enough, we stop and return lower bound, which will not spike
        if depth > self.max_depth:
            logging.debug(
                "Exiting search due to reaching max_depth. The required voltage precision "
                "was not reached."
            )
            return lower_bound

        if voltage is not None and abs(voltage - self.holding_voltage) < self.voltage_precision:
            logger.debug("Depth of holding search: %s", depth)
            return mid_bound

        # if voltage is None, it means we spike at mid_bound, so we try with lower side
        if voltage is None or voltage > self.holding_voltage:
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


class SearchThresholdCurrent(ProtocolWithDependencies):
    """Protocol used to find the threshold current (rheobase) of a model"""

    def __init__(
        self,
        name,
        location,
        target_threshold=None,
        current_precision=1e-2,
        stimulus_delay=500.0,
        stimulus_duration=2000.0,
        stimulus_totduration=3000.0,
        max_threshold_voltage=-30,
        spikecount_timeout=50,
        max_depth=10,
        no_spikes=True,
        efel_threshold=None,
        output_key="bpo_threshold_current",
        hold_key="bpo_holding_current",
        rmp_key="bpo_rmp",
        rin_key="bpo_rin",
        hold_prot_name="SearchHoldingCurrent",
        rmp_prot_name="RMPProtocol",
        rin_prot_name="RinProtocol",
    ):
        """Constructor.

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_threshold (Efeature): target for the threshold_current
            current_precision (float): size of search interval in current to stop the search
            stimulus_delay (float): delay before the beginning of the step
                used to create the protocol
            stimulus_duration (float): duration of the step used to create the
                protocol
            stimulus_totduration (float): total duration of the protocol
            max_threshold_voltage (float): maximum voltage used as upper
                bound in the threshold current search
            spikecount_timeout (float): timeout for spikecount computation, if timeout is reached,
                we set spikecount=2 as if many spikes were present, to speed up bisection search.
            max_depth (int): maximum depth for the binary search
            no_spikes (bool): if True, will check that the holding current (lower bound) does not
                trigger spikes.
            efel_threshold: spike threshold for the efel settings.
                Set to None to keep the default value (currently -20 mV in efel)
        """
        # pylint: disable=too-many-arguments

        dependencies = {
            "stimulus.holding_current": [hold_prot_name, hold_key],
            "rin": [rin_prot_name, rin_key],
            "rmp": [rmp_prot_name, rmp_key],
        }

        stimulus_definition = {
            "delay": stimulus_delay,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": stimulus_totduration,
            "holding_current": None,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]
        self.output_key = output_key
        self.hold_key = hold_key

        super().__init__(
            dependencies=dependencies,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.rin = None
        self.rmp = None

        self.target_threshold = target_threshold
        self.max_threshold_voltage = max_threshold_voltage
        self.current_precision = current_precision
        self.no_spikes = no_spikes
        self.max_depth = max_depth

        self.spike_feature = ephys.efeatures.eFELFeature(
            name=f"{name}.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"{name}.{location.name}.v"},
            stim_start=stimulus_delay,
            stim_end=stimulus_delay + stimulus_duration,
            exp_mean=1,
            exp_std=0.1,
            threshold=efel_threshold,
        )
        self.spikecount_timeout = spikecount_timeout

    def return_none_responses(self):
        return {self.output_key: None, self.recording_name: None}

    def _get_spikecount(self, current, cell_model, param_values, sim, isolate):
        """Get spikecount at a given current."""

        self.stimulus.amp = current

        response = self._run(
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            timeout=self.spikecount_timeout,
        )
        if response[self.recording_name] is None:
            logger.debug(
                "Trace computation for threshold timed out at %s",
                self.spikecount_timeout,
            )
            return 2

        return self.spike_feature.calculate_feature(response)

    def run(
        self,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        """Run protocol"""
        if not self.set_dependencies(responses):
            return self.return_none_responses()

        lower_bound, upper_bound = self.define_search_bounds(
            cell_model, param_values, sim, isolate, responses
        )
        if lower_bound is None or upper_bound is None:
            logger.debug("Threshold search bounds are not good")
            return {self.output_key: None}

        threshold = self.bisection_search(
            cell_model,
            param_values,
            sim,
            isolate,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            timeout=timeout,
        )

        response = {self.output_key: threshold}
        if threshold is None:
            return response

        self.stimulus.amp = threshold
        response.update(
            self._run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)
        )

        return response

    def max_threshold_current(self):
        """Find the current necessary to get to max_threshold_voltage"""
        max_threshold_current = (self.max_threshold_voltage - self.rmp) / self.rin
        max_threshold_current = numpy.min([max_threshold_current, 2.0])
        logger.debug("Max threshold current: %.6g", max_threshold_current)
        return max_threshold_current

    def define_search_bounds(self, cell_model, param_values, sim, isolate, responses):
        """Define the bounds and check their validity"""

        upper_bound = self.max_threshold_current()
        spikecount = self._get_spikecount(upper_bound, cell_model, param_values, sim, isolate)
        if spikecount == 0:
            logger.debug("No spikes at upper bound during threshold search")
            return None, None

        lower_bound = responses[self.hold_key]
        spikecount = self._get_spikecount(lower_bound, cell_model, param_values, sim, isolate)

        if spikecount > 0:
            if self.no_spikes:
                logger.debug("Spikes at lower bound during threshold search")
                return None, None
            lower_bound -= 0.5

        if lower_bound > upper_bound:
            logger.debug("lower bound higher than upper bound in threshold search")
            return lower_bound, lower_bound

        return lower_bound, upper_bound

    def bisection_search(
        self,
        cell_model,
        param_values,
        sim,
        isolate,
        upper_bound,
        lower_bound,
        timeout=None,
        depth=0,
    ):
        """Do bisection search to find threshold current."""
        mid_bound = (upper_bound + lower_bound) * 0.5
        spikecount = self._get_spikecount(mid_bound, cell_model, param_values, sim, isolate)
        if abs(lower_bound - upper_bound) < self.current_precision:
            logger.debug("Depth of threshold search: %s", depth)
            return upper_bound

        if depth > self.max_depth:
            return upper_bound

        if spikecount == 0:
            return self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                lower_bound=mid_bound,
                upper_bound=upper_bound,
                timeout=timeout,
                depth=depth + 1,
            )
        return self.bisection_search(
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            lower_bound=lower_bound,
            upper_bound=mid_bound,
            timeout=timeout,
            depth=depth + 1,
        )


class ProtocolRunner(ephys.protocols.Protocol):
    """Meta-protocol in charge of running the other protocols in the correct order"""

    def __init__(self, protocols, name="ProtocolRunner"):
        """Initialize the protocol runner

        Args:
            protocols (dict): Dictionary of protocols to run
            name (str): Name of the current protocol runner
        """

        super().__init__(name=name)

        self.protocols = protocols
        self.execution_order = self.compute_execution_order()

    def _add_to_execution_order(self, protocol, execution_order, before_index=None):
        """Recursively adds protocols to the execution order while making sure that their
        dependencies are added before them. Warning: Does not solve all types of dependency graph.
        """

        if protocol.name not in execution_order:
            if before_index is None:
                execution_order.append(protocol.name)
            else:
                execution_order.insert(before_index, protocol.name)

        if hasattr(protocol, "dependencies"):
            for dep in protocol.dependencies.values():
                if dep[0] not in execution_order:
                    self._add_to_execution_order(
                        self.protocols[dep[0]],
                        execution_order,
                        before_index=execution_order.index(protocol.name),
                    )

    def compute_execution_order(self):
        """Compute the execution order of the protocols by taking into account their dependencies"""

        execution_order = []

        for protocol in self.protocols.values():
            self._add_to_execution_order(protocol, execution_order)

        return execution_order

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        responses = OrderedDict()
        cell_model.freeze(param_values)

        for protocol_name in self.execution_order:
            logger.debug("Computing protocol %s", protocol_name)
            new_responses = self.protocols[protocol_name].run(
                cell_model,
                param_values={},
                sim=sim,
                isolate=isolate,
                timeout=timeout,
                responses=responses,
            )

            if new_responses is None or any(v is None for v in new_responses.values()):
                logger.debug("None in responses, exiting evaluation")
                break

            responses.update(new_responses)

        cell_model.unfreeze(param_values.keys())
        return responses

    def __str__(self):
        """String representation"""

        content = f"Sequence protocol {self.name}:\n"

        content += f"{len(self.protocols)} subprotocols:\n"
        for protocol in self.protocols:
            content += f"{protocol}\n"

        return content
