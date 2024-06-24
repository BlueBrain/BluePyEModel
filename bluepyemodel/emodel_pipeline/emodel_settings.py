"""EModelPipelineSettings class"""

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

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-statements

logger = logging.getLogger(__name__)


class EModelPipelineSettings:
    """Container for the settings used during the different steps of the e-model building pipeline.
    When using the "local" access point, these settings will be coming from the recipes.json file.
    When using the "nexus" access point, these settings will be coming from a resource of type
    EModelPipelineSettings.

    This is a backend container class, not meant to be used directly by the user."""

    def __init__(
        self,
        extraction_reader=None,
        extraction_threshold_value_save=1,
        plot_extraction=True,
        pickle_cells_extraction=False,
        extract_absolute_amplitudes=False,
        rheobase_strategy_extraction="absolute",
        rheobase_settings_extraction=None,
        interpolate_RMP_extraction=False,
        default_std_value=1e-3,
        bound_max_std=False,
        efel_settings=None,
        minimum_protocol_delay=0.0,
        stochasticity=False,
        morph_modifiers=None,
        threshold_based_evaluator=None,
        start_from_emodel=None,
        optimiser="IBEA",
        optimizer=None,
        optimisation_params=None,
        optimisation_timeout=600.0,
        threshold_efeature_std=None,
        max_ngen=100,
        optimisation_checkpoint_period=None,
        use_stagnation_criterion=True,
        validation_function="max_score",
        validation_threshold=5.0,
        plot_optimisation=True,
        use_ProbAMPANMDA_EMS=False,
        compile_mechanisms=False,
        n_model=3,
        optimisation_batch_size=5,
        max_n_batch=3,
        path_extract_config=None,
        name_Rin_protocol=None,
        name_rmp_protocol=None,
        validation_protocols=None,
        name_gene_map=None,
        plot_currentscape=True,
        plot_parameter_evolution=True,
        plot_bAP_EPSP=False,
        currentscape_config=None,
        save_recordings=False,
        neuron_dt=None,
        cvode_minstep=0.0,
        use_params_for_seed=True,
        max_threshold_voltage=-30,
        strict_holding_bounds=True,
        max_depth_holding_search=7,
        max_depth_threshold_search=10,
        spikecount_timeout=50,
        files_for_extraction=None,
        targets=None,
        protocols_rheobase=None,
        auto_targets=None,
        auto_targets_presets=None,
    ):
        """Creator of the EModelPipelineSettings class.

        Args:
            extraction_reader (function or list): function used to read the ephys data during
                efeature extraction. If it is a list, it must contain the path to the file
                containing the function and name of the function. E.g: ``["path_to_module",
                "name_of_function"]``. If None, BluePyEfe will try to automatically use the correct
                default reader.
            extraction_threshold_value_save (int): during extraction, define the minimum number of
                values (data points) needed for an e-feature to be considered valid and be returned
                in the output of the extraction process. If the number of data point is lower than
                this value, the e-feature will be ignored.
            plot_extraction (bool): at the end of extraction, should the e-features and
                experimental traces be plotted.
            pickle_cells_extraction (bool): at the end of extraction, should the BluePyEfe objects
                be saved in pickle files for further analysis. The pickle files will be
                saved in the folder "./figures/{emodel_name}/efeatures_extraction/".
            extract_absolute_amplitudes (bool): if True, during extraction, BluePyEfe will assume
                that the targets are expressed in absolute current amplitudes (nA) instead of the
                relative amplitudes when checking if a recording fits a given target.
            rheobase_strategy_extraction (str): during extraction, sets which function is used to
                compute the rheobase of the experimental cells. Can be 'absolute' (amplitude of
                the lowest current amplitude inducing at least a spike) or 'flush' (amplitude of
                the lowest current amplitude inducing at least a spike followed by another
                recording also inducing a spike).
            rheobase_settings_extraction (dict): settings related to the rheobase computation
                strategy. Keys have to match the arguments expected by the rheobase computation
                function present in the module bluepyefe.rheobase.
            interpolate_RMP_extraction (bool): whether to set the RMP after extraction as
                V_hold - R_in*I_Hold, which is an approximation of the RMP.
                This should be used when there is no protocol without holding current
                in the experimental data.
            default_std_value (float): At the end of e-features extraction, all features
                presenting a standard deviation of 0, will see their standard deviation
                replaced by the present value.
            bound_max_std (bool): If set to True, the std from extraction will be set to
                the mean value from extraction if it goes above it.
            efel_settings (dict): efel settings in the form {setting_name: setting_value} to be
                used during extraction. If settings are also informed in the targets on a per
                efeature basis, the latter will have priority.
            minimum_protocol_delay (float): during optimisation, if a protocol has an initial
                delay shorter than this value, the delay will be set to minimum_protocol_delay.
            stochasticity (bool or list of str): should channels behave stochastically if they can.
                If True, the mechanisms will be stochastic for all protocols. If a list of protocol
                names is provided, the mechanisms will be stochastic only for these protocols.
            morph_modifiers (list of str or list of list):
                If str, name of the morph modifier to use from bluepyemodel.evaluation.modifiers.
                If List of morphology modifiers. Each modifier is defined by a list
                that includes the following elements:
                1. The path to the file that contains the modifier.
                2. The name of the function that applies the modifier.
                3. Optionally, a "hoc_string" that represents the hoc code for the modifier.
                For example, morph_modifiers could be defined as follows:

                    .. code-block::

                        morph_modifiers = [["path_to_module",
                                            "name_of_function",
                                            "hoc_string"], ...].

                If the "hoc_string" is not provided, the system will search within
                the specified module for a string that matches the function name appended
                with "_hoc".
                If ``None``, the default modifier will replace the axon with a tappered axon
                initial segment (replace_axon_with_taper). If you do not wish to use any modifier,
                set the present argument to ``[]``.
                If ``["bluepyopt_replace_axon"]``, the replace_axon function from
                bluepyopt.ephys.morphologies.NrnFileMorphology will be used
                and no other morph modifiers will be used.
            threshold_based_evaluator (bool): not used. To be deprecated.
            start_from_emodel (dict): If informed, the optimisation for the present e-model will
                be instantiated using as values for the model parameters the ones from the
                e-model specified in the present dict. That option can be used for example
                to perform a two-steps optimisations. Example:

                .. code-block::

                        {
                            "emodel": "bNAC",
                            "etype": "bNAC",
                            "iteration_tag": "mytest"
                        }.
            optimiser (str): name of the algorithm used for optimisation, can be "IBEA", "SO-CMA"
                or "MO-CMA".
            optimizer (str): here for backward compatibility. Use optimiser instead.
            optimisation_params (dict): parameter for the optimisation process. Keys have to match
                the call of the optimiser. Here are the possible options and default values for the
                different optimisers:

                .. code-block::

                    - IBEA: {
                        "offspring_size": 100 # number of individuals in the population
                    }
                    - SO-CMA: {
                        "offspring_size": 20, # number of individuals in the population
                        sigma=0.4 # initial standard deviation of the gaussian distribution
                    }
                    - MO-CMA: {
                        "offspring_size": 20, # number of individuals in the population
                        sigma=0.4, # initial standard deviation of the gaussian distribution
                        weight_hv=0.5 # weight of the hypervolume score in the selection process
                    }

                For more details, see the documentation of the bluepyopt.deapext package.
            optimisation_timeout (float): duration (in second) after which the evaluation
                of a protocol will be interrupted. When a protocol is interrupted, its response
                will be considered as invalid and the score of all the related e-features will
                be set to the maximum value (250 per default).
            threshold_efeature_std (float): if informed, the standard deviations of the
                efeatures will be thresholded at a minimum of ``abs(threshold_efeature_std
                * efeature_mean)``. Note that this will not overwrite the original standard
                deviations, but only modify them for the optimisation process.
            max_ngen (int): maximum number of generations of the evolutionary process of the
                optimisation.
            optimisation_checkpoint_period (float): minimum time (in s) between checkpoint save.
                None to save checkpoint independently of the time between them
            use_stagnation_criterion (bool): whether to use the stagnation stopping criterion on
                top of the maximum generation criterion during optimisation.
            validation_function (str or list): if str, can be ``max_score`` or ``mean_score``.
                If list, must contain the path to the file containing the function and name
                of the function. E.g: ``["path_to_module", "name_of_function"]``
            validation_threshold (float): if ``max_score`` or ``mean_score`` were specified as the
                validation_function, this parameter will set the threshold under which the e-models
                will be considered to pass validation successfully.
            optimisation_batch_size (int): number of optimisation seeds to run in parallel. Only
                used by the Luigi workflow.
            max_n_batch (int): maximum number of optimisation batches. Only used by the Luigi
                workflow.
            name_gene_map (str): name of the gene mapping csv file. Only required when using the
                Nexus access_point and the IC-selector.
            n_model (int): minimum number of models to pass validation to consider the e-model
                building task done. Only used by the Luigi workflow.
            plot_optimisation (bool): should the e-models scores and traces be plotted. Only used
                by the Luigi workflow.
            use_ProbAMPANMDA_EMS (bool): True to link ProbAMPANMDA_EMS in EMC on nexus,
                and download ProbAMPANMDA from nexus along with other mechanisms.
            compile_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory. Only used by the Luigi workflow.
            path_extract_config (str): specify the path to the .json file containing the targets
                for the extraction process. Defaults to the current directory.
                If an 'iteration_tag' is provided, the file will be located in
                './run/iteration_tag'. This is only applicable with a local access point.
                See example emodel_pipeline_local_python for more details.
            name_Rin_protocol (list or str): name and amplitude of the protocol from which the
                input resistance should be selected from. The matching protocol should have
                "ohmic_input_resistance_vb_ssse" in its feature targets .E.g: ``["IV", -20]`` or
                ``IV_-20``.
                This setting has to be set before efeature extraction if you wish to run
                a threshold based evaluator.
            name_rmp_protocol (list or str): name and amplitude of the protocol from which the
                resting membrane potential should be selected from, e.g: ``["IV", 0]`` or ``IV_0``.
                The matching protocol should have "voltage_base" in its feature targets.
                This setting has to be set before efeature extraction if you wish to run
                a threshold based evaluator.
            validation_protocols (list of str): name of the protocols to be used for validation
                only. E.g. ``["APWaveform_300"]``. These protocols will not be used during
                optimisation.
            plot_currentscape (bool): during the plotting, should the currentscapes be
                plotted for the recordings.
            plot_parameter_evolution (bool): during the plotting, should the evolution of the
                parameters be plotted.
            plot_bAP_EPSP (bool): during the plotting, should ready-to-use back-propagating AP
                and EPSP protocols be run and plotted.
                Should be True only for pyramidal cells,
                since it depends on the presence of apical dendrite.
            currentscape_config (dict): currentscape configuration according to the currentscape
                documentation (https://github.com/BlueBrain/Currentscape).
                Note that current.names, output.savefig, output.fname and output.dir
                do not need to be set, since they are automatically overwritten by BluePyEModel.
                If current.names is set nonetheless, it will be used as the subset of available
                currents to be selected for the plot.
            save_recordings (bool): Whether to save the responses data under a folder
                named `recordings`.
            neuron_dt (float): time step of the NEURON simulator. If ``None``, cvode will be used.
            cvode_minstep (float): minimum time step allowed when using cvode.
            use_params_for_seed (bool): use a hashed version of the parameter
                dictionary as a seed for the simulator
            max_threshold_voltage (float): upper bound for the voltage during the
                search for the threshold or rheobase current (see SearchThresholdProtocol).
            strict_holding_bounds (bool): if True, the minimum and maximum values for the current
                used during the holding current search will be fixed. Otherwise, they will be
                widened dynamically if the holding current is beyond the initial bounds.
            max_depth_holding_search (int): maximum depth for the binary search for the
                holding current.
            max_depth_threshold_search (int): maximum depth for the binary search for the
                threshold current.
            spikecount_timeout (float): during the search of the threshold current, if the present
                timeout is reached, we set spikecount=2 as if many spikes were present, to speed
                up bisection search.
            files_for_extraction (list): temporary
            targets (list): temporary
            protocols_rheobase (list): temporary
            auto_targets (list): temporary
            auto_targets_presets (list): temporary
        """

        # Settings related to E-features extraction
        self.extraction_reader = extraction_reader
        self.extraction_threshold_value_save = extraction_threshold_value_save
        self.plot_extraction = plot_extraction
        self.extract_absolute_amplitudes = extract_absolute_amplitudes
        self.pickle_cells_extraction = pickle_cells_extraction
        self.efel_settings = efel_settings  # Also used during optimisation
        if self.efel_settings is None:
            self.efel_settings = {"interp_step": 0.025, "strict_stiminterval": True}
        self.path_extract_config = path_extract_config
        self.rheobase_strategy_extraction = rheobase_strategy_extraction
        self.rheobase_settings_extraction = rheobase_settings_extraction
        self.interpolate_RMP_extraction = interpolate_RMP_extraction
        self.default_std_value = default_std_value
        self.bound_max_std = bound_max_std
        self.minimum_protocol_delay = minimum_protocol_delay

        # Settings related to the evaluator
        self.max_threshold_voltage = max_threshold_voltage
        self.threshold_efeature_std = threshold_efeature_std
        self.max_depth_holding_search = max_depth_holding_search
        self.max_depth_threshold_search = max_depth_threshold_search
        self.spikecount_timeout = spikecount_timeout
        self.stochasticity = stochasticity
        self.neuron_dt = neuron_dt
        self.cvode_minstep = cvode_minstep
        self.use_params_for_seed = use_params_for_seed

        # Settings related to the optimisation
        self.start_from_emodel = start_from_emodel
        self.optimisation_timeout = optimisation_timeout
        self.optimiser = optimiser if optimizer is None else optimizer
        self.optimisation_params = optimisation_params
        if self.optimisation_params is None:
            if self.optimiser == "IBEA":
                self.optimisation_params = {"offspring_size": 100}
            else:
                self.optimisation_params = {"offspring_size": 20}

        self.max_ngen = max_ngen
        self.optimisation_checkpoint_period = optimisation_checkpoint_period
        self.use_stagnation_criterion = use_stagnation_criterion
        self.plot_optimisation = plot_optimisation
        self.compile_mechanisms = compile_mechanisms

        # Specific to threshold based optimisation
        self.name_Rin_protocol = name_Rin_protocol
        self.name_rmp_protocol = name_rmp_protocol
        self.strict_holding_bounds = strict_holding_bounds

        # Settings related to the validation
        self.validation_threshold = validation_threshold
        self.validation_function = validation_function

        if isinstance(validation_protocols, dict):
            self.validation_protocols = [
                f"{name}_{amp}"
                for name in validation_protocols
                for amp in validation_protocols[name]
            ]
        elif validation_protocols is None:
            self.validation_protocols = []
        else:
            self.validation_protocols = validation_protocols

        # Settings specific to the Luigi pipeline
        self.n_model = n_model
        self.optimisation_batch_size = optimisation_batch_size
        self.max_n_batch = max_n_batch
        self.name_gene_map = name_gene_map

        # Settings specific to the currentscape plotting
        self.plot_currentscape = plot_currentscape
        self.currentscape_config = currentscape_config

        self.plot_parameter_evolution = plot_parameter_evolution
        self.plot_bAP_EPSP = plot_bAP_EPSP

        # Settings specific to the recordings
        self.save_recordings = save_recordings

        # Settings specific to the cell model
        self.morph_modifiers = morph_modifiers

        if threshold_based_evaluator is not None:
            logger.warning(
                "Setting threshold_based_evaluator is not used anymore and will be deprecated"
            )

        # Settings for targets configuration
        # Temporarily in pipeline_settings - will come from SBO in the future
        self.files_for_extraction = files_for_extraction
        self.targets = targets
        self.protocols_rheobase = protocols_rheobase
        self.auto_targets = auto_targets
        self.auto_targets_presets = auto_targets_presets

        # Settings used with nexus
        self.use_ProbAMPANMDA_EMS = use_ProbAMPANMDA_EMS

    def as_dict(self):
        return vars(self)
