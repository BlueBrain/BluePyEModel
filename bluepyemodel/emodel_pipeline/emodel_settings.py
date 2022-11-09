"""EModelPipelineSettings class"""
import logging

# pylint: disable=too-many-arguments,too-many-locals

logger = logging.getLogger(__name__)


class EModelPipelineSettings:

    """Container for the settings of the E-Model building pipeline"""

    def __init__(
        self,
        extraction_reader=None,
        extraction_threshold_value_save=1,
        plot_extraction=True,
        pickle_cells_extraction=False,
        rheobase_strategy_extraction="absolute",
        rheobase_settings_extraction=None,
        efel_settings=None,
        stochasticity=False,
        morph_modifiers=None,
        threshold_based_evaluator=None,
        optimiser="IBEA",
        optimizer=None,
        optimisation_params=None,
        optimisation_timeout=600.0,
        threshold_efeature_std=None,
        max_ngen=100,
        validation_threshold=5.0,
        validation_function="max_score",
        plot_optimisation=True,
        compile_mechanisms=False,
        n_model=3,
        optimisation_batch_size=5,
        max_n_batch=3,
        path_extract_config=None,
        name_Rin_protocol=None,
        name_rmp_protocol=None,
        validation_protocols=None,
        name_gene_map=None,
        plot_currentscape=False,
        currentscape_config=None,
        neuron_dt=None,
        cvode_minstep=0.0,
        max_threshold_voltage=-30,
        strict_holding_bounds=True,
        max_depth_holding_search=7,
        max_depth_threshold_search=10,
        spikecount_timeout=50,
    ):
        """Init

        Args:
            extraction_reader (function or list): function used to read the ephys data during
                efeature extraction. If list, must contain the path to the file containing the
                function and name of the function. E.g: ["path_to_module", "name_of_function"]
            extraction_threshold_value_save (int): minimum number of values (data points)
                needed for an efeature to be returned in the output of the extraction process.
            plot_extraction (bool): should the efeatures and experimental traces be plotted.
            pickle_cells_extraction (bool): sould the cells object be saved as a pickle file for
                further analysis during extraction.
            rheobase_strategy_extraction (str): function used to compute the rheobase during
                extraction. Can be 'absolute' (amplitude of the lowest amplitude inducing at
                least a spike) or 'majority' (amplitude of the bin in which a majority of
                sweeps induced at least one spike).
            rheobase_settings_extraction (dict): settings related to the rheobase computation.
                Keys have to match the arguments expected by the rheobase computation function.
            efel_settings (dict): efel settings in the form {setting_name: setting_value}.
                If settings are also informed in the targets per efeature, the latter
                will have priority.
            stochasticity (bool or list of str): should channels behave stochastically if they can.
                If a list of protocol names is provided, the runs will be stochastic
                for these protocols, and deterministic for the other ones.
            morph_modifiers (list): List of morphology modifiers. Each modifier has to be
                informed by the path the file containing the modifier and the name of the
                function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
            threshold_based_evaluator (bool): not used. To be deprecated.
            optimiser (str): algorithm used for optimisation, can be "IBEA", "SO-CMA",
                "MO-CMA" (use cma option in pip install for CMA optimisers).
            optimizer (str): for legacy reasons, overwrites optimiser when not None.
            optimisation_params (dict): optimisation parameters. Keys have to match the
                optimiser's call. E.g., for optimiser MO-CMA:
                {"offspring_size": 10, "weight_hv": 0.4}
            optimisation_timeout (float): duration (in second) after which the evaluation
                of a protocol will be interrupted.
            threshold_efeature_std (float): if informed, during extraction, the std of the
                features will be computed as abs(threshold_efeature_std * mean) if
                std is < threshold_efeature_std * min.
            max_ngen (int): maximum number of generations of the evolutionary process of the
                optimisation.
            validation_threshold (float): score threshold under which the emodel passes
                validation.
            validation_function (str or list): if str, can be "max_score" or "mean_score".
                If list, must contain the path to the file containing the function and name
                of the function. E.g: ["path_to_module", "name_of_function"]
            optimisation_batch_size (int): number of optimisation seeds to run in parallel.
            max_n_batch (int): maximum number of optimisation batches.
            name_gene_map (str): name of the gene mapping csv file. Only required when using the
                Nexus access_point.
            n_model (int): minimum number of models to pass validation
                to consider the EModel building task done.
            plot_optimisation (bool): should the EModel scores and traces be plotted.
            compile_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            path_extract_config (str): path to the .json containing the extraction targets, files
                metadata and the name of the protocols used to compute the threshold of the cell.
                Only used with local access point.
            name_Rin_protocol (list or str): name and amplitude of the protocol associated
                with the efeatures used for the computation of the input resistance scores
                during optimisation, e.g: ["IV", -20] or "IV_-20".
                This setting has to be set before efeature extraction if you wish to run
                a threshold based evaluator.
            name_rmp_protocol (list or str): name and amplitude of the protocol associated
                with the efeatures used for the computation of the resting membrane potential
                scores during optimisation, e.g: ["IV", 0] or "IV_0".
                This setting has to be set before efeature extraction if you wish to run
                a threshold based evaluator.
            validation_protocols (list of str): name of the protocols used for validation only.
                E.g. ["APWaveform_300"]
            plot_currentscape (bool): should the EModel currentscapes be plotted
            currentscape_config (dict): currentscape config
                according to the currentscape documentation
                (https://bbpgitlab.epfl.ch/cells/currentscape#about-the-config)
                Note that current.names, output.savefig, output.fname and output.dir
                do not need to be set, since they are automatically rewritten by BPEM.
            neuron_dt (float): dt of the NEURON simulator. If None, cvode will be used.
            cvode_minstep (float): minimum time step allowed for a CVODE step.
            max_threshold_voltage (float): maximum voltage at which the SearchThresholdProtocol
                will search for the rheobase.
            strict_holding_bounds (bool): if True, the minimum and maximum values for the current
                used during the holding current search will be fixed. Otherwise, they will be
                widened dynamically.
            max_depth_holding_search (int): maximum depth for the binary search for the
                holding current
            max_depth_threshold_search (int): maximum depth for the binary search for the
                threshold current
            spikecount_timeout (float): timeout for spikecount computation, if timeout is reached,
                we set spikecount=2 as if many spikes were present, to speed up bisection search.
        """

        # Settings related to E-features extraction
        self.extraction_reader = extraction_reader
        self.extraction_threshold_value_save = extraction_threshold_value_save
        self.plot_extraction = plot_extraction
        self.pickle_cells_extraction = pickle_cells_extraction
        self.efel_settings = efel_settings  # Also used during optimisation
        if self.efel_settings is None:
            self.efel_settings = {"interp_step": 0.025, "strict_stiminterval": True}
        self.path_extract_config = path_extract_config
        self.rheobase_strategy_extraction = rheobase_strategy_extraction
        self.rheobase_settings_extraction = rheobase_settings_extraction

        # Settings related to the evaluator
        self.max_threshold_voltage = max_threshold_voltage
        self.threshold_efeature_std = threshold_efeature_std
        self.max_depth_holding_search = max_depth_holding_search
        self.max_depth_threshold_search = max_depth_threshold_search
        self.spikecount_timeout = spikecount_timeout
        self.stochasticity = stochasticity
        self.neuron_dt = neuron_dt
        self.cvode_minstep = cvode_minstep

        # Settings related to the optimisation
        self.optimisation_timeout = optimisation_timeout
        self.optimiser = optimiser if optimizer is None else optimizer
        self.optimisation_params = optimisation_params
        if self.optimisation_params is None:
            self.optimisation_params = {"offspring_size": 100}

        self.max_ngen = max_ngen
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

        # Settings specific to the cell model
        self.morph_modifiers = morph_modifiers

        if threshold_based_evaluator is not None:
            logger.warning(
                "Setting threshold_based_evaluator is not used anymore and will be deprecated"
            )

    def as_dict(self):
        return vars(self)
