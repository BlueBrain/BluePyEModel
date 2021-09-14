"""EModelPipelineSettings class"""

# pylint: disable=too-many-arguments


class EModelPipelineSettings:

    """Container for the settings of the E-Model building pipeline"""

    def __init__(
        self,
        extraction_reader=None,
        extraction_threshold_value_save=1,
        plot_extraction=True,
        efel_settings=None,
        stochasticity=False,
        morph_modifiers=None,
        threshold_based_evaluator=True,
        optimizer="IBEA",
        optimisation_params=None,
        optimisation_timeout=600.0,
        threshold_efeature_std=0.05,
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
        additional_protocols=None,
    ):
        """Init

        Args:
            extraction_reader (function or list): function used to read the ephys data during
                efeature extraction. If list, must contain the path to the file containing the
                function and name of the function. E.g: ["path_to_module", "name_of_function"]
            extraction_threshold_value_save (int): name of the mechanism.
            efel_settings (dict): efel settings in the form {setting_name: setting_value}.
                If settings are also informed in the targets per efeature, the latter
                will have priority.
            stochasticity (bool): should channels behave stochastically if they can.
            morph_modifiers (list): List of morphology modifiers. Each modifier has to be
                informed by the path the file containing the modifier and the name of the
                function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
            threshold_based_evaluator (bool): if the evaluator is threshold-based. All
                protocol's amplitude and holding current will be rescaled by the ones of the
                models. If True, name_Rin_protocol and name_rmp_protocol have to be informed.
            optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
                "MO-CMA" (use cma option in pip install for CMA optimizers).
            optimisation_params (dict): optimisation parameters. Keys have to match the
                optimizer's call. E.g., for optimizer MO-CMA:
                {"offspring_size": 10, "weight_hv": 0.4}
            optimisation_timeout (float): duration (in second) after which the evaluation
                of a protocol will be interrupted.
            max_ngen (int): maximum number of generations of the evolutionary process of the
                optimization.
            validation_threshold (float): score threshold under which the emodel passes
                validation.
            validation_function (str or list): if str, can be "max_score" or "mean_score".
                If list, must contain the path to the file containing the function and name
                of the function. E.g: ["path_to_module", "name_of_function"]
            optimisation_batch_size (int): number of optimisation seeds to run in parallel.
            max_n_batch (int): maximum number of optimisation batches.
            n_model (int): minimum number of models to pass validation
                to consider the EModel building task done.
            plot_extraction (bool): should the efeatures and experimental traces be plotted.
            plot_optimisation (bool): should the EModel scores and traces be plotted.
            compile_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            path_extract_config (str): path to the .json containing the extraction targets, files
                metadata and the name of the protocols used to compute the threshold of the cell.
            name_Rin_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the input resistance scores during optimisation, e.g: IV_-20.
                This settings as to be set before efeature extraction if you wish to run a
                threshold based evaluator.
            name_rmp_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the resting membrane potential scores during optimisation,
                e.g: IV_0. This settings has to be set before efeature extraction if you wish
                to run a threshold based evaluator. Can also be 'all', in which case the RMP
                will be estimated as the mean of the voltage_base for all the protocols.
            validation_protocols (dict): names and targets of the protocol that will be used for
                validation only. This settings has to be set before efeature extraction if you
                wish to run validation.
            additional_protocols (dict): definition of supplementary protocols.
        """

        # Settings related to E-features extraction
        self.extraction_reader = extraction_reader
        self.extraction_threshold_value_save = extraction_threshold_value_save
        self.plot_extraction = plot_extraction
        self.efel_settings = efel_settings  # Also used during optimisation
        if self.efel_settings is None:
            self.efel_settings = {"interp_step": 0.025, "strict_stiminterval": True}
        self.path_extract_config = path_extract_config  # only when using local access point
        self.name_Rin_protocol = name_Rin_protocol  # only when using local access point
        self.name_rmp_protocol = name_rmp_protocol  # only when using local access point

        # Settings related to the optimisation
        self.threshold_based_evaluator = threshold_based_evaluator
        self.stochasticity = stochasticity
        self.morph_modifiers = morph_modifiers
        self.optimizer = optimizer
        self.optimisation_params = optimisation_params
        if self.optimisation_params is None:
            self.optimisation_params = {"offspring_size": 100}
        self.optimisation_timeout = optimisation_timeout
        self.threshold_efeature_std = threshold_efeature_std
        self.max_ngen = max_ngen
        self.plot_optimisation = plot_optimisation
        self.compile_mechanisms = compile_mechanisms

        # Settings related to the validation
        self.validation_threshold = validation_threshold
        self.validation_function = validation_function
        self.validation_protocols = validation_protocols  # only when using local access point
        if self.validation_protocols is None:
            self.validation_protocols = []
        self.additional_protocols = additional_protocols  # for plotting

        # Settings specific to the Luigi pipeline
        self.n_model = n_model
        self.optimisation_batch_size = optimisation_batch_size
        self.max_n_batch = max_n_batch
