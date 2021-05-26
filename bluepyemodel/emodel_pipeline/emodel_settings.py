"""EModelPipelineSettings class"""

# pylint: disable=too-many-arguments


class EModelPipelineSettings:

    """Container for the settings of the E-Model building pipeline"""

    def __init__(
        self,
        extraction_threshold_value_save=1,
        plot_extraction=True,
        efel_settings=None,
        stochasticity=False,
        morph_modifiers=None,
        optimizer="IBEA",
        optimisation_params=None,
        optimisation_timeout=600.0,
        threshold_efeature_std=0.05,
        max_ngen=100,
        validation_threshold=5.0,
        validation_function=None,
        plot_optimisation=True,
        n_model=3,
        optimisation_batch_size=5,
        max_n_batch=3,
        path_extract_config=None,
        name_Rin_protocol=None,
        name_rmp_protocol=None,
        validation_protocols=None,
    ):
        """Init

        Args:
            extraction_threshold_value_save (int): name of the mechanism.
            efel_settings (dict): efel settings in the form {setting_name: setting_value}.
                If settings are also informed in the targets per efeature, the latter
                will have priority.
            stochasticity (bool): should channels behave stochastically if they can.
            morph_modifiers (list): List of morphology modifiers. Each modifier has to be
                informed by the path the file containing the modifier and the name of the
                function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
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
            validation_function (): TO DO
            optimisation_batch_size (int): number of optimisation seeds to run in parallel.
            max_n_batch (int): maximum number of optimisation batches.
            n_model (int): minimum number of models to pass validation
                to consider the EModel building task done.
            plot_extraction (bool): should the efeatures and experimental traces be plotted.
            plot_optimisation (bool): should the EModel scores and traces be plotted.
            path_extract_config (str): path to the .json containing the extraction targets, files
                metadata and the name of the protocols used to compute the threshold of the cell.
            name_Rin_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the input resistance scores during optimisation. This settings
                has to be set before efeature extraction if you wish to run a threshold based
                optimisation.
            name_rmp_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the resting membrane potential scores during optimisation. This
                settings has to be set before efeature extraction if you wish to run a threshold
                based optimisation.
            validation_protocols (dict): names and targets of the protocol that will be used for
                validation only. This settings has to be set before efeature extraction if you
                wish to run validation.
        """

        # Settings related to E-features extraction
        self.extraction_threshold_value_save = extraction_threshold_value_save
        self.plot_extraction = plot_extraction
        self.efel_settings = efel_settings  # Also used during optimisation
        if self.efel_settings is None:
            self.efel_settings = {"interp_step": 0.025, "strict_stiminterval": True}
        self.path_extract_config = path_extract_config  # only when using local access point
        self.name_Rin_protocol = name_Rin_protocol  # only when using local access point
        self.name_rmp_protocol = name_rmp_protocol  # only when using local access point

        # Settings related to the optimisation
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

        # Settings related to the validation
        self.validation_threshold = validation_threshold
        self.validation_function = validation_function
        self.validation_protocols = validation_protocols  # only when using local access point

        # Settings specific to the Luigi pipeline
        self.n_model = n_model
        self.optimisation_batch_size = optimisation_batch_size
        self.max_n_batch = max_n_batch
