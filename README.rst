# BluePyEModel: Bluebrain Python E-Model Building Library

Introduction
============

The Bluebrain Python E-Model Building Library (BluePyEModel) is a python package
dedicated to simplify the configuration and execution of E-Model building tasks, 
more specifically, the tasks related to feature extraction, model optimisation, 
validation, model management and AIS synthesis. As such, it builds ontop of BluePyEfe, 
BluePyOpt and BluePyMM.

To get started with the E-Model building pipeline
==============

The data access points
---------------------

TO COME

Running the pipeline with Python
--------------------------------

TO COME

Running the pipeline with the CLI
--------------------------------

TO COME

Running the pipeline with Luigi
--------------------------------

TO COME

Pipeline settings:
-----------------

The settings of the pipeline are set as follows:
- When using the 'local' data access point, settings are to be informed in the field "pipeline_settings" of the recipes for each emodel independently.
- When using the 'nexus' data access point, settings have to be registered as a Resource of type PipelineSettings. This can be done using the function store_pipeline_settings from bluepyemodel.access_point.nexus.

List of settings:

extraction_threshold_value_save (int, default: 1): during extraction, minimum number of values needed for an efeatures to be returned in the output.
plot_extraction (bool, default: True): should the e-features and traces be plotted at the end of the e-features extraction. Can be lengthly if their is a lot of data.
efel_settings (dict, default: {'interp_step': 0.025, 'strict_stiminterval': True}): eFEl settings used during efeatures extraction and optimisation. If settings are also informed per e-feature, the latter will have priority.
stochasticity (bool, default: False): should stochasticity be enabled for the the channels that can be stochastic.
morph_modifiers (list, default: replace_axon_with_taper): List of morphology modifiers. Each modifier has to be informed by the path the file containing the modifier and the name of the function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
optimizer (str, default: "IBEA"): algorithm used for optimization, can be "IBEA", "SO-CMA" or "MO-CMA". If the optimizer is "SO-CMA" or "MO-CMA", please pip install bluepyemodel with the 'cma' option.
optimisation_params (dict, default: None): parameters used by BluePyOpt during optimisation. The keys have to match the optimizer's call. E.g., for optimizer MO-CMA: {"offspring_size": 10, "weight_hv": 0.4}.
optimisation_timeout (float, default: 600.0): maximum time in second during which a protocol is allowed to run before being killed.
threshold_efeature_std (float, default: 0.05): lower bound used for the standard deviation of the e-features in the cell evaluator. If informed, the stds are computed as threshold_efeature_std * mean if std is < threshold_efeature_std * min.
max_ngen (int, default: 100): maximum number of generations of the evolutionary process of the optimization.
validation_threshold (float, default: 5.0): used by the default validation function. Threshold under which each score of the emodel has to be for it to pass validation.
plot_optimisation (bool, default: True, Luigi only): should the EModel scores and traces be plotted.

n_model (int, default: 3, nexus access point only, Luigi only): number of models optimized an validated to consider the EModel building task done. The Luigi pipeline will continue launching optimisation batch until this value is reached.
optimisation_batch_size (int, default: 5, nexus access point only, Luigi only): number of optimisation seeds to run in parallel for each batch of optimization launched by the Luigi pipeline.
max_n_batch (int, default: 3, nexus access point only, Luigi only): maximum number of optimisation batches that the Luigi pipeline will run if n_model is not reached.

path_extract_config (str, default: None, local access point only): path to the .json containing the extraction targets, files metadata and the name of the protocols used to compute the threshold of the cell.
name_Rin_protocol (str, default: None, local access point only): name of the protocol associated with the efeatures used for the computation of the input resistance scores during optimisation. This settings has to be set before efeature extraction if you wish to run a threshold based optimisation.
name_rmp_protocol (str, default: None, local access point only): name of the protocol associated with the efeatures used for the computation of the resting membrane potential scores during optimisation. This settings has to be set before efeature extraction if you wish to run a threshold based optimisation. 
validation_protocols (dict default: None, local access point only): names and targets of the protocol that will be used for validation only. This settings has to be set before efeature extraction if you wish to run validation.
