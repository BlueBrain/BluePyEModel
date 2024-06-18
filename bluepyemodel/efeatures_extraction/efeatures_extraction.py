"""Efeatures extraction functions"""

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
import pathlib
from importlib.machinery import SourceFileLoader

import bluepyefe.extract
import numpy

from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.tools.search_pdfs import search_figure_efeatures

logger = logging.getLogger(__name__)


def interpolate_RMP(fitness_calculator_configuration):
    """If we do not have recordings with no holding, we need to estimate the RMP as:
    RMP = V_hold - R_in*I_Hold
    """
    rin = None
    holding_current = None
    holding_voltages = []

    for f in fitness_calculator_configuration.efeatures:
        if (
            f.efel_feature_name == "ohmic_input_resistance_vb_ssse"
            and f.protocol_name == "RinProtocol"
        ):
            rin = f.mean
        if f.efel_feature_name == "voltage_base" and (
            "iv" in f.protocol_name.lower()
            or "idrest" in f.protocol_name.lower()
            or "idthres" in f.protocol_name.lower()
        ):
            holding_voltages.append(f.mean)
        if f.efel_feature_name == "bpo_holding_current":
            holding_current = f.mean

    if rin is None:
        raise TypeError("Impossible to interpolate the RMP as Rin is also missing")
    if holding_current is None:
        raise TypeError("Impossible to interpolate the RMP as the holding current is also missing")
    if not holding_voltages:
        raise ValueError("Impossible to interpolate the RMP as no voltage base is available")

    holding_voltage = numpy.median(holding_voltages)
    rmp = holding_voltage - (rin * holding_current)

    logger.debug("The RMP was computed to be: %s", rmp)

    for i, f in enumerate(fitness_calculator_configuration.efeatures):
        if (
            f.efel_feature_name == "steady_state_voltage_stimend"
            and f.protocol_name == "RMPProtocol"
        ):
            fitness_calculator_configuration.efeatures[i].mean = rmp
            break
    else:
        raise ValueError("RMP feature not found and cannot be replaced")


def threshold_efeatures_std(fitness_calculator_configuration, default_std_value):
    """Set std as the mean value if std is higher than mean value."""
    for i, f in enumerate(fitness_calculator_configuration.efeatures):
        if f.original_std > abs(1.0 * f.mean) and f.original_std != default_std_value:
            logger.debug("Thresholding %s: %s -> %s", f.name, f.original_std, abs(1.0 * f.mean))
            fitness_calculator_configuration.efeatures[i].original_std = abs(1.0 * f.mean)


def define_extraction_reader_function(access_point):
    """Define the function used to read the ephys data during efeature extraction"""

    extraction_reader = access_point.pipeline_settings.extraction_reader

    if extraction_reader is None or not extraction_reader:
        logger.warning(
            "Extraction reader function not specified, BluePyEfe will use automatic "
            "detection based on file extension."
        )
        return None

    if isinstance(extraction_reader, list) and len(extraction_reader) == 2:
        # pylint: disable=deprecated-method,no-value-for-parameter
        function_module = SourceFileLoader(
            pathlib.Path(extraction_reader[0]).stem, extraction_reader[0]
        ).load_module()
        extraction_reader = getattr(function_module, extraction_reader[1])

    elif not callable(extraction_reader):
        raise TypeError("Extraction reader function is not callable nor a list of two strings")

    return extraction_reader


def attach_efeatures_pdf(emodel, efeatures):
    """If the efeatures are plotted, attach the path to the plot to the related efeature"""

    for protocol in efeatures:
        for efeat in efeatures[protocol]["soma"]:
            pdfs = {}

            pdf_amp, pdf_amp_rel = search_figure_efeatures(emodel, protocol, efeat["feature"])

            if pdf_amp:
                pdfs["amp"] = pdf_amp
            if pdf_amp_rel:
                pdfs["amp_rel"] = pdf_amp_rel

            if pdfs:
                efeat["pdfs"] = pdfs


def update_minimum_protocols_delay(access_point, config):
    """Threshold the minimum delay to the value of minimum_protocol_delay provided in the
    settings."""

    min_delay = access_point.pipeline_settings.minimum_protocol_delay

    for i, protocol in enumerate(config.protocols):
        if "delay" in protocol.stimuli[0] and protocol.stimuli[0]["delay"] < min_delay:
            logger.debug(
                "Replacing delay %s with %s in protocol %s",
                protocol.stimuli[0]["delay"],
                min_delay,
                protocol.name,
            )

            delta_delay = min_delay - protocol.stimuli[0]["delay"]
            config.protocols[i].stimuli[0]["delay"] = min_delay
            config.protocols[i].stimuli[0]["totduration"] = (
                protocol.stimuli[0]["totduration"] + delta_delay
            )

            for j, f in enumerate(config.efeatures):
                if f.protocol_name == protocol.name:
                    if "stim_start" in f.efel_settings:
                        config.efeatures[j].efel_settings["stim_start"] += delta_delay
                    if "stim_end" in f.efel_settings:
                        config.efeatures[j].efel_settings["stim_end"] += delta_delay

    return config


def extract_save_features_protocols(access_point, mapper=map):
    """Extract the efeatures and saves the results as a configuration for the fitness calculator.

    Args:
        access_point (DataAccessPoint): access point to the model's data
        mapper (map): mapper for parallel computations.
    """

    targets_configuration = access_point.get_targets_configuration()
    if (
        access_point.pipeline_settings.name_rmp_protocol is not None
        and access_point.pipeline_settings.name_Rin_protocol is not None
    ):
        targets_configuration.check_presence_RMP_Rin_efeatures(
            access_point.pipeline_settings.name_rmp_protocol,
            access_point.pipeline_settings.name_Rin_protocol,
        )

    reader_function = define_extraction_reader_function(access_point)

    threshold_nvalue_save = access_point.pipeline_settings.extraction_threshold_value_save
    plot = access_point.pipeline_settings.plot_extraction
    output_directory = f"./figures/{access_point.emodel_metadata.emodel}/efeatures_extraction/"

    efeatures, stimuli, current = bluepyefe.extract.extract_efeatures(
        output_directory=output_directory,
        files_metadata=targets_configuration.files_metadata_BPE,
        targets=targets_configuration.targets_BPE,
        auto_targets=targets_configuration.auto_targets_BPE,
        absolute_amplitude=access_point.pipeline_settings.extract_absolute_amplitudes,
        threshold_nvalue_save=threshold_nvalue_save,
        protocols_rheobase=targets_configuration.protocols_rheobase_BPE,
        recording_reader=reader_function,
        map_function=mapper,
        write_files=False,
        plot=plot,
        efel_settings=access_point.pipeline_settings.efel_settings,
        pickle_cells=access_point.pipeline_settings.pickle_cells_extraction,
        rheobase_strategy=access_point.pipeline_settings.rheobase_strategy_extraction,
        rheobase_settings=access_point.pipeline_settings.rheobase_settings_extraction,
        default_std_value=access_point.pipeline_settings.default_std_value,
    )

    if plot:
        attach_efeatures_pdf(access_point.emodel_metadata.emodel, efeatures)

    fitness_calculator_config = FitnessCalculatorConfiguration(
        name_rmp_protocol=access_point.pipeline_settings.name_rmp_protocol,
        name_rin_protocol=access_point.pipeline_settings.name_Rin_protocol,
        validation_protocols=access_point.pipeline_settings.validation_protocols,
        stochasticity=access_point.pipeline_settings.stochasticity,
        default_std_value=access_point.pipeline_settings.default_std_value,
    )

    fitness_calculator_config.init_from_bluepyefe(
        efeatures,
        stimuli,
        current,
        access_point.pipeline_settings.threshold_efeature_std,
        targets_configuration.protocols_mapping,
    )

    # correct for any weight present in features
    for target in targets_configuration.targets:
        if target.weight != 1.0:
            try:
                feat = next(
                    feat
                    for feat in fitness_calculator_config.efeatures
                    if feat.efel_feature_name == target.efeature
                    and feat.protocol_name == f"{target.protocol}_{target.amplitude}"
                )
                feat.weight = target.weight
            except StopIteration:
                logger.debug(
                    "Could not find efeature with name %s and protocol %s",
                    target.efeature,
                    target.protocol,
                )

    if access_point.pipeline_settings.bound_max_std:
        threshold_efeatures_std(
            fitness_calculator_config, access_point.pipeline_settings.default_std_value
        )

    fitness_calculator_config.protocols += fitness_calculator_config.initialise_protocols(
        targets_configuration.additional_fitness_protocols
    )

    fitness_calculator_config.efeatures += fitness_calculator_config.initialise_efeatures(
        targets_configuration.additional_fitness_efeatures,
        threshold_efeature_std=access_point.pipeline_settings.threshold_efeature_std,
    )

    if access_point.pipeline_settings.interpolate_RMP_extraction:
        interpolate_RMP(fitness_calculator_config)

    fitness_calculator_config = update_minimum_protocols_delay(
        access_point, fitness_calculator_config
    )

    access_point.store_fitness_calculator_configuration(fitness_calculator_config)

    return fitness_calculator_config
