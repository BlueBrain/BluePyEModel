"""Efeatures extraction functions"""
import logging
import pathlib
from importlib.machinery import SourceFileLoader

import bluepyefe.extract

from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.tools.search_pdfs import search_figure_efeatures

logger = logging.getLogger(__name__)


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
        raise Exception("Extraction reader function is not callable nor a list of two strings")

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


def extract_save_features_protocols(
    access_point,
    emodel,
    mapper=map,
):
    """Extract the efeatures and saves the results as a configuration for the fitness calculator.

    Args:
        access_point (DataAccessPoint): object which contains API to access emodel data
        emodel (str): name of the emodel.
        mapper (map): mapper for parallel computations.
    """

    (files_metadata, targets, protocols_threshold) = access_point.get_extraction_metadata()

    if files_metadata is None or targets is None or protocols_threshold is None:
        raise Exception("Could not get the extraction metadata from the api.")

    reader_function = define_extraction_reader_function(access_point)

    threshold_nvalue_save = access_point.pipeline_settings.extraction_threshold_value_save
    plot = access_point.pipeline_settings.plot_extraction
    output_directory = f"./figures/{emodel}/efeatures_extraction/"

    efeatures, stimuli, current = bluepyefe.extract.extract_efeatures(
        output_directory=output_directory,
        files_metadata=files_metadata,
        targets=targets,
        threshold_nvalue_save=threshold_nvalue_save,
        protocols_rheobase=protocols_threshold,
        recording_reader=reader_function,
        map_function=mapper,
        write_files=False,
        plot=plot,
    )

    if plot:
        attach_efeatures_pdf(emodel, efeatures)

    fitness_calculator_config = FitnessCalculatorConfiguration(
        name_rmp_protocol=access_point.pipeline_settings.name_rmp_protocol,
        name_rin_protocol=access_point.pipeline_settings.name_Rin_protocol,
        threshold_efeature_std=access_point.pipeline_settings.threshold_efeature_std,
        validation_protocols=access_point.pipeline_settings.validation_protocols,
    )

    fitness_calculator_config.init_from_bluepyefe(efeatures, stimuli, current)

    access_point.store_fitness_calculator_configuration(fitness_calculator_config)

    return fitness_calculator_config
