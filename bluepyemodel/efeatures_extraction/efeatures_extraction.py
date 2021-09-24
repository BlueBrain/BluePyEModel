"""Efeatures extraction functions"""

import copy
import logging
import pathlib
from importlib.machinery import SourceFileLoader

import bluepyefe.extract
import numpy

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


def format_threshold_based_efeatures(
    protocols, features, currents, name_Rin_protocol, name_rmp_protocol
):
    """Used when performing threshold-based optimisation. The efeatures associated to Rin,
        RMP, holding current and threshold current are taken from their original protocols to be
        assigned to their respective special protocol instead

    Args:
        protocols (dict): description of each protocols
        features (dict): features name and values for each protocols
    """

    out_protocols = {}
    out_features = {}

    if name_rmp_protocol not in features and name_rmp_protocol != "all":
        raise Exception(
            f"The stimulus {name_rmp_protocol} requested for RMP "
            "computation couldn't be extracted from the ephys data."
        )
    if name_Rin_protocol not in features:
        raise Exception(
            f"The stimulus {name_Rin_protocol} requested for Rin "
            "computation couldn't be extracted from the ephys data."
        )

    out_features = {
        "SearchHoldingCurrent": {
            "soma.v": [
                {
                    "feature": "bpo_holding_current",
                    "val": currents["holding_current"],
                    "strict_stim": True,
                }
            ]
        },
        "SearchThresholdCurrent": {
            "soma.v": [
                {
                    "feature": "bpo_threshold_current",
                    "val": currents["threshold_current"],
                    "strict_stim": True,
                }
            ]
        },
    }

    for protocol in features:
        for efeat in features[protocol]["soma"]:

            if protocol == name_rmp_protocol and efeat["feature"] == "voltage_base":
                out_features["RMPProtocol"] = {
                    "soma.v": [
                        {
                            "feature": "steady_state_voltage_stimend",
                            "val": efeat["val"],
                            "strict_stim": True,
                        }
                    ]
                }
                if "pdfs" in efeat:
                    out_features["RMPProtocol"]["soma.v"][0]["pdfs"] = efeat["pdfs"]

            elif protocol == name_Rin_protocol and efeat["feature"] == "voltage_base":
                out_features["SearchHoldingCurrent"]["soma.v"].append(
                    {
                        "feature": "steady_state_voltage_stimend",
                        "val": efeat["val"],
                        "strict_stim": True,
                    }
                )
                if "pdfs" in efeat:
                    out_features["SearchHoldingCurrent"]["soma.v"][0]["pdfs"] = efeat["pdfs"]

            elif (
                protocol == name_Rin_protocol
                and efeat["feature"] == "ohmic_input_resistance_vb_ssse"
            ):
                out_features["RinProtocol"] = {"soma.v": [copy.copy(efeat)]}

            elif protocol not in [name_rmp_protocol, name_Rin_protocol]:
                if protocol not in out_features:
                    out_features[protocol] = {"soma.v": []}
                out_features[protocol]["soma.v"].append(efeat)

    for protocol in protocols:
        if protocol in out_features:
            out_protocols[protocol] = protocols[protocol]

    if name_rmp_protocol == "all":

        voltage_bases = []
        pdfs = []
        for protocol in features:
            for efeat in features[protocol]["soma"]:
                if efeat["feature"] == "voltage_base":
                    voltage_bases.append(efeat["val"])
                    if "pdfs" in efeat:
                        pdfs += efeat["pdfs"]

        if not voltage_bases:
            raise Exception("name_rmp_protocol is 'all' but no voltage_base were extracted.")

        voltage_bases = numpy.asarray(voltage_bases)

        out_features["RMPProtocol"] = {
            "soma.v": [
                {
                    "feature": "steady_state_voltage_stimend",
                    "val": [numpy.mean(voltage_bases[:, 0]), numpy.mean(voltage_bases[:, 1])],
                    "strict_stim": True,
                    "pdfs": pdfs,
                }
            ]
        }

    return out_protocols, out_features


def tag_validation(dict_, validation_protocols):
    """Mark the validation protocols and efeatures as such"""

    for protocol in dict_:

        if protocol in [
            "SearchHoldingCurrent",
            "SearchThresholdCurrent",
            "RinProtocol",
            "RMPProtocol",
        ]:
            continue

        ecode_name = str(protocol.split("_")[0])
        stimulus_target = float(protocol.split("_")[1])

        if ecode_name in validation_protocols:
            for target in validation_protocols[ecode_name]:
                if int(target) == int(stimulus_target):
                    dict_[protocol]["validation"] = True
                    break

        if "validation" not in dict_[protocol]:
            dict_[protocol]["validation"] = False


def extract_save_features_protocols(
    access_point,
    emodel,
    mapper=map,
):
    """

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

    # extract features
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

    # Reformat the features & protocols in case of threshold-based optimization
    if access_point.pipeline_settings.threshold_based_evaluator:
        stimuli, efeatures = format_threshold_based_efeatures(
            stimuli,
            efeatures,
            current,
            access_point.pipeline_settings.name_Rin_protocol,
            access_point.pipeline_settings.name_rmp_protocol,
        )

    tag_validation(efeatures, access_point.pipeline_settings.validation_protocols)
    tag_validation(stimuli, access_point.pipeline_settings.validation_protocols)

    # store features & protocols
    access_point.store_efeatures(efeatures)
    access_point.store_protocols(stimuli)

    return efeatures, stimuli, current
