"""Efeatures extraction functions"""

import bluepyefe.extract


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

    threshold_nvalue_save = access_point.pipeline_settings.extraction_threshold_value_save
    plot = access_point.pipeline_settings.plot_extraction

    # extract features
    efeatures, stimuli, current = bluepyefe.extract.extract_efeatures(
        output_directory="./figures/efeatures_extraction/{emodel}",
        files_metadata=files_metadata,
        targets=targets,
        threshold_nvalue_save=threshold_nvalue_save,
        protocols_rheobase=protocols_threshold,
        recording_reader=None,
        map_function=mapper,
        write_files=False,
        plot=plot,
    )

    # store features & protocols
    access_point.store_efeatures(efeatures, current)
    access_point.store_protocols(stimuli)

    return efeatures, stimuli, current
