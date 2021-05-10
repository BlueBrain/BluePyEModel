"""Efeatures extraction functions"""

import bluepyefe.extract

# pylint: disable=unused-argument


def extract_save_features_protocols(
    emodel_db,
    emodel,
    files_metadata=None,
    targets=None,
    protocols_threshold=None,
    threshold_nvalue_save=1,
    mapper=map,
    name_Rin_protocol=None,
    name_rmp_protocol=None,
    validation_protocols=None,
    plot=False,
):
    """

    Args:
        emodel_db (DatabaseAPI): object which contains API to access emodel data
        emodel (str): name of the emodel.
        files_metadata (dict): define for which cell and protocol each file
            has to be used.
        targets (dict): define the efeatures to extract for each protocols
            and the amplitude around which these features should be
            averaged.
        protocols_threshold (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        threshold_nvalue_save (int): lower bounds of the number of values required
            to save an efeature.
        mapper (map): mapper for parallel computations.
        name_Rin_protocol (str): name of the protocol that should be used to compute
            the input resistance. Only used when db_api is 'singlecell'
        name_rmp_protocol (str): name of the protocol that should be used to compute
            the resting membrane potential. Only used when db_api is 'singlecell'.
        validation_protocols (dict): Of the form {"ecodename": [targets]}. Only used
            when db_api is 'singlecell'.
        plot (bool): True to plot the efeatures and the traces.
    """

    if files_metadata is None or targets is None or protocols_threshold is None:

        (
            files_metadata,
            targets,
            protocols_threshold,
        ) = emodel_db.get_extraction_metadata()

        if files_metadata is None or targets is None or protocols_threshold is None:
            raise Exception("Could not get the extraction metadata from the api.")

    # extract features
    efeatures, stimuli, current = bluepyefe.extract.extract_efeatures(
        output_directory=emodel,
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
    emodel_db.store_efeatures(
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    )
    emodel_db.store_protocols(stimuli, validation_protocols)

    return efeatures, stimuli, current
