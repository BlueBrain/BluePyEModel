"""Data utils"""

import csv
import pkg_resources
import numpy as np


def get_dendritic_data_filepath(data_type):
    """Gets a dendritic data file path.
    
    Args:
        data_type (str): can be 'ISI_CV' or 'rheobase'

    Raises:
        ValueError if data_type is not 'ISI_CV' nor 'rheobase'
    """
    if data_type == "ISI_CV":
        return pkg_resources.resource_filename(__name__, "ISI_CV_Shai2015.csv")
    if data_type == "rheobase":
        return pkg_resources.resource_filename(
            __name__, "spike_rheobase_pA_BeaulieuLaroche2021.csv"
        )
    raise ValueError(f"read_data expects 'ISI_CV' or 'rheobase' but got {data_type}")


def read_dendritic_data(data_type):
    """Reads a dendritic data file and returns distance and data values.

    rheobase values are returned in nA.
    
    Args:
        data_type (str): can be 'ISI_CV' or 'rheobase'
    """
    file_path = get_dendritic_data_filepath(data_type)

    data = []
    distances = []
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for distance, dat in csvreader:
            distances.append(float(distance))
            data.append(float(dat))

    if data_type == "rheobase":
        data = np.asarray(data) / 1000. # pA -> nA
    return np.asarray(distances), data