"""
This script allows to provide the channel related genes expression profiles
for all met-types.
"""

"""
Copyright 2024, EPFL/Blue Brain Project

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

import json

import numpy as np
import pandas as pd

# Resources

# PATH_TO_SC_RNA_SEQ_DATA = './input/medians.csv'
PATH_TO_SC_RNA_SEQ_DATA = "./input/Yao_et_al_trim_mean25.csv.20240325101358"
PATH_TO_SELECTED_CHANNELS = "./input/Channels_genes_(correspondance_channels)_v2.csv"
PATH_TO_BBP_M_TYPE_LIST = "./input/BBP_mtype_list.csv"
PATH_TO_INH_MAP_L1 = "./input/P(BBPmarker_metype)_L1_(Gouw+pseq_BBP)April_16_2021.csv"
PATH_TO_INH_MAP_L26 = "./input/P(BBPmarker_metype)_L26_(Gouw+pseq_BBP)April_16_2021.csv"

IC_CONSTRAINTS = pd.read_csv(
    "./Ion_Channel_Constraints-Area_and_Conductance.csv", index_col=["Gene name"]
)


def count_elements(array):
    """
    Return as and pandas DataFrame unique elements and the associated counts of an array.
    :param array: array of elements
    :return: panda DataFrame
    """
    unq = np.unique(array)

    return pd.DataFrame(
        [len(array[[y == x for y in array]]) for x in unq], index=unq, columns=["counts"]
    )


def generate_panda(data, t_type):
    """
    Generate a panda DataFrame for a given t-type from the dictionary of channel expression""
    :param data: dictionary of channel expression with t-types as keys
    :param t_type: string, t-type name
    :return: panda DataFrame
    """
    data_df = pd.DataFrame(
        data[t_type]["values"], index=data[t_type]["index"], columns=data[t_type]["columns"]
    )
    return data_df


def preprocess_df(data_df):
    """
    Split scRNAseq data into major classes of neuronal cells
    :param data_df: panda DataFrame
    :return: panda DataFrames for all neuronal cells, inhibitory neurons, excitatory neurons
    and non-neuronal cells
    """
    # dict_class = {}
    # for x in data_df.index:
    #     if (
    #         ("Lamp5" in x)
    #         | ("Sncg" in x)
    #         | ("Serpinf1" in x)
    #         | ("Vip" in x)
    #         | ("Sst" in x)
    #         | ("Pvalb" in x)
    #     ):
    #         dict_class[x] = "GABAergic"
    #     elif (
    #         ("IT" in x)
    #         | ("Car3" in x)
    #         | ("ET" in x)
    #         | ("NP" in x)
    #         | ("CT" in x)
    #         | ("L6b" in x)
    #         | ("Ly6g6e" in x)
    #     ):
    #         dict_class[x] = "Glutamatergic"
    #     elif "CA" in x:
    #         dict_class[x] = "Hippocampus"
    #     else:
    #         dict_class[x] = "Non-neuronal"

    dict_class = {}
    for x in data_df.index:
        if "Gaba" in x:
            dict_class[x] = "GABAergic"
        elif "Glut" in x:
            dict_class[x] = "Glutamatergic"
        elif (
            ("Chol" in x)
            | ("Dopa" in x)
            | ("Sero" in x)
            | ("Gly" in x)
            | ("Glyc" in x)
            | ("Hist" in x)
        ):
            dict_class[x] = "modulatory"
        else:
            dict_class[x] = "Non-neuronal"

    msk_0 = [
        (c == "Glutamatergic") | (c == "GABAergic") for c in data_df.rename(index=dict_class).index
    ]
    msk_1 = [c == "GABAergic" for c in data_df.rename(index=dict_class).index]
    msk_2 = [c == "Glutamatergic" for c in data_df.rename(index=dict_class).index]
    msk_3 = [c == "Non-neuronal" for c in data_df.rename(index=dict_class).index]

    data_nrn_df = data_df[msk_0]
    data_inh_df = data_df[msk_1]
    data_exc_df = data_df[msk_2]
    data_non_nrn_df = data_df[msk_3]

    return data_nrn_df, data_inh_df, data_exc_df, data_non_nrn_df


def make_binary(data_df, thresholding="zero"):
    """
    Make scRNAseq data (i.e. RNA counts) binary by applying a threshold ('zero' or '1_percent')
    :param data_df: panda Data Frame of single cell RNA seq data
    :param thresholding: {'zero', '1_percent'}, default='zero'
    :return: panda DataFrame, gene as columns and t-type as index.
    # TO DO: Implement gaussian thresh
    """
    binary = []

    for t_type in data_df.index:

        channels_expr_ = data_df.T[t_type]

        if thresholding == "1_percent":
            threshold = 0.01 * np.sum(channels_expr_.values)
        else:
            threshold = 0.0

        binary.append(np.where(channels_expr_ > threshold, 1, 0))

    binary_df = pd.DataFrame(np.asarray(binary), index=data_df.index, columns=data_df.columns)
    return binary_df


def compute_default_distribution_file(binary_df):
    """
    Compute a panda DataFrame of the same shape than binary_df (gene as columns and t-type as index)
    with 'uniform' string as values.
    :param binary_df: panda DataFrame
    :return: panda DataFrame
    """
    distribution_df = pd.DataFrame(
        np.asarray([["uniform"] * len(binary_df.columns)] * len(binary_df.index)),
        columns=binary_df.columns,
        index=binary_df.index,
    )
    return distribution_df


def compute_default_g_bar_values(binary_df, ion_channels_constraints):
    """
    Compute a panda DataFrame of the same shape than binary_df
    (gene as columns and t-type as index) with default maximal g_bar values (when available).
    Values taken from Darshan Mandge (Cells team).
    :param binary_df: panda DataFrame
    :param ion_channels_constraints: panda DataFrame
    :return: panda DataFrame
    """
    g_bar_vals = []

    for gene in binary_df.columns:
        try:
            if ion_channels_constraints["Maximal Conductance (S/cm²)"][gene.upper()]:
                v_ = ion_channels_constraints["Maximal Conductance (S/cm²)"][gene.upper()]
                g_bar_vals.append(float(v_.replace(",", ".")))
        except KeyError:
            if "Scn" in gene:
                g_bar_vals.append(0.5)
            elif "Kcn" in gene:
                g_bar_vals.append(0.005)
            elif "Hcn" in gene:
                g_bar_vals.append(1e-4)
            else:
                g_bar_vals.append(np.nan)

    g_bar_df = pd.DataFrame(
        np.asarray([g_bar_vals] * len(binary_df.index)),
        columns=binary_df.columns,
        index=binary_df.index,
    )

    return g_bar_df


def compute_ic_data(binary_df, distribution_df, g_bar_df):
    """
    Compute a panda dictionary merging presence, distribution and
    max g_bar values of ion channels for all t-types in somatic, dendritic and axonic sections.
    :param binary_df: panda DataFrame
    :param distribution_df: panda DataFrame
    :param g_bar_df: panda DataFrame
    :return: panda DataFrame
    """
    ic_data = {}
    for t_type in binary_df.index:
        res_df = pd.concat(
            [
                binary_df.T[t_type],
                distribution_df.T[t_type],
                distribution_df.T[t_type],
                distribution_df.T[t_type],
                g_bar_df.T[t_type],
            ],
            axis=1,
        )
        res_df.columns = [
            "presence",
            "dendrites distribution",
            "axon distribution",
            "soma distribution",
            "g_bar_max",
        ]
        ic_data[t_type] = {
            "values": res_df.values.tolist(),
            "index": res_df.index.tolist(),
            "columns": res_df.columns.tolist(),
        }

    return ic_data


def compute_default_exc_map(path_to_bbp_m_type_list, t_type_list):
    """
    Compute binary map between excitatory t-types and excitatory BBP me-types.
    Mapping based on layers (e.g. all layer 2/3 t-types are mapped with all me-type from layer 2/3)
    :param path_to_bbp_m_type_list: string
    :param t_type_list: list of t-types names as strings
    :return: dictionary
    """
    bbp_m_types = pd.read_csv(path_to_bbp_m_type_list, index_col=0)["m-type"].values
    msk_exc_bbp = np.asarray(
        [
            ("TPC" in x) | ("BPC" in x) | ("UPC" in x) | ("IPC" in x) | ("SSC" in x) | ("HPC" in x)
            for x in bbp_m_types
        ]
    )

    lay_list_dict = {
        "L2": ["L2/3"],
        "L23": ["L2/3"],
        "L3": ["L2/3"],
        "L4": ["L4", "L4/5"],
        "L5": ["L4/5", "L5", "NP"],
        "L6": ["L5/6", "L6"],
    }

    map_exc_t_type = {}

    for morpho_type in bbp_m_types[msk_exc_bbp]:
        lay = morpho_type.split("_")[0]
        t_type_match = []
        for lay_el in lay_list_dict[lay]:
            msk_tmp = [lay_el in x for x in t_type_list]
            t_type_match += np.asarray(t_type_list)[msk_tmp].tolist()
        t_type_match = np.unique(np.asarray(t_type_match))
        map_exc_t_type[morpho_type] = t_type_match.tolist()

    return map_exc_t_type


def make_inh_map_binary(path_to_inh_map_l1, path_to_inh_map_l26):
    """
    Convert probabilistic mapping  between inhibitory t-types and inhibitory me-types
    into binary map by applying a zero threshold.
    Original probabilistic mapping  was obtained from the "cross-species mapping" pipeline
    :param path_to_inh_map_l1: string
    :param path_to_inh_map_l26: string
    :return: dictionary
    """
    map_inh_t_type_l1 = pd.read_csv(path_to_inh_map_l1, index_col=0)
    map_inh_t_type_l26 = pd.read_csv(path_to_inh_map_l26, index_col=0)
    map_inh_t_type_l26 = map_inh_t_type_l26.reindex(
        ["Vip", "Lamp5", "Pvalb", "Sst", "Sncg", "Serpinf1"]
    )

    map_inh_t_type = pd.concat([map_inh_t_type_l1, map_inh_t_type_l26], axis=1, sort=True).fillna(0)
    map_inh_t_type_binary = pd.DataFrame(
        np.where(map_inh_t_type > 0.0, 1, 0),
        index=map_inh_t_type.index,
        columns=map_inh_t_type.columns,
    )
    return map_inh_t_type_binary


def combine_exc_inh_data(ic_data_exc, ic_data_inh):
    """
    Combine data from excitatory and inhibitory neurons in a single panda DataFrame.
    :param ic_data_exc: dictionary
    :param ic_data_inh: dictionary
    :return: panda DataFrame
    """
    level_1 = []
    level_2 = []
    level_3 = []
    df_list = []

    for x in ic_data_exc.keys():
        for y in ic_data_exc[x].keys():
            level_1 += [x + "_cADpyr"] * len(
                ic_data_exc[x][y]["columns"]
            )  # Same e-type for all excitatory neurons: cADpyr
            level_2 += [y] * len(ic_data_exc[x][y]["columns"])
            level_3 += ic_data_exc[x][y]["columns"]
            expression_df = generate_panda(ic_data_exc[x], y)
            df_list.append(expression_df)

    for x in ic_data_inh.keys():
        for y in ic_data_inh[x].keys():
            level_1 += [x] * len(ic_data_inh[x][y]["columns"])
            level_2 += [y] * len(ic_data_inh[x][y]["columns"])
            level_3 += ic_data_inh[x][y]["columns"]
            expression_df = generate_panda(ic_data_inh[x], y)
            df_list.append(expression_df)

    arrays = [level_1, level_2, level_3]
    idx = pd.MultiIndex.from_arrays(arrays, names=("me-type", "t-type", "modality"))

    res = pd.DataFrame(pd.concat(df_list, axis=1).T.values, columns=expression_df.index, index=idx)

    return res


if __name__ == "__main__":

    medians = pd.read_csv(PATH_TO_SC_RNA_SEQ_DATA, index_col=0)
    Channels_genes = pd.read_csv(PATH_TO_SELECTED_CHANNELS, index_col="gene_symbol")
    Channels_genes = Channels_genes.rename({"Scn2a1": "Scn2a", "Clcn4-2": "Clcn4"}, axis=0).drop(
        "Clca4c-ps", axis=0
    )
    msk_genes = [g in Channels_genes.index.tolist() for g in medians.index]
    expr_df = medians.T[medians.index[msk_genes]]
    expr_nrn_df = preprocess_df(expr_df)[0]
    Binary_0thresh_df = make_binary(expr_nrn_df)
    Distribution_df = compute_default_distribution_file(Binary_0thresh_df)
    g_bars_df = compute_default_g_bar_values(Binary_0thresh_df, IC_CONSTRAINTS)
    IC_data = compute_ic_data(Binary_0thresh_df, Distribution_df, g_bars_df)

    with open("./output/t_types_IC_expression.json", "w") as fp:
        json.dump(IC_data, fp)

    ttype_list = list(IC_data)
    subclass_list = np.unique([x.split("_")[0] for x in IC_data])

    map_exc_ttype = compute_default_exc_map(PATH_TO_BBP_M_TYPE_LIST, ttype_list)

    with open("./output/map_exc_ttype.json", "w") as fp:
        json.dump(map_exc_ttype, fp)

    compatible_profiles = {}
    for m_type, ttypes in map_exc_ttype.items():
        compatible_profiles[m_type] = {}
        for ttype in ttypes:
            df = generate_panda(IC_data, ttype)
            compatible_profiles[m_type][ttype] = {
                "values": df.values.tolist(),
                "index": df.index.tolist(),
                "columns": df.columns.tolist(),
            }

    with open("./output/exc_compatible_profiles.json", "w") as fp:
        json.dump(compatible_profiles, fp)

    map_inh_ttype_binary = make_inh_map_binary(PATH_TO_INH_MAP_L1, PATH_TO_INH_MAP_L26)

    msk_clstr_Lamp5 = [("Lamp5" in c) for c in medians.columns]
    msk_clstr_Sncg = [("Sncg" in c) for c in medians.columns]
    msk_clstr_Serpinf = [("Serpinf" in c) for c in medians.columns]
    msk_clstr_Vip = [("Vip" in c) for c in medians.columns]
    msk_clstr_Sst = [("Sst" in c) for c in medians.columns]
    msk_clstr_Pvalb = [("Pvalb" in c) for c in medians.columns]

    dict_mkr_ttype = {
        "Lamp5": medians.columns[msk_clstr_Lamp5],
        "Pvalb": medians.columns[msk_clstr_Pvalb],
        "Serpinf1": medians.columns[msk_clstr_Serpinf],
        "Sncg": medians.columns[msk_clstr_Sncg],
        "Sst": medians.columns[msk_clstr_Sst],
        "Vip": medians.columns[msk_clstr_Vip],
    }

    compatible_inh_profiles = {}

    for me_type in map_inh_ttype_binary.columns:
        compatible_inh_profiles[me_type] = {}
        compatible_markers = map_inh_ttype_binary.index[
            np.where(map_inh_ttype_binary[me_type] == 1)[0]
        ]

        compatible_ttype = []
        for mkr in compatible_markers:
            compatible_ttype += dict_mkr_ttype[mkr].tolist()

        for ttype in compatible_ttype:
            df = generate_panda(IC_data, ttype)
            compatible_inh_profiles[me_type][ttype] = {
                "values": df.values.tolist(),
                "index": df.index.tolist(),
                "columns": df.columns.tolist(),
            }

    with open("./output/inh_compatible_profiles.json", "w") as fp:
        json.dump(compatible_inh_profiles, fp)

    combined_data = combine_exc_inh_data(compatible_profiles, compatible_inh_profiles)
    combined_data.to_csv("./output/met_type_ion_channel_gene_expression.csv")
