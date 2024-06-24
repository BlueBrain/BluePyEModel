"""Morphology utils."""

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


def get_apical_point_soma_distance(morph_path):
    """Get the euclidian distance between soma and apical point given a morphology.

    Args:
        morph_path (str): path to the morphology

    Returns:
        float: euclidian distance between soma and apical point
    """
    import morph_tool
    import morphio
    import neurom

    morphio_morph = morphio.Morphology(morph_path)
    apical_point = morph_tool.apical_point.apical_point_position(morphio_morph)
    return neurom.morphmath.point_dist(apical_point, morphio_morph.soma.center)


def get_apical_max_radial_distance(morph_path):
    """Returns the max radial distance of the apical dendrites."""
    import neurom

    neurom_morph = neurom.load_morphology(morph_path)
    return neurom.features.get(
        "max_radial_distance", neurom_morph, neurite_type=neurom.APICAL_DENDRITE
    )


def get_basal_and_apical_max_radial_distances(morph_path):
    """Returns the max radial distances of the apical and basal dendrites."""
    import neurom

    neurom_morph = neurom.load_morphology(morph_path)
    basal_length = neurom.features.get(
        "max_radial_distance", neurom_morph, neurite_type=neurom.BASAL_DENDRITE
    )
    apical_length = neurom.features.get(
        "max_radial_distance", neurom_morph, neurite_type=neurom.APICAL_DENDRITE
    )
    return basal_length, apical_length


def get_hotspot_location(morph_path, hotspot_percent=20.0):
    """Get hot spot begin and end in terms of soma distance.

    Calcium hot spot should be in distal apical trunk, and in primary and secondary tufts,
    i.e. around apical point, according to Larkum and Zhu, 2002.

    Attention! Apical point detection is not always accurate.

    Args:
        morph_path (str): path to the morphology
        hotspot_percent (float): percentage of the radial apical distance that is in the hot spot.
            Here, we assume that the hotspot size is dependent on the apical radial distance.
            20% is in accordance with experiments from Larkum and Zhu, 2002
    """
    ap_soma_dist = get_apical_point_soma_distance(morph_path)
    apical_length = get_apical_max_radial_distance(morph_path)
    hotspot_halfsize = apical_length * hotspot_percent / 2.0 / 100.0

    return max((0.0, ap_soma_dist - hotspot_halfsize)), ap_soma_dist + hotspot_halfsize
