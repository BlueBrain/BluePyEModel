"""Morphology utils."""


def get_apical_point_soma_distance(morph_path):
    """Get the euclidian distance between soma and apical point given a morphology.

    Args:
        morph_path (str): path to the morphology

    Returns:
        float: euclidian distance between soma and apical point
    """
    import morphio
    import morph_tool
    import neurom

    morphio_morph = morphio.Morphology(morph_path)
    apical_point = morph_tool.apical_point.apical_point_position(morphio_morph)
    return neurom.morphmath.point_dist(apical_point, morphio_morph.soma.center)

def get_apical_length(morph_path):
    """Returns the max radial distance of the apical dendrites."""
    import neurom

    neurom_morph = neurom.load_morphology(morph_path)
    return neurom.features.get("max_radial_distance", neurom_morph, neurite_type=neurom.APICAL_DENDRITE)

def get_basal_and_apical_lengths(morph_path):
    """Returns the max radial distance of the apical and basal dendrites."""
    import neurom

    neurom_morph = neurom.load_morphology(morph_path)
    basal_length = neurom.features.get("max_radial_distance", neurom_morph, neurite_type=neurom.BASAL_DENDRITE)
    apical_length = neurom.features.get("max_radial_distance", neurom_morph, neurite_type=neurom.APICAL_DENDRITE)
    return basal_length, apical_length

def get_hotspot_location(morph_path, hotspot_percent=20.):
    """Get hot spot begin and end in terms of soma distance.

    Calcium hot spot should be in distal apical trunk, and in primary and secondary tufts,
    i.e. around apical point, according to Larkum and Zhu, 2002.

    Attention! Apical point detection is not always accurate.

    Args:
        morph_path (str): path to the morphology
        hotspot_percent (float): percentage of the radial apical length that is in the hot spot.
            Here, we assume that the hotspot size is dependent on the apical length.
            20% is in accordance with experiments from Larkum and Zhu, 2002
    """
    ap_soma_dist = get_apical_point_soma_distance(morph_path)
    apical_length = get_apical_length(morph_path)
    hotspot_halfsize = apical_length * hotspot_percent / 2.0 / 100.0

    return max((0.0, ap_soma_dist - hotspot_halfsize)), ap_soma_dist + hotspot_halfsize
