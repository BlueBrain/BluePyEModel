"""E-model access_point module"""


def get_access_point(access_point, emodel, **kwargs):
    """Returns a DataAccessPoint object.

    Args:
        access_point (str): name of the access_point to use, can be 'nexus' or 'local'.
        emodel (str): name of the emodel.
        kwargs (dict): extra arguments to pass to access_point constructors, see below.

    Optional:
        etype (str): name of the electric type.
        ttype (str): name of the transcriptomic type.
        mtype (str): name of the morphology type.
        species (str): name of the species.
        brain_region (str): name of the brain location.
        iteration (str): tag associated to the current run.
        morph_class (str): name of the morphology class, has to be "PYR", "INT".
        synapse_class (str): name of the synapse class, has to be "EXC", "INH".
        layer (str): layer of the model.

    For local:
        emodel_dir (str): path of the directory containing the parameters,
            features and parameters config files.
        recipe_path (str, optional): path to the file containing the recipes.
        final_path (str, optional): path to the final.json, if different from the one in emodel_dir
        legacy_dir_structure (bool, optional): uses legacy folder structure
        with_seed (bool): allows for emodel_seed type of emodel names in final.json (not in recipes)

    For nexus:
        project (str): name of the Nexus project.
        organisation (str): name of the Nexus organization to which the project belong.
        endpoint (str): Nexus endpoint.
        forge_path (str): path to a .yml used as configuration by nexus-forge.
        access_token (str, optional): Nexus connection token.

    Returns:
        DataAccessPoint
    """

    etype = kwargs.get("etype", None)
    etype = etype.replace("__", " ") if etype else None

    ttype = kwargs.get("ttype", None)
    ttype = ttype.replace("__", " ") if ttype else None

    mtype = kwargs.get("mtype", None)
    mtype = mtype.replace("__", " ") if mtype else None

    brain_region = kwargs.get("brain_region", None)
    brain_region = brain_region.replace("__", " ") if brain_region else None

    if access_point == "nexus":
        from bluepyemodel.access_point.nexus import NexusAccessPoint

        return NexusAccessPoint(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=kwargs.get("species", None),
            brain_region=brain_region,
            iteration_tag=kwargs.get("iteration_tag", None),
            morph_class=kwargs.get("morph_class", None),
            synapse_class=kwargs.get("synapse_class", None),
            layer=kwargs.get("layer", None),
            project=kwargs.get("project", "ncmv3"),
            organisation=kwargs.get("organisation", "bbp"),
            endpoint=kwargs.get("endpoint", "https://bbp.epfl.ch/nexus/v1"),
            forge_path=kwargs.get("forge_path", None),
            access_token=kwargs.get("access_token", None),
        )

    if access_point == "local":
        from bluepyemodel.access_point.local import LocalAccessPoint

        return LocalAccessPoint(
            emodel=emodel,
            emodel_dir=kwargs["emodel_dir"],
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=kwargs.get("species", None),
            brain_region=brain_region,
            iteration_tag=kwargs.get("iteration_tag", None),
            morph_class=kwargs.get("morph_class", None),
            synapse_class=kwargs.get("synapse_class", None),
            layer=kwargs.get("layer", None),
            recipes_path=kwargs.get("recipes_path", None),
            final_path=kwargs.get("final_path", None),
            legacy_dir_structure=kwargs.get("legacy_dir_structure", False),
            with_seeds=kwargs.get("with_seeds", False),
        )

    raise Exception(f"Unknown access point: {access_point}")
