"""E-model access_point module"""


def get_access_point(access_point, emodel, **kwargs):
    """Returns a DataAccessPoint object.

    Args:
        access_point (str): name of the access_point to use, can be 'nexus' or 'local'.
        emodel (str): name of the emodel.
        kwargs (dict): extra arguments to pass to access_point constructors, see below.

    Optional:
        ttype (str): name of the t-type.
        iteration_tag (str): tag associated to the current run. Used to tag the
            Resources generated during the different run.

    For local:
        emodel_dir (str): path of the directory containing the parameters,
            features and parameters config files.
        recipe_path (str, optional): path to the file containing the recipes.
        final_path (str, optional): path to the final.json, if different from the one in emodel_dir
        legacy_dir_structure (bool, optional): uses legacy folder structure
        with_seed (bool): allows for emodel_seed type of emodel names in final.json (not in recipes)

    For nexus:
        species (str): name of the species.
        brain_region (str): name of the brain region.
        project (str): name of the Nexus project.
        organisation (str): name of the Nexus organization to which the project belong.
        endpoint (str): Nexus endpoint.
        forge_path (str): path to a .yml used as configuration by nexus-forge.

    Returns:
        DataAccessPoint
    """

    ttype = kwargs.get("ttype", None)
    if ttype:
        ttype = ttype.replace("__", " ")

    if access_point == "nexus":
        from bluepyemodel.access_point.nexus import NexusAccessPoint

        return NexusAccessPoint(
            emodel=emodel,
            species=kwargs.get("species", "rat"),
            brain_region=kwargs.get("brain_region", None),
            project=kwargs.get("project", "emodel_pipeline"),
            organisation=kwargs.get("organisation", "demo"),
            endpoint=kwargs.get("endpoint", "https://bbp.epfl.ch/nexus/v1"),
            forge_path=kwargs.get("forge_path", None),
            ttype=ttype,
            iteration_tag=kwargs.get("iteration_tag", None),
        )

    if access_point == "local":
        from bluepyemodel.access_point.local import LocalAccessPoint

        return LocalAccessPoint(
            emodel=emodel,
            emodel_dir=kwargs["emodel_dir"],
            recipes_path=kwargs.get("recipes_path", None),
            final_path=kwargs.get("final_path", None),
            legacy_dir_structure=kwargs.get("legacy_dir_structure", False),
            with_seeds=kwargs.get("with_seeds", False),
            ttype=ttype,
            iteration_tag=kwargs.get("iteration_tag", None),
        )

    raise Exception(f"Unknown access point: {access_point}")
