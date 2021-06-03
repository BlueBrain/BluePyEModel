"""Emodel api module"""


def get_db(api, emodel, **kwargs):
    """Returns a DatabaseAPI object.

    Args:
        api (str): name of the api to use, can be nexus' or 'singlecell'.
        emodel (str): name of the emodel.
        kwargs (dict): extra arguments to pass to api constructors, see below.

    For singlecell:
        emodel_dir (str): path of the directory containing the parameters,
            features and parameters config files.
        recipe_path (str, optional): path to the file containing the recipes.
        final_path (str, optional): path to the final.json, if different from the one in emodel_dir
        legacy_dir_structure (bool, optional): uses legacy folder structure

    For nexus:
        species (str): name of the species.
        brain_region (str): name of the brain region.
        project (str): name of the Nexus project.
        organisation (str): name of the Nexus organization to which the project belong.
        endpoint (str): Nexus endpoint.
        forge_path (str): path to a .yml used as configuration by nexus-forge.
        ttype (str): name of the t-type.
        version_tag (str): tag associated to the current run. Used to tag the
            Resources generated during the different run.

    Returns:
        DatabaseAPI
    """

    if api == "nexus":
        from bluepyemodel.api.nexus import NexusAPI

        return NexusAPI(
            emodel=emodel,
            species=kwargs.get("species", "rat"),
            brain_region=kwargs.get("brain_region", None),
            project=kwargs.get("project", "emodel_pipeline"),
            organisation=kwargs.get("organisation", "demo"),
            endpoint=kwargs.get("endpoint", "https://bbp.epfl.ch/nexus/v1"),
            forge_path=kwargs.get("forge_path", None),
            ttype=kwargs.get("ttype", None),
            version_tag=kwargs.get("version_tag", None),
        )

    if api == "singlecell":
        from bluepyemodel.api.singlecell import SinglecellAPI

        return SinglecellAPI(
            emodel=emodel,
            emodel_dir=kwargs["emodel_dir"],
            recipes_path=kwargs.get("recipes_path", None),
            final_path=kwargs.get("final_path", None),
            legacy_dir_structure=kwargs.get("legacy_dir_structure", False),
            extract_config=kwargs.get("extract_config", None),
        )

    raise Exception(f"Unknown api: {api}")
