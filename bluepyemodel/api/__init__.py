"""Emodel api module"""


def get_db(api, **kwargs):
    """Returns a DatabaseAPI object.

    Args:
        api (str): name of the api to use, can be 'sql', 'nexus' or 'singlecell'.
        kwargs (dict): extra arguments to pass to api constructors, see below.

    For singlecell:
        emodel_dir (str): path of the directory containing the parameters,
            features and parameters config files.
        recipe_path (str, optional): path to the file containing the recipes.
        final_path (str, optional): path to the final.json, if different from the one in emodel_dir
        legacy_dir_structure (bool, optional): uses legacy folder structure

    For sql:
        project_name (str): name of the project. Used as prefix to create the tables
            of the postgreSQL database.

    For nexus:
        forge_path (str): path to nexus forge project

    Returns:
        DatabaseAPI

    """
    if api == "sql":
        from bluepyemodel.api.postgreSQL import PostgreSQL_API

        return PostgreSQL_API(project_name=kwargs["project_name"])

    if api == "nexus":
        from bluepyemodel.api.nexus import Nexus_API

        return Nexus_API(kwargs["forge_path"])

    if api == "singlecell":
        from bluepyemodel.api.singlecell import Singlecell_API

        return Singlecell_API(
            emodel_dir=kwargs["emodel_dir"],
            recipes_path=kwargs.get("recipes_path", None),
            final_path=kwargs.get("final_path", None),
            legacy_dir_structure=kwargs.get("legacy_dir_structure", False),
        )

    raise Exception(f"Unknown api: {api}")
