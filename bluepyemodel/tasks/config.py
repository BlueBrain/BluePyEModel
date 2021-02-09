"""Luigi config classes."""
import luigi


class EmodelAPIConfig(luigi.Config):
    """Configuration of emodel api database."""

    api = luigi.Parameter(default="singlecell")

    # singlecell parameters
    emodel_dir = luigi.Parameter(default=None)
    recipes_path = luigi.Parameter(default=None)
    final_path = luigi.Parameter(default=None)
    legacy_dir_structure = luigi.BoolParameter(default=False)
    extract_config = luigi.Parameter(default=None)

    # sql parameters
    project_name = luigi.Parameter(default=None)

    # nexus parameters
    forge_path = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.api == "singlecell":
            if self.emodel_dir is None:
                raise Exception("Please provide at least emodel_dir with singlcell api")
            self.api_args = {
                "emodel_dir": self.emodel_dir,
                "recipes_path": self.recipes_path,
                "final_path": self.final_path,
                "legacy_dir_structure": self.legacy_dir_structure,
                "extract_config": self.extract_config,
            }

        if self.api == "sql":
            if self.project_name is None:
                raise Exception("Please provide at least project_name with sql api")
            self.api_args = {"project_name": self.project_name}

        if self.api == "nexus":
            if self.forge_path is None:
                raise Exception("Please provide at least forge_path with sql api")
            self.api_args = {"forge_path": self.forge_path}
