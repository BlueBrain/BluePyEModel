"""Luigi config classes."""
import luigi


class EmodelAPIConfig(luigi.Config):
    """Configuration of emodel api database."""

    api = luigi.Parameter(default="singlecell")

    # singlecell parameters
    emodel_dir = luigi.Parameter(default="./")
    species = luigi.Parameter(default="rat")
    recipes_path = luigi.Parameter(default=None)
    final_path = luigi.Parameter(default="./final.json")
    legacy_dir_structure = luigi.BoolParameter(default=False)
    extract_config = luigi.Parameter(default=None)

    # nexus parameters
    forge_path = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default="")
    nexus_poject = luigi.Parameter(default="emodel_pipeline")
    nexus_organisation = luigi.Parameter(default="demo")
    nexus_endpoint = luigi.Parameter(default="https://bbp.epfl.ch/nexus/v1")
    ttype = luigi.Parameter(default=None)
    nexus_version_tag = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.api == "singlecell":
            self.api_args = {
                "emodel_dir": self.emodel_dir,
                "recipes_path": self.recipes_path,
                "final_path": self.final_path,
                "legacy_dir_structure": self.legacy_dir_structure,
                "extract_config": self.extract_config,
            }

        if self.api == "nexus":
            self.api_args = {
                "brain_region": self.brain_region,
                "forge_path": self.forge_path,
                "species": self.species,
                "project": self.nexus_poject,
                "organisation": self.nexus_organisation,
                "endpoint": self.nexus_endpoint,
                "ttype": self.ttype,
                "version_tag": self.nexus_version_tag,
            }
