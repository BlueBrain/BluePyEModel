"""Nexus Forge access point used by the Nexus access point"""

import getpass
import logging
import pathlib

import jwt
from entity_management.state import refresh_token
from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource
from kgforge.specializations.resources import Dataset

logger = logging.getLogger("__main__")


# pylint: disable=bare-except


class AccessPointException(Exception):
    """For Exceptions related to the NexusForgeAccessPoint"""


class NexusForgeAccessPoint:
    """Access point to Nexus Knowledge Graph using Nexus Forge"""

    def __init__(
        self,
        project="emodel_pipeline",
        organisation="demo",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=None,
        limit=1000,
        debug=False,
        cross_bucket=True,
        access_token=None,
        iteration_tag=None,
    ):

        self.limit = limit
        self.debug = debug
        self.cross_bucket = cross_bucket

        self.access_token = access_token
        if not self.access_token:
            self.access_token = self.get_access_token()

        bucket = organisation + "/" + project
        self.forge = self.connect_forge(bucket, endpoint, self.access_token, forge_path)

        self.iteration_tag = iteration_tag

        decoded_token = jwt.decode(self.access_token, options={"verify_signature": False})
        self.agent = self.forge.reshape(
            self.forge.from_json(decoded_token), keep=["name", "email", "sub", "preferred_username"]
        )
        self.agent.id = decoded_token["sub"]
        self.agent.type = ["Person", "Agent"]

    @staticmethod
    def get_access_token():
        """Define access token either from bbp-workflow or provided by the user"""

        try:
            access_token = refresh_token()
        except:  # noqa
            logger.info("Please get your Nexus access token from https://bbp.epfl.ch/nexus/web/.")
            access_token = getpass.getpass()

        return access_token

    @staticmethod
    def connect_forge(bucket, endpoint, access_token, forge_path=None):
        """Creation of a forge session"""

        if not forge_path:
            forge_path = (
                "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
                + "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
            )

        forge = KnowledgeGraphForge(
            forge_path, bucket=bucket, endpoint=endpoint, token=access_token
        )

        return forge

    def add_contribution(self, resource):
        """Add the contributing agent to the resource"""

        if self.agent:

            if isinstance(resource, Dataset):
                resource.add_contribution(self.agent, versioned=False)
            elif isinstance(resource, Resource):
                resource.contribution = Resource(type="Contribution", agent=self.agent)

        return resource

    def register(self, resource_description, filters_existance=None, replace=False, tag=True):
        """Register a resource from its dictionary description."""

        if "type" not in resource_description:
            raise AccessPointException("The resource description should contain 'type'.")

        previous_resources = None
        if filters_existance:

            if tag and self.iteration_tag:
                filters_existance["iteration"] = self.iteration_tag

            previous_resources = self.fetch(filters_existance)

        if filters_existance and previous_resources:

            if replace:
                for resource in previous_resources:
                    self.forge.deprecate(resource)

            else:
                logger.warning(
                    "The resource you are trying to register already exist and will be ignored."
                )
                return

        if tag and self.iteration_tag:
            resource_description["iteration"] = self.iteration_tag

        resource_description["objectOfStudy"] = {
            "@id": "http://bbp.epfl.ch/neurosciencegraph/taxonomies/objectsofstudy/singlecells",
            "@type": "nsg:ObjectOfStudy",
            "label": "Single Cell",
        }

        logger.debug("Registering resources: %s", resource_description)

        resource = self.forge.from_json(resource_description)
        resource = self.add_contribution(resource)

        self.forge.register(resource)

    def retrieve(self, id_):
        """Retrieve a resource based on its id"""

        resource = self.forge.retrieve(id=id_, cross_bucket=self.cross_bucket)

        if resource:
            return resource

        logger.debug("Could not retrieve resource of id: %s", id_)

        return None

    def fetch(self, filters, use_version=True):
        """Fetch resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include "type" or "id".
            use_version (bool): if True, the search is restricted to the Resources that include
                the current version tag.

        Returns:
            resources (list): list of resources
        """

        if "type" not in filters and "id" not in filters:
            raise AccessPointException("Search filters should contain either 'type' or 'id'.")

        if use_version and self.iteration_tag:
            filters["iteration"] = self.iteration_tag

        logger.debug("Searching: %s", filters)
        resources = self.forge.search(
            filters, cross_bucket=self.cross_bucket, limit=self.limit, debug=self.debug
        )

        if resources:
            return resources

        logger.debug("No resources for filters: %s", filters)

        return None

    def fetch_one(self, filters, use_version=True, strict=True):
        """Fetch one and only one resource based on filters."""

        resources = self.fetch(filters, use_version=use_version)

        if resources is None:
            if strict:
                raise AccessPointException(f"Could not get resource for filters {filters}")
            return None

        if len(resources) > 1:
            if strict:
                raise AccessPointException(f"More than one resource for filters {filters}")
            return resources[0]

        return resources[0]

    def download(self, resource_id, download_directory="./nexus_temp"):
        """Download datafile from nexus if it doesn't already exist."""

        resource = self.forge.retrieve(resource_id, cross_bucket=True)

        if resource is None:
            raise AccessPointException(f"Could not find resource for id: {resource_id}")

        if isinstance(resource.distribution, list):
            filename = resource.distribution[0].name
        else:
            filename = resource.distribution.name

        if isinstance(resource.distribution, list):
            filename = resource.distribution[0].name
        else:
            filename = resource.distribution.name

        file_path = pathlib.Path(download_directory) / filename

        if not file_path.is_file():
            self.forge.download(resource, "distribution.contentUrl", download_directory)

        return str(file_path)

    def deprecate(self, filters, use_version=True):
        """Deprecate resources based on filters."""

        resources = self.fetch(filters, use_version=use_version)

        if resources:
            for resource in resources:
                self.forge.deprecate(resource)

    def resource_location(self, resource):
        """Get the path of the files attached to a resource. If the resource is
        not located on gpfs, download it instead"""

        paths = []

        if not hasattr(resource, "distribution"):
            raise AccessPointException(f"Resource {resource} does not have distribution")

        if isinstance(resource.distribution, list):
            distribution_iter = resource.distribution
        else:
            distribution_iter = [resource.distribution]

        for distrib in distribution_iter:

            filepath = None

            if hasattr(distrib, "atLocation"):
                loc = self.forge.as_json(distrib.atLocation)
                if "location" in loc:
                    filepath = loc["location"].replace("file:/", "")

            if filepath is None:
                filepath = self.download(resource.id)

            paths.append(filepath)

        return paths
