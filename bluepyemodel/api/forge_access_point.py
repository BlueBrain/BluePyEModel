"""Neus Forge access point used by the Nexus API"""

import getpass
import logging
import pathlib

from entity_management.state import refresh_token
from kgforge.core import KnowledgeGraphForge

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
    ):

        self.limit = limit
        self.debug = debug
        self.cross_bucket = cross_bucket

        self.access_token = access_token
        if not self.access_token:
            self.access_token = self.get_access_token()

        bucket = organisation + "/" + project
        self.forge = self.connect_forge(bucket, endpoint, self.access_token, forge_path)

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

    def register(self, resource_description, filters_existance=None, force_replace=False):
        """Register a resource from its dictionary description."""

        if "type" not in resource_description:
            raise AccessPointException("The resource description should contain 'type'.")

        if filters_existance:

            previous_resources = self.fetch(filters_existance)

            if previous_resources:

                if not force_replace:
                    logger.warning(
                        "The resource you are trying to register already exist and will be ignored."
                    )
                    return

                for resource in previous_resources:
                    self.forge.deprecate(resource)

        self.forge.register(self.forge.from_json(resource_description))

    def retrieve(self, id_):
        """Retrieve a resource based on its id"""

        resource = self.forge.retrieve(id=id_, cross_bucket=self.cross_bucket)

        if resource:
            return resource

        logger.debug("Could not retrieve resource of id: %s", id_)

        return None

    def fetch(self, filters):
        """Fetch resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include "type" or "id".

        Returns:
            resources (list): list of resources
        """

        if "type" not in filters and "id" not in filters:
            raise AccessPointException("Search filters should contain either 'type' or 'id'.")

        resources = self.forge.search(
            filters, cross_bucket=self.cross_bucket, limit=self.limit, debug=self.debug
        )

        if resources:
            return resources

        logger.debug("No resources for filters: %s", filters)

        return None

    def fetch_one(self, filters):
        """Fetch one and only one resource based on filters."""

        resources = self.fetch(filters)

        if resources is None:
            raise AccessPointException("Could not get resource for filters %s" % filters)

        if len(resources) > 1:
            raise AccessPointException("More than one resource for filters %s" % filters)

        return resources[0]

    def download(self, resource_id, download_directory="./nexus_temp"):
        """Download datafile from nexus if it doesn't already exist."""

        resource = self.forge.retrieve(resource_id, cross_bucket=True)

        if resource is None:
            raise AccessPointException("Could not find resource for id: %s" % resource_id)

        filename = resource.distribution.name
        file_path = pathlib.Path(download_directory) / filename

        if not file_path.is_file():
            self.forge.download(resource, "distribution.contentUrl", download_directory)

        return str(file_path)

    def deprecate(self, filters):
        """Deprecate resources based on filters."""

        resources = self.fetch(filters)

        if resources:
            for resource in resources:
                self.forge.deprecate(resource)

    def resource_location(self, resource):
        """Get the path of the files attached to a resource. If the resource is
        not located on gpfs, download it instead"""

        paths = []

        if not hasattr(resource, "distribution"):
            raise AccessPointException("Resource %s does not have distribution" % resource)

        if isinstance(resource.distribution, list):
            distribution_iter = resource.distribution
        else:
            distribution_iter = [resource.distribution]

        for distrib in distribution_iter:

            if hasattr(distrib, "atLocation"):
                loc = self.forge.as_json(distrib.atLocation)
                paths.append(loc["location"].replace("file:/", ""))
            else:
                filepath = self.download(resource.id)
                paths.append(filepath)

        return paths
