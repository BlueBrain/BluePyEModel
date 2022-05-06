"""Nexus Forge access point used by the Nexus access point"""

import getpass
import logging
import pathlib

import jwt
from entity_management.state import refresh_token
from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource
from kgforge.core.commons.strategies import ResolvingStrategy
from kgforge.specializations.resources import Dataset

from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.model.distribution_configuration import DistributionConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.tools.utils import yesno

logger = logging.getLogger("__main__")


# pylint: disable=bare-except,consider-iterating-dictionary

CLASS_TO_NEXUS_TYPE = {
    "TargetsConfiguration": "ExtractionTargetsConfiguration",
    "EModelPipelineSettings": "EModelPipelineSettings",
    "FitnessCalculatorConfiguration": "FitnessCalculatorConfiguration",
    "NeuronModelConfiguration": "EModelConfiguration",
    "EModel": "EModel",
    "DistributionConfiguration": "EModelChannelDistribution",
    "EModelWorkflow": "EModelWorkflow",
}

NEXUS_TYPE_TO_CLASS = {
    "ExtractionTargetsConfiguration": TargetsConfiguration,
    "EModelPipelineSettings": EModelPipelineSettings,
    "FitnessCalculatorConfiguration": FitnessCalculatorConfiguration,
    "EModelConfiguration": NeuronModelConfiguration,
    "EModel": EModel,
    "EModelChannelDistribution": DistributionConfiguration,
    "EModelWorkflow": EModelWorkflow,
}

NEXUS_ENTRIES = ["objectOfStudy", "contribution", "type", "id", "distribution", "@type"]


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
        limit=5000,
        debug=False,
        cross_bucket=True,
        access_token=None,
        search_endpoint="sparql",
    ):

        self.limit = limit
        self.debug = debug
        self.cross_bucket = cross_bucket
        self.search_endpoint = search_endpoint

        self.access_token = access_token
        if not self.access_token:
            self.access_token = self.get_access_token()

        bucket = organisation + "/" + project
        self.forge = self.connect_forge(bucket, endpoint, self.access_token, forge_path)

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

    def resolve(self, text, scope="ontology", strategy="all", limit=1):
        """Resolves a string to find the matching ontology"""

        if strategy == "all":
            resolving_strategy = ResolvingStrategy.ALL_MATCHES
        elif strategy == "best":
            resolving_strategy = ResolvingStrategy.BEST_MATCH
        elif strategy == "exact":
            resolving_strategy = ResolvingStrategy.EXACT_MATCH
        else:
            raise Exception(
                f"Resolving strategy {strategy} does not exist. "
                "Strategy should be 'all', 'best' or 'exact'"
            )

        return self.forge.resolve(text, scope=scope, strategy=resolving_strategy, limit=limit)

    def register(
        self,
        resource_description,
        filters_existance=None,
        replace=False,
        distributions=None,
    ):
        """Register a resource from its dictionary description."""

        if "type" not in resource_description:
            raise AccessPointException("The resource description should contain 'type'.")

        previous_resources = None
        if filters_existance:

            previous_resources = self.fetch(filters_existance)

        if filters_existance and previous_resources:

            if replace:
                for resource in previous_resources:
                    rr = self.retrieve(resource.id)
                    self.forge.deprecate(rr)

            else:
                logger.warning(
                    "The resource you are trying to register already exist and will be ignored."
                )
                return

        resource_description["objectOfStudy"] = {
            "@id": "http://bbp.epfl.ch/neurosciencegraph/taxonomies/objectsofstudy/singlecells",
            "@type": "nsg:ObjectOfStudy",
            "label": "Single Cell",
        }

        logger.debug("Registering resources: %s", resource_description)

        resource = self.forge.from_json(resource_description)
        resource = self.add_contribution(resource)

        if distributions:
            resource = Dataset.from_resource(self.forge, resource)
            for path in distributions:
                resource.add_distribution(path)

        self.forge.register(resource)

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

        logger.debug("Searching: %s", filters)

        resources = self.forge.search(
            filters,
            cross_bucket=self.cross_bucket,
            limit=self.limit,
            debug=self.debug,
            search_endpoint=self.search_endpoint,
        )

        if resources:
            return resources

        logger.debug("No resources for filters: %s", filters)

        return None

    def fetch_one(self, filters, strict=True):
        """Fetch one and only one resource based on filters."""

        resources = self.fetch(filters)

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

        file_path = pathlib.Path(download_directory) / filename

        if not file_path.is_file():
            self.forge.download(resource, "distribution.contentUrl", download_directory)

        return str(file_path)

    def deprecate(self, filters):  # TODO: THATS VERY DANGEROUS TO REWORK
        """Deprecate resources based on filters."""

        resources = self.fetch(filters)

        if resources:
            for resource in resources:
                rr = self.retrieve(resource.id)
                self.forge.deprecate(rr)

    def deprecate_all(self, metadata):
        """Deprecate all resources used or produced by BluePyModel. Use with extreme caution."""

        if not yesno("Confirm deprecation of all BluePyEmodel resources in Nexus project"):
            return

        for type_ in NEXUS_TYPE_TO_CLASS.keys():

            filters = {"type": type_}
            filters.update(metadata)

            self.deprecate(filters)

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

    def object_to_nexus(self, object_, metadata, replace=True):
        """Transform a BPEM object into a dict which gets registered into Nexus as
        a Resource of the matching type. The metadata are also attached to the object
        to be able to retrieve the Resource."""

        type_ = CLASS_TO_NEXUS_TYPE[object_.__class__.__name__]

        base_payload = {"type": ["Entity", type_]}
        payload_existance = {"type": type_}

        base_payload.update(metadata)
        payload_existance.update(metadata)

        payload = {**base_payload, **object_.as_dict()}

        distributions = payload.pop("nexus_distributions", None)

        self.register(
            payload,
            filters_existance=payload_existance,
            replace=replace,
            distributions=distributions,
        )

    def resource_to_object(self, type_, resource, metadata):
        """Transform a Resource into a BPEM object of the matching type"""

        payload = self.forge.as_json(resource)

        for k in metadata:
            payload.pop(k, None)

        for k in NEXUS_ENTRIES:
            payload.pop(k, None)

        return NEXUS_TYPE_TO_CLASS[type_](**payload)

    def nexus_to_object(self, type_, metadata):
        """Search for a single Resource matching the type_ and metadata and return it
        as a BPEM object of the matching type"""

        filters = {"type": type_}
        filters.update(metadata)

        resource = self.fetch_one(filters)

        return self.resource_to_object(type_, resource, metadata)

    def nexus_to_objects(self, type_, metadata):
        """Search for Resources matching the type_ and metadata and return them
        as BPEM objects of the matching type"""

        filters = {"type": type_}
        filters.update(metadata)

        resources = self.fetch(filters)

        objects_ = []

        if resources:
            for resource in resources:
                objects_.append(self.resource_to_object(type_, resource, metadata))

        return objects_

    def get_nexus_id(self, type_, metadata):
        """Search for a single Resource matching the type_ and metadata and return its id"""
        filters = {"type": type_}
        filters.update(metadata)

        resource = self.fetch_one(filters)

        return resource.id


def ontology_forge_access_point(access_token=None):
    """Returns an access point targeting the project containing the ontology for the
    species and brain regions"""

    access_point = NexusForgeAccessPoint(
        project="datamodels",
        organisation="neurosciencegraph",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=None,
        access_token=access_token,
    )

    return access_point


def get_all_brain_regions(access_token=None):
    """Returns a list of all the brain regions available"""

    access_point = ontology_forge_access_point(access_token)

    filters = {
        "isDefinedBy": {"id": "http://bbp.epfl.ch/neurosciencegraph/ontologies/mba"},
        "type": "Class",
    }

    resources = access_point.forge.search(filters, limit=10000, cross_bucket=True)

    return sorted(set(r.label for r in resources))


def raise_brain_region_exception(base_text, brain_region, access_point):
    """Raise an exception mentioning the possible brain region names available on nexus

    Arguments:
        base_text (str): text to display in the Exception
        brain_region (str): name of the brain region to search for
        access_point (NexusForgeAccessPoint)
    """
    if not base_text.endswith("."):
        base_text = f"{base_text}."
    resources = access_point.resolve(brain_region, strategy="all", limit=20)
    if resources is None:
        raise AccessPointException(base_text)

    # make sure that resources is iterable
    if not isinstance(resources, list):
        resources = [resources]
    brain_regions = "\n".join(
        set(
            r.label
            for r in resources
            if hasattr(r, "subClassOf") and r.subClassOf == "nsg:BrainRegion"
        )
    )
    raise AccessPointException(f"{base_text} Maybe you meant one of those:\n{brain_regions}")


def get_brain_region(brain_region, access_token=None):
    """Returns a json dict of the resource corresponding to the brain region

    If the brain region name is not present in nexus,
    raise an exception mentioning the possible brain region names available on nexus

    Arguments:
        brain_region (str): name of the brain region to search for
        access_token (str): nexus connection token

    Returns:
        the nexus resource of the brain region as a json dict
    """

    access_point = ontology_forge_access_point(access_token)

    if brain_region in ["SSCX", "sscx"]:
        brain_region = "somatosensory areas"

    resource = access_point.resolve(brain_region, strategy="exact")
    # try with capital 1st letter, or every letter lowercase
    if resource is None:
        # do not use capitalize, because it also make every other letter lowercase
        if len(brain_region) > 1:
            brain_region = f"{brain_region[0].upper()}{brain_region[1:]}"
        elif len(brain_region) == 1:
            brain_region.upper()
        resource = access_point.resolve(brain_region, strategy="exact")

        if resource is None:
            brain_region.lower()
            resource = access_point.resolve(brain_region, strategy="exact")

    # raise Exception if resource was not found
    if resource is None:
        base_text = f"Could not find any brain region with name {brain_region}"
        raise_brain_region_exception(base_text, brain_region, access_point)

    # if resource found but not a brain region, also raise Exception
    if not hasattr(resource, "subClassOf") or resource.subClassOf != "nsg:BrainRegion":
        base_text = f"Resource {brain_region} is not a brain region"
        raise_brain_region_exception(base_text, brain_region, access_point)

    # if no exception was raised, return brain region as a json dict
    return access_point.forge.as_json(resource)


def get_all_species(access_token=None):

    access_point = ontology_forge_access_point(access_token)

    resources = access_point.forge.search({"subClassOf": "nsg:Species"}, limit=100)

    return sorted(set(r.label for r in resources))
