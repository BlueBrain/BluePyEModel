"""Nexus Forge access point used by the Nexus access point"""

import getpass
import json
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

CLASS_TO_RESOURCE_NAME = {
    "TargetsConfiguration": "ETC",
    "EModelPipelineSettings": "EMPS",
    "FitnessCalculatorConfiguration": "FCC",
    "NeuronModelConfiguration": "EMC",
    "EModel": "EM",
    "DistributionConfiguration": "EMCD",
    "EModelWorkflow": "EMW",
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

NEXUS_ENTRIES = [
    "objectOfStudy",
    "contribution",
    "type",
    "id",
    "distribution",
    "@type",
    "annotation",
    "name",
]


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

        self._available_etypes = None
        self._available_mtypes = None
        self._available_ttypes = None

    @property
    def available_etypes(self):
        """List of ids of available etypes in this forge graph"""
        if self._available_etypes is None:
            self._available_etypes = self.get_available_etypes()
        return self._available_etypes

    @property
    def available_mtypes(self):
        """List of ids of available mtypes in this forge graph"""
        if self._available_mtypes is None:
            self._available_mtypes = self.get_available_mtypes()
        return self._available_mtypes

    @property
    def available_ttypes(self):
        """List of ids of available ttypes in this forge graph"""
        if self._available_ttypes is None:
            self._available_ttypes = self.get_available_ttypes()
        return self._available_ttypes

    def get_available_etypes(self):
        """Returns a list of nexus ids of all the etype resources using sparql"""
        query = """
            SELECT ?e_type_id

            WHERE {{
                    ?e_type_id label ?e_type ;
                        subClassOf* EType ;
            }}
        """
        # should we use self.limit here?
        resources = self.forge.sparql(query, limit=self.limit)
        if resources is None:
            return []
        return [r.e_type_id for r in resources]

    def get_available_mtypes(self):
        """Returns a list of nexus ids of all the mtype resources using sparql"""
        query = """
            SELECT ?m_type_id

            WHERE {{
                    ?m_type_id label ?m_type ;
                        subClassOf* MType ;
            }}
        """
        # should we use self.limit here?
        resources = self.forge.sparql(query, limit=self.limit)
        if resources is None:
            return []
        return [r.m_type_id for r in resources]

    def get_available_ttypes(self):
        """Returns a list of nexus ids of all the ttype resources using sparql"""
        query = """
            SELECT ?t_type_id

            WHERE {{
                    ?t_type_id label ?t_type ;
                        subClassOf* BrainCellTranscriptomeType ;
            }}
        """
        # should we use self.limit here?
        resources = self.forge.sparql(query, limit=self.limit)
        if resources is None:
            return []
        return [r.t_type_id for r in resources]

    @staticmethod
    def get_access_token():
        """Define access token either from bbp-workflow or provided by the user"""

        try:
            access_token = refresh_token()
        except:  # noqa: E722
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

        resource = self.forge.from_json(resource_description, na="None")
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

    def download(self, resource_id, download_directory=None, metadata_str=None):
        """Download datafile from nexus if it doesn't already exist."""
        if download_directory is None:
            if metadata_str is None:
                raise AccessPointException("download_directory or metadata_str should be other than None")
            download_directory = pathlib.Path("./nexus_temp") / metadata_str
        resource = self.forge.retrieve(resource_id, cross_bucket=True)

        if resource is None:
            raise AccessPointException(f"Could not find resource for id: {resource_id}")

        if hasattr(resource, "distribution"):

            file_paths = []
            if isinstance(resource.distribution, list):
                for dist in resource.distribution:
                    if hasattr(dist, "name"):
                        file_paths.append(pathlib.Path(download_directory) / dist.name)
                    else:
                        raise AttributeError(
                            f"A distribution of the resource {resource.name} does "
                            "not have a file name."
                        )
            else:
                file_paths = [pathlib.Path(download_directory) / resource.distribution.name]

            if any(not fp.is_file() for fp in file_paths):
                self.forge.download(
                    resource, "distribution.contentUrl", download_directory, cross_bucket=True
                )

            return [str(fp) for fp in file_paths]

        return []

    def deprecate(self, filters):
        """Deprecate resources based on filters."""

        tmp_cross_bucket = self.cross_bucket
        self.cross_bucket = False

        resources = self.fetch(filters)

        if resources:
            for resource in resources:
                rr = self.retrieve(resource.id)
                if rr is not None:
                    self.forge.deprecate(rr)

        self.cross_bucket = tmp_cross_bucket

    def deprecate_all(self, metadata):
        """Deprecate all resources used or produced by BluePyModel. Use with extreme caution."""

        if not yesno("Confirm deprecation of all BluePyEmodel resources in Nexus project"):
            return

        for type_ in NEXUS_TYPE_TO_CLASS.keys():

            filters = {"type": type_}
            filters.update(metadata)

            self.deprecate(filters)

    def resource_location(self, resource, metadata_str):
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
                filepath = self.download(resource.id, metadata_str=metadata_str)[0]

            paths.append(filepath)

        return paths

    @staticmethod
    def resource_name(class_name, metadata, seed=None):
        """Create a resource name from the class name and the metadata."""
        name_parts = [CLASS_TO_RESOURCE_NAME[class_name]]
        if "iteration" in metadata:
            name_parts.append(metadata["iteration"])
        if "emodel" in metadata:
            name_parts.append(metadata["emodel"])
        if "ttype" in metadata:
            name_parts.append(metadata["ttype"])
        if seed is not None:
            name_parts.append(str(seed))

        return "__".join(name_parts)

    def object_to_nexus(self, object_, metadata_dict, metadata_str, replace=True, seed=None):
        """Transform a BPEM object into a dict which gets registered into Nexus as
        the distribution of a Dataset of the matching type. The metadata
        are also attached to the object to be able to retrieve the Resource."""

        class_name = object_.__class__.__name__
        type_ = CLASS_TO_NEXUS_TYPE[class_name]

        seed = None
        if class_name == "EModel":
            seed = object_.seed

        base_payload = {
            "type": ["Entity", type_],
            "name": self.resource_name(class_name, metadata_dict, seed=seed),
        }
        payload_existance = {
            "type": type_,
            "name": self.resource_name(class_name, metadata_dict, seed=seed),
        }

        base_payload.update(metadata_dict)
        payload_existance.update(metadata_dict)
        json_payload = object_.as_dict()

        path_json = f"{CLASS_TO_RESOURCE_NAME[class_name]}"
        if seed is not None:
            path_json += f"__{seed}"
        path_json = str((pathlib.Path("./nexus_temp") / metadata_str / f"{path_json}.json").resolve())

        distributions = [path_json]
        if "nexus_distributions" in json_payload:
            distributions += json_payload.pop("nexus_distributions")

        with open(path_json, "w") as fp:
            json.dump(json_payload, fp, indent=2)

        payload_existance.pop("annotation", None)

        self.register(
            base_payload,
            filters_existance=payload_existance,
            replace=replace,
            distributions=distributions,
        )

    def resource_to_object(self, type_, resource, metadata, metadata_str):
        """Transform a Resource into a BPEM object of the matching type"""

        file_paths = self.download(resource.id, metadata_str)
        json_path = next((fp for fp in file_paths if pathlib.Path(fp).suffix == ".json"), None)

        if json_path is None:
            # legacy case where the payload is in the Resource
            payload = self.forge.as_json(resource)

            for k in metadata:
                payload.pop(k, None)

            for k in NEXUS_ENTRIES:
                payload.pop(k, None)

        else:
            # Case in which the payload is in a .json distribution
            with open(json_path, "r") as f:
                payload = json.load(f)

        return NEXUS_TYPE_TO_CLASS[type_](**payload)

    def nexus_to_object(self, type_, metadata, metadata_str):
        """Search for a single Resource matching the type_ and metadata and return it
        as a BPEM object of the matching type"""

        filters = {"type": type_}
        filters.update(metadata)

        resource = self.fetch_one(filters)

        return self.resource_to_object(type_, resource, metadata, metadata_str)

    def nexus_to_objects(self, type_, metadata, metadata_str):
        """Search for Resources matching the type_ and metadata and return them
        as BPEM objects of the matching type"""

        filters = {"type": type_}
        filters.update(metadata)

        resources = self.fetch(filters)

        objects_ = []

        if resources:
            for resource in resources:
                objects_.append(self.resource_to_object(type_, resource, metadata, metadata_str))

        return objects_

    def get_nexus_id(self, type_, metadata):
        """Search for a single Resource matching the type_ and metadata and return its id"""
        filters = {"type": type_}
        filters.update(metadata)

        resource = self.fetch_one(filters)

        return resource.id

    @staticmethod
    def brain_region_filter(resources):
        """Filter resources to keep only brain regions

        Arguments:
            resources (list of Resource): resources to be filtered

        Returns:
            list of Resource: the filtered resources
        """
        return [
            r for r in resources if hasattr(r, "subClassOf") and r.subClassOf == "nsg:BrainRegion"
        ]

    def type_filter(self, resources, filter):
        """Filter resources to keep only etypes/mtypes/ttypes

        Arguments:
            resources (list of Resource): resources to be filtered
            filter (str): can be "etype", "mytype" or "ttype"

        Returns:
            list of Resource: the filtered resources
        """
        if filter == "etype":
            available_names = self.available_etypes
        elif filter == "mtype":
            available_names = self.available_mtypes
        elif filter == "ttype":
            available_names = self.available_ttypes
        else:
            raise AccessPointException(
                f'filter is {filter} but should be in ["etype", "mtype", "ttype"]'
            )
        return [r for r in resources if r.id in available_names]

    def filter_resources(self, resources, filter):
        """Filter resources

        Arguments:
            resources (list of Resource): resources to be filtered
            filter (str): which filter to use
                can be "brain_region", "etype", "mtype", "ttype"

        Returns:
            list of Resource: the filtered resources

        Raises:
            AccessPointException if filter not in ["brain_region", "etype", "mtype", "ttype"]
        """
        if filter == "brain_region":
            return self.brain_region_filter(resources)
        if filter in ["etype", "mtype", "ttype"]:
            return self.type_filter(resources, filter)

        filters = ["brain_region", "etype", "mtype", "ttype"]
        raise AccessPointException(
            f"Filter not expected in filter_resources: {filter}"
            f"Please choose among the following filters: {filters}"
        )


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


def raise_not_found_exception(base_text, label, access_point, filter, limit=30):
    """Raise an exception mentioning the possible appropriate resource names available on nexus

    Arguments:
        base_text (str): text to display in the Exception
        label (str): name of the resource to search for
        access_point (NexusForgeAccessPoint)
        filter (str): which filter to use
            can be "brain_region", "etype", "mtype", or "ttype"
        limit (int): maximum number of resources to fetch when looking up
            for resource name suggestions
    """
    if not base_text.endswith("."):
        base_text = f"{base_text}."

    resources = access_point.resolve(label, strategy="all", limit=limit)
    if resources is None:
        raise AccessPointException(base_text)

    # make sure that resources is iterable
    if not isinstance(resources, list):
        resources = [resources]
    filtered_names = "\n".join(
        set(r.label for r in access_point.filter_resources(resources, filter))
    )
    if filtered_names:
        raise AccessPointException(f"{base_text} Maybe you meant one of those:\n{filtered_names}")

    raise AccessPointException(base_text)


def check_resource(label, category, access_point=None, access_token=None):
    """Checks that resource is present on nexus and is part of the provided category

    Arguments:
        label (str): name of the resource to search for
        category (str): can be "etype", "mtype" or "ttype"
        access_point (str):  ontology_forge_access_point(access_token)
    """
    allowed_categories = ["etype", "mtype", "ttype"]
    if category not in allowed_categories:
        raise AccessPointException(f"Category is {category}, but should be in {allowed_categories}")

    if access_point is None:
        access_point = ontology_forge_access_point(access_token)

    resource = access_point.resolve(label, strategy="exact")
    # raise Exception if resource was not found
    if resource is None:
        base_text = f"Could not find {category} with name {label}"
        raise_not_found_exception(base_text, label, access_point, category)

    # if resource found but not of the appropriate category, also raise Exception
    available_names = []
    if category == "etype":
        available_names = access_point.available_etypes
    elif category == "mtype":
        available_names = access_point.available_mtypes
    elif category == "ttype":
        available_names = access_point.available_ttypes
    if resource.id not in available_names:
        base_text = f"Resource {label} is not a {category}"
        raise_not_found_exception(base_text, label, access_point, category)


def get_brain_region(brain_region, access_token=None):
    """Returns a dict with id and label of the resource corresponding to the brain region

    If the brain region name is not present in nexus,
    raise an exception mentioning the possible brain region names available on nexus

    Arguments:
        brain_region (str): name of the brain region to search for
        access_token (str): nexus connection token

    Returns:
        dict: the id and label of the nexus resource of the brain region
    """

    filter = "brain_region"
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
            brain_region = brain_region.upper()
        resource = access_point.resolve(brain_region, strategy="exact")

        if resource is None:
            resource = access_point.resolve(brain_region.lower(), strategy="exact")

    # raise Exception if resource was not found
    if resource is None:
        base_text = f"Could not find any brain region with name {brain_region}"
        raise_not_found_exception(base_text, brain_region, access_point, filter)

    # if resource found but not a brain region, also raise Exception
    if not hasattr(resource, "subClassOf") or resource.subClassOf != "nsg:BrainRegion":
        base_text = f"Resource {brain_region} is not a brain region"
        raise_not_found_exception(base_text, brain_region, access_point, filter)

    # if no exception was raised, filter to get id and label and return them
    brain_region_dict = access_point.forge.as_json(resource)
    return {
        "id": brain_region_dict["id"],
        "label": brain_region_dict["label"],
    }


def get_all_species(access_token=None):

    access_point = ontology_forge_access_point(access_token)

    resources = access_point.forge.search({"subClassOf": "nsg:Species"}, limit=100)

    return sorted(set(r.label for r in resources))
