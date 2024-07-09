"""NexusForgeAccessPoint class."""

"""
Copyright 2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import getpass
import json
import logging
import pathlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import jwt
from entity_management.state import refresh_token
from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource
from kgforge.core.commons.strategies import ResolvingStrategy
from kgforge.specializations.resources import Dataset

from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.emodel_pipeline.emodel_script import EModelScript
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from bluepyemodel.emodel_pipeline.memodel import MEModel
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
    "EModelScript": "EModelScript",
    "MEModel": "MEModel",
}

CLASS_TO_RESOURCE_NAME = {
    "TargetsConfiguration": "ETC",
    "EModelPipelineSettings": "EMPS",
    "FitnessCalculatorConfiguration": "FCC",
    "NeuronModelConfiguration": "EMC",
    "EModel": "EM",
    "DistributionConfiguration": "EMCD",
    "EModelWorkflow": "EMW",
    "EModelScript": "EMS",
    "MEModel": "MEM",
}

NEXUS_TYPE_TO_CLASS = {
    "ExtractionTargetsConfiguration": TargetsConfiguration,
    "EModelPipelineSettings": EModelPipelineSettings,
    "FitnessCalculatorConfiguration": FitnessCalculatorConfiguration,
    "EModelConfiguration": NeuronModelConfiguration,
    "EModel": EModel,
    "EModelChannelDistribution": DistributionConfiguration,
    "EModelWorkflow": EModelWorkflow,
    "EModelScript": EModelScript,
    "MEModel": MEModel,
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

NEXUS_PROJECTS_TRACES = [
    {"project": "lnmce", "organisation": "bbp"},
    {"project": "thalamus", "organisation": "public"},
    {"project": "mmb-point-neuron-framework-model", "organisation": "bbp"},
]


class AccessPointException(Exception):
    """For Exceptions related to the NexusForgeAccessPoint"""


class NexusForgeAccessPoint:
    """Access point to Nexus Knowledge Graph using Nexus Forge"""

    forges = {}

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

        self.endpoint = endpoint
        self.bucket = organisation + "/" + project
        self.forge_path = forge_path

        # reuse token to avoid redundant user prompts
        self.access_token = access_token

        if not self.access_token:
            self.access_token = self.get_access_token()
        decoded_token = jwt.decode(self.access_token, options={"verify_signature": False})
        self.agent = self.forge.reshape(
            self.forge.from_json(decoded_token),
            keep=["name", "email", "sub", "preferred_username"],
        )
        username = decoded_token["preferred_username"]
        self.agent.id = f"https://bbp.epfl.ch/nexus/v1/realms/bbp/users/{username}"
        self.agent.type = ["Person", "Agent"]

        self._available_etypes = None
        self._available_mtypes = None
        self._available_ttypes = None
        self._atlas_release = None

    def refresh_token(self, offset=300):
        """refresh token if token is expired or will be soon. Returns new expiring time.

        Args:
            offset (int): offset to apply to the expiring time in s.
        """
        # Check if the access token has expired
        decoded_token = jwt.decode(self.access_token, options={"verify_signature": False})
        token_exp_timestamp = decoded_token["exp"]
        # Get the current UTC time as a timezone-aware datetime object
        utc_now = datetime.now(timezone.utc)
        current_timestamp = int(utc_now.timestamp())
        if current_timestamp > token_exp_timestamp - offset:
            logger.info("Nexus access token has expired, refreshing token...")
            self.access_token = self.get_access_token()
            decoded_token = jwt.decode(self.access_token, options={"verify_signature": False})
            token_exp_timestamp = decoded_token["exp"]

        return token_exp_timestamp

    @property
    def forge(self):
        key = f"{self.endpoint}|{self.bucket}|{self.forge_path}"
        if key in self.__class__.forges:
            expiry, forge = self.__class__.forges[key]
            if expiry > datetime.now(timezone.utc):
                return forge

        token_exp_timestamp = self.refresh_token()
        forge = KnowledgeGraphForge(
            self.forge_path,
            bucket=self.bucket,
            endpoint=self.endpoint,
            token=self.access_token,
        )

        self.__class__.forges[key] = (
            datetime.fromtimestamp(token_exp_timestamp, timezone.utc) - timedelta(minutes=15),
            forge,
        )
        return forge

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

    @property
    def atlas_release(self):
        """Hard-coded atlas release fields for metadata"""
        # pylint: disable=protected-access
        atlas_def = {
            "id": "https://bbp.epfl.ch/neurosciencegraph/data/4906ab85-694f-469d-962f-c0174e901885",
            "type": ["BrainAtlasRelease", "AtlasRelease"],
        }

        if self._atlas_release is None:
            self.refresh_token()
            atlas_access_point = atlas_forge_access_point(
                access_token=self.access_token, forge_path=self.forge_path
            )
            atlas_resource = atlas_access_point.retrieve(atlas_def["id"])
            atlas_def["_rev"] = atlas_resource._store_metadata["_rev"]
            self._atlas_release = atlas_def
        return self._atlas_release

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
        if access_token is None:
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
            raise ValueError(
                f"Resolving strategy {strategy} does not exist. "
                "Strategy should be 'all', 'best' or 'exact'"
            )

        return self.forge.resolve(text, scope=scope, strategy=resolving_strategy, limit=limit)

    def register(
        self,
        resource_description,
        filters_existence=None,
        legacy_filters_existence=None,
        replace=False,
        distributions=None,
        images=None,
    ):
        """Register a resource from its dictionary description.

        Args:
            resource_description (dict): contains resource type, name and metadata
            filters_existence (dict): contains resource type, name and metadata,
                can be used to search for existence of resource on nexus
            legacy_filters_existence (dict): same as filters_existence,
                but with legacy nexus metadata
            replace (bool): whether to replace resource if found with filters_existence
            distributions (list): paths to resource object as json and other distributions
            images (list): paths to images to be attached to the resource
        """

        if "type" not in resource_description:
            raise AccessPointException("The resource description should contain 'type'.")

        previous_resources = None
        if filters_existence:
            previous_resources = self.fetch_legacy_compatible(
                filters_existence, legacy_filters_existence
            )

        if previous_resources:
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
            "label": "Single Cell",
        }
        if resource_description.get("brainLocation", None) is not None:
            resource_description["atlasRelease"] = self.atlas_release

        logger.debug("Registering resources: %s", resource_description)

        resource = self.forge.from_json(resource_description, na="None")
        resource = self.add_contribution(resource)

        if distributions:
            resource = Dataset.from_resource(self.forge, resource)
            for path in distributions:
                resource.add_distribution(path, content_type=f"application/{path.split('.')[-1]}")

        if images:
            for path in images:
                try:
                    resource_type = path.split("__")[-1].split(".")[0]
                except IndexError:
                    resource_type = filters_existence.get("type", None)
                # Do NOT do this BEFORE turning resource into a Dataset.
                # That would break the storing LazyAction into a string
                resource.add_image(
                    path=path,
                    content_type=f"application/{path.split('.')[-1]}",
                    about=resource_type,
                )

        self.forge.register(resource)

    def retrieve(self, id_):
        """Retrieve a resource based on its id"""

        resource = self.forge.retrieve(id=id_, cross_bucket=self.cross_bucket)

        if resource:
            return resource

        logger.debug("Could not retrieve resource of id: %s", id_)

        return None

    def fetch(self, filters, cross_bucket=None):
        """Fetch resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include "type" or "id".
            cross_bucket (bool): whether to also fetch from other projects or not.

        Returns:
            resources (list): list of resources
        """
        if "type" not in filters and "id" not in filters:
            raise AccessPointException("Search filters should contain either 'type' or 'id'.")

        if cross_bucket is None:
            cross_bucket = self.cross_bucket

        logger.debug("Searching: %s", filters)

        resources = self.forge.search(
            filters,
            cross_bucket=cross_bucket,
            limit=self.limit,
            debug=self.debug,
            search_endpoint=self.search_endpoint,
        )

        if resources:
            return resources

        logger.debug("No resources for filters: %s", filters)

        return None

    def fetch_legacy_compatible(self, filters, legacy_filters=None):
        """Fetch resources based on filters. Use legacy filters if no resources are found.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include "type" or "id".
            legacy_filters (dict): same as filters, with legacy nexus metadata

        Returns:
            resources (list): list of resources
        """
        resources = self.fetch(filters)
        if not resources and legacy_filters is not None:
            resources = self.fetch(legacy_filters)

        if resources:
            return resources
        return None

    def fetch_one(self, filters, legacy_filters=None, strict=True):
        """Fetch one and only one resource based on filters."""

        resources = self.fetch_legacy_compatible(filters, legacy_filters)

        if resources is None:
            if strict:
                raise AccessPointException(f"Could not get resource for filters {filters}")
            return None

        if len(resources) > 1:
            if strict:
                raise AccessPointException(f"More than one resource for filters {filters}")
            return resources[0]

        return resources[0]

    def download(self, resource_id, download_directory, content_type=None):
        """Download datafile from nexus."""
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

            self.forge.download(
                resource,
                "distribution.contentUrl",
                download_directory,
                cross_bucket=True,
                content_type=content_type,
                overwrite=True,
            )

            # Verify that each datafile for the resource was successfully downloaded
            for fp in file_paths:
                if not fp.exists():
                    raise AccessPointException(
                        f"Download failed: file {fp} does not exist for resource {resource_id}"
                    )

            return [str(fp) for fp in file_paths]

        return []

    def deprecate(self, filters, legacy_filters=None):
        """Deprecate resources based on filters."""

        tmp_cross_bucket = self.cross_bucket
        self.cross_bucket = False

        resources = self.fetch_legacy_compatible(filters, legacy_filters)

        if resources:
            for resource in resources:
                rr = self.retrieve(resource.id)
                if rr is not None:
                    self.forge.deprecate(rr)

        self.cross_bucket = tmp_cross_bucket

    def deprecate_all(self, metadata, metadata_legacy=None):
        """Deprecate all resources used or produced by BluePyModel. Use with extreme caution."""

        if not yesno("Confirm deprecation of all BluePyEmodel resources in Nexus project"):
            return

        for type_ in NEXUS_TYPE_TO_CLASS.keys():
            filters = {"type": type_}
            filters.update(metadata)

            if metadata_legacy is None:
                legacy_filters = None
            else:
                legacy_filters = {"type": type_}
                legacy_filters.update(metadata_legacy)

            self.deprecate(filters, legacy_filters)

    def resource_location(self, resource, download_directory):
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
                filepath = self.download(resource.id, download_directory)[0]

            paths.append(filepath)

        return paths

    @staticmethod
    def resource_name(class_name, metadata, seed=None):
        """Create a resource name from the class name and the metadata."""
        name_parts = [CLASS_TO_RESOURCE_NAME[class_name]]
        if "iteration" in metadata:
            name_parts.append(metadata["iteration"])
        if "eModel" in metadata:
            name_parts.append(metadata["eModel"])
        if "tType" in metadata:
            name_parts.append(metadata["tType"])
        # legacy nexus emodel metadata
        if "emodel" in metadata:
            name_parts.append(metadata["emodel"])
        # legacy nexus ttype metadata
        if "ttype" in metadata:
            name_parts.append(metadata["ttype"])
        if seed is not None:
            name_parts.append(str(seed))

        return "__".join(name_parts)

    @staticmethod
    def dump_json_and_get_distributions(object_, class_name, metadata_str, seed=None):
        """Write object as json dict, and get distribution paths (obj as json and others)"""
        json_payload = object_.as_dict()

        path_json = f"{CLASS_TO_RESOURCE_NAME[class_name]}__{metadata_str}"
        if seed is not None:
            path_json += f"__{seed}"
        path_json = str(
            (pathlib.Path("./nexus_temp") / metadata_str / f"{path_json}.json").resolve()
        )

        distributions = [path_json]
        json_payload.pop("nexus_images", None)  # remove nexus_images from payload
        if "nexus_distributions" in json_payload:
            distributions += json_payload.pop("nexus_distributions")

        with open(path_json, "w") as fp:
            json.dump(json_payload, fp, indent=2)

        return distributions

    @staticmethod
    def get_seed_from_object(object_, class_name):
        """Get the seed from the object if it has one else None."""
        seed = None
        if class_name in ("EModel", "EModelScript"):
            seed = object_.seed
        return seed

    def object_to_nexus(
        self,
        object_,
        metadata_dict,
        metadata_str,
        metadata_dict_legacy,
        replace=True,
        currents=None,
    ):
        """Transform a BPEM object into a dict which gets registered into Nexus as
        the distribution of a Dataset of the matching type. The metadata
        are also attached to the object to be able to retrieve the Resource."""

        class_name = object_.__class__.__name__
        type_ = CLASS_TO_NEXUS_TYPE[class_name]

        seed = self.get_seed_from_object(object_, class_name)
        score = None
        if class_name == "EModel":
            score = object_.fitness

        base_payload = {
            "type": ["Entity", type_],
            "name": self.resource_name(class_name, metadata_dict, seed=seed),
        }
        payload_existence = {
            "type": type_,
            "name": self.resource_name(class_name, metadata_dict, seed=seed),
        }
        payload_existence_legacy = {
            "type": type_,
            "name": self.resource_name(class_name, metadata_dict_legacy, seed=seed),
        }

        base_payload.update(metadata_dict)
        if score is not None:
            base_payload["score"] = score
        if currents is not None:
            base_payload["holding_current"] = currents["holding"]
            base_payload["threshold_current"] = currents["threshold"]
        if hasattr(object_, "get_related_nexus_ids"):
            related_nexus_ids = object_.get_related_nexus_ids()
            if related_nexus_ids:
                base_payload.update(related_nexus_ids)

        payload_existence.update(metadata_dict)
        payload_existence.pop("annotation", None)

        payload_existence_legacy.update(metadata_dict_legacy)
        payload_existence_legacy.pop("annotation", None)

        nexus_images = object_.as_dict().get("nexus_images", None)
        distributions = self.dump_json_and_get_distributions(
            object_=object_, class_name=class_name, metadata_str=metadata_str, seed=seed
        )

        self.register(
            base_payload,
            filters_existence=payload_existence,
            legacy_filters_existence=payload_existence_legacy,
            replace=replace,
            distributions=distributions,
            images=nexus_images,
        )

    def update_distribution(self, resource, metadata_str, object_):
        """Update a resource distribution using python object.

        Cannot update resource that has more than one resource."""
        class_name = object_.__class__.__name__
        seed = self.get_seed_from_object(object_, class_name)

        distributions = self.dump_json_and_get_distributions(
            object_=object_,
            class_name=class_name,
            metadata_str=metadata_str,
            seed=seed,
        )

        path_json = distributions[0]

        resource = Dataset.from_resource(self.forge, resource, store_metadata=True)
        # Nexus behavior:
        # - if only one element, gives either a dict or a list
        # - if multiple elements, returns a list of elements
        # Here, we want to be sure that we only have one element
        if isinstance(resource.distribution, list):
            if len(resource.distribution) != 1:
                raise TypeError(
                    f"'update_distribution' method cannot be used on {class_name} {metadata_str} "
                    "with more than 1 distribution."
                )
        elif not isinstance(resource.distribution, dict):
            raise TypeError(
                "'update_distribution' method requires a dict or a single-element list for "
                f"{class_name} {metadata_str}, got {type(resource.distribution)} instead."
            )

        # add distribution from object and remove old one from resource
        resource.add_distribution(path_json, content_type=f"application/{path_json.split('.')[-1]}")
        resource.distribution = [resource.distribution[1]]
        return resource

    def resource_to_object(self, type_, resource, metadata, download_directory):
        """Transform a Resource into a BPEM object of the matching type"""

        file_paths = self.download(resource.id, download_directory)
        json_path = next((fp for fp in file_paths if pathlib.Path(fp).suffix == ".json"), None)

        if json_path is None:
            # legacy case where the payload is in the Resource
            # can no longer use this for recent resources
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

    def nexus_to_object(self, type_, metadata, download_directory, legacy_metadata=None):
        """Search for a single Resource matching the ``type_`` and metadata and return it
        as a BPEM object of the matching type"""

        filters = {"type": type_}
        filters.update(metadata)

        legacy_filters = None
        if legacy_metadata:
            legacy_filters = {"type": type_}
            legacy_filters.update(legacy_metadata)

        resource = self.fetch_one(filters, legacy_filters)

        return self.resource_to_object(type_, resource, metadata, download_directory)

    def nexus_to_objects(self, type_, metadata, download_directory, legacy_metadata=None):
        """Search for Resources matching the ``type_`` and metadata and return them
        as BPEM objects of the matching type"""

        filters = {"type": type_}
        filters.update(metadata)

        legacy_filters = None
        if legacy_metadata:
            legacy_filters = {"type": type_}
            legacy_filters.update(legacy_metadata)

        resources = self.fetch_legacy_compatible(filters, legacy_filters)

        objects_ = []
        ids = []

        if resources:
            for resource in resources:
                objects_.append(
                    self.resource_to_object(type_, resource, metadata, download_directory)
                )
                ids.append(resource.id)

        return objects_, ids

    def get_nexus_id(self, type_, metadata, legacy_metadata=None):
        """Search for a single Resource matching the ``type_`` and metadata and return its id"""
        filters = {"type": type_}
        filters.update(metadata)

        legacy_filters = None
        if legacy_metadata:
            legacy_filters = {"type": type_}
            legacy_filters.update(legacy_metadata)

        resource = self.fetch_one(filters, legacy_filters)

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


def ontology_forge_access_point(
    access_token=None, forge_path=None, endpoint="https://bbp.epfl.ch/nexus/v1"
):
    """Returns an access point targeting the project containing the ontology for the
    species and brain regions"""

    access_point = NexusForgeAccessPoint(
        project="datamodels",
        organisation="neurosciencegraph",
        endpoint=endpoint,
        forge_path=forge_path,
        access_token=access_token,
    )

    return access_point


def atlas_forge_access_point(access_token=None, forge_path=None):
    """Returns an access point targeting the project containing the atlas"""

    access_point = NexusForgeAccessPoint(
        project="atlas",
        organisation="bbp",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=forge_path,
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


def check_resource(
    label,
    category,
    access_point=None,
    access_token=None,
    forge_path=None,
    endpoint="https://bbp.epfl.ch/nexus/v1",
):
    """Checks that resource is present on nexus and is part of the provided category

    Arguments:
        label (str): name of the resource to search for
        category (str): can be "etype", "mtype" or "ttype"
        access_point (str):  ontology_forge_access_point(access_token)
        forge_path (str): path to a .yml used as configuration by nexus-forge.
        endpoint (str): nexus endpoint
    """
    allowed_categories = ["etype", "mtype", "ttype"]
    if category not in allowed_categories:
        raise AccessPointException(f"Category is {category}, but should be in {allowed_categories}")

    if access_point is None:
        access_point = ontology_forge_access_point(access_token, forge_path, endpoint)

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


def get_available_traces(species=None, brain_region=None, access_token=None, forge_path=None):
    """Returns a list of Resources of type Traces from the bbp/lnmce Nexus project"""

    filters = {"type": "Trace", "distribution": {"encodingFormat": "application/nwb"}}

    if species:
        filters["subject"] = species
    if brain_region:
        filters["brainLocation"] = brain_region

    resources = []
    for proj_traces in NEXUS_PROJECTS_TRACES:
        access_point = NexusForgeAccessPoint(
            project=proj_traces["project"],
            organisation=proj_traces["organisation"],
            endpoint="https://bbp.epfl.ch/nexus/v1",
            forge_path=forge_path,
            access_token=access_token,
            cross_bucket=True,
        )
        tmp_resources = access_point.fetch(filters=filters)
        if tmp_resources:
            resources += tmp_resources

    return resources


def get_brain_region(
    brain_region, access_token=None, forge_path=None, endpoint="https://bbp.epfl.ch/nexus/v1"
):
    """Returns the resource corresponding to the brain region

    If the brain region name is not present in nexus,
    raise an exception mentioning the possible brain region names available on nexus

    Arguments:
        brain_region (str): name of the brain region to search for
        access_token (str): nexus connection token
        forge_path (str): path to a .yml used as configuration by nexus-forge.
        endpoint (str): nexus endpoint
    """

    filter = "brain_region"
    access_point = ontology_forge_access_point(access_token, forge_path, endpoint)

    if brain_region in ["SSCX", "sscx"]:
        brain_region = "somatosensory areas"
    if brain_region == "all":
        # http://api.brain-map.org/api/v2/data/Structure/8
        brain_region = "Basic cell groups and regions"

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

    if isinstance(resource, list):
        resource = resource[0]

    # raise Exception if resource was not found
    if resource is None:
        base_text = f"Could not find any brain region with name {brain_region}"
        raise_not_found_exception(base_text, brain_region, access_point, filter)

    return resource


def get_brain_region_dict(
    brain_region, access_token=None, forge_path=None, endpoint="https://bbp.epfl.ch/nexus/v1"
):
    """Returns a dict with id and label of the resource corresponding to the brain region

    Arguments:
        brain_region (str): name of the brain region to search for
        access_token (str): nexus connection token
        forge_path (str): path to a .yml used as configuration by nexus-forge.
        endpoint (str): nexus endpoint

    Returns:
        dict: the id and label of the nexus resource of the brain region
    """
    br_resource = get_brain_region(brain_region, access_token, forge_path, endpoint)

    access_point = ontology_forge_access_point(access_token, forge_path, endpoint)

    # if no exception was raised, filter to get id and label and return them
    brain_region_dict = access_point.forge.as_json(br_resource)
    return {
        "id": brain_region_dict["id"],
        "label": brain_region_dict["label"],
    }


def get_brain_region_notation(
    brain_region, access_token=None, forge_path=None, endpoint="https://bbp.epfl.ch/nexus/v1"
):
    """Get the ontology of the brain location."""
    if brain_region is None:
        return None

    brain_region_resource = get_brain_region(
        brain_region, access_token=access_token, forge_path=forge_path, endpoint=endpoint
    )

    return brain_region_resource.notation


def get_nexus_brain_region(
    brain_region, access_token=None, forge_path=None, endpoint="https://bbp.epfl.ch/nexus/v1"
):
    """Get the ontology of the brain location."""
    if brain_region is None:
        return None

    brain_region_from_nexus = get_brain_region_dict(
        brain_region, access_token=access_token, forge_path=forge_path, endpoint=endpoint
    )

    return {
        "type": "BrainLocation",
        "brainRegion": brain_region_from_nexus,
    }


def get_all_species(access_token=None, forge_path=None, endpoint="https://bbp.epfl.ch/nexus/v1"):
    access_point = ontology_forge_access_point(access_token, forge_path, endpoint)

    resources = access_point.forge.search({"subClassOf": "nsg:Species"}, limit=100)

    return sorted(set(r.label for r in resources))


def get_curated_morphology(resources):
    """Get curated morphology from multiple resources with same morphology name"""
    for r in resources:
        if hasattr(r, "annotation"):
            annotations = r.annotation if isinstance(r.annotation, list) else [r.annotation]
            for annotation in annotations:
                if "QualityAnnotation" in annotation.type:
                    if annotation.hasBody.label == "Curated":
                        return r
        if hasattr(r, "derivation"):
            return r
    return None


def filter_mechanisms_with_brain_region(forge, resources, brain_region_label, br_visited):
    """Filter mechanisms by brain region"""
    br_visited.add(brain_region_label)
    filtered_resources = [
        r
        for r in resources
        if hasattr(r, "brainLocation") and r.brainLocation.brainRegion.label == brain_region_label
    ]
    if len(filtered_resources) > 0:
        return filtered_resources, br_visited

    query = (
        """
        SELECT DISTINCT ?br ?label
        WHERE{
            ?id label \""""
        + f"{brain_region_label}"
        + """\" ;
            isPartOf ?br .
            ?br label ?label .
        }
    """
    )
    brs = forge.sparql(query, limit=1000)
    # when fails can be None or empty list
    if brs:
        new_brain_region_label = brs[0].label
        return filter_mechanisms_with_brain_region(
            forge, resources, new_brain_region_label, br_visited
        )

    # if no isPartOf present, try with isLayerPartOf
    query = (
        """
        SELECT DISTINCT ?br ?label
        WHERE{
            ?id label \""""
        + f"{brain_region_label}"
        + """\" ;
            isLayerPartOf ?br .
            ?br label ?label .
        }
    """
    )
    brs = forge.sparql(query, limit=1000)
    # when fails can be None or empty list
    if brs:
        # can have multiple brain regions
        for br in brs:
            new_brain_region_label = br.label
            resources, br_visited = filter_mechanisms_with_brain_region(
                forge, resources, new_brain_region_label, br_visited
            )
            if resources is not None:
                return resources, br_visited

    return None, br_visited
