import unittest

from kgforge.core import Resource

from bluepyemodel.api.nexus import Nexus_API


class TestNexusAPI(unittest.TestCase):
    def test_condition_to_filter(self):
        pass

    def test_register_ressource(self):
        #         access_point = Nexus_API(
        #             project="emodel_pipeline",
        #             organisation="Cells",
        #             endpoint="https://staging.nexus.ocp.bbp.epfl.ch/v1",
        #             forge_path=None,
        #         )

        #         extraction_target = Resource(
        #             type="ElectrophysiologyFeatureExtractionTarget",
        #             eModel="L23_PC",
        #             subject={
        #                 "type": "Subject",
        #                 "species": {
        #                     "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
        #                     "label": "Homo sapiens",
        #                 },
        #             },
        #             stimulus={
        #                 "stimulusType": {
        #                     "id": "http://bbp.epfl.ch/neurosciencegraph/ontologies/stimulustypes/IDRest",
        #                     "label": "IDRest",
        #        },
        #        "target": [150],
        #        "tolerance": [10],
        #        "threshold": false,
        #        "recordingLocation": "soma",
        #    },
        #    feature=[{"name": "voltage_base"}],
        # )

        # access_point.register(extraction_target)
        pass

    def test_fetch_ressource(self):
        pass
        # access_point = Nexus_API(
        #    project="emodel_pipeline",
        #    organisation="Cells",
        #    endpoint="https://staging.nexus.ocp.bbp.epfl.ch/v1",
        #    forge_path=None,
        # )
        #
        # extraction_target = access_point.fetch(
        #    type_="ElectrophysiologyFeatureExtractionTarget", conditions={"eModel": "L23_PC"}
        # )


if __name__ == "__main__":
    unittest.main()
