import unittest

from bluepyemodel.api.nexus import NexusAPI


class TestNexusAPI(unittest.TestCase):
    def test_morphology(self):
        pass
        # access_point = NexusAPI()
        #
        # emodel = "test_emodel"
        # name = "test_name"
        # species = "rat"
        #
        # resource = access_point.forge.from_json({
        #     "type": [
        #         "Entity",
        #         "ElectrophysiologyFeatureOptimisationNeuronMorphology",
        #     ],
        #     "name": name,
        #     "description": "Neuron morphology used for optimisation",
        #     "eModel": emodel,
        #     "subject": access_point.get_subject(species),
        #     "morphology": {
        #         "id": "https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies/377dacba-ec8d-4c63-981e-78320f1054ad",
        #         "type": "NeuronMorphology",
        #         "name": name,
        #         "distribution": {
        #             "contentUrl": "https://bbp.epfl.ch/nexus/v1/files/public/sscx/819017b1-545d-47a8-9d41-24d8c0415176"
        #         }
        #     },
        #     "sectionListNames": None,
        #     "sectionArrayNames": None,
        #     "sectionIndex": None
        # })
        #
        # access_point.register(resource)
        #
        # access_point.get_morphologies(emodel, species)


if __name__ == "__main__":

    unittest.main()
