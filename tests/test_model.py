import unittest
from bluepyemodel.evaluation import model
import bluepyopt.ephys as ephys


class TestModel(unittest.TestCase):
    def test_multi_locations(self):

        section = "alldend"
        locations = model.multi_locations(section, {})

        self.assertIsInstance(locations[0], ephys.locations.NrnSeclistLocation)
        self.assertEqual(len(locations), 2)

    def test_define_parameters(self):

        definitions = {
            "distributions": {
                "exp": {"fun": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}"},
                "decay": {
                    "fun": "math.exp({distance}*{constant})*{value}",
                    "parameters": ["constant"],
                },
            },
            "parameters": {
                "global": [{"name": "celsius", "val": 34}],
                "distribution_decay": [{"name": "constant", "val": [-0.1, 0.0]}],
                "somadend": [{"name": "gIhbar_Ih", "val": [0, 2e-4], "dist": "exp"}],
            },
        }

        parameters = model.define_parameters(definitions)

        for param in parameters:
            self.assertIsInstance(param, ephys.parameters.NrnParameter)

            if param.name == "distribution_decay":
                self.assertEqual(param.bounds[0], -0.1)
                self.assertEqual(param.bounds[1], -0.0)
