import unittest

from bluepyemodel.ecode import *

from bluepyopt.ephys.locations import (
    NrnSeclistCompLocation,
    NrnSomaDistanceCompLocation,
    NrnSecSomaDistanceCompLocation,
)

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)


class TestProtocols(unittest.TestCase):
    def test_subwhitenoise(self):

        prot_def = {"amp": 0.2, "holding_current": -0.001}
        stimulus = eCodes["subwhitenoise"](location=soma_loc, **prot_def)
        t, i = stimulus.generate()

        self.assertEqual(stimulus.name, "SubWhiteNoise")
        self.assertEqual(stimulus.total_duration, 5099.8)
        self.assertEqual(len(i), 50999)


if __name__ == "__main__":
    unittest.main()
