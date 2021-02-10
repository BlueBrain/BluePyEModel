import unittest
import pathlib
import numpy
import os
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline


class PipelineSetUp(unittest.TestCase):
    def setUp(self):
        os.chdir("./tests")

        emodel = "cADpyr_L5TPC"
        species = "rat"
        recipes_path = "./config/recipes.json"

        self.pipeline = EModel_pipeline(
            emodel=emodel,
            species=species,
            db_api="singlecell",
            recipes_path=recipes_path,
        )

        self.model_result = self.pipeline.compute_responses(
            stochasticity=False,
            copy_mechanisms=False,
            compile_mechanisms=True,
            additional_protocols=None,
        )[0]

    def tearDown(self):
        os.chdir("./../")


class TestProtocols(PipelineSetUp):
    def test_protocols(self):
        for prot_name in [
            "RMPProtocol.soma.v",
            "bpo_rmp",
            "bpo_holding_current",
            "RinProtocol.soma.v",
            "bpo_rin",
            "bpo_threshold_current",
            "bAP.soma.v",
            "bAP.dend1.v",
            "bAP.dend2.v",
            "bAP.ca_prox_apic.cai",
            "bAP.ca_prox_basal.cai",
            "bAP.ca_soma.cai",
            "bAP.ca_ais.cai",
            "Step_200.soma.v",
            "Step_280.soma.v",
            "APWaveform_320.soma.v",
            "IV_-100.soma.v",
            "SpikeRec_600.soma.v",
        ]:
            self.assertIn(prot_name, self.model_result["responses"])
            self.assertIsNot(self.model_result["responses"][prot_name], None)

    def test_Rin(self):
        exp = 37.37
        model = self.model_result["responses"]["bpo_rin"]
        self.assertLess(numpy.abs(exp - model), 1.0)

    def test_RMP(self):
        exp = -77.23
        model = self.model_result["responses"]["bpo_rmp"]
        self.assertLess(numpy.abs(exp - model), 1.0)

    def test_holding_current(self):
        exp = -0.1475
        model = self.model_result["responses"]["bpo_holding_current"]
        self.assertLess(numpy.abs(exp - model), 0.02)

    def test_threshold_current(self):
        exp = 0.475
        model = self.model_result["responses"]["bpo_threshold_current"]
        self.assertLess(numpy.abs(exp - model), 0.02)


if __name__ == "__main__":
    unittest.main()
