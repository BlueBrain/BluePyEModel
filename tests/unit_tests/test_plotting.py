from bluepyemodel.emodel_pipeline.plotting import get_ordered_currentscape_keys


def test_get_ordered_currentscape_keys():
    keys = ["RMPProtocol.soma.v", "Step_300.soma.cai", "Step_300.soma.ica_TC_iL", "Step_300.soma.v"]
    expected_keys = {
        "Step_300": {
            "soma": {
                "voltage_key": "Step_300.soma.v",
                "current_keys": ["Step_300.soma.ica_TC_iL"],
                "current_names": ["ica_TC_iL"],
                "ion_conc_keys": ["Step_300.soma.cai"],
                "ion_conc_names": ["cai"],
            }
        }
    }
    ordered_keys = get_ordered_currentscape_keys(keys)
    assert ordered_keys == expected_keys
