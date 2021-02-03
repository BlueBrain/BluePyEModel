"""Write down a config_dict.json."""
import json
import glob
import pathlib
import re


def write_config_dict(output_path, max_cells=10):
    traces_path = "/gpfs/bbp.cscs.ch/project/proj38/singlecell/expdata/LNMC/_data/sandrine/L5PC_P14"

    protocols = {
        "IDthresh": {
            "tolerances": [10.0],
            "targets": [0],
            "efeatures": ["Spikecount"],
            "location": "soma",
        },
        "IDRest": {
            "tolerances": [10.0],
            "targets": [150],
            "efeatures": ["mean_frequency", "ISI_CV", "AP_amplitude", "AP_begin_width"],
            "location": "soma",
        },
        "APWaveform": {
            "tolerances": [10.0],
            "targets": [300],
            "efeatures": [
                "AP_amplitude",
                "AP_begin_width",
            ],
            "location": "soma",
        },
        "IV": {
            "tolerances": [10.0],
            "targets": [0, -40],
            "efeatures": ["voltage_base", "ohmic_input_resistance_vb_ssse"],
            "location": "soma",
        },
    }

    cells_metadata = {}

    ton = {
        "IDthresh": 700,
        "IDRest": 700,
        "APWaveform": 5,
        "IV": 20,
    }

    toff = {
        "IDthresh": 2700,
        "IDRest": 2700,
        "APWaveform": 55,
        "IV": 1020,
    }
    for trace_path in glob.glob(traces_path + "/**/*.ibw", recursive=True):
        # keep only a given number of cells cells
        if len(cells_metadata.keys()) > max_cells:
            break

        trace_path = pathlib.Path(trace_path)

        for protocol_name in protocols:
            # keep only a given number of cells cells
            if len(cells_metadata.keys()) > max_cells:
                break

            if protocol_name.lower() in trace_path.name.lower():

                ch = re.findall("ch\d{1,2}", trace_path.stem)

                if ch and int(ch[0][2:]) % 2:

                    cell_id = trace_path.parent.name

                    v_file = str(trace_path)
                    i_file = v_file.replace(ch[0], "ch{}".format(int(ch[0][2:]) - 1))

                    tracedata = {
                        "i_file": i_file,
                        "v_file": v_file,
                        "ljp": 14.0,
                        "i_unit": "A",
                        "v_unit": "V",
                        "t_unit": "s",
                        "ton": ton[protocol_name],
                        "toff": toff[protocol_name],
                    }

                    if cell_id not in cells_metadata:
                        cells_metadata[cell_id] = {}

                    cells_metadata[cell_id].setdefault(protocol_name, []).append(
                        tracedata
                    )

    from bluepyemodel.feature_extraction.extract import get_config

    protocols_threshold = ["IDRest"]
    threshold_nvalue_save = 1

    config_dict = get_config(
        cells_metadata,
        protocols,
        protocols_threshold=protocols_threshold,
        threshold_nvalue_save=threshold_nvalue_save,
    )

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=4)


if __name__ == "__main__":
    output_path = pathlib.Path("config") / "config_dict.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_config_dict(output_path)
