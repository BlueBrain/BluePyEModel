"""Extract features."""
import logging
import bluepyefe

logger = logging.getLogger(__name__)


def get_config(
    cells, protocols, file_format, protocols_threshold=None, threshold_nvalue_save=1
):
    """Create configuration dictionnary."""
    if protocols_threshold is None:
        protocols_threshold = []

    config = bluepyefe.tools.default_config_dict()

    config["cells"] = cells
    config["protocols"] = protocols
    config["options"]["format"] = file_format
    config["options"]["protocols_threshold"] = protocols_threshold
    config["options"]["threshold_nvalue_save"] = threshold_nvalue_save

    return config


def extract_efeatures(
    config, emodel, trace_reader=None, map_function=None, write_files=False
):
    """Extract efeatures."""
    extractor = bluepyefe.Extractor(
        config, mainname=emodel, trace_reader=trace_reader, map_function=map_function
    )
    extractor.extract_efeatures()
    extractor.analyse_threshold()
    for j, cell in enumerate(extractor.cells):

        # Remove the IV traces containing spikes
        extractor.cells[j].traces = [
            t
            for i, t in enumerate(cell.traces)
            if not (t.protocol_name == "IV" and t.spikecount > 0)
        ]
        # Remove the APWaveform traces without spikes
        extractor.cells[j].traces = [
            t
            for i, t in enumerate(cell.traces)
            if not (t.protocol_name == "APWaveform" and t.spikecount == 0)
        ]

    extractor.mean_efeatures()
    efeatures, stimuli, current = extractor.create_feature_protocol_files(
        write_files=write_files
    )

    return efeatures, stimuli, current


def plot_efeatures_and_traces(extractor):
    """Plot features and traces."""
    plotter = bluepyefe.Plotter(
        extractor.cells, extractor.maindirname, plot_per_column=5
    )
    plot_efeatures(extractor, plotter)
    plot_traces(extractor, plotter)


def plot_efeatures(extractor, plotter=None):
    """Plot efeatures."""
    if plotter is None:
        plotter = bluepyefe.Plotter(
            extractor.cells, extractor.maindirname, plot_per_column=5
        )

    for cell in extractor.cells:
        for prot_name in cell.get_protocol_names():
            plotter.individual_cell_efeatures(
                cell,
                prot_name,
                extractor.config_protocols[prot_name]["efeatures"],
                key_amp="amp_rel",
            )
    plotter.cells_efeatures(
        extractor.cells,
        extractor.protocols,
        extractor.config_protocols,
        key_amp="amp_rel",
    )
    plotter.cells_efeatures(
        extractor.cells, extractor.protocols, extractor.config_protocols, key_amp="amp"
    )


def plot_traces(extractor, plotter=None):
    """Plot traces."""
    if plotter is None:
        plotter = bluepyefe.Plotter(
            extractor.cells, extractor.maindirname, plot_per_column=5
        )

    # Warning: slow due to tight_layout
    for cell in extractor.cells:
        for exp in cell.get_protocol_names():
            plotter.individual_cell_traces(cell, exp)
