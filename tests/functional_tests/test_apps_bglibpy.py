"""Test apps module."""
import numpy.testing as npt
import os
from pathlib import Path
import pandas as pd

from pandas.testing import assert_frame_equal
from click.testing import CliRunner

from voxcell import CellCollection

runner = CliRunner()

DATA = Path(__file__).parents[1] / "test_data_apps"
output_path = Path("out_emodel_release")

from bluepyemodel.apps.emodel_release import cli


def test_get_me_combos_currents():
    response = runner.invoke(
        cli,
        [
            "get_me_combos_currents",
            "--input-sonata-path",
            "sonata_ais.h5",
            "--release-path",
            str(output_path),
            "--morphology-path",
            str(DATA / "morphologies"),
            "--protocol-config-path",
            str(DATA / "protocol_config.yaml"),
            "--output-sonata-path",
            "sonata_currents.h5",
            "--emodel-api",
            "local",
        ],
    )

    # ensures cli run
    assert response.exit_code == 0

    # ensures currents are correct
    df = CellCollection().load_sonata("sonata_currents.h5").as_dataframe()
    npt.assert_allclose(
        df["@dynamics:holding_current"].to_list(),
        [-0.03673185397730094, -0.04284688945972448],
        rtol=1e-6,
    )
    npt.assert_allclose(
        df["@dynamics:threshold_current"].to_list(), [0.115187, 0.131859375], rtol=1e-5
    )


def test_detect_stuck_cells():

    response = runner.invoke(
        cli,
        [
            "detect_stuck_cells",
            "--sonata-path",
            "sonata_ais.h5",
            "--morphology-path",
            str(DATA / "morphologies"),
            "--emodel-hoc-path",
            str(output_path / "hoc_files"),
            "--protocol-config-path",
            str(DATA / "protocol_config.yaml"),
        ],
    )

    # ensures cli run
    assert response.exit_code == 0

    df = pd.read_csv("stuck_df/stuck_df_None.csv")
    expected_df = pd.read_csv(DATA / "stuck_df.csv")
    assert_frame_equal(df, expected_df)
