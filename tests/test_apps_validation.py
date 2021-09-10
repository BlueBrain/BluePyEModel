"""Test apps module."""
import os
from pathlib import Path
import pandas as pd

from pandas.testing import assert_frame_equal
from click.testing import CliRunner

runner = CliRunner()

DATA = Path(__file__).parent / "test_apps"
output_path = Path("out_emodel_release")
os.environ["USE_NEURODAMUS"] = str(1)

from bluepyemodel.apps.emodel_release import cli


def test_evaluate_emodels():
    response = runner.invoke(
        cli,
        [
            "evaluate_emodels",
            "--sonata-path",
            "sonata_ais.h5",
            "--morphology-path",
            str(DATA / "morphologies"),
            "--emodel-api",
            "local",
            "--emodel-path",
            str(DATA / "configs"),
            "--final-path",
            str(DATA / "final.json"),
            "--ais-emodels-path",
            str(output_path / "ais_models.yaml"),
            "--megate-thresholds-path",
            str(DATA / "megate_thresholds.yaml"),
        ],
    )

    # ensures cli run
    assert response.exit_code == 0

    df_exemplar = pd.read_csv("exemplar_evaluations.csv")
    # df_exemplar.to_csv(DATA / "exemplar_evaluations.csv", index=False)
    expected_df_exemplar = pd.read_csv(DATA / "exemplar_evaluations.csv")
    assert_frame_equal(df_exemplar, expected_df_exemplar)

    # remove path column to avoid issues with absolute paths
    df_eval = pd.read_csv("results/region_None/results.csv").drop(columns=["path"])
    # df_eval.to_csv(DATA / "cell_evaluations.csv", index=False)
    expected_df_eval = pd.read_csv(DATA / "cell_evaluations.csv")
    assert_frame_equal(df_eval, expected_df_eval)
