# """Test apps module."""
# import numpy.testing as npt
# import shutil
# import os
# from pathlib import Path
# import pandas as pd

# from pandas.testing import assert_frame_equal
# from click.testing import CliRunner

# from bluepyemodel.apps.emodel_release import cli
# from voxcell import CellCollection

# runner = CliRunner()

# DATA = Path(__file__).parent / "test_apps"
# output_path = Path("out_emodel_release")


# def test_create_emodel_release():
#     os.environ["USE_NEURODAMUS"] = str(1)
#     if output_path.exists():
#         shutil.rmtree(output_path)
#     response = runner.invoke(
#         cli,
#         [
#             "create_emodel_release",
#             "--ais-models",
#             str(DATA / "ais_models.yaml"),
#             "--target-rho-factors",
#             str(DATA / "target_rhos.yaml"),
#             "--emodel-api",
#             "local",
#             "--emodel-path",
#             str(DATA / "configs"),
#             "--final-path",
#             str(DATA / "final.json"),
#             "--template",
#             str(DATA / "cell_template_neurodamus_AIS.jinja2"),
#             "--etype-emodel-map-path",
#             str(DATA / "best_emodels.csv"),
#             "--emodel-release",
#             str(output_path),
#         ],
#     )
#     # ensures cli runs
#     assert response.exit_code == 0

#     # ensure all needed files exist
#     assert (output_path / "target_rho_factors.yaml").exists()
#     assert (output_path / "emodels").is_dir()
#     assert (output_path / "emodels" / "final.json").exists()
#     assert (output_path / "emodels" / "cADpyr_L5TPC").is_dir()
#     assert (output_path / "etype_emodel_map.csv").exists()
#     assert (output_path / "ais_models.yaml").exists()
#     assert (output_path / "hoc_files/cADpyr_L5TPC.hoc").exists()

#     # ensures hoc file is correct
#     with open(output_path / "hoc_files/cADpyr_L5TPC.hoc") as f:
#         hoc_file = f.readlines()[2:]
#     with open(DATA / "cADpyr_L5TPC.hoc") as f:
#         expected_hoc_file = f.readlines()[2:]
#     assert expected_hoc_file == hoc_file


# def test_get_me_combos_scales():
#     os.environ["USE_NEURODAMUS"] = str(1)
#     response = runner.invoke(
#         cli,
#         [
#             "get_me_combos_scales",
#             "--cells-path",
#             str(DATA / "sonata.h5"),
#             "--release-path",
#             str(output_path),
#             "--morphology-path",
#             str(DATA / "morphologies"),
#             "--output-sonata-path",
#             "sonata_ais.h5",
#             "--emodel-api",
#             "local",
#         ],
#     )

#     # ensures cli run
#     assert response.exit_code == 0

#     # check AIS_scaler and model_template are assigned correctly
#     df = CellCollection().load_sonata("sonata_ais.h5").as_dataframe()
#     assert df["model_template"].to_list() == ["hoc:cADpyr_L5TPC", "hoc:cADpyr_L5TPC"]
#     npt.assert_allclose(
#         df["@dynamics:AIS_scaler"].to_list(), [1.964936407296357, 2.1514433186516744], rtol=1e-6
#     )


# def test_get_me_combos_currents():
#     response = runner.invoke(
#         cli,
#         [
#             "get_me_combos_currents",
#             "--input-sonata-path",
#             "sonata_ais.h5",
#             "--release-path",
#             str(output_path),
#             "--morphology-path",
#             str(DATA / "morphologies"),
#             "--protocol-config-path",
#             str(DATA / "protocol_config.yaml"),
#             "--output-sonata-path",
#             "sonata_currents.h5",
#             "--emodel-api",
#             "local",
#         ],
#     )

#     # ensures cli run
#     assert response.exit_code == 0

#     # ensures currents are correct
#     df = CellCollection().load_sonata("sonata_currents.h5").as_dataframe()
#     npt.assert_allclose(
#         df["@dynamics:holding_current"].to_list(),
#         [-0.03673185397730094, -0.04284688945972448],
#         rtol=1e-6,
#     )
#     npt.assert_allclose(
#         df["@dynamics:threshold_current"].to_list(), [0.1159453125, 0.131859375], rtol=1e-6
#     )


# def test_evaluate_emodels():
#     os.environ["USE_NEURODAMUS"] = str(1)
#     response = runner.invoke(
#         cli,
#         [
#             "evaluate_emodels",
#             "--sonata-path",
#             "sonata_ais.h5",
#             "--morphology-path",
#             str(DATA / "morphologies"),
#             "--emodel-api",
#             "local",
#             "--emodel-path",
#             str(DATA / "configs"),
#             "--final-path",
#             str(DATA / "final.json"),
#             "--ais-emodels-path",
#             str(output_path / "ais_models.yaml"),
#             "--megate-thresholds-path",
#             str(DATA / "megate_thresholds.yaml"),
#         ],
#     )

#     # ensures cli run
#     assert response.exit_code == 0

#     df_exemplar = pd.read_csv("exemplar_evaluations.csv")
#     expected_df_exemplar = pd.read_csv(DATA / "exemplar_evaluations.csv")
#     assert_frame_equal(df_exemplar, expected_df_exemplar, rtol=1e-3)

#     # remove path column to avoid issues with absolute paths
#     df_eval = pd.read_csv("results/region_None/results.csv").drop(columns=["path"])
#     expected_df_eval = pd.read_csv(DATA / "cell_evaluations.csv")
#     assert_frame_equal(df_eval, expected_df_eval, rtol=1e-3)


# def test_detect_stuck_cells():
#     del os.environ["USE_NEURODAMUS"]
#     response = runner.invoke(
#         cli,
#         [
#             "detect_stuck_cells",
#             "--sonata-path",
#             "sonata_ais.h5",
#             "--morphology-path",
#             str(DATA / "morphologies"),
#             "--emodel-hoc-path",
#             str(output_path / "hoc_files"),
#             "--protocol-config-path",
#             str(DATA / "protocol_config.yaml"),
#         ],
#     )

#     # ensures cli run
#     assert response.exit_code == 0

#     df = pd.read_csv("stuck_df/stuck_df_None.csv")
#     expected_df = pd.read_csv(DATA / "stuck_df.csv")
#     assert_frame_equal(df, expected_df)
