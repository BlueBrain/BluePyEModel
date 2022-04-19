# """Test apps module."""
# import numpy.testing as npt
# import shutil
# import os
# from pathlib import Path

# from click.testing import CliRunner

# from voxcell import CellCollection

# runner = CliRunner()

# DATA = Path(__file__).parents[1] / "test_data_apps"
# output_path = Path("out_emodel_release")
# os.environ["USE_NEURODAMUS"] = str(1)

# from bluepyemodel.apps.emodel_release import cli

# def test_create_emodel_release():
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
