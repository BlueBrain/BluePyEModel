from pathlib import Path
from bluepyemodel.generalisation.combodb import ComboDB
from bluepyemodel import api


if __name__ == "__main__":
    xml_path = "/gpfs/bbp.cscs.ch/project/proj82/home/gevaert/morphology_release/mouse-scaled/scaled_output/06_RepairUnravel-asc/neuronDB.xml"
    composition = "/gpfs/bbp.cscs.ch/project/proj83/data/recipes/cell_composition.yaml"
    emodel_dir = "/gpfs/bbp.cscs.ch/project/proj38/sscx_emodel_paper/data/emodels/configs"
    emodel_db = api.get_db("singlecell", emodel_dir=emodel_dir)
    db = ComboDB.from_neurondb(Path(xml_path), cell_composition=composition, emodel_db=emodel_db)
    print(db.combo_df)
