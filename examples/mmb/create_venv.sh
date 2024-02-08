deactivate
module purge all

module load unstable python
python -m venv myvenv
module purge all
source myvenv/bin/activate

pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow
pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow-cli

pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bluepyemodelnexus

deactivate
