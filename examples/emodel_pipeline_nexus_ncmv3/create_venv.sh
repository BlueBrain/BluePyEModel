deactivate
module purge all

export VENV=myvenv
module load unstable python
python -m venv $VENV
module purge all
source $VENV/bin/activate

mkdir softwares
cd softwares

../$VENV/bin/pip install --upgrade setuptools pip
../$VENV/bin/pip install setuptools==57
../$VENV/bin/pip install wheel
../$VENV/bin/pip install --ignore-installed --no-deps luigi

../$VENV/bin/pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy[all]
../$VENV/bin/pip install -i https://bbpteam.epfl.ch/repository/devpi/simple entity-management
../$VENV/bin/pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow
../$VENV/bin/pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow-cli
../$VENV/bin/pip install -i https://bbpteam.epfl.ch/repository/devpi/simple icselector
../$VENV/bin/pip install nexusforge

git clone https://bbpgitlab.epfl.ch/cells/bluepyemodel.git
../$VENV/bin/pip install -e ./bluepyemodel[cma,luigi,nexus]

cd ..

deactivate
module purge all

export VENV=myvenv_cli
module load unstable python
python -m venv $VENV
module purge all
source $VENV/bin/activate
$VENV/bin/pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow-cli

deactivate
