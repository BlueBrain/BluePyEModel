deactivate
module purge all

export VENV=myvenv
module load unstable python
python -m venv $VENV
module purge all
source $VENV/bin/activate

mkdir software
cd software

../$VENV/bin/pip install --upgrade setuptools pip
../$VENV/bin/pip install wheel

git clone https://github.com/BlueBrain/BluePyEModel.git
../$VENV/bin/pip install -e ./BluePyEModel

cd ..
source $VENV/bin/activate
