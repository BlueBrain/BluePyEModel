deactivate
module purge all # comment out this line if the script is not running on BB5

module load unstable python # comment out this line if the script is not running on BB5
python -m venv myvenv
module purge all # comment out this line if the script is not running on BB5
source myvenv/bin/activate

pip install bluepyemodel