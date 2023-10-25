#####################################################################
# Copyright 2023, EPFL/Blue Brain Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################

source ./myvenv/bin/activate

export OPT_EMODEL="L5PC"

for seed in {1..5}; do
    export OPT_SEED=${seed}
    sbatch -J "export_hoc_${OPT_EMODEL}_${OPT_SEED}"  ./export_hoc.sbatch
done
