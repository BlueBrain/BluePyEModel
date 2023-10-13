#!/bin/bash

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

deactivate
module purge all # comment out this line if you are not using spack packages

module load unstable python # comment out this line if you are not using spack packages
python -m venv myvenv
module purge all # comment out this line if you are not using spack packages
source myvenv/bin/activate

pip install bluepyemodel