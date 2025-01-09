#!/bin/bash

#####################################################################
# Copyright 2023-2024 Blue Brain Project / EPFL

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

# Download the input traces from the repository https://github.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction/input-traces/C060109A1-SR-C1
USER="BlueBrain"
REPO="SSCxEModelExamples"
BRANCH="main"
FOLDER="feature_extraction/input-traces/C060109A1-SR-C1"
BASE_API="https://api.github.com/repos/$USER/$REPO/git/trees/$BRANCH?recursive=1"
DEST_DIR="ephys_data/C060109A1-SR-C1" # should match the path in the targets.py file

# List of files to download
FILES_TO_DOWNLOAD=(
    "X_IDrest_ch0_326.ibw"
    "X_IDrest_ch0_327.ibw"
    "X_IDrest_ch0_328.ibw"
    "X_IDrest_ch0_329.ibw"
    "X_IDrest_ch0_330.ibw"
    "X_IDrest_ch1_326.ibw"
    "X_IDrest_ch1_327.ibw"
    "X_IDrest_ch1_328.ibw"
    "X_IDrest_ch1_329.ibw"
    "X_IDrest_ch1_330.ibw"
    "X_IDthresh_ch0_349.ibw"
    "X_IDthresh_ch0_350.ibw"
    "X_IDthresh_ch0_351.ibw"
    "X_IDthresh_ch0_352.ibw"
    "X_IDthresh_ch0_353.ibw"
    "X_IDthresh_ch1_349.ibw"
    "X_IDthresh_ch1_350.ibw"
    "X_IDthresh_ch1_351.ibw"
    "X_IDthresh_ch1_352.ibw"
    "X_IDthresh_ch1_353.ibw"
    "X_IV_ch0_266.ibw"
    "X_IV_ch0_267.ibw"
    "X_IV_ch0_268.ibw"
    "X_IV_ch0_269.ibw"
    "X_IV_ch0_270.ibw"
    "X_IV_ch1_266.ibw"
    "X_IV_ch1_267.ibw"
    "X_IV_ch1_268.ibw"
    "X_IV_ch1_269.ibw"
    "X_IV_ch1_270.ibw"
)

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Fetch the list of files from the GitHub API and filter for required files
FILES=$(curl -s "$BASE_API" | grep -oE "\"path\": \"$FOLDER/[^\"]*\.ibw")

# Download each file if it matches the list to download
for file in $FILES; do
    filename=$(basename "$file")

    if [[ " ${FILES_TO_DOWNLOAD[*]} " == *" $filename "* ]]; then
        if [ ! -f "$DEST_DIR/$filename" ]; then
            wget "https://raw.githubusercontent.com/$USER/$REPO/$BRANCH/$FOLDER/$filename" -P "$DEST_DIR"
        else
            echo "$filename already exists. Skipping download."
        fi
    fi
done
