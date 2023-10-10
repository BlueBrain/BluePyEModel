# Dowlnoad the input traces from the repository https://github.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction/input-traces/C060109A1-SR-C1
USER="BlueBrain"
REPO="SSCxEModelExamples"
BRANCH="main"
FOLDER="feature_extraction/input-traces/C060109A1-SR-C1"
BASE_API="https://api.github.com/repos/$USER/$REPO/git/trees/$BRANCH?recursive=1"
DEST_DIR="ephys_data/C060109A1-SR-C1" # should match the path in the targets.py file

# Fetch the list of files
FILES=$(curl -s $BASE_API | python -c "
import sys, json
files = [item['path'] for item in json.load(sys.stdin)['tree'] if item['path'].startswith('$FOLDER/')]
for f in files:
    print(f)
")

# Download each file
for file in $FILES; do
    filename=$(basename $file)

    if [ ! -f "$DEST_DIR/$filename" ]; then
        wget "https://raw.githubusercontent.com/$USER/$REPO/$BRANCH/$file" -P "$DEST_DIR/$(dirname $filename)"
    else
        echo "$filename already exists. Skipping download."
    fi
done