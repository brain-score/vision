#!/bin/sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TARGET_DIRECTORY=$(realpath "$SCRIPT_DIR"/../brainscore_vision/models/)
TEMPORARY_UNZIP_DIRECTORY=$(realpath "$TARGET_DIRECTORY"/../models_unzip_temp)
SUBMISSIONS_DIRECTORY=/braintree/data2/active/common/brainscore_submissions

# retrieve submission ids to work with (these are all the successful submissions with a score only)
SUBMISSION_IDS=()
while IFS=',' read -ra array; do
  SUBMISSION_IDS+=("${array[0]}")
done <"$SCRIPT_DIR"/20230211_submission_ids.csv
NUM_SUBMISSIONS="${#SUBMISSION_IDS[@]}"
echo "Number of submissions: $NUM_SUBMISSIONS"

echo "Unzipping to $TEMPORARY_UNZIP_DIRECTORY, moving to $TARGET_DIRECTORY"

counter=1
failures=()
for submission_id in "${SUBMISSION_IDS[@]}"; do
  submission_zip="$SUBMISSIONS_DIRECTORY"/submission_"$submission_id".zip
  # log
  echo "Processing $submission_zip ($counter/$NUM_SUBMISSIONS)"
  counter=$((counter + 1))

  # unzip
  rm -rf --- "${TEMPORARY_UNZIP_DIRECTORY:?}"/*                      # clear temporary directory
  unzip "$submission_zip" -d "$TEMPORARY_UNZIP_DIRECTORY" >/dev/null # unzip, do not print stdout
  chmod -R ugo+w "$TEMPORARY_UNZIP_DIRECTORY"
  rm -rf "${TEMPORARY_UNZIP_DIRECTORY:?}"/__MACOSX  # clear Apple leftovers

  # retrieve contents of zip file
  num_directories=$(find "$TEMPORARY_UNZIP_DIRECTORY"/* -maxdepth 0 -type d | wc -l)
  zip_contents=$(find "$TEMPORARY_UNZIP_DIRECTORY"/* -maxdepth 0 -type d)
  if [ "$num_directories" -ne 1 ]; then
    echo "Expected exactly one directory, got $num_directories: $zip_contents"
    exit 1
  fi

  # check if contents already exist in target directory -- if it does, rename with suffix until no conflicts
  target_base=$(basename "$zip_contents")
  target_name="$target_base"
  target_suffix=1
  while [ -d "$TARGET_DIRECTORY"/"$target_name" ]; do
    target_suffix=$((target_suffix + 1))
    target_name="$target_base"_"$target_suffix"
  done

  # move into target directory
  plugin_dir="$TARGET_DIRECTORY"/"$target_name"
  mv "$zip_contents" "$plugin_dir"

  # restructure: move `models/base_models.py` -> `model.py`, delete `brain_models.py`, and delete `models/` if empty
  models_file="$plugin_dir"/model.py
  mv "$plugin_dir"/models/base_models.py "$models_file"
  rm -f "$plugin_dir"/models/brain_models.py # -f to avoid error message if doesn't exist
  if [ -z "$(ls -A "$plugin_dir"/models)" ]; then
    rm -r "$plugin_dir"/models
  fi

  # restructure: delete `.git` directories as well as README.md and `test/` directory
  # since the latter two typically stem from the sample model submission
  rm -rf "$plugin_dir"/.git
  rm -f "$plugin_dir"/README.md
  rm -rf "$plugin_dir"/test/

  # restructure setup.py: remove model-tools, brain-score and result_caching dependencies from setup.py requirements
  # also clear standard parts from submission template
  setup_file="$plugin_dir"/setup.py
  sed -i '/model-tools @ /d' "$setup_file"
  sed -i '/brain-score @ /d' "$setup_file"
  sed -i '/result_caching @/d' "$setup_file"

  sed -i "/name='model-template'/d" "$setup_file"
  sed -i "/version='0.1.0'/d" "$setup_file"
  sed -i '/description="An example project for adding brain or base model implementation"/d' "$setup_file"
  sed -i '/author="Franziska Geiger"/d' "$setup_file"
  sed -i "/author_email='fgeiger@mit.edu'/d" "$setup_file"
  sed -i "/url='https:\\/\\/github\\.com\\/brain-score/d" "$setup_file"

  # restructure: add `__init__.py` with registry
  init_file="$plugin_dir"/__init__.py
  echo "from brainscore_vision import model_registry" >"$init_file"
  echo "from brainscore_vision.model_helpers import ModelCommitment" >>"$init_file"
  echo "from .model import get_model, get_layers" >>"$init_file"
  echo "" >>"$init_file"

  # retrieve all the identifiers
  # Since this part is in a python file, the easiest would be to load the file. But that requires loading dependencies
  # which takes forever, and is unstable and can lead to inconsistencies. Instead we will create a temporary file
  # that includes only the list for us to load.
  starting_line=$(sed -n '/^def get_model_list/{=;q;}' "$models_file")
  end_line=$(tail -n+"$((starting_line + 1))" "$models_file" | sed -n -E '/^(def|class) /{=;q;}')
  end_line=$((end_line + starting_line - 1))
  model_list_python=$(sed -n "$starting_line,$end_line"p "$models_file")
  identifiers=$(python -c "from typing import List
$model_list_python
print(' '.join(get_model_list()))") || { printf "\n>> FAILED: %s\n" "$submission_zip"; failures+=("$submission_zip"); }
  for identifier in $identifiers; do
    echo "model_registry['$identifier'] = ModelCommitment(identifier='$identifier', activations_model=get_model('$identifier'), layers=get_layers('$identifier'))" >>"$init_file"
  done

  # add empty `test.py`
  test_file="$plugin_dir"/test.py
  if [[ ! -e "$test_file" ]]; then
    echo "# Left empty as part of 2023 models migration" > "$test_file"
  fi
done

echo "Failures: ${failures[*]}"
