#!/bin/sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODELS_DIRECTORY=$(realpath "$SCRIPT_DIR"/../brainscore_vision/models/)
MODEL_PLUGINS=($(ls -d -- "$MODELS_DIRECTORY"/*))
for plugin_dir in "${MODEL_PLUGINS[@]}"; do
  models_dir="$plugin_dir"/models
  init_file="$models_dir"/__init__.py

  # if init file present and empty, delete it
  if [ -f "$init_file" ]; then
    if ! grep -q '[^[:space:]]' "$init_file"; then
      echo "init file empty: $init_file --> deleting"
      rm "$init_file"
    fi
  fi

  # if models directory exists and empty, delete it
  if [ -d "$models_dir" ]; then
    if [ -z "$(ls -A "$models_dir")" ]; then
      echo "models dir empty: $models_dir --> deleting"
      rm -r "$models_dir"
    fi
  fi
done
