#!/usr/bin/env bash
set -euo pipefail

BASE="data"
DATA_ROOT="$BASE/data"
SCRIPTS="$BASE/scripts"

# Process all folders in data/equipocket
for folder in "$DATA_ROOT"/*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        python3 "$SCRIPTS/protein_features.py" -p "$folder" -j "$(nproc)" -b processes
        python3 "$SCRIPTS/extract_binding_atoms.py" -p "$folder" -j "$(nproc)" -t 4.0 -b processes
    fi
done

echo "All folders processed."
