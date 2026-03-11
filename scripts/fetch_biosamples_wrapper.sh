#!/usr/bin/env bash
set -euo pipefail

# run with relative folder as option, e.g. bash fetch_biosamples_consensus.sh CIRMBUYJFK_f__CWCCKOQCWZ_r/consensus1
FOLDER="$1"

# only runs if files are not there already
find "$FOLDER" -type f -name "*.ids.txt" | while read -r ids_file; do
    output_file="${ids_file%.ids.txt}.ncbi_results.tsv"
    if [ ! -f "$output_file" ]; then
        echo "Missing output for $(basename "$ids_file"), rerunning..."
        bash fetch_biosample.sh "$ids_file" "$output_file"
    fi
done

