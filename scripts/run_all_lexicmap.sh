#!/usr/bin/env bash

# Usage: bash run_all_lexicmap.sh /path/to/fasta/folder
FOLDER="$1"

# Pattern for your files (adjust folder if needed)
find "$FOLDER" -type f -name 'NZ_*_segment_*.fasta' | while read -r f; do

    # Extract length from FASTA header
    L=$(grep -m1 '^>' "$f" | sed -E 's/.*length([0-9]+).*/\1/')
    if [[ -z "$L" ]]; then
        echo "Could not extract length for $f, skipping."
        continue
    fi

    PIDENT=98

    if (( L < 500 )); then
        echo "Skipping $f (length=$L < 500)"
        continue
    elif (( L > 2500 )); then
        LEN_PARAM=2000
    elif (( L > 1200 )); then
        LEN_PARAM=1000
    else
        LEN_PARAM=400
    fi

    echo "Submitting: $f (length=$L) -> pident=$PIDENT len=$LEN_PARAM"
    sbatch lexicmap_query.sh "$f" "$PIDENT" "$LEN_PARAM"

done
