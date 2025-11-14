#!/usr/bin/env bash

# Usage: bash run_all_lexicmap.sh /path/to/fasta/folder
FOLDER="$1"

# Pattern for your files (adjust folder if needed)
for f in "${FOLDER}"/NZ_*_segment_*.fasta; do
    [ -e "$f" ] || continue  # skip if no file matches name pattern

    # Extract the number after "length" from the FASTA header line
    # Example header: >NZ_CP018970.1|segment_0 path[...] length767
    L=$(grep -m1 '^>' "$f" | sed -E 's/.*length([0-9]+).*/\1/')
    if [[ -z "$L" ]]; then
        echo "Could not extract length for $f, skipping."
        continue
    fi

    # Keep pident fixed at 99 (adjust if you like)
    PIDENT=98

    # Simple rule for align-min-match-len:
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
  sbatch lexicmap_query_short.sh "${FOLDER}"/"$f" "$PIDENT" "$LEN_PARAM"
done
