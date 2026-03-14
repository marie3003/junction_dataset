#!/usr/bin/env bash
# Usage: bash run_all_lexicmap_array.sh /path/to/fasta/folder
#
# Generates a jobs list file and submits a SLURM array job (100 concurrent).
# Each line in the jobs list contains the arguments for one lexicmap search.

FOLDER="$1"
JOBS_FILE="$(realpath "$FOLDER")/lexicmap_jobs.txt"

# --- Build jobs list ---
> "$JOBS_FILE"

find "$FOLDER" -type f -name '*.fasta' | sort | while read -r f; do

    L=$(grep -m1 '^>' "$f" | sed -E 's/.*length([0-9]+).*/\1/')
    if [[ -z "$L" ]]; then
        echo "Could not extract length for $f, skipping."
        continue
    fi

    if (( L < 200 )); then
        echo "Skipping $f (length=$L < 200)"
        continue
    fi

    if (( L <= 1000 )); then
        QCOV_HSP=90; QCOV_GENOME=90; ALIGN_LEN=200; SEED_P=15; SEED_PP=17
    elif (( L <= 2500 )); then
        QCOV_HSP=70; QCOV_GENOME=90; ALIGN_LEN=1000; SEED_P=17; SEED_PP=19
    else
        QCOV_HSP=50; QCOV_GENOME=80; ALIGN_LEN=1000; SEED_P=19; SEED_PP=21
    fi

    echo "$f 98 $QCOV_HSP $QCOV_GENOME $ALIGN_LEN $SEED_P $SEED_PP" >> "$JOBS_FILE"

done

N=$(wc -l < "$JOBS_FILE")
if (( N == 0 )); then
    echo "No valid query files found in $FOLDER. Exiting."
    exit 1
fi

echo "Submitting array job with $N tasks (100 concurrent) using jobs list: $JOBS_FILE"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sbatch --array=1-${N}%100 "$SCRIPT_DIR/lexicmap_query_array.sh" "$JOBS_FILE"
