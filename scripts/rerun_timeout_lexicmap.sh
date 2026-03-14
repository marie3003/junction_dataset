#!/usr/bin/env bash
# Usage: bash rerun_timeout_lexicmap.sh <job_id> [logs_dir]
#
# Scans SLURM .err log files matching lexicmap_array_<job_id>_*.err for jobs
# cancelled due to time limit, extracts the input file path, rebuilds the
# parameter set, and submits a new array job with a longer time limit.
#
# Run from the lexicmap_index directory (same as the original jobs).

JOB_ID="${1:?Usage: bash rerun_timeout_lexicmap.sh <job_id> [logs_dir]}"
LOGS_DIR="${2:-logs}"
WORK_DIR="/scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index"
JOBS_FILE="${WORK_DIR}/lexicmap_timeout_rerun_jobs.txt"

> "$JOBS_FILE"

for err_file in "$LOGS_DIR"/lexicmap_array_${JOB_ID}_*.err; do
    # Check if this job timed out
    if ! grep -q "DUE TO TIME LIMIT" "$err_file"; then
        continue
    fi

    # Extract the relative input file path from the log
    REL_PATH=$(grep "input file given:" "$err_file" | sed -E 's/.*input file given: (.*)/\1/' | tr -d '[:space:]')
    if [[ -z "$REL_PATH" ]]; then
        echo "Warning: could not extract file path from $err_file, skipping."
        continue
    fi

    # Resolve to absolute path (paths in logs are relative to WORK_DIR)
    if [[ "$REL_PATH" = /* ]]; then
        f="$REL_PATH"
    else
        f="${WORK_DIR}/${REL_PATH#./}"
    fi

    # Extract length from FASTA header
    L=$(grep -m1 '^>' "$f" | sed -E 's/.*length([0-9]+).*/\1/')
    if [[ -z "$L" ]]; then
        echo "Warning: could not extract length for $f, skipping."
        continue
    fi

    if (( L < 200 )); then
        echo "Skipping $f (length=$L < 200)"
        continue
    fi

    # Assign parameters by length regime (same as original)
    if (( L <= 1000 )); then
        QCOV_HSP=90; QCOV_GENOME=90; ALIGN_LEN=200; SEED_P=15; SEED_PP=17
    elif (( L <= 2500 )); then
        QCOV_HSP=70; QCOV_GENOME=90; ALIGN_LEN=1000; SEED_P=17; SEED_PP=19
    else
        QCOV_HSP=50; QCOV_GENOME=80; ALIGN_LEN=1000; SEED_P=19; SEED_PP=21
    fi

    echo "$f 98 $QCOV_HSP $QCOV_GENOME $ALIGN_LEN $SEED_P $SEED_PP" >> "$JOBS_FILE"
    echo "Queued: $f (length=$L)"
done

N=$(wc -l < "$JOBS_FILE")
if (( N == 0 )); then
    echo "No timed-out jobs found in $LOGS_DIR."
    exit 0
fi

echo ""
echo "Submitting $N rerun jobs (100 concurrent, 60 min time limit)."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sbatch --time=01:00:00 --array=1-${N}%100 "$SCRIPT_DIR/lexicmap_query_array.sh" "$JOBS_FILE"
