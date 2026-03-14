#!/usr/bin/env bash
# Usage: bash collect_and_fetch_array.sh /path/to/folder
#
# 1. Collects all unique sgenome IDs from *.lexicmap.tsv files in the folder.
# 2. Splits them into chunks of 10,000 IDs each.
# 3. Submits a SLURM array job where each task runs fetch_biosample.sh on one chunk.
# 4. After all tasks finish, merge chunk results into all_ncbi_results.tsv.

FOLDER="${1:?Please provide folder path}"
CHUNKS_DIR="${FOLDER}/biosample_chunks"
ALL_IDS="${FOLDER}/all_ids.txt"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$CHUNKS_DIR"

# --- Step 1: collect all unique sgenome IDs ---
find "$FOLDER" -type f -name "*.lexicmap.tsv" | while read -r f; do
    awk -F'\t' 'NR==1 {
        for (i=1; i<=NF; i++) if ($i == "sgenome") { col=i; next }
    } NR>1 && col { print $col }' "$f"
done | awk 'NF > 0' | sort -u > "$ALL_IDS"

count=$(wc -l < "$ALL_IDS")
echo "Collected $count unique IDs into $ALL_IDS"

if (( count == 0 )); then
    echo "No IDs found. Exiting."
    exit 1
fi

# --- Step 2: split into chunks of 10,000 ---
split -l 10000 -d --suffix-length=4 "$ALL_IDS" "${CHUNKS_DIR}/chunk_"

N=$(ls "${CHUNKS_DIR}"/chunk_[0-9]* | wc -l)
echo "Split into $N chunks of up to 10,000 IDs each."

# --- Step 3: submit array job ---
echo "Submitting array job with $N tasks."
sbatch --array=1-${N} "$SCRIPT_DIR/fetch_biosample_array.sh" "$CHUNKS_DIR"
