#!/bin/bash
#SBATCH -J fetch_biosample
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -D /scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index/

module load Miniconda3/24.7.1-0
source "${EBROOTMINICONDA3}/etc/profile.d/conda.sh"
conda activate awscli

CHUNKS_DIR="$1"

# Get the chunk file for this array task (sorted, 1-indexed)
CHUNK_FILE=$(ls "$CHUNKS_DIR"/chunk_[0-9]* | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [[ -z "$CHUNK_FILE" ]]; then
    echo "No chunk file found for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

BASENAME=$(basename "$CHUNK_FILE")
OUTPUT_FILE="${CHUNKS_DIR}/${BASENAME}_results.tsv"

echo "Task $SLURM_ARRAY_TASK_ID: processing $CHUNK_FILE -> $OUTPUT_FILE"
bash fetch_biosample.sh "$CHUNK_FILE" "$OUTPUT_FILE"
