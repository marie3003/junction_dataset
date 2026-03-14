#!/bin/bash
#SBATCH -J lexicmap_array
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -D /scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index/

module load Miniconda3/24.7.1-0
source "${EBROOTMINICONDA3}/etc/profile.d/conda.sh"
conda activate awscli

JOBS_FILE="$1"

# Read the line corresponding to this array task
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")
read -r INPUT_FILE PIDENT QCOV_HSP QCOV_GENOME ALIGN_LEN SEED_P SEED_PP <<< "$LINE"

INPUT_DIR=$(dirname "$INPUT_FILE")
BASENAME=$(basename "$INPUT_FILE" .fasta)
OUTPUT_FILE="${INPUT_DIR}/${BASENAME}.lexicmap.tsv"

lexicmap search \
-d atb.lmi \
"$INPUT_FILE" \
-o "$OUTPUT_FILE" \
--align-min-match-pident "$PIDENT" \
--min-qcov-per-hsp "$QCOV_HSP" \
--min-qcov-per-genome "$QCOV_GENOME" \
--align-min-match-len "$ALIGN_LEN" \
-p "$SEED_P" \
-P "$SEED_PP" \
--top-n-genomes 10000
