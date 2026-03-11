#!/bin/bash
#SBATCH -J lexicmap_job
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:05:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -D /scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index/

module load Miniconda3/24.7.1-0
source "${EBROOTMINICONDA3}/etc/profile.d/conda.sh"
conda activate awscli

# Input arguments
INPUT_FILE="$1"
PIDENT="$2"
QCOV_HSP="$3"
QCOV_GENOME="$4"
ALIGN_LEN="$5"
SEED_P="$6"
SEED_PP="$7"

# Derive output filename
INPUT_DIR=$(dirname "$INPUT_FILE")
BASENAME=$(basename "$INPUT_FILE" .fasta)
OUTPUT_FILE="${INPUT_DIR}/${BASENAME}.lexicmap.tsv"
IDS_FILE="${INPUT_DIR}/${BASENAME}.ids.txt"

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
--top-n-genomes 1000

awk -F'\t' '
NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i == "sgenome") col=i
    }
    if (!col) {
        print "Error: no sgenome column found in " FILENAME > "/dev/stderr"
        exit 1
    }
    next
}
{ print $col }
' "$OUTPUT_FILE" > "$IDS_FILE"
