#!/bin/bash
#SBATCH -J lexicmap_job
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# set working directory
#SBATCH -D /scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index/

module load Miniconda3/24.7.1-0
source "${EBROOTMINICONDA3}/etc/profile.d/conda.sh"
conda activate awscli

# Input argument (FASTA file)
FILENAME="$1"
# Derive output filename
INPUT_FILE="queries/${FILENAME}"
BASENAME=$(basename "$FILENAME" .fasta)
OUTPUT_FILE="results/${BASENAME}.lexicmap.tsv"

lexicmap search \
-d atb.lmi \
"$INPUT_FILE" \
-o "$OUTPUT_FILE" \
--align-min-match-pident 98 \
-p 17 -P 19 \
--min-qcov-per-hsp 90 \
--min-qcov-per-genome 90 \
--align-min-match-len 1000 \
--top-n-genomes 100