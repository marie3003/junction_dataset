#!/bin/bash
#SBATCH -J lexicmap_job
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -D /scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index/

module load Miniconda3/24.7.1-0
source "${EBROOTMINICONDA3}/etc/profile.d/conda.sh"
conda activate awscli

# Input arguments
INPUT_FILE="$1"
ALIGN_MIN_MATCH_PIDENT="$2"
ALIGN_MIN_MATCH_LEN="$3"

# Conditionally add parameters if align_min_match_len > 2000
EXTRA_PARAMS=""
if (( ALIGN_MIN_MATCH_LEN > 2000 )); then
  EXTRA_PARAMS="-p 17 -P 19"
fi

# Derive output filename
INPUT_DIR=$(dirname "$INPUT_FILE")
BASENAME=$(basename "$INPUT_FILE" .fasta)
OUTPUT_FILE="${INPUT_DIR}/${BASENAME}.lexicmap.tsv"
IDS_FILE="${INPUT_DIR}/${BASENAME}.ids.txt"

lexicmap search \
-d atb.lmi \
"$INPUT_FILE" \
-o "$OUTPUT_FILE" \
--align-min-match-pident "$ALIGN_MIN_MATCH_PIDENT" \
$EXTRA_PARAMS \
--min-qcov-per-hsp 90 \
--min-qcov-per-genome 90 \
--align-min-match-len "$ALIGN_MIN_MATCH_LEN" \
--top-n-genomes 200

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