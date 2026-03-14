#!/bin/bash
#SBATCH -J collect_ids_and_fetch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -D /scicore/home/neher/GROUP/data/2025_all_the_bacteria/lexicmap_index/

set -uo pipefail
set +e

module load Miniconda3/24.7.1-0
source "${EBROOTMINICONDA3}/etc/profile.d/conda.sh"
conda activate awscli

mkdir -p logs

# Usage: sbatch collect_ids_and_fetch.sh /path/to/folder
# Collects all unique SAM... IDs from the sgenome column of *.lexicmap.tsv
# files in the folder, writes them to all_ids.txt, then runs fetch_biosample.sh.

FOLDER="${1:?Please provide folder path}"
ALL_IDS="${FOLDER}/all_ids.txt"
ALL_RESULTS="${FOLDER}/all_ncbi_results.tsv"

# Extract sgenome column from all *.lexicmap.tsv files, skip header, deduplicate
find "$FOLDER" -type f -name "*.lexicmap.tsv" | while read -r f; do
    awk -F'\t' 'NR==1 {
        for (i=1; i<=NF; i++) if ($i == "sgenome") { col=i; next }
    } NR>1 && col { print $col }' "$f"
done | awk 'NF > 0' | sort -u > "$ALL_IDS"

count=$(wc -l < "$ALL_IDS")
echo "Collected $count unique IDs into $ALL_IDS"

bash fetch_biosample.sh "$ALL_IDS" "$ALL_RESULTS"
