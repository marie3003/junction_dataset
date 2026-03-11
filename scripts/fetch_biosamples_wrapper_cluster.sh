#!/bin/bash
#SBATCH -J fetch_biosamples
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

FOLDER="${1:?Please provide folder path}"

while read -r ids_file; do
    output_file="${ids_file%.ids.txt}.ncbi_results.tsv"
    if [ ! -f "$output_file" ]; then
        echo "Processing $ids_file ..."
        if ! bash fetch_biosample.sh "$ids_file" "$output_file" < /dev/null; then
            echo "Failed: $ids_file" >&2
        fi
    fi
done < <(find "$FOLDER" -type f -name "*.ids.txt")