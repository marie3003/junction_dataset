#!/usr/bin/env bash
# Usage: bash merge_biosample_results.sh /path/to/folder
#
# Merges all chunk_*_results.tsv files from biosample_chunks/ into
# all_ncbi_results.tsv, keeping the header from the first file only.

FOLDER="${1:?Please provide folder path}"
CHUNKS_DIR="${FOLDER}/biosample_chunks"
OUTPUT="${FOLDER}/all_ncbi_results.tsv"

first=1
for f in $(ls "$CHUNKS_DIR"/chunk_*_results.tsv | sort); do
    if (( first )); then
        cat "$f" > "$OUTPUT"
        first=0
    else
        tail -n +2 "$f" >> "$OUTPUT"
    fi
done

count=$(( $(wc -l < "$OUTPUT") - 1 ))
echo "Merged $count result rows into $OUTPUT"
