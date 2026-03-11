#!/bin/bash
set -euo pipefail

INPUT_FILE="${1:?give input file}"
OUTPUT_FILE="${2:-biosample_summary.tsv}"

INITIAL_BATCH_SIZE=100
MAX_ROUNDS=4

tmp_dir=$(mktemp -d)
tmp_ids="$tmp_dir/ids.txt"
tmp_remaining="$tmp_dir/remaining.txt"
tmp_found="$tmp_dir/found.tsv"
tmp_found_ids="$tmp_dir/found_ids.txt"

trap 'rm -rf "$tmp_dir"' EXIT

printf "sgenome\torganism\tstrain\tserovar\tsequence_type\n" > "$OUTPUT_FILE"

# unique non-empty IDs for querying
awk 'NF > 0' "$INPUT_FILE" | sort -u > "$tmp_ids"
cp "$tmp_ids" "$tmp_remaining"
: > "$tmp_found"

run_batch() {
    local batch_file="$1"
    local query

    query=$(awk '{printf "%s%s[ACCN]", (NR==1?"":" OR "), $0}' "$batch_file")

    esearch -db biosample -query "$query" | \
      efetch -format docsum | \
      xtract -pattern DocumentSummary -def NA \
        -element Accession Organism \
        -block Attribute -if "@attribute_name" -equals strain -element . \
        -block Attribute -if "@attribute_name" -equals serovar -element . \
        -block Attribute -if "@attribute_name" -equals sequence_type -element .
}

batch_query_file() {
    local ids_file="$1"
    local batch_size="$2"
    local round="$3"

    local split_prefix="$tmp_dir/split_r${round}_"
    split -l "$batch_size" "$ids_file" "$split_prefix"

    for chunk in "${split_prefix}"*; do
        [[ -s "$chunk" ]] || continue
        echo "Round $round: processing $(basename "$chunk") with batch size $batch_size" >&2

        local out="$chunk.out"
        if run_batch "$chunk" > "$out" 2>/dev/null; then
            cat "$out" >> "$tmp_found"
        else
            echo "Round $round: batch failed: $(basename "$chunk")" >&2
        fi

        sleep 0.5
    done
}

for round in $(seq 1 "$MAX_ROUNDS"); do
    if [[ ! -s "$tmp_remaining" ]]; then
        break
    fi

    case "$round" in
        1) batch_size=$INITIAL_BATCH_SIZE ;;
        2) batch_size=25 ;;
        3) batch_size=5 ;;
        4) batch_size=1 ;;
        *) batch_size=1 ;;
    esac

    batch_query_file "$tmp_remaining" "$batch_size" "$round"

    # deduplicate found rows by accession, keep first occurrence
    awk -F'\t' '!seen[$1]++' "$tmp_found" > "$tmp_found.tmp"
    mv "$tmp_found.tmp" "$tmp_found"

    # collect successfully returned accessions
    cut -f1 "$tmp_found" | sort -u > "$tmp_found_ids"

    # recompute unresolved IDs
    comm -23 <(sort "$tmp_ids") "$tmp_found_ids" > "$tmp_remaining"

    remaining_count=$(wc -l < "$tmp_remaining")
    found_count=$(wc -l < "$tmp_found_ids")
    echo "After round $round: found=$found_count remaining=$remaining_count" >&2

    sleep 1
done

# final lookup map
awk -F'\t' 'BEGIN{OFS="\t"} !seen[$1]++ {print}' "$tmp_found" > "$tmp_dir/final_found.tsv"

# restore original input order, including duplicates, and only now fill unresolved with NA
while IFS= read -r ID; do
    row=$(awk -F'\t' -v id="$ID" '$1==id {print; exit}' "$tmp_dir/final_found.tsv")
    if [[ -z "$row" ]]; then
        printf "%s\tNA\tNA\tNA\tNA\n" "$ID" >> "$OUTPUT_FILE"
    else
        printf "%s\n" "$row" >> "$OUTPUT_FILE"
    fi
done < "$INPUT_FILE"