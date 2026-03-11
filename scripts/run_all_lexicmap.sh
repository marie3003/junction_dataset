#!/usr/bin/env bash

# Usage: bash run_all_lexicmap.sh /path/to/fasta/folder
FOLDER="$1"

find "$FOLDER" -type f -name '*.fasta' | while read -r f; do

    # Extract length from FASTA header
    L=$(grep -m1 '^>' "$f" | sed -E 's/.*length([0-9]+).*/\1/')
    if [[ -z "$L" ]]; then
        echo "Could not extract length for $f, skipping."
        continue
    fi

    # Skip queries below 200 bp (likely pangenome graph artefacts)
    if (( L < 200 )); then
        echo "Skipping $f (length=$L < 200)"
        continue
    fi

    # Assign parameters by length regime
    if (( L <= 1000 )); then
        # Short: 200–1000 bp
        QCOV_HSP=90; QCOV_GENOME=90; ALIGN_LEN=200; SEED_P=15; SEED_PP=17
    elif (( L <= 2500 )); then
        # Medium: 1000–2500 bp
        QCOV_HSP=70; QCOV_GENOME=90; ALIGN_LEN=1000; SEED_P=17; SEED_PP=19
    else
        # Large: >2500 bp
        QCOV_HSP=50; QCOV_GENOME=80; ALIGN_LEN=1000; SEED_P=19; SEED_PP=21
    fi

    echo "Submitting: $f (length=$L) -> qcov_hsp=$QCOV_HSP qcov_genome=$QCOV_GENOME align_len=$ALIGN_LEN -p $SEED_P -P $SEED_PP"
    sbatch lexicmap_query.sh "$f" 99 "$QCOV_HSP" "$QCOV_GENOME" "$ALIGN_LEN" "$SEED_P" "$SEED_PP"

done
