from Bio import SeqIO
from Bio.Seq import Seq


def reverse_complement(sequence):
    """
    Return the reverse complement of a DNA sequence.

    Args:
        sequence: Bio.Seq.Seq object or string

    Returns:
        Bio.Seq.Seq: Reverse complement of the input sequence
    """
    if isinstance(sequence, str):
        sequence = Seq(sequence)
    return sequence.reverse_complement()


def extract_circular_sequence(sequence, start, end):
    """
    Extract a subsequence from a circular sequence, handling wraparound.

    Args:
        sequence: Bio.Seq.Seq object (the complete circular sequence)
        start: int, 0-based start position
        end: int, 0-based end position (exclusive)

    Returns:
        Bio.Seq.Seq: Extracted subsequence
    """
    seq_len = len(sequence)

    # Validate positions
    if start < 0 or start >= seq_len:
        raise ValueError(f"Start position {start} out of range [0, {seq_len})")
    if end < 0 or end > seq_len:
        raise ValueError(f"End position {end} out of range [0, {seq_len}]")

    # Normal case: no wraparound
    if start < end:
        return sequence[start:end]

    # Wraparound case: extract from start to end of sequence,
    # then from beginning to end position
    elif start > end:
        return sequence[start:] + sequence[:end]

    # Edge case: start == end (could mean full circle or empty)
    # Treating as empty sequence
    else:
        return Seq("")


def extract_genbank_region(genbank_file, start, end, forward_orientation=True):
    """
    Extract a region from a circular GenBank plasmid with optional reverse complement.

    Args:
        genbank_file: str, path to GenBank file
        start: int, 0-based start position
        end: int, 0-based end position (exclusive, Python-style)
        forward_orientation: bool, if False, return reverse complement

    Returns:
        Bio.Seq.Seq: Extracted (and optionally reverse-complemented) sequence

    Example:
        # Extract positions 100-200 on forward strand
        seq = extract_genbank_region("plasmid.gb", 100, 200, True)

        # Extract wraparound region 9900-100 on reverse strand
        seq = extract_genbank_region("plasmid.gb", 9900, 100, False)
    """
    # Parse the GenBank file (take the first record)
    record = SeqIO.read(genbank_file, "genbank")

    # Check if the sequence is marked as circular
    if not record.annotations.get("topology") == "circular":
        print(
            f"Warning: Sequence topology is '{record.annotations.get('topology')}', "
            f"not 'circular'. Proceeding anyway."
        )

    # Extract the sequence from the circular plasmid
    extracted_seq = extract_circular_sequence(record.seq, start, end)

    # Apply reverse complement if needed
    if not forward_orientation:
        extracted_seq = reverse_complement(extracted_seq)

    return extracted_seq


def overlaps_interval(feat_start, feat_end, interval_start, interval_end, seq_len):
    """
    Check if a feature overlaps with an interval in a circular sequence.

    Args:
        feat_start: int, feature start position (0-based)
        feat_end: int, feature end position (0-based, exclusive)
        interval_start: int, interval start position
        interval_end: int, interval end position
        seq_len: int, length of the circular sequence

    Returns:
        bool: True if there's overlap
    """
    # Normalize positions to handle wraparound features
    # A feature wraps around if start > end
    feat_wraps = feat_start >= feat_end
    interval_wraps = interval_start >= interval_end

    if not feat_wraps and not interval_wraps:
        # Simple case: neither wraps
        return not (feat_end <= interval_start or feat_start >= interval_end)

    elif feat_wraps and not interval_wraps:
        # Feature wraps, interval doesn't
        # Feature covers [feat_start, seq_len) and [0, feat_end)
        return feat_start < interval_end or feat_end > interval_start

    elif not feat_wraps and interval_wraps:
        # Interval wraps, feature doesn't
        # Interval covers [interval_start, seq_len) and [0, interval_end)
        return interval_start < feat_end or interval_end > feat_start

    else:
        # Both wrap - they must overlap
        return True


def transform_coordinates(feat_start, feat_end, interval_start, interval_end, seq_len):
    """
    Transform feature coordinates to be relative to the extracted interval.

    Args:
        feat_start: int, original feature start
        feat_end: int, original feature end
        interval_start: int, interval start
        interval_end: int, interval end
        seq_len: int, sequence length

    Returns:
        tuple: (new_start, new_end, is_partial) where is_partial indicates if the
               feature only partially overlaps with the interval (has overhangs outside)
    """
    interval_wraps = interval_start >= interval_end

    if interval_wraps:
        interval_len = (seq_len - interval_start) + interval_end
    else:
        interval_len = interval_end - interval_start

    # Transform coordinates relative to interval start
    def relative_pos(pos):
        if interval_wraps:
            if pos >= interval_start:
                return pos - interval_start
            else:
                return (seq_len - interval_start) + pos
        else:
            return pos - interval_start

    new_start = relative_pos(feat_start)
    new_end = relative_pos(feat_end)

    # Check if feature extends beyond interval boundaries before clipping
    is_partial = new_start < 0 or new_end > interval_len

    # Clip to interval boundaries
    new_start = max(0, min(new_start, interval_len))
    new_end = max(0, min(new_end, interval_len))

    return new_start, new_end, is_partial


def extract_genbank_annotations(genbank_file, start, end, forward_orientation=True):
    """
    Extract annotations from a GenBank file that overlap with a specified interval.
    Returns annotations in GFF3-compatible format with coordinates relative to the interval.

    Args:
        genbank_file: str, path to GenBank file
        start: int, 0-based start position
        end: int, 0-based end position (exclusive)
        forward_orientation: bool, if False, coordinates and strands are reversed

    Returns:
        list of dict: Each dict contains GFF3 fields:
            - seqid: sequence identifier
            - source: annotation source
            - type: feature type (e.g., 'CDS', 'gene')
            - start: start position (1-based, GFF3 convention)
            - end: end position (1-based, inclusive, GFF3 convention)
            - score: score (usually '.')
            - strand: '+' or '-'
            - phase: phase for CDS (usually '.')
            - attributes: dict of attributes (e.g., {'ID': 'gene1', 'Name': 'lacZ'})

    Example:
        annotations = extract_genbank_annotations("plasmid.gb", 100, 200, True)
        for ann in annotations:
            print(f"{ann['type']} at {ann['start']}-{ann['end']} ({ann['strand']})")
    """
    # Parse the GenBank file
    record = SeqIO.read(genbank_file, "genbank")
    seq_len = len(record.seq)

    # Calculate interval length for coordinate transformation
    if start >= end:
        interval_len = (seq_len - start) + end
    else:
        interval_len = end - start

    annotations = []

    for feature in record.features:
        # Get feature coordinates
        feat_start = int(feature.location.start)
        feat_end = int(feature.location.end)
        feat_strand = feature.location.strand  # 1 for +, -1 for -

        # Check if feature overlaps with interval
        if not overlaps_interval(feat_start, feat_end, start, end, seq_len):
            continue

        # Transform coordinates to be relative to the interval
        new_start, new_end, is_partial = transform_coordinates(
            feat_start, feat_end, start, end, seq_len
        )

        # If extracting reverse strand, flip coordinates and strand
        if not forward_orientation:
            # Reverse the coordinates: position p becomes (interval_len - p)
            new_start_fwd = new_start
            new_end_fwd = new_end
            new_start = interval_len - new_end_fwd
            new_end = interval_len - new_start_fwd
            # Flip strand
            feat_strand = -feat_strand if feat_strand else 0

        # Convert to GFF3 format (1-based, inclusive coordinates)
        gff_start = new_start + 1
        gff_end = new_end

        # Handle strand
        if feat_strand == 1:
            strand = "+"
        elif feat_strand == -1:
            strand = "-"
        else:
            strand = "."

        # Build attributes dictionary
        attributes = {}

        # Add common qualifiers
        for key, value in feature.qualifiers.items():
            # Handle lists (most qualifiers are lists in Biopython)
            if isinstance(value, list):
                if len(value) == 1:
                    attributes[key] = value[0]
                else:
                    attributes[key] = ",".join(str(v) for v in value)
            else:
                attributes[key] = str(value)

        # Add ID and Name attributes if available
        if "locus_tag" in feature.qualifiers:
            attributes["ID"] = feature.qualifiers["locus_tag"][0]
        elif "gene" in feature.qualifiers:
            attributes["ID"] = feature.qualifiers["gene"][0]
        elif "label" in feature.qualifiers:
            attributes["ID"] = feature.qualifiers["label"][0]
        else:
            attributes["ID"] = f"{feature.type}_{feat_start}_{feat_end}"

        if "gene" in attributes:
            attributes["Name"] = attributes["gene"]
        elif "label" in attributes:
            attributes["Name"] = attributes["label"]
        elif "locus_tag" in attributes:
            attributes["Name"] = attributes["locus_tag"]
        else:
            attributes["Name"] = "unknown"

        # Indicate if the feature is partial (has overhangs)
        attributes["is_partial"] = str(is_partial)

        # Create GFF3 entry
        gff_entry = {
            "seqid": record.id,
            "source": "GenBank",
            "type": feature.type,
            "start": gff_start,
            "end": gff_end,
            "score": ".",
            "strand": strand,
            "phase": ".",
            "attributes": attributes,
        }

        annotations.append(gff_entry)

    return annotations


def url_encode_gff3(text):
    """
    URL encode special characters in GFF3 fields according to RFC 3986.
    Required for: tab, newline, carriage return, %, ;, =, &, comma in attributes.

    Args:
        text: str, text to encode

    Returns:
        str: URL-encoded text
    """
    if not isinstance(text, str):
        text = str(text)

    # Define characters that must be escaped in GFF3
    replacements = {
        "\t": "%09",
        "\n": "%0A",
        "\r": "%0D",
        "%": "%25",
        ";": "%3B",
        "=": "%3D",
        "&": "%26",
        ",": "%2C",
    }

    result = text
    for char, encoded in replacements.items():
        result = result.replace(char, encoded)

    return result


def write_gff3(annotations, output_file, sequence_region_lengths):
    """
    Write annotations to a GFF3 file following official specifications.

    Args:
        annotations: list of dict, output from extract_genbank_annotations
        output_file: str, path to output GFF3 file
        sequence_region_lengths: dict, mapping of seqid to region length
                                 e.g., {'NZ_CP070593.1': 11739, 'NC_000913.3': 4641652}

    GFF3 Format specifications:
    - Tab-delimited, 9 columns
    - Special characters must be URL-encoded (RFC 3986)
    - First line must be ##gff-version 3
    - ##sequence-region directives describe sequence extents
    - Attributes are tag=value pairs separated by semicolons
    - Standard attribute tags (capitalized) include: ID, Name, Alias, Parent, Target, etc.
    """
    with open(output_file, "w") as f:
        # Write mandatory GFF3 version header
        f.write("##gff-version 3\n")

        # Group annotations by seqid to write sequence-region directives
        if annotations:
            seqids_seen = set()

            # Calculate ranges for each seqid
            for ann in annotations:
                seqid = ann["seqid"]
                if seqid not in seqids_seen:
                    seqids_seen.add(seqid)

            # Write ##sequence-region directives for each seqid
            for seqid in sorted(seqids_seen):
                if seqid in sequence_region_lengths:
                    # Use provided length from dictionary
                    f.write(
                        f"##sequence-region {seqid} 1 {sequence_region_lengths[seqid]}\n"
                    )
                else:
                    # raise error if length not provided
                    raise ValueError(
                        f"Length for sequence '{seqid}' not provided in sequence_region_lengths."
                    )

        # Write annotation lines
        for ann in annotations:
            # Format attributes according to GFF3 specifications
            # Standard tags (capitalized): ID, Name, Alias, Parent, Target, Gap,
            # Derives_from, Note, Dbxref, Ontology_term
            # Custom tags should be lowercase

            attr_list = []

            # Process attributes in a specific order for readability
            # Standard attributes first
            priority_attrs = [
                "ID",
                "Name",
                "Alias",
                "Parent",
                "Target",
                "Gap",
                "Derives_from",
                "Note",
                "Dbxref",
                "Ontology_term",
            ]

            # Add priority attributes first
            for key in priority_attrs:
                if key in ann["attributes"]:
                    value = url_encode_gff3(str(ann["attributes"][key]))
                    attr_list.append(f"{key}={value}")

            # Add remaining attributes
            for key, value in sorted(ann["attributes"].items()):
                if key not in priority_attrs:
                    # Custom attributes should be lowercase (convert if needed)
                    key_formatted = (
                        key
                        if key[0].islower() or key in priority_attrs
                        else key.lower()
                    )
                    value_encoded = url_encode_gff3(str(value))
                    attr_list.append(f"{key_formatted}={value_encoded}")

            # Join attributes with semicolons (no spaces after semicolons per spec)
            attr_str = ";".join(attr_list) if attr_list else "."

            # Build the GFF3 line (tab-separated)
            # Ensure seqid doesn't contain invalid characters
            seqid_clean = (
                url_encode_gff3(ann["seqid"])
                if any(c in ann["seqid"] for c in "\t\n\r ;=,")
                else ann["seqid"]
            )

            line = "\t".join(
                [
                    seqid_clean,
                    ann["source"],
                    ann["type"],
                    str(ann["start"]),
                    str(ann["end"]),
                    str(ann["score"]),
                    ann["strand"],
                    str(ann["phase"]),
                    attr_str,
                ]
            )
            f.write(line + "\n")
