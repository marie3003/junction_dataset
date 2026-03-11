import scipy.cluster.hierarchy as sch
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import Phylo, SeqIO, Align, AlignIO
from Bio.Phylo.BaseTree import Tree
import copy
from Bio.motifs import Motif
import os, re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

def repo_root(start: Path | None = None) -> Path:
    """
    Walk upwards from `start` (or from this file) until we find the repo root.
    Repo root is identified by containing: junction_analysis/, results/, data/
    """
    if start is None:
        start = Path(__file__).resolve()
    start = start.resolve()

    for p in [start] + list(start.parents):
        if (p / "junction_analysis").is_dir() and (p / "results").is_dir() and (p / "data").is_dir():
            return p

    raise RuntimeError("Repo root not found (expected junction_analysis/, results/, data/ in a parent dir).")

def get_hierarchical_order(distance_df):
    linkage_matrix = sch.linkage(distance_df, method="ward")
    ordered_indices = sch.leaves_list(linkage_matrix)
    order = distance_df.index[ordered_indices]
    return order

def get_tree_order():
    root = repo_root()
    _fname = str( root / "config" / "polished_tree.nwk")
    _tree = Phylo.read(_fname, "newick")
    _tree.root_at_midpoint()
    _tree.ladderize()
    # extract the order of isolates from the tree
    leaf_order = [leaf.name for leaf in _tree.get_terminals()]
    return leaf_order


def get_isolate_name_from_node_id(example_pangraph, node_id):
    return example_pangraph.paths.idx_to_name[example_pangraph.nodes[node_id].path_id]

def write_isolate_fasta(example_pangraph, block, output_path):
    """
    Write sequences to a FASTA file, using isolate names derived from node IDs.

    Parameters
    ----------
    example_pangraph : Pangraph object
        Object containing mapping from node IDs to isolate names.
    block : Block object
    output_path : str or Path
        Path to the FASTA file to write.
    """
    records = block.to_biopython_records()
    fasta_records = []
    for record in records:
        node_id = record.id
        isolate_name = get_isolate_name_from_node_id(example_pangraph, node_id)

        # Create a new SeqRecord with a readable isolate name
        new_record = SeqRecord(
            record.seq,
            id=f"{isolate_name}__{node_id}",        # what appears after ">" in FASTA
            description=""          # no extra text
        )
        fasta_records.append(new_record)

    # Write all sequences to a FASTA file
    SeqIO.write(fasta_records, output_path, "fasta")


def convert_gbk_fasta(gbk_folder, fasta_folder):

    os.makedirs(fasta_folder, exist_ok=True)

    # List all .gbk files
    for fname in os.listdir(gbk_folder):
        if not fname.endswith(".gbk"):
            continue

        input_gbk = os.path.join(gbk_folder, fname)
        genome_fasta = os.path.join(fasta_folder, fname.replace(".gbk", ".fasta"))

        print(f"Converting: {input_gbk} → {genome_fasta}")

        with open(genome_fasta, "w") as out_f:
            for record in SeqIO.parse(input_gbk, "genbank"):
                SeqIO.write(record, out_f, "fasta")


def convert_gbk_fasta_plasmids(gbk_folder, fasta_folder):
    """
    Converts plasmid GenBank files from a nested directory structure into FASTA format,
    preserving the directory structure.

    For each plasmid, a FASTA record is created with a header in the format:
    >{plasmid_name}|{isolate_name}

    The plasmid name is extracted from the GenBank record's source feature if available,
    otherwise it defaults to the filename. The isolate name is derived from the
    directory path relative to the input folder.

    Parameters
    ----------
    gbk_folder : str
        The root directory containing the plasmid GenBank files (.gbk or .gb).
    fasta_folder : str
        The root directory where the output FASTA files will be saved.
    """
    os.makedirs(fasta_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(gbk_folder):
        for filename in filenames:
            if not (filename.endswith(".gbk") or filename.endswith(".gb")):
                continue

            input_gbk = os.path.join(dirpath, filename)

            # Determine output path while preserving structure
            relative_dir = os.path.relpath(dirpath, gbk_folder)
            output_dir = os.path.join(fasta_folder, relative_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_fasta = os.path.join(output_dir, os.path.splitext(filename)[0] + ".fasta")

            print(f"Converting: {input_gbk} -> {output_fasta}")

            records_to_write = []
            for record in SeqIO.parse(input_gbk, "genbank"):
                # Determine isolate name from directory structure
                isolate_name = "unknown_isolate"
                if relative_dir and relative_dir != ".":
                    isolate_name = relative_dir.replace(os.path.sep, "_")

                # Determine plasmid name from GenBank record
                plasmid_name = os.path.splitext(filename)[0]  # Default to filename
                if record.features:
                    for feature in record.features:
                        if feature.type == "source":
                            if "plasmid" in feature.qualifiers:
                                plasmid_name = feature.qualifiers["plasmid"][0]
                                break
                
                # Clean up names to be filesystem-friendly and header-friendly
                plasmid_name = plasmid_name.replace(" ", "_").replace("|", "-")
                isolate_name = isolate_name.replace(" ", "_").replace("|", "-")

                # Create a new record with the desired header format
                new_record = SeqRecord(
                    record.seq,
                    id=f"{plasmid_name}|{isolate_name}",
                    description=""
                )
                records_to_write.append(new_record)

            # Write the new record(s) to the FASTA file
            if records_to_write:
                with open(output_fasta, "w") as out_f:
                    SeqIO.write(records_to_write, out_f, "fasta")


def cluster_by_tree(
    tree_file,
    n_clusters=None,
    distance_threshold=None,
    frac_of_max=None,
    verbose=True,
):
    """
    Cluster isolates in a Newick tree using:
    - A) fixed number of clusters (n_clusters)
    - B) absolute distance threshold (distance_threshold)
    - C) relative distance threshold (frac_of_max * max_distance)

    Returns:
        dict {isolate_name: cluster_id}
    """

    modes = [n_clusters is not None,
             distance_threshold is not None,
             frac_of_max is not None]

    if sum(modes) != 1:
        raise ValueError(
            "Specify exactly ONE of the following:\n"
            " - n_clusters\n"
            " - distance_threshold\n"
            " - frac_of_max"
        )

    # Read tree 
    tree = Phylo.read(tree_file, "newick")
    terminals = tree.get_terminals()
    n = len(terminals)

    # Build patristic distance matrix 
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(terminals[i], terminals[j])
            dist[i, j] = dist[j, i] = d

    # Determine distance threshold if using frac_of_max 
    if frac_of_max is not None:
        max_d = dist.max()
        distance_threshold = frac_of_max * max_d
        if verbose:
            print(
                f"Using distance threshold = {distance_threshold:.6f} "
                f"(= {frac_of_max*100:.1f}% of max distance {max_d:.6f})"
            )

    # Configure clustering
    if n_clusters is not None:
        # Mode A: fixed number of clusters
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

    else:
        # Mode B/C: distance-based clustering
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

    # Remap labels to isolate names
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique, start=1)}

    cluster_map = {
        tip.name: int(remap[label]) for tip, label in zip(terminals, labels)
    }

    if verbose:
        print(f"Formed {len(unique)} clusters")

    return cluster_map


def strip_newick_suffixes(input_path, output_path):
    """
    Remove everything after '__' in all node names inside a Newick tree string.
    Example: 'A__123' -> 'A'
    """
    with open(input_path, "r") as f:
        newick_str = f.read()
    pattern = re.compile(r'([^,():;]+)__[^,():;]+')
    cleaned = pattern.sub(r'\1', newick_str)

    with open(output_path, "w") as f:
        f.write(cleaned)

def get_isolate_sequence(pangraph, block_id, node_id):
    sequence = pangraph.blocks[block_id].alignment.generate_alignment()[str(node_id)]
    return sequence.replace("-", "")

def write_shared_nodes_fasta(pangraph, path_dict, shared_nodes, shared_node_isolates, output_path):
    """
    Docstring for write_shared_nodes_fasta. Blocks are added in the order of shared_nodes list.
    Only isolates that contain all shared nodes are included.
    
    :param pangraph: Pangraph object
    :param path_dict: Dictionary of all paths (can be deduplicated)
    :param shared_nodes: List of Node objects that are shared between isolates
    :param shared_node_isolates: List of isolates that potentially share all nodes
    :param output_path: Path where resulting fasta file is stored
    """
    shared_id_set = {node for node in shared_nodes}

    fasta_records = []
    for isolate in shared_node_isolates:
        # skip isolates that do not contain all shared blocks
        isolate_blocks = {block for block in path_dict[isolate].nodes}
        if not shared_id_set.issubset(isolate_blocks):
            continue

        seq = ""
        for shared_block in shared_nodes:
            for block in path_dict[isolate].nodes:
                if block == shared_block:
                    block_seq = get_isolate_sequence(pangraph, block.id, block.nid)
                    if not block.strand:
                        block_seq = str(Seq(block_seq).reverse_complement())
                    seq = seq + block_seq
                    break # end low level loop once one sequence was added (since blocks are unique should only happen once)

        record = SeqRecord(Seq(seq), id = isolate, description = f"blocks{shared_nodes} length{len(seq)}")
        fasta_records.append(record)

    if not fasta_records:
        raise ValueError(
            "No isolate contains all shared nodes — FASTA file was not written."
        )

    SeqIO.write(fasta_records, output_path, "fasta")

def simplify_cluster_keys(cluster_map):
    """
    Return a new cluster_map where keys are truncated at the first '__'.
    
    Example:
    'NZ_CP080117.1__5727886839925671655' -> 'NZ_CP080117.1'
    """
    new_map = {}
    for key, value in cluster_map.items():
        simple_key = key.split("__")[0]
        new_map[simple_key] = value
    return new_map

def get_consensus_seq_from_alignment(aln_path, threshold=0.5):
    alignment = Align.read(aln_path, "fasta")
    motif = Motif("acgt-", alignment)
    consensus = motif.counts.calculate_consensus(
        identity=threshold
    )
    return consensus

def snp_positions(alignment_file, fmt="fasta"):
    alignment = AlignIO.read(alignment_file, fmt)
    snps = []

    for i in range(alignment.get_alignment_length()):
        column = set(alignment[:, i]) - {"-"}
        if len(column) > 1:
            snps.append(i)

    return snps

def get_block_length(alignment_file, fmt="fasta"):
    alignment = AlignIO.read(alignment_file, fmt)
    return alignment.get_alignment_length()


def build_subtree(tree, isolate_list):
    # Only keep isolates that actually appear in the tree
    tree_tips = {tip.name for tip in tree.get_terminals()}
    isolate_list = [iso for iso in isolate_list if iso in tree_tips]

    if not isolate_list:
        raise ValueError("No isolates in isolate_list are present in the tree.")

    mrca = tree.common_ancestor(isolate_list)
    subtree = Tree(root=copy.deepcopy(mrca), rooted=True)
    subtree.root.branch_length = 0.0  # there is no parent anymore

    # prune unwanted tips by name to avoid stale clade-object references
    keep = set(isolate_list)
    tips_to_remove = [tip.name for tip in subtree.get_terminals() if tip.name not in keep]
    for name in tips_to_remove:
        # re-check: earlier prunes may have already collapsed this tip away
        current_tip_names = {tip.name for tip in subtree.get_terminals()}
        if name in current_tip_names:
            subtree.prune(name)

    return subtree

def read_gff3_annotations(gff_path: str) -> pd.DataFrame:
    rows = []
    with open(gff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            start, end = int(start), int(end)

            # parse attributes into dict
            ad = {}
            for item in attrs.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    ad[k] = v

            # IS subtype from ID like: NZ_CP102061.1|IS66|1992
            is_subtype = None
            if ftype == "IS":
                _id = ad.get("ID", "")
                toks = _id.split("|")
                if len(toks) >= 2:
                    is_subtype = toks[1]  # e.g. IS66 / IS3 / ISL3 / new

            rows.append(
                dict(
                    seqid=seqid,
                    feature=ftype,      # "IS", "defense_system", "prophage", ...
                    start=start,
                    end=end,
                    length = end - start,
                    attrs=ad,
                    is_subtype=is_subtype,
                )
            )
    return pd.DataFrame(rows)


def read_gff3_trna(gff_path: str) -> pd.DataFrame:
    """
    Read tRNA and tmRNA entries from a GFF3 annotation file.

    Returns a DataFrame with columns:
        seqid, start, end, strand, feature (tRNA/tmRNA),
        product, locus_tag, anticodon, is_partial, gene_id
    """
    rows = []
    with open(gff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue

            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype not in ("tRNA", "tmRNA"):
                continue

            start, end = int(start), int(end)

            ad = {}
            for item in attrs.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    ad[k] = v

            rows.append(dict(
                seqid=seqid,
                start=start,
                end=end,
                length=end - start,
                strand=strand,
                feature=ftype,
                product=ad.get("product") or ad.get("Name") or ad.get("ID"),
                locus_tag=ad.get("locus_tag"),
                anticodon=ad.get("anticodon"),
                is_partial=ad.get("is_partial"),
                gene_id=ad.get("ID"),
            ))

    return pd.DataFrame(rows)


def count_trna_per_junction(annotations_dir: str) -> pd.DataFrame:
    """
    For each junction GFF file in annotations_dir, count the number of
    annotated tRNA and tmRNA entries.

    Returns a DataFrame with columns:
        junction_name, n_tRNA, n_tmRNA, n_total
    sorted by n_total descending.
    """
    rows = []
    for gff_path in sorted(Path(annotations_dir).glob("*.gff")):
        junction_name = gff_path.stem
        tdf = read_gff3_trna(str(gff_path))
        if tdf.empty:
            n_trna, n_tmrna = 0, 0
        else:
            n_trna  = int((tdf["feature"] == "tRNA").sum())
            n_tmrna = int((tdf["feature"] == "tmRNA").sum())
        rows.append(dict(
            junction_name=junction_name,
            n_tRNA=n_trna,
            n_tmRNA=n_tmrna,
            n_total=n_trna + n_tmrna,
        ))
    return pd.DataFrame(rows).sort_values("n_total", ascending=False).reset_index(drop=True)


def read_gff3_cds_products(gff_path: str) -> pd.DataFrame:
    rows = []
    with open(gff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue

            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype != "CDS":
                continue

            start, end = int(start), int(end)

            ad = {}
            for item in attrs.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    ad[k] = v

            # label: prefer product; fallback to Name/ID
            label = ad.get("product") or ad.get("Name") or ad.get("ID") or "CDS"

            rows.append(dict(
                seqid=seqid,
                start=start,
                end=end,
                length=end - start,
                strand=strand,
                product=label,
                attrs=ad,
            ))

    return pd.DataFrame(rows)