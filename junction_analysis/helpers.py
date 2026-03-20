import scipy.cluster.hierarchy as sch
import pypangraph as pp
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
import warnings

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

def snp_gap_lengths(alignment_file, fmt="fasta"):
    """
    Calculate the gap lengths between consecutive SNP positions in an alignment.

    Parameters
    ----------
    alignment_file : str or Path
    fmt : str
        Alignment format. Default: "fasta".

    Returns
    -------
    dict with keys:
        snp_positions  : list of int  — column indices of SNP positions
        gap_lengths    : list of int  — distances between consecutive SNPs
                         (len = len(snp_positions) - 1)
        aln_length     : int          — total alignment length
    """
    alignment = AlignIO.read(alignment_file, fmt)
    aln_length = alignment.get_alignment_length()

    snps = []
    for i in range(aln_length):
        column = set(alignment[:, i]) - {"-"}
        if len(column) > 1:
            snps.append(i)

    gaps = []
    if snps:
        gaps.append(snps[0])                                          # before first SNP
        gaps.extend(snps[i + 1] - snps[i] for i in range(len(snps) - 1))  # between SNPs
        gaps.append(aln_length - 1 - snps[-1])                       # after last SNP

    return {
        "snp_positions": snps,
        "gap_lengths": gaps,
        "aln_length": aln_length,
    }


def core_block_snp_gaps(results_dir: str, save_path: str = None) -> pd.DataFrame:
    """
    For each junction in results/block_alignments, identify core blocks via the
    pangraph, then compute SNP positions and inter-SNP gap lengths for each.

    Parameters
    ----------
    results_dir : str or Path
        Base results directory. Expects:
          - ``results_dir/block_alignments/<junction>/block_<id>_aln.fa``
          - ``results_dir/junction_pangraphs/<junction>.json``

    Returns
    -------
    pd.DataFrame with columns:
        junction_name, block_id, aln_length, snp_positions, gap_lengths
    One row per core block per junction.
    """
    results_dir = Path(results_dir)
    aln_base = results_dir / "block_alignments"
    pangraph_base = results_dir / "junction_pangraphs"

    rows = []
    for junction_dir in sorted(aln_base.iterdir()):
        if not junction_dir.is_dir():
            continue
        jname = junction_dir.name

        pangraph_path = pangraph_base / f"{jname}.json"
        if not pangraph_path.exists():
            continue

        pangraph = pp.Pangraph.from_json(str(pangraph_path))
        blockstats = pangraph.to_blockstats_df()
        core_ids = set(blockstats[blockstats["core"] == True].index.astype(str))

        for aln_file in sorted(junction_dir.glob("block_*_aln.fa")):
            # extract block id from filename: block_<id>_aln.fa
            block_id = aln_file.stem.removeprefix("block_").removesuffix("_aln")
            if block_id not in core_ids:
                continue
            result = snp_gap_lengths(str(aln_file))
            rows.append(dict(
                junction_name=jname,
                block_id=block_id,
                aln_length=result["aln_length"],
                snp_positions=result["snp_positions"],
                gap_lengths=result["gap_lengths"],
            ))
        print(f"Successfully processed junction {junction_dir}")

    df = pd.DataFrame(rows)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


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


def count_trna_per_junction(
    annotations_dir: str,
    category_bins: list = None,
    category_labels: list = None,
) -> pd.DataFrame:
    """
    For each junction GFF file in annotations_dir, count the number of
    annotated tRNA and tmRNA entries.

    Returns a DataFrame with columns:
        junction_name, n_tRNA, n_tmRNA, n_total_trna
    sorted by n_total_trna descending.

    If `category_bins` is provided, also adds:
        trna_cat : ordered categorical column based on n_total_trna

    Parameters
    ----------
    annotations_dir : str
    category_bins : list of numeric or None
        Bin edges passed to pd.cut (e.g. [-1, 0, 9, 49, inf]).
        The first edge should be below 0 so that 0 falls in the first bin.
    category_labels : list of str or None
        Labels for each bin. Must have length len(category_bins) - 1.
        Defaults to string representations of the bin edges.
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
            n_total_trna=n_trna + n_tmrna,
        ))
    df = pd.DataFrame(rows).sort_values("n_total_trna", ascending=False).reset_index(drop=True)

    if category_bins is not None:
        df["trna_cat"] = pd.cut(df["n_total_trna"], bins=category_bins, labels=category_labels)

    return df


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



def compare_consensus_paths(
    path1,
    path2,
    junction_name: str,
    cluster1: str,
    cluster2: str,
    base_alignment_dir=None,
) -> dict:
    """
    Compare two consensus paths and return shared/exclusive block lengths.

    Parameters
    ----------
    path1, path2 : Path objects (pangraph_utils.Path)
        The two consensus paths to compare.
    junction_name : str
        Junction name, used to locate the alignment stats CSV.
    cluster1, cluster2 : str
        Cluster labels for path1 and path2 respectively.
    base_alignment_dir : str or Path or None
        Root of the block_alignments directory. Defaults to
        repo_root() / "results" / "block_alignments".

    Returns
    -------
    dict with keys:
        junction_name, cluster1, cluster2,
        shared_length, exclusive_length_1, exclusive_length_2
    """
    if base_alignment_dir is None:
        base_alignment_dir = repo_root() / "results" / "block_alignments"

    stats_csv = Path(base_alignment_dir) / junction_name / f"{junction_name}_alignment_stats.csv"
    stats_df = pd.read_csv(stats_csv)
    block_length = dict(zip(stats_df["block_id"].astype(str), stats_df["alignment_len"]))

    nodes1 = set(path1.nodes)
    nodes2 = set(path2.nodes)

    shared    = nodes1 & nodes2
    only_in_1 = nodes1 - nodes2
    only_in_2 = nodes2 - nodes1

    shared_length      = sum(block_length.get(str(n.id), 0) for n in shared)
    exclusive_length_1 = sum(block_length.get(str(n.id), 0) for n in only_in_1)
    exclusive_length_2 = sum(block_length.get(str(n.id), 0) for n in only_in_2)

    return {
        "junction_name":      junction_name,
        "cluster1":           cluster1,
        "cluster2":           cluster2,
        "n_shared_blocks":    len(shared),
        "n_exclusive_blocks_1": len(only_in_1),
        "n_exclusive_blocks_2": len(only_in_2),
        "shared_length":      shared_length,
        "exclusive_length_1": exclusive_length_1,
        "exclusive_length_2": exclusive_length_2,
    }


def compute_within_between_cluster_distances(
    cluster_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    save_path: str = None
) -> pd.DataFrame:
    meta_cols = {"junction_name", "n_clusters", "n_isolates"}
    isolate_cols = [c for c in cluster_df.columns if c not in meta_cols]

    rows = []

    for _, jrow in cluster_df.iterrows():
        jname = jrow["junction_name"]
        n_clusters = int(jrow["n_clusters"])

        iso_to_cl = {
            iso: int(jrow[iso])
            for iso in isolate_cols
            if pd.notna(jrow[iso])
        }
        if not iso_to_cl:
            continue

        jdist = dist_df[dist_df["junction_name"] == jname].copy()
        if jdist.empty:
            continue

        jdist = jdist[
            jdist["isolate_1"].isin(iso_to_cl) &
            jdist["isolate_2"].isin(iso_to_cl)
        ].copy()

        if jdist.empty:
            continue

        dist_lookup = {}
        for _, r in jdist.iterrows():
            dist_lookup[(r["isolate_1"], r["isolate_2"])] = r["distance"]
            dist_lookup[(r["isolate_2"], r["isolate_1"])] = r["distance"]

        cl_to_isos = {}
        for iso, cl in iso_to_cl.items():
            cl_to_isos.setdefault(cl, []).append(iso)

        for iso, cl in iso_to_cl.items():
            same_cl = [o for o in cl_to_isos[cl] if o != iso]

            if not same_cl:
                rows.append({
                    "junction_name": jname,
                    "n_clusters": n_clusters,
                    "isolate": iso,
                    "cluster": cl,
                    "a": np.nan,
                    "b": np.nan,
                    "silhouette": np.nan
                })
                continue

            same_dists = [dist_lookup[(iso, o)] for o in same_cl if (iso, o) in dist_lookup]
            a = np.mean(same_dists) if same_dists else np.nan

            other_cls = [c for c in cl_to_isos if c != cl]
            if not other_cls:
                rows.append({
                    "junction_name": jname,
                    "n_clusters": n_clusters,
                    "isolate": iso,
                    "cluster": cl,
                    "a": a,
                    "b": np.nan,
                    "silhouette": np.nan
                })
                continue

            mean_dists = []
            for other_cl in other_cls:
                dists = [dist_lookup[(iso, o)] for o in cl_to_isos[other_cl] if (iso, o) in dist_lookup]
                if dists:
                    mean_dists.append(np.mean(dists))

            b = min(mean_dists) if mean_dists else np.nan

            if pd.notna(a) and pd.notna(b):
                denom = max(a, b)
                if denom > 0:
                    s = (b - a) / denom
                else:
                    # all distances are zero → no cluster separation
                    s = 0.0
            else:
                s = np.nan

            rows.append({
                "junction_name": jname,
                "n_clusters": n_clusters,
                "isolate": iso,
                "cluster": cl,
                "a": a,
                "b": b,
                "silhouette": s
            })

    result = pd.DataFrame(rows)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)

    return result


def silhouette_score_from_ab(a: float, b: float) -> float:
    """
    Compute silhouette score from pre-computed a and b values.

    Parameters
    ----------
    a : float
        Mean intra-cluster distance.
    b : float
        Mean nearest-cluster distance.

    Returns
    -------
    float : silhouette score (b - a) / max(a, b), or NaN if inputs are NaN.
    """
    # only one cluster (no between-cluster distance) → silhouette is 0
    if np.isnan(b):
        return 0.0
    # within is NaN (e.g. all clusters are singletons) → undefined
    if np.isnan(a):
        return np.nan
    denom = max(a, b)
    if denom == 0:
        return 0.0
    return (b - a) / denom


def add_silhouette_scores(
    df: pd.DataFrame,
    within_col: str = "mean_within_dist",
    between_col: str = "mean_between_dist",
    is_similarity: bool = False,
    result_col: str = "silhouette_score",
) -> pd.DataFrame:
    """
    Add a silhouette score column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    within_col : str
        Column to use as a (intra-cluster distance/similarity).
    between_col : str
        Column to use as b (inter-cluster distance/similarity).
    is_similarity : bool
        If True, converts similarity to distance via 1 - value before scoring.
    result_col : str
        Name of the output column.

    Returns
    -------
    pd.DataFrame with added silhouette score column.
    """
    result = df.copy()
    a = result[within_col]
    b = result[between_col]
    if is_similarity:
        a = 1 - a
        b = 1 - b
    result[result_col] = [silhouette_score_from_ab(ai, bi) for ai, bi in zip(a, b)]
    return result


def summarize_clustering_thresholds(results_dir: str, save_path: str = None) -> pd.DataFrame:
    """
    Read all cluster_maps_*.csv files in results_dir/consensus_analysis/, extract
    the branch-length threshold from the filename, and compute the total number of
    additional clusters (n_clusters summed over all junctions minus the number of
    junctions, since every junction has at least 1 cluster).

    Parameters
    ----------
    results_dir : str or Path
    save_path : str or None

    Returns
    -------
    pd.DataFrame with columns: threshold, n_additional_clusters
    sorted by threshold ascending.
    """
    rows = []
    pattern = re.compile(r"cluster_maps_(\d+)_(\d+)\.csv")

    for csv_path in sorted((Path(results_dir) / "consensus_analysis").glob("cluster_maps_*.csv")):
        m = pattern.match(csv_path.name)
        if not m:
            continue
        threshold = float(f"{m.group(1)}.{m.group(2)}")
        df = pd.read_csv(csv_path, usecols=["n_clusters"])
        n_additional = int(df["n_clusters"].sum()) - len(df)
        rows.append({"threshold": threshold, "n_additional_clusters": n_additional})

    result = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)

    return result


def extract_biosample_ids(gbk_dir: str, plasmids_dir: str) -> pd.DataFrame:
    """
    Extract BioSample IDs for each isolate from chromosome and plasmid GBK files.

    Scans:
      - `gbk_dir`      : flat folder of <isolate_id>.gbk files (chromosomes)
      - `plasmids_dir` : folder with one subfolder per isolate (<isolate_id>/<plasmid_id>.gbk)

    Returns a DataFrame with columns:
        isolate_id, accession, type (chromosome/plasmid), biosample
    """
    rows = []

    def _parse_biosample(gbk_path: str) -> str | None:
        with open(gbk_path) as fh:
            for line in fh:
                m = re.match(r'\s+BioSample:\s+(\S+)', line)
                if m:
                    return m.group(1)
                # stop searching after KEYWORDS line (BioSample is always before it)
                if line.startswith("KEYWORDS"):
                    break
        return None

    # Chromosomes
    for gbk_path in sorted(Path(gbk_dir).glob("*.gbk")):
        isolate_id = gbk_path.stem
        biosample = _parse_biosample(str(gbk_path))
        rows.append(dict(isolate_id=isolate_id, accession=isolate_id, type="chromosome", biosample=biosample))

    # Plasmids
    for isolate_dir in sorted(Path(plasmids_dir).iterdir()):
        if not isolate_dir.is_dir():
            continue
        isolate_id = isolate_dir.name
        for gbk_path in sorted(isolate_dir.glob("*.gbk")):
            accession = gbk_path.stem
            biosample = _parse_biosample(str(gbk_path))
            rows.append(dict(isolate_id=isolate_id, accession=accession, type="plasmid", biosample=biosample))

    return pd.DataFrame(rows)


def get_core_alignment_lengths(
    consensus_analysis_dir: str,
    results_dir: str = None,
    save_path: str = None,
    block_save_path: str = None,
) -> tuple:
    """
    For each junction in `consensus_analysis_dir`, read core_blocks_aln.fa
    and record the alignment length with and without gaps.

    If `results_dir` is provided, also iterates over
    ``results_dir/block_alignments/<junction>/block_<id>_aln.fa`` and returns
    per-block lengths for core blocks (identified via the pangraph).

    Parameters
    ----------
    consensus_analysis_dir : str
    results_dir : str or None
        Base results directory containing block_alignments/ and junction_pangraphs/.
    save_path : str or None
        If provided, save the junction-level DataFrame as CSV.
    block_save_path : str or None
        If provided, save the per-block DataFrame as CSV.

    Returns
    -------
    junction_df : pd.DataFrame
        Columns: junction_name, aln_length, aln_length_nogap
    block_df : pd.DataFrame or None
        Columns: junction_name, block_id, aln_length, aln_length_nogap
        (only if results_dir is provided)
    """
    rows = []
    for aln_path in sorted(Path(consensus_analysis_dir).glob("*/core_blocks_aln.fa")):
        junction_name = aln_path.parent.name
        seqs = [str(r.seq) for r in SeqIO.parse(str(aln_path), "fasta")]
        aln_length = len(seqs[0])
        aln_length_nogap = sum(
            1 for i in range(aln_length)
            if all(s[i] != "-" for s in seqs)
        )
        rows.append(dict(
            junction_name=junction_name,
            aln_length=aln_length,
            aln_length_nogap=aln_length_nogap,
        ))
    junction_df = pd.DataFrame(rows).sort_values("junction_name").reset_index(drop=True)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        junction_df.to_csv(save_path, index=False)

    block_df = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        aln_base = results_dir / "block_alignments"
        pangraph_base = results_dir / "junction_pangraphs"

        block_rows = []
        for junction_dir in sorted(aln_base.iterdir()):
            if not junction_dir.is_dir():
                continue
            jname = junction_dir.name

            pangraph_path = pangraph_base / f"{jname}.json"
            if not pangraph_path.exists():
                continue

            pangraph = pp.Pangraph.from_json(str(pangraph_path))
            blockstats = pangraph.to_blockstats_df()
            core_ids = set(blockstats[blockstats["core"] == True].index.astype(str))

            for aln_file in sorted(junction_dir.glob("block_*_aln.fa")):
                block_id = aln_file.stem.removeprefix("block_").removesuffix("_aln")
                if block_id not in core_ids:
                    continue
                seqs = [str(r.seq) for r in SeqIO.parse(str(aln_file), "fasta")]
                aln_length = len(seqs[0])
                aln_length_nogap = sum(
                    1 for i in range(aln_length)
                    if all(s[i] != "-" for s in seqs)
                )
                block_rows.append(dict(
                    junction_name=jname,
                    block_id=block_id,
                    aln_length=aln_length,
                    aln_length_nogap=aln_length_nogap,
                ))

        block_df = pd.DataFrame(block_rows).sort_values(["junction_name", "block_id"]).reset_index(drop=True)
        if block_save_path is not None:
            Path(block_save_path).parent.mkdir(parents=True, exist_ok=True)
            block_df.to_csv(block_save_path, index=False)

    return junction_df, block_df


def load_all_block_alignment_stats(results_dir: str, save_path: str = None) -> pd.DataFrame:
    """
    Load avg_pairwise_dist (and other alignment stats) for every block across all
    junctions by reading the per-junction alignment stats CSVs:
      results_dir/block_alignments/<junction>/<junction>_alignment_stats.csv

    Returns
    -------
    pd.DataFrame with all columns from the stats CSVs plus a junction_name column.
    """
    results_dir = Path(results_dir)
    aln_base = results_dir / "block_alignments"

    dfs = []
    for junction_dir in sorted(aln_base.iterdir()):
        if not junction_dir.is_dir():
            continue
        csv_path = junction_dir / f"{junction_dir.name}_alignment_stats.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "junction_name", junction_dir.name)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
    return result


def add_singleton_gain_loss_isolate_columns(jdf_all, pangraph_dir="../results/junction_pangraphs"):
    """
    Add columns:
      - singleton_gain_isolate
      - singleton_loss_isolate

    Values:
      - isolate name if detected
      - comma-separated isolate names if multiple blocks point to different isolates
      - pd.NA otherwise

    Rules:
    - non-singleton junctions -> pd.NA
    - singleton junctions with n_iso == 2 -> pd.NA
    - inspect all non-core blocks in the pangraph
    """
    jdf_all = jdf_all.copy()

    jdf_all["singleton_gain_isolate"] = pd.Series(pd.NA, index=jdf_all.index, dtype="object")
    jdf_all["singleton_loss_isolate"] = pd.Series(pd.NA, index=jdf_all.index, dtype="object")

    singleton_mask = jdf_all["singleton"]

    for idx, row in jdf_all.loc[singleton_mask].iterrows():
        junction_name = row["junction_name"]
        n_iso = int(row["n_iso"])

        if n_iso == 2:
            continue

        pangraph_path = Path(pangraph_dir) / f"{junction_name}.json"
        if not pangraph_path.exists():
            print(f"Skipping {junction_name}: file not found")
            continue

        try:
            pangraph = pp.Pangraph.from_json(str(pangraph_path))

            blockstats_df = pangraph.to_blockstats_df()
            blockcount_df = pangraph.to_blockcount_df()

            accessory_block_ids = blockstats_df.loc[~blockstats_df["core"]].index

            gain_isolates = set()
            loss_isolates = set()

            for block_id in accessory_block_ids:
                counts = blockcount_df.loc[block_id]   # Series: isolate_name -> count

                # Most common copy number across isolates
                baseline = counts.mode().iloc[0]

                gain_candidates = counts[counts == baseline + 1].index.tolist()
                loss_candidates = counts[counts == baseline - 1].index.tolist()

                # Only call it singleton if exactly one isolate deviates that way
                if len(gain_candidates) == 1:
                    gain_isolates.add(gain_candidates[0])

                if len(loss_candidates) == 1:
                    loss_isolates.add(loss_candidates[0])

            if len(gain_isolates) == 1:
                jdf_all.at[idx, "singleton_gain_isolate"] = next(iter(gain_isolates))
            elif len(gain_isolates) > 1:
                jdf_all.at[idx, "singleton_gain_isolate"] = ",".join(sorted(gain_isolates))

            if len(loss_isolates) == 1:
                jdf_all.at[idx, "singleton_loss_isolate"] = next(iter(loss_isolates))
            elif len(loss_isolates) > 1:
                jdf_all.at[idx, "singleton_loss_isolate"] = ",".join(sorted(loss_isolates))

        except Exception as e:
            print(f"Skipping {junction_name}: {e}")

    return jdf_all

def add_singleton_has_own_cluster(jdf_all, cluster_df):
    """
    Adds column:
        - singleton_has_own_cluster

    True  -> singleton isolate is alone in its cluster
    False -> cluster contains other isolates
    <NA>  -> not applicable (non-singleton or n_iso == 2 or no isolate info)
    """
    jdf_all = jdf_all.copy()
    jdf_all["singleton_has_own_cluster"] = pd.Series(pd.NA, index=jdf_all.index, dtype="boolean")

    # isolate columns = everything except metadata
    meta_cols = {"junction_name", "n_clusters", "n_isolates"}
    isolate_cols = [c for c in cluster_df.columns if c not in meta_cols]

    # make lookup faster
    cluster_df = cluster_df.set_index("junction_name")

    for idx, row in jdf_all.iterrows():
        if not row["singleton"] or row["n_iso"] == 2:
            continue

        junction = row["junction_name"]

        # get singleton isolate (prefer gain, else loss)
        isolate = row.get("singleton_gain_isolate")
        if pd.isna(isolate):
            isolate = row.get("singleton_loss_isolate")

        if pd.isna(isolate):
            continue

        if junction not in cluster_df.index:
            continue

        cluster_row = cluster_df.loc[junction]

        if isolate not in isolate_cols:
            continue

        cluster_id = cluster_row[isolate]
        if pd.isna(cluster_id):
            continue

        # count how many isolates share this cluster
        same_cluster = (cluster_row[isolate_cols] == cluster_id).sum()

        jdf_all.at[idx, "singleton_has_own_cluster"] = (same_cluster == 1)

    return jdf_all

def add_binary_event_info(jdf_all, cluster_df, pangraph_dir="../results/junction_pangraphs"):
    """
    For binary junctions (n_categories == 2), add:
      - binary_gain_isolates
      - binary_loss_isolates
      - binary_gain_frequency
      - binary_loss_frequency
      - binary_event_type
      - binary_gain_has_own_cluster
      - binary_loss_has_own_cluster
      - binary_gain_in_multiple_clusters
      - binary_loss_in_multiple_clusters
      - binary_event_in_multiple_clusters

    Event logic per non-core block:
      - subtract the minimum count across isolates
      - count how many isolates have adjusted count > 0
      - if this frequency is < n_iso/2: gain in those isolates
      - if this frequency is > n_iso/2: loss in the complementary isolates
      - if this frequency is == n_iso/2: ambiguous
    """
    jdf_all = jdf_all.copy()

    # output columns
    for col in ["binary_gain_isolates", "binary_loss_isolates", "binary_event_type"]:
        jdf_all[col] = pd.Series(pd.NA, index=jdf_all.index, dtype="object")

    for col in ["binary_gain_frequency", "binary_loss_frequency"]:
        jdf_all[col] = pd.Series(pd.NA, index=jdf_all.index, dtype="Int64")

    for col in [
        "binary_gain_has_own_cluster",
        "binary_loss_has_own_cluster",
        "binary_gain_in_multiple_clusters",
        "binary_loss_in_multiple_clusters",
        "binary_event_in_multiple_clusters",
    ]:
        jdf_all[col] = pd.Series(pd.NA, index=jdf_all.index, dtype="boolean")

    meta_cols = {"junction_name", "n_clusters", "n_isolates"}
    isolate_cols = [c for c in cluster_df.columns if c not in meta_cols]
    cluster_df = cluster_df.set_index("junction_name")

    def _cluster_info(cluster_row, event_isolates):
        if not event_isolates:
            return pd.NA, pd.NA

        cluster_ids = cluster_row[event_isolates].dropna().unique()
        if len(cluster_ids) == 0:
            return pd.NA, pd.NA

        if len(cluster_ids) > 1:
            return False, True

        cluster_id = cluster_ids[0]
        cluster_members = set(cluster_row[isolate_cols][cluster_row[isolate_cols] == cluster_id].index)
        return cluster_members == set(event_isolates), False

    binary_mask = jdf_all["n_categories"] == 2

    for idx, row in jdf_all.loc[binary_mask].iterrows():
        junction_name = row["junction_name"]
        n_iso = int(row["n_iso"])

        if n_iso == 2:
            continue

        pangraph_path = Path(pangraph_dir) / f"{junction_name}.json"
        if not pangraph_path.exists():
            print(f"Skipping {junction_name}: file not found")
            continue

        try:
            pangraph = pp.Pangraph.from_json(str(pangraph_path))
            blockstats_df = pangraph.to_blockstats_df()
            blockcount_df = pangraph.to_blockcount_df()

            accessory_block_ids = blockstats_df.loc[~blockstats_df["core"]].index

            gain_isolates = set()
            loss_isolates = set()
            gain_frequency = None
            loss_frequency = None
            ambiguous = False

            for block_id in accessory_block_ids:
                counts = blockcount_df.loc[block_id]
                min_count = counts.min()
                adjusted = counts - min_count
                freq = int((adjusted > 0).sum())

                if freq == 0:
                    continue

                if 2 * freq == n_iso:
                    warnings.warn(
                        f"Binary junction {junction_name}: block {block_id} occurs in exactly half "
                        f"of isolates after min-subtraction. Marking as ambiguous."
                    )
                    ambiguous = True
                    break

                present_isolates = set(adjusted[adjusted > 0].index)
                absent_isolates = set(counts.index) - present_isolates

                if freq < n_iso / 2:
                    if gain_frequency is None:
                        gain_frequency = freq
                    elif gain_frequency != freq:
                        warnings.warn(
                            f"Binary junction {junction_name}: inconsistent gain frequencies across blocks."
                        )
                        ambiguous = True
                        break

                    gain_isolates.update(present_isolates)

                else:
                    current_loss_frequency = n_iso - freq
                    if loss_frequency is None:
                        loss_frequency = current_loss_frequency
                    elif loss_frequency != current_loss_frequency:
                        warnings.warn(
                            f"Binary junction {junction_name}: inconsistent loss frequencies across blocks."
                        )
                        ambiguous = True
                        break

                    loss_isolates.update(absent_isolates)

            if ambiguous:
                continue

            if gain_isolates:
                jdf_all.at[idx, "binary_gain_isolates"] = ",".join(sorted(gain_isolates))
                jdf_all.at[idx, "binary_gain_frequency"] = gain_frequency

            if loss_isolates:
                jdf_all.at[idx, "binary_loss_isolates"] = ",".join(sorted(loss_isolates))
                jdf_all.at[idx, "binary_loss_frequency"] = loss_frequency

            if gain_isolates and loss_isolates:
                jdf_all.at[idx, "binary_event_type"] = "gain_and_loss"
            elif gain_isolates:
                jdf_all.at[idx, "binary_event_type"] = "gain"
            elif loss_isolates:
                jdf_all.at[idx, "binary_event_type"] = "loss"

            if junction_name in cluster_df.index:
                cluster_row = cluster_df.loc[junction_name]

                gain_has_own, gain_multi = _cluster_info(cluster_row, sorted(gain_isolates))
                loss_has_own, loss_multi = _cluster_info(cluster_row, sorted(loss_isolates))

                jdf_all.at[idx, "binary_gain_has_own_cluster"] = gain_has_own
                jdf_all.at[idx, "binary_loss_has_own_cluster"] = loss_has_own
                jdf_all.at[idx, "binary_gain_in_multiple_clusters"] = gain_multi
                jdf_all.at[idx, "binary_loss_in_multiple_clusters"] = loss_multi
                jdf_all.at[idx, "binary_event_in_multiple_clusters"] = (
                    bool(gain_multi) if pd.notna(gain_multi) else False
                ) or (
                    bool(loss_multi) if pd.notna(loss_multi) else False
                )

        except Exception as e:
            print(f"Skipping {junction_name}: {e}")

    return jdf_all