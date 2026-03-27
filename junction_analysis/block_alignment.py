from pathlib import Path
import subprocess
import re

import pandas as pd
import numpy as np

from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import pypangraph as pp
from Bio import SeqIO

from junction_analysis.helpers import write_isolate_fasta, repo_root


# create block fasta files
def create_block_msas(example_junction, only_core=False): # 2 min 14
    """
    Parameters
    ----------
    example_junction : str
    only_core : bool
        If True, only write and align core blocks. Default: False.
    """
    example_pangraph = pp.Pangraph.from_json(f"../results/junction_pangraphs/{example_junction}.json")

    if only_core:
        blockstats_df = example_pangraph.to_blockstats_df()
        core_ids = set(blockstats_df[blockstats_df["core"] == True].index)

    parent_dir = Path(f"../results/block_fastas/{example_junction}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    aligned_dir = Path(f"../results/block_alignments/{example_junction}")
    aligned_dir.mkdir(parents=True, exist_ok=True)

    for block in example_pangraph.blocks:
        if only_core and block.id not in core_ids:
            continue
        output_path = parent_dir / f"block_{block.id}_sequences.fa"
        write_isolate_fasta(example_pangraph, block, output_path)

    for fasta_file in parent_dir.glob("block_*_sequences.fa"):
        aln_file = aligned_dir / fasta_file.name.replace("_sequences.fa", "_aln.fa")

        # Run MAFFT quietly (you can remove '--quiet' if you want verbose output)
        subprocess.run(
            ["mafft", "--quiet", str(fasta_file)],
            stdout=open(aln_file, "w"),
            check=True
        )

def _context_to_filename(context: str) -> str:
    """Convert a block context string to a safe filename component."""
    return context if context else "nocontext"


def create_block_msas_for_cluster(example_junction, isolate_list, cl, ambiguous_blocks,
                                            path_dict_cluster_dedup=None,
                                            in_fasta_dir=None, out_base_dir=None,
                                            mafft_bin="mafft"):
    """
    Only (re)align blocks whose block_id appears in ambiguous_blocks.
    When path_dict_cluster_dedup is provided, sequences are filtered by the nid
    corresponding to each block's specific context copy, so different context copies
    of the same duplicated block get separate alignment files.
    File naming: block_{block_id}_ctx_{context}_sequences_cluster_aln.fa
    """

    isolate_set = set(isolate_list)

    root = repo_root()
    results_dir = root / "results"
    if in_fasta_dir is None:
        in_fasta_dir = results_dir / "block_fastas" / example_junction

    if out_base_dir is None:
        out_base_dir = results_dir / "block_alignments_cluster"

    out_parent = Path(out_base_dir) / str(example_junction) / f"cluster_{cl}"
    out_fasta_dir = out_parent / "fastas"
    out_aln_dir = out_parent / "alns"
    out_fasta_dir.mkdir(parents=True, exist_ok=True)
    out_aln_dir.mkdir(parents=True, exist_ok=True)

    for block in ambiguous_blocks:
        fasta_file = Path(in_fasta_dir) / f"block_{block.id}_sequences.fa"
        ctx_tag = _context_to_filename(block.context)

        out_aln = out_aln_dir / f"block_{block.id}_ctx_{ctx_tag}_sequences_cluster_aln.fa"
        if out_aln.exists():
            continue

        # Build the set of nids that correspond to this (block_id, context) copy
        if path_dict_cluster_dedup is not None:
            valid_nids = {
                str(node.nid)
                for path in path_dict_cluster_dedup.values()
                for node in path.nodes
                if node.id == block.id and node.context == block.context and node.nid is not None
            }

        # Filter records: by isolate, and by nid when path_dict is available
        records = []
        for rec in SeqIO.parse(str(fasta_file), "fasta"):
            parts = rec.id.split("__", 1)
            iso = parts[0]
            nid = parts[1] if len(parts) > 1 else None
            if iso not in isolate_set:
                continue
            if path_dict_cluster_dedup is not None and nid not in valid_nids:
                continue
            records.append(rec)

        if len(records) < 2:
            print(f"Block {block.id} ctx={block.context} only appeared once in cluster. Assuming gain with ancestral state 0.")
            continue

        out_fa = out_fasta_dir / f"block_{block.id}_ctx_{ctx_tag}_sequences_cluster.fa"
        SeqIO.write(records, str(out_fa), "fasta")

        with open(out_aln, "w") as aln_out:
            subprocess.run([mafft_bin, "--quiet", str(out_fa)],
                           stdout=aln_out, check=True)

def read_fasta_alignment(path):
    for rec in SeqIO.parse(path, "fasta"):
        yield rec.id, str(rec.seq)

def core_bounds(records):
    """Return (start, end) of intersection of non-gap spans across sequences.
    - start: position of first non-gap character
    - end : position of last non-gap character
    Returns (0, 0) if records is empty or all sequences are gap-only.
    """
    seqs = [s for _, s in records]
    if not seqs:
        return (0, 0)

    L = len(seqs[0])

    non_gap_seqs = [s for s in seqs if set(s) != {'-'}]
    if not non_gap_seqs:
        return (0, L - 1)

    firsts = [s.find(s.replace('-', '')[0]) if '-' in s else 0 for s in non_gap_seqs]
    lasts  = [s.rfind(s.replace('-', '')[-1]) if '-' in s else L-1 for s in non_gap_seqs]

    start, end = max(firsts), min(lasts)
    return (start, end)

def avg_pairwise_distance(arr):
    """
    Compute relative Hamming distances for all pairs of sequences in an alignment.
    Only ACGT positions are considered; gaps and ambiguous characters are ignored.
    Returns the mean distance across all pairs and an array of per-pair distances.
    """
    # ignores gap positions
    # shape: (n, L_core)
    n, L = arr.shape

    if n < 2 or L == 0:
        return 0.0, np.array([])

    # valid ACGT positions
    valid = np.isin(arr, list("ACGT"))          # (n, L)

    # broadcast to pairwise (i, j, pos)
    arr_i = arr[:, None, :]                     # (n, 1, L)
    arr_j = arr[None, :, :]                     # (1, n, L)
    valid_i = valid[:, None, :]                 # (n, 1, L)
    valid_j = valid[None, :, :]                 # (1, n, L)

    pair_valid = valid_i & valid_j              # (n, n, L)
    pair_mismatch = (arr_i != arr_j) & pair_valid

    valid_counts = pair_valid.sum(axis=2)       # (n, n)
    mism_counts = pair_mismatch.sum(axis=2)     # (n, n)

    # only upper triangle (i < j), no self-pairs
    iu, ju = np.triu_indices(n, k=1)
    vc = valid_counts[iu, ju]
    mc = mism_counts[iu, ju]

    # keep only pairs with at least one valid site
    mask = vc > 0
    if not np.any(mask):
        return 0.0, np.array([])

    dists = mc[mask] / vc[mask]
    return float(dists.mean()), dists

def calc_consensus_seq(arr):
    """
    Majority/plurality consensus:
    - counts A,C,G,T and '-' as valid characters
    - chooses the most frequent symbol (in tie chooses in this order ACGT_)
    """
    n, L = arr.shape

    # allowed consensus symbols (gaps included)
    symbols = np.array(list("ACGT-"))   # shape (5,)

    # count occurrences: shape (5, L)
    counts = np.vstack([(arr == sym).sum(axis=0) for sym in symbols])
    # find most frequent symbol, if tie returns the first maximal one
    max_idx = counts.argmax(axis=0)     # index in symbols
    consensus = symbols[max_idx]

    # guard a gainst positions with no nucleotide in acgt, shouldn't happen
    no_info = counts.sum(axis=0) == 0
    consensus[no_info] = "N"

    return "".join(consensus)

def avg_distance_to_consensus(arr, consensus):
    """
    arr: (n, L) array of single characters (A/C/G/T/-)
    consensus: string of length L
    Ignores positions where either seq or consensus is not A/C/G/T.
    Returns average Hamming distance (mismatches / valid sites) over sequences.
    """
    n, L = arr.shape
    if n == 0 or L == 0:
        return 0.0

    bases = np.array(list("ACGT"))

    cons_arr = np.array(list(consensus))
    cons_arr = np.char.upper(cons_arr)

    # valid positions: both in A/C/G/T (no gaps)
    valid_seq  = np.isin(arr, bases)          # (n, L)
    valid_cons = np.isin(cons_arr, bases)     # (L,)
    valid = valid_seq & valid_cons[None, :]   # (n, L)

    mismatches = (arr != cons_arr[None, :]) & valid

    valid_counts = valid.sum(axis=1)          # per sequence
    mism_counts  = mismatches.sum(axis=1)

    mask = valid_counts > 0
    if not np.any(mask):
        return 0.0

    dists = mism_counts[mask] / valid_counts[mask]
    return float(dists.mean())


def analyze_alignment(path, return_pairwise_dists=False):
    recs = list(read_fasta_alignment(path))
    seqs = [s for _, s in recs]

    if not seqs:
        empty_stats = dict(
            file=path.name, block_id=int(path.name.split("_", 2)[1]), context="",
            n_seqs=0, alignment_len=0, core_len=0,
            left_overhang=0, right_overhang=0,
            mismatch_columns=0, mismatch_fraction=0.0,
            avg_pairwise_dist=0.0, avg_consensus_dist=0.0,
        )
        if return_pairwise_dists:
            return empty_stats, np.array([])
        return empty_stats

    start, end = core_bounds(recs)

    length = len(seqs[0])

    left_overhang = start
    right_overhang = length - end - 1
    core_len = end - start + 1

    mismatch_cols = 0
    for j in range(start, end + 1):
        bases = {s[j] for s in seqs}
        if len(bases) >= 2:           # ≥2 distinct symbols
            mismatch_cols += 1

    seqs_arr = np.array([list(s) for s in seqs])
    seqs_arr = np.char.upper(seqs_arr) 

    avg_pair_dist, dists = avg_pairwise_distance(seqs_arr)

    consensus_seq = calc_consensus_seq(seqs_arr)
    avg_cons_dist = avg_distance_to_consensus(seqs_arr, consensus_seq)

    # filename: block_{block_id}_ctx_{context}_sequences_cluster_aln.fa
    # or legacy: block_{block_id}_sequences_cluster_aln.fa
    fname = path.name
    block_id = int(fname.split("_", 2)[1])
    ctx_match = re.search(r"_ctx_(.+?)_sequences", fname)
    context = ctx_match.group(1) if ctx_match else ""
    if context == "nocontext":
        context = ""

    stats = dict(
        file=fname,
        block_id=block_id,
        context=context,
        n_seqs=len(recs),
        alignment_len=length,
        core_len=core_len,
        left_overhang=left_overhang,
        right_overhang=right_overhang,
        mismatch_columns=mismatch_cols,
        mismatch_fraction=(mismatch_cols / core_len) if core_len > 0 else 0.0,
        avg_pairwise_dist=avg_pair_dist,
        avg_consensus_dist=avg_cons_dist,
    )

    if return_pairwise_dists:
        return stats, dists
    else:
        return stats
    
def summarize_block_msas(junction_name, cl=None, save_df=True, return_pairwise_dists=False,
                         context_sensitive=False,
                         base_full=None,
                         base_cluster=None):
    """
    Creates summary statistics for block alignments.
    If cl is None: summarize ../results/block_alignments/<junction_name>/block_*_aln.fa
    If cl is not None: summarize ../results/block_alignments_cluster/<junction_name>/cluster_<cl>/alns/block_*_aln.fa

    context_sensitive : bool
        If False (default), pairwise_dict is keyed by block_id only — old behaviour,
        suitable when each block appears at most once per alignment directory.
        If True, pairwise_dict is keyed by (block_id, context) — use this when the
        directory contains separate alignment files per context copy of a duplicated block.
        The returned DataFrame always contains a 'context' column regardless of this flag.
    """

    results_dir = repo_root() / "results"
    if base_full is None:
        base_full = results_dir / "block_alignments"
    if base_cluster is None:
        base_cluster = results_dir / "block_alignments_cluster"

    if cl is None:
        aligned_dir = Path(base_full) / str(junction_name)
        out_csv = aligned_dir / f"{junction_name}_alignment_stats.csv"
    else:
        aligned_dir = Path(base_cluster) / str(junction_name) / f"cluster_{cl}" / "alns"
        out_csv = aligned_dir / f"{junction_name}_cluster_{cl}_alignment_stats.csv"

    results = []
    pairwise_dict = {} if return_pairwise_dists else None

    for p in sorted(aligned_dir.glob("block_*_aln.fa")):
        if return_pairwise_dists:
            stats, dists = analyze_alignment(p, return_pairwise_dists=True)
            if stats["n_seqs"] == 0:
                continue
            results.append(stats)
            key = (stats["block_id"], stats["context"]) if context_sensitive else stats["block_id"]
            pairwise_dict[key] = dists
        else:
            stats = analyze_alignment(p)
            if stats["n_seqs"] == 0:
                continue
            results.append(stats)

    df = pd.DataFrame(results)

    # if there was no block file found, return empty results and save an empty dataframe
    if df.empty:
        df = pd.DataFrame(columns=["block_id", "avg_pairwise_dist"])
        if save_df:
            df.to_csv(out_csv, index=False)
        if return_pairwise_dists:
            return df, pairwise_dict
        else:
            return df

    df = df.sort_values("block_id")

    pangraph = pp.Pangraph.from_json(str(repo_root() / "results" / "junction_pangraphs" / f"{junction_name}.json"))
    blockstats_df = pangraph.to_blockstats_df().reset_index()
    df = df.merge(blockstats_df, on="block_id", how="left")

    if save_df:
        df.to_csv(out_csv, index=False)

    if return_pairwise_dists:
        return df, pairwise_dict

    return df


def cluster_alignment(alignment_path):
    # gap positions are ignored in identity calculation (that's why one doesn't have to worry about missing parts on the ends)
    aln = AlignIO.read(alignment_path, "fasta")
    calc = DistanceCalculator('identity') # counts non identical positions, ignoring gaps
    dm = calc.get_distance(aln)

    distance_df = pd.DataFrame(dm.matrix, index=dm.names, columns=dm.names)
    distance_matrix = distance_df.to_numpy()
    distance_matrix = np.nan_to_num(distance_matrix, nan = 0.0)
    distance_matrix = distance_matrix + distance_matrix.T

    # --- hierarchical clustering (UPGMA/average by default) ---
    Z = linkage(squareform(distance_matrix), method="average")  # use 'single'/'complete' if preferred

    return distance_matrix, Z, dm.names

def retrieve_cluster_assignments(Z, names, n_clusters):
    labels_k = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Create dictionary mapping isolate name -> cluster number
    cluster_assignments = {name.split("__", 1)[0]: int(label) for name, label in zip(names, labels_k)}

    return cluster_assignments

def build_block_trees(
    junction_name,
    fasttree_exe="fasttree",
    overwrite=False,
    min_seqs=3,
    min_core_len=500,
):
    """
    For a given junction, run FastTree on all block alignment files that pass
    sequence count and core length thresholds, and write the resulting trees
    into the same directory.

    Parameters
    ----------
    junction_name : str
        Name of the junction (subfolder under ../results/block_alignments/).
    fasttree_exe : str, optional
        Path to the FastTree executable (default: "fasttree" on PATH).
    overwrite : bool, optional
        If False, skip tree files that already exist.
    min_seqs : int, optional
        Minimum number of sequences (n_seqs) required to run FastTree.
    min_core_len : int, optional
        Minimum core alignment length (core_len) required to run FastTree.
    """
    aligned_dir = repo_root() / "results" / "block_alignments" / junction_name
    aln_files = sorted(aligned_dir.glob("block_*_aln.fa"))

    if not aln_files:
        print(f"No alignments found for junction {junction_name} in {aligned_dir}")
        return

    stats_csv = aligned_dir / f"{junction_name}_alignment_stats.csv"

    stats_df = pd.read_csv(stats_csv)
    stats_df = stats_df.set_index("file")
    stats_df["tree_built"] = False

    for aln_file in aln_files:
        fname = aln_file.name

        if fname not in stats_df.index:
            print(f"[{junction_name}] No stats for {fname}, skipping.")
            continue

        row = stats_df.loc[fname]
        n_seqs = int(row["n_seqs"])
        core_len = int(row["core_len"])

        # Apply thresholds
        if n_seqs < min_seqs or core_len < min_core_len:
            print(
                f"[{junction_name}] Skipping {fname}: "
                f"n_seqs={n_seqs} (min {min_seqs}), "
                f"core_len={core_len} (min {min_core_len})"
            )
            continue

        # Tree filename: block_X_aln.tree (same directory)
        tree_file = aln_file.with_suffix(".tree")

        if tree_file.exists() and not overwrite:
            print(f"[{junction_name}] Skipping existing tree: {tree_file.name}")
            continue

        print(
            f"[{junction_name}] Building tree for {fname} "
            f"(n_seqs={n_seqs}, core_len={core_len}) -> {tree_file.name}"
        )
        with tree_file.open("w") as out_f:
            subprocess.run(
                [fasttree_exe, "-nt", "-gtr", str(aln_file)],
                stdout=out_f,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        stats_df.loc[fname, "tree_built"] = True
        
    stats_df.to_csv(stats_csv)