import numpy as np
import pandas as pd

import string
import os
from itertools import combinations
from pathlib import Path

from Bio import Align

import pypangraph as pp

from junction_analysis.helpers import get_hierarchical_order, get_tree_order
from junction_analysis.consensus import make_deduplicated_paths
import junction_analysis.pangraph_utils as pu


def _path_to_node_set(path):
    """Return frozenset of DeduplicatedNodes in a path."""
    return frozenset(path.nodes)


def _path_to_edge_set(path):
    """
    Return frozenset of Edges from consecutive DeduplicatedNode pairs.
    Returns None if the path has fewer than 2 nodes (no edges).
    """
    nodes = path.nodes
    if len(nodes) < 2:
        return None
    return frozenset(pu.Edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))


def _jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 1.0


def calculate_dedup_block_ji_matrix(pangraph):
    """
    Compute pairwise block Jaccard index between all isolate pairs using
    deduplicated paths (DeduplicatedNode comparison: id + strand + context).

    Returns
    -------
    pd.DataFrame of shape (n_isolates, n_isolates) with JI values.
    """
    dedup_paths, _ = make_deduplicated_paths(pangraph)
    isolates = sorted(dedup_paths.keys())
    node_sets = {iso: _path_to_node_set(dedup_paths[iso]) for iso in isolates}

    n = len(isolates)
    mat = np.ones((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        ji = _jaccard(node_sets[isolates[i]], node_sets[isolates[j]])
        mat[i, j] = ji
        mat[j, i] = ji

    return pd.DataFrame(mat, index=isolates, columns=isolates)


def calculate_dedup_edge_ji_matrix(pangraph):
    """
    Compute pairwise edge Jaccard index between all isolate pairs using
    deduplicated paths. Edge equality ignores context (id + strand only),
    and is bidirectional (edge == its reverse complement).

    Junctions with only one block (no edges) return a matrix of NaN.

    Returns
    -------
    pd.DataFrame of shape (n_isolates, n_isolates) with JI values,
    or NaN-filled DataFrame if no edges exist in any path.
    """
    dedup_paths, _ = make_deduplicated_paths(pangraph)
    isolates = sorted(dedup_paths.keys())

    edge_sets = {iso: _path_to_edge_set(dedup_paths[iso]) for iso in isolates}

    # if all paths have no edges (single-block junction) → return NaN matrix
    if all(es is None for es in edge_sets.values()):
        return pd.DataFrame(np.full((len(isolates), len(isolates)), np.nan), index=isolates, columns=isolates)

    # treat missing edge sets (single-node paths) as empty sets
    edge_sets = {iso: (es if es is not None else frozenset()) for iso, es in edge_sets.items()}

    n = len(isolates)
    mat = np.ones((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        ji = _jaccard(edge_sets[isolates[i]], edge_sets[isolates[j]])
        mat[i, j] = ji
        mat[j, i] = ji

    return pd.DataFrame(mat, index=isolates, columns=isolates)


def compare_sequences_by_shared_proportion(sequence_comparison_df, order="hierarchical"):
    sequence_comparison_df['similarity'] = 1 - sequence_comparison_df['diff'] / (sequence_comparison_df['shared'] + sequence_comparison_df['diff'])

    # reshape to full matrices with the same index/columns
    sim = sequence_comparison_df['similarity'].unstack()

    diff = sequence_comparison_df['diff'].unstack().reindex_like(sim)
    shared = sequence_comparison_df['shared'].unstack().reindex_like(sim)

    # set diagonal to 1
    np.fill_diagonal(sim.values, 1)

    if order == "hierarchical":
        isolate_order = get_hierarchical_order(sim)
    elif order == "tree":
        isolate_order = get_tree_order()
    else:
        return sim, diff, shared
    
    sim = sim.loc[isolate_order, isolate_order]
    diff = diff.loc[isolate_order, isolate_order]
    shared = shared.loc[isolate_order, isolate_order]

    return sim, diff, shared

def encode_blocks_all(paths_blocks):
    """
    Takes the paths block dictionary and returns:
      - encoded versions of each path's block list as strings
      - a mapping dict from original block ID -> character
      - an inverse mapping
    """
    unique_blocks = sorted({b for lst in paths_blocks.values() for b in lst}) #builds a set, {}
    symbols = (
        list(string.ascii_letters)                     # 52 (a–z, A–Z)
        + list(string.digits)                          # 10 (0–9)
        + list("!@#$%^&*()[]{}<>?|;:,./~`+-=_")        # 31 basic punctuations
        + list("§±€£¥©®°µ¶¿×÷")                        # 12 extra Latin/technical symbols
        + list("αβγδΔΣΩπλφψχ")                        # 10 Greek letters
        + list("←↑→↓⇌∞∑∏√≈≠≤≥")                        # 11 math symbols
    )
    if len(unique_blocks) > len(symbols):
        raise ValueError(
            f"Too many unique blocks ({len(unique_blocks)}); only {len(symbols)} symbols defined."
        )
    block_to_char = {b: symbols[i] for i, b in enumerate(unique_blocks)}
    char_to_block = {v: k for k, v in block_to_char.items()}
    encoded = {name: ''.join(block_to_char[b] for b in lst)
               for name, lst in paths_blocks.items()}
    return encoded, block_to_char, char_to_block

def divergence_points_from_aligned(aln_a: str, aln_b: str) -> int:
    """Count contiguous mismatching (or gap) runs between two aligned strings."""
    assert len(aln_a) == len(aln_b)
    count = 0
    in_mismatch = False
    for a, b in zip(aln_a, aln_b):
        is_match = (a == b)  # (no double-gap columns in pairwise2)
        if not is_match:
            if not in_mismatch:
                count += 1
                in_mismatch = True
        else:
            in_mismatch = False
    return count

def mean_divergence(seq1: str, seq2: str, tie_break: bool = True) -> float:
    """
    Compute average divergence points across all optimal alignments.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -0.0001
    aligner.open_gap_score = -0.0003
    aligner.extend_gap_score = -0.0002
    alignments = aligner.align(seq1, seq2)
    #return divergence_points_from_aligned(alignments[0].sequences[0], alignments[0].sequences[1])
    #alignments = pairwise2.align.globalms(seq1, seq2, 1, 0, 0, 0, one_alignment_only=True)
    #alignments = pairwise2.align.globalms(seq1, seq2, 1, -1e-6, -1e-6, -1e-6)
    vals = []
    # TODO: potentially only consider first alinment instead of all
    for aln in alignments:
        aln1, aln2 = aln
        vals.append(divergence_points_from_aligned(aln1, aln2))
    #    vals.append(divergence_points_from_aligned(aln[0], aln[1]))
    return sum(vals) / len(vals)

def calculate_divergence_matrix(pangraph, order="hierarchical"):
    """Calculate divergence points matrix for all paths in the pangraph."""

    path_dict = pangraph.to_path_dictionary() #always true, never false, if more blocks might have more entries
    paths_blocks = {name: [bid for bid, _ in lst] for name, lst in path_dict.items()}
    encoded_paths, block_to_char, char_to_block = encode_blocks_all(paths_blocks)

    names = sorted(encoded_paths.keys())
    divergence_points_mat = pd.DataFrame(0.0, index=names, columns=names)
    #counter = 0
    for i, j in combinations(range(len(names)), 2):
        n1, n2 = names[i], names[j]
        d = mean_divergence(encoded_paths[n1], encoded_paths[n2])
        #if counter % 100 == 0:
        #    print(i, j, n1, n2, d)
        #counter += 1
        divergence_points_mat.loc[n1, n2] = d
        divergence_points_mat.loc[n2, n1] = d

    if order == "hierarchical":
        isolate_order = get_hierarchical_order(divergence_points_mat)
    elif order == "tree":
        isolate_order = get_tree_order()
    else:
        return divergence_points_mat
    
    divergence_points_mat = divergence_points_mat.loc[isolate_order, isolate_order]

    return divergence_points_mat

def calculate_block_ji_matrix(pangraph, order="hierarchical"):
    """Calculate Jaccard index matrix based on shared blocks in paths. Duplicate blocks in paths ignored."""

    # gives whether block in present or not but doesn't give order of blocks in path
    blockcount_mat = pangraph.to_blockcount_df()
    blockcount_mat = (blockcount_mat > 0).astype(int) # ignores duplicate blocks in paths

    shared_blocks_arr = np.dot(blockcount_mat.T, blockcount_mat)
    blocks_per_isolate = blockcount_mat.sum(axis = 0)
    union_blocks_arr = blocks_per_isolate.values[:, None] + blocks_per_isolate.values[None, :] - shared_blocks_arr

    names = blockcount_mat.columns

    jaccard_blocks_arr = shared_blocks_arr / union_blocks_arr
    jaccard_blocks_df = pd.DataFrame(
        jaccard_blocks_arr,
        index=names,   # isolate names as row labels
        columns=names  # isolate names as column labels
    )
    jaccard_blocks_df.index.name = "path_i"
    jaccard_blocks_df.columns.name = "path_j"

    if order == "hierarchical":
        isolate_order = get_hierarchical_order(jaccard_blocks_df)
    elif order == "tree":
        isolate_order = get_tree_order()
    else:
        return jaccard_blocks_df
    
    jaccard_blocks_df = jaccard_blocks_df.loc[isolate_order, isolate_order]

    shared_df = pd.DataFrame(shared_blocks_arr, index=names, columns=names)
    union_df  = pd.DataFrame(union_blocks_arr,  index=names, columns=names)
    shared_df = shared_df.loc[isolate_order, isolate_order]
    union_df = union_df.loc[isolate_order, isolate_order]

    return jaccard_blocks_df, union_df - shared_df, shared_df

def edges_from_path(blocks):
    return set([(blocks[i], blocks[i+1]) for i in range(len(blocks)-1)])

def calculate_edge_ji_matrix(pangraph, order="hierarchical"):
    """Calculate Jaccard index matrix based on shared edges in paths. """
    path_dict = pangraph.to_path_dictionary()
    paths_blocks = {name: [bid for bid, _ in lst] for name, lst in path_dict.items()}
    edge_sets = {name: edges_from_path(blocks) for name, blocks in paths_blocks.items()}

    names = sorted(edge_sets.keys())
    n_isolates = len(names)

    sizes = {name: len(edge_sets[name]) for name in names}

    # allocate matrices
    inter_mat = np.zeros((n_isolates, n_isolates), dtype=int)
    union_mat = np.zeros((n_isolates, n_isolates), dtype=int)

    # diagonal = size of each set
    for idx, name in enumerate(names):
        inter_mat[idx, idx] = sizes[name]
        union_mat[idx, idx] = sizes[name]

    # fill upper triangle, mirror to lower
    for i, j in combinations(range(n_isolates), 2):
        a, b = names[i], names[j]
        edges_a, edges_b = edge_sets[a], edge_sets[b]
        inter = len(edges_a & edges_b)
        union = sizes[a] + sizes[b] - inter
        inter_mat[i, j] = inter_mat[j, i] = inter
        union_mat[i, j] = union_mat[j, i] = union

    edges_ji_matrix = np.empty((n_isolates, n_isolates), dtype=float)
    np.divide(inter_mat, union_mat, out=edges_ji_matrix, where=(union_mat != 0))
    edges_ji_matrix[union_mat == 0] = 1.0

    #edges_from_path(paths_blocks['NZ_CP096110.1'])
    jaccard_edges_df = pd.DataFrame(
            edges_ji_matrix,
            index=names,   # isolate names as row labels
            columns=names # isolate names as column labels
    )
    jaccard_edges_df.index.name = "path_i"
    jaccard_edges_df.columns.name = "path_j"

    if order == "hierarchical":
        isolate_order = get_hierarchical_order(jaccard_edges_df)
    elif order == "tree":
        isolate_order = get_tree_order()
    else:
        return jaccard_edges_df
    
    jaccard_edges_df = jaccard_edges_df.loc[isolate_order, isolate_order]

    shared_df = pd.DataFrame(inter_mat, index=names, columns=names)
    union_df  = pd.DataFrame(union_mat,  index=names, columns=names)
    shared_df = shared_df.loc[isolate_order, isolate_order]
    union_df = union_df.loc[isolate_order, isolate_order]

    return jaccard_edges_df, union_df - shared_df, shared_df


def _mean_upper_triangle(df):
    """Return the mean of the strictly upper triangle of a square DataFrame."""
    arr = df.values.astype(float)
    n = arr.shape[0]
    if n < 2:
        return float("nan")
    idx = np.triu_indices(n, k=1)
    return arr[idx].mean()


_ALL_METRICS = ["block_ji", "edge_ji", "dedup_block_ji", "dedup_edge_ji", "seq_similarity", "n_divergence_points"]


def _upper_triangle_pairs(mat_df):
    """
    Extract upper-triangle pairs from a square DataFrame.
    Returns list of (name_i, name_j, value).
    """
    names = list(mat_df.index)
    arr = mat_df.values.astype(float)
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        a, b = names[i], names[j]
        # canonicalise order so pairs from differently-sorted matrices merge correctly
        if a > b:
            a, b = b, a
            val = arr[j, i]
        else:
            val = arr[i, j]
        pairs.append((a, b, val))
    return pairs


def compute_diversity_all_junctions(
    pangraph_dir,
    junction_names=None,
    metrics=None,
    save_path=None,
    pairwise_save_path=None,
):
    """
    For every junction pangraph in `pangraph_dir`, compute pairwise diversity
    metrics across all isolate pairs and return summary (mean) and pairwise DataFrames.

    Available metrics (pass subset to ``metrics`` to skip others):
      - ``block_ji``            block Jaccard index
      - ``edge_ji``             edge Jaccard index
      - ``seq_similarity``      accessory genome sequence similarity
      - ``n_divergence_points`` divergence point count

    Parameters
    ----------
    pangraph_dir : str or Path
    junction_names : list of str or None
    metrics : list of str or None
        Subset of metrics to compute. Defaults to all four.
    save_path : str or None
        CSV path for the per-junction mean-diversity summary.
    pairwise_save_path : str or None
        CSV path for the long-format pairwise distances:
        columns junction_name, isolate_1, isolate_2, <metric>, ...

    Returns
    -------
    summary_df : pd.DataFrame
        Columns: junction_name, mean_<metric>, ...
    pairwise_df : pd.DataFrame
        Columns: junction_name, isolate_1, isolate_2, <metric>, ...
    """
    pangraph_dir = Path(pangraph_dir)

    if metrics is None:
        metrics = list(_ALL_METRICS)
    else:
        unknown = set(metrics) - set(_ALL_METRICS)
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Choose from {_ALL_METRICS}")

    if junction_names is None:
        junction_names = [f.stem for f in sorted(pangraph_dir.glob("*.json"))]

    summary_rows = []
    pairwise_rows = []

    for jname in junction_names:
        fpath = pangraph_dir / f"{jname}.json"
        if not fpath.exists():
            print(f"Pangraph not found for {jname}, skipping.")
            continue

        try:
            pan = pp.Pangraph.from_json(str(fpath))
        except Exception as e:
            print(f"Error loading pangraph for {jname}: {e}")
            continue

        # compute requested metric matrices
        metric_matrices = {}
        for metric in metrics:
            try:
                mat = None
                if metric == "block_ji":
                    mat = calculate_block_ji_matrix(pan, order=None)
                elif metric == "edge_ji":
                    mat = calculate_edge_ji_matrix(pan, order=None)
                    if isinstance(mat, tuple):
                        mat = mat[0]
                elif metric == "dedup_block_ji":
                    mat = calculate_dedup_block_ji_matrix(pan)
                elif metric == "dedup_edge_ji":
                    mat = calculate_dedup_edge_ji_matrix(pan)
                elif metric == "seq_similarity":
                    mat = compare_sequences_by_shared_proportion(
                        pan.pairwise_accessory_genome_comparison(), order=None
                    )[0]
                elif metric == "n_divergence_points":
                    mat = calculate_divergence_matrix(pan, order=None)
                metric_matrices[metric] = mat
            except Exception as e:
                import traceback
                print(f"  {jname} [{metric}]: {e}")
                traceback.print_exc()
                metric_matrices[metric] = None

        # summary row
        summary_row = {"junction_name": jname}
        for metric, mat in metric_matrices.items():
            summary_row[f"mean_{metric}"] = _mean_upper_triangle(mat) if mat is not None else float("nan")
        summary_rows.append(summary_row)

        # pairwise rows — collect all pairs across all computed metrics
        # use the first available matrix to get the list of pairs
        pair_data = {}  # (iso_i, iso_j) -> {metric: value}
        for metric, mat in metric_matrices.items():
            if mat is None:
                continue
            for iso_i, iso_j, val in _upper_triangle_pairs(mat):
                key = (iso_i, iso_j)
                if key not in pair_data:
                    pair_data[key] = {}
                pair_data[key][metric] = val

        for (iso_i, iso_j), vals in pair_data.items():
            row = {"junction_name": jname, "isolate_1": iso_i, "isolate_2": iso_j}
            row.update(vals)
            pairwise_rows.append(row)

    summary_cols = ["junction_name"] + [f"mean_{m}" for m in metrics]
    summary_df = pd.DataFrame(summary_rows, columns=summary_cols)

    pairwise_cols = ["junction_name", "isolate_1", "isolate_2"] + list(metrics)
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        for col in pairwise_cols:
            if col not in pairwise_df.columns:
                pairwise_df[col] = float("nan")
        pairwise_df = pairwise_df[pairwise_cols]

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        print(f"Saved diversity summary to {save_path}")

    if pairwise_save_path is not None:
        Path(pairwise_save_path).parent.mkdir(parents=True, exist_ok=True)
        pairwise_df.to_csv(pairwise_save_path, index=False)
        print(f"Saved pairwise distances to {pairwise_save_path}")

    return summary_df, pairwise_df


def compute_dedup_ji_all_junctions(pangraph_dir, junction_names=None, save_path=None):
    """
    For every junction pangraph compute pairwise block and edge Jaccard indices
    using deduplicated paths.

    Parameters
    ----------
    pangraph_dir : str or Path
    junction_names : list of str or None
    save_path : str or None

    Returns
    -------
    pd.DataFrame with columns:
        junction_name, isolate_1, isolate_2, block_ji_dist, edge_ji_dist
    """
    pangraph_dir = Path(pangraph_dir)
    if junction_names is None:
        junction_names = [f.stem for f in sorted(pangraph_dir.glob("*.json"))]

    rows = []
    for jname in junction_names:
        fpath = pangraph_dir / f"{jname}.json"
        if not fpath.exists():
            print(f"Pangraph not found for {jname}, skipping.")
            continue
        try:
            pan = pp.Pangraph.from_json(str(fpath))
        except Exception as e:
            print(f"Error loading {jname}: {e}")
            continue

        try:
            dedup_paths, _ = make_deduplicated_paths(pan)
        except Exception as e:
            print(f"  {jname} [make_deduplicated_paths]: {e}")
            continue

        isolates = sorted(dedup_paths.keys())

        # block JI
        try:
            block_df = calculate_dedup_block_ji_matrix(pan)
        except Exception as e:
            print(f"  {jname} [dedup_block_ji]: {e}")
            block_df = None

        # edge JI
        try:
            edge_df = calculate_dedup_edge_ji_matrix(pan)
        except Exception as e:
            print(f"  {jname} [dedup_edge_ji]: {e}")
            edge_df = None

        for i, j in combinations(range(len(isolates)), 2):
            iso_a, iso_b = isolates[i], isolates[j]
            block_val = block_df.loc[iso_a, iso_b] if block_df is not None else np.nan
            edge_val = edge_df.loc[iso_a, iso_b] if edge_df is not None else np.nan
            rows.append({
                "junction_name": jname,
                "isolate_1": iso_a,
                "isolate_2": iso_b,
                "block_ji_dist": block_val,
                "edge_ji_dist": edge_val,
            })

    df = pd.DataFrame(rows, columns=["junction_name", "isolate_1", "isolate_2", "block_ji_dist", "edge_ji_dist"])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")

    return df