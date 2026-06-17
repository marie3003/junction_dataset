import pandas as pd
import numpy as np
import math
import re
import copy
import matplotlib.pyplot as plt

from pathlib import Path
from IPython.display import display

from collections import Counter, defaultdict, OrderedDict

from Bio import Phylo, SeqIO
import pypangraph as pp
import junction_analysis.pangraph_utils as pu
from junction_analysis.plotting import plot_junction_pangraph_interactive, plot_pairwise_distance_hist, plot_snp_pos_distribution, plot_block_distance_distribution, plot_fitch_tree
from junction_analysis.junction_trees import build_tree_from_block_list, cluster_tree_by_branch_length, compute_pairwise_distances
from junction_analysis.helpers import get_block_length, snp_positions, build_subtree, repo_root
from junction_analysis.block_alignment import create_block_msas_for_cluster, summarize_block_msas, avg_pairwise_distance


def find_invertible_ids(paths: dict) -> set:
    """
    Returns a set of block ids that appear in both orientations across all paths.
    A block that only ever appears on the minus strand is not considered invertible —
    it is simply a block that is consistently reversed. Only blocks seen in both
    plus and minus orientation are truly invertible.
    This is needed for context definition: context anchors must never be invertible.
    """
    plus_ids  = set()
    minus_ids = set()

    for path in paths.values():
        for node in path:
            if node.strand:
                plus_ids.add(node.id)
            else:
                minus_ids.add(node.id)

    return plus_ids & minus_ids


def filter_isolates_by_anchor_count(path_dict: dict, rare_context_thresh: float = 0.2,
                                     z_score_threshold: float = 3.5, verbose: bool = False,
                                     figsize=(8, 4)) -> list:
    """
    Filter isolates whose anchor block count is an outlier on the low end,
    detected via modified z-score on the per-isolate anchor count.

    A block is a potential anchor if it is:
      - not duplicated in this isolate
      - not rare (appears in >= floor(rare_context_thresh * n_isolates) isolates)
      - on the majority strand for that block id

    Parameters
    ----------
    path_dict : dict
        isolate -> Path (iterable of nodes with .id and .strand attributes)
    rare_context_thresh : float
        Fraction threshold below which a block is considered too rare to anchor.
    z_score_threshold : float
        Modified z-score below which an isolate is flagged as a low-anchor outlier.
        Default 3.5 (standard threshold for modified z-score).
    verbose : bool
        If True, plot the distribution with outliers highlighted.
    figsize : tuple

    Returns
    -------
    list of isolate names that are not low-anchor outliers.
    """
    isolate_counts: Counter = Counter()
    for path in path_dict.values():
        for bid in set(n.id for n in path):
            isolate_counts[bid] += 1
    thresh = np.floor(rare_context_thresh * len(path_dict))
    rare_ids = {bid for bid, cnt in isolate_counts.items() if cnt < thresh}

    strand_counts = defaultdict(Counter)
    for path in path_dict.values():
        for n in path:
            strand_counts[n.id][n.strand] += 1
    majority_strand = {bid: counts.most_common(1)[0][0] for bid, counts in strand_counts.items()}

    anchor_counts = []
    for isolate, path in path_dict.items():
        id_counts = Counter(n.id for n in path.nodes)
        duplicated_in_isolate = {bid for bid, cnt in id_counts.items() if cnt > 1}
        non_anchor_ids = duplicated_in_isolate | rare_ids
        n_anchor = sum(1 for n in path.nodes if n.id not in non_anchor_ids and n.strand == majority_strand.get(n.id))
        anchor_counts.append(n_anchor)

    counts = np.array(anchor_counts)
    median = np.median(counts)
    mad = np.median(np.abs(counts - median))
    modified_z = np.zeros_like(counts, dtype=float) if mad == 0 else 0.6745 * (counts - median) / mad

    outlier_set = {iso for iso, z in zip(path_dict.keys(), modified_z) if z < -z_score_threshold}
    qualified   = [iso for iso in path_dict.keys() if iso not in outlier_set]

    if outlier_set:
        print(f"Filtered {len(outlier_set)} low-anchor outliers: {sorted(outlier_set)}")

    if verbose:
        bins = range(0, max(anchor_counts) + 2)
        included_counts = [c for iso, c in zip(path_dict.keys(), anchor_counts) if iso not in outlier_set]
        outlier_counts  = [c for iso, c in zip(path_dict.keys(), anchor_counts) if iso in outlier_set]
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(included_counts, bins=bins, color="#7394c2", edgecolor="white", linewidth=0.5, label="included")
        if outlier_counts:
            ax.hist(outlier_counts, bins=bins, color="#a6444f", edgecolor="white", linewidth=0.5, label="outlier")
        ax.set_xlabel("number of potential anchor blocks")
        ax.set_ylabel("number of isolates")
        ax.set_title("Distribution of anchor block counts per isolate")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return qualified


def filter_isolates_by_anchor_fraction(path_dict: dict, min_anchor_fraction: float = 0.7,
                                        rare_context_thresh: float = 0.1) -> list:
    """
    For each isolate, compute the fraction of blocks that are usable as anchors.
    A block is a non-anchor if it is:
      - duplicated: the same block id appears more than once in that isolate's path, OR
      - rare: the block id appears in fewer than floor(rare_context_thresh * n_isolates)
              isolates across the full path_dict, OR
      - inverted: the block appears on the minus strand in that isolate

    Parameters
    ----------
    path_dict : dict
        isolate -> Path (iterable of nodes with .id and .strand attributes)
    min_anchor_fraction : float
        Minimum fraction of anchor blocks required to keep an isolate (default 0.7).
    rare_context_thresh : float
        A block is rare if it appears in fewer than
        floor(rare_context_thresh * n_isolates) isolates (default 0.1).

    Returns
    -------
    list of isolate names that meet the threshold.
    """
    # Compute rare ids globally (same threshold as used during deduplication)
    isolate_counts: Counter = Counter()
    for path in path_dict.values():
        for bid in set(n.id for n in path):
            isolate_counts[bid] += 1
    thresh = np.floor(rare_context_thresh * len(path_dict))
    rare_ids = {bid for bid, cnt in isolate_counts.items() if cnt < thresh}

    qualified = []

    for isolate, path in path_dict.items():
        nodes = list(path)
        if not nodes:
            continue

        id_counts = Counter(n.id for n in nodes)
        duplicated_in_isolate = {bid for bid, cnt in id_counts.items() if cnt > 1}
        non_anchor_ids = duplicated_in_isolate | rare_ids

        n_anchor = sum(
            1 for n in nodes
            if n.id not in non_anchor_ids and n.strand
        )
        fraction = n_anchor / len(nodes)

        if fraction >= min_anchor_fraction:
            qualified.append(isolate)
        else:
            print(f"Isolate {isolate} filtered out: {n_anchor}/{len(nodes)} anchor blocks ({fraction:.2%})")

    return qualified


def _compute_path_dict_stats(path_dict: dict, rare_context_thresh: float):
    """
    Compute duplicated_ids, rare_ids, and invertible_ids directly from a path_dict.

    - duplicated_ids : block ids that appear more than once in at least one isolate's path
    - rare_ids       : block ids that appear in fewer than
                       floor(rare_context_thresh * n_isolates) isolates
    - invertible_ids : block ids seen in both orientations across all paths
                       (via find_invertible_ids)
    """
    isolate_counts: Counter = Counter()   # how many isolates contain each block id
    duplicated_ids: set = set()

    for path in path_dict.values():
        seen_in_isolate: Counter = Counter(n.id for n in path)
        for bid, cnt in seen_in_isolate.items():
            isolate_counts[bid] += 1
            if cnt > 1:
                duplicated_ids.add(bid)

    thresh = np.floor(rare_context_thresh * len(path_dict))
    rare_ids = {bid for bid, cnt in isolate_counts.items() if cnt < thresh}
    invertible_ids = find_invertible_ids(path_dict)

    return duplicated_ids, rare_ids, invertible_ids


def _deduplicate_path_dict(path_dict: dict, duplicated_ids: set, rare_ids: set, invertible_ids: set):
    """
    Core deduplication logic.  Assigns context labels to duplicated blocks based
    on the nearest upstream non-duplicated, non-invertible, non-rare block.
    Nodes must already have nid set on them before calling this function.

    Parameters
    ----------
    path_dict      : dict[isolate -> Path(Node,...)] with nid set on each node
    duplicated_ids : block ids that are duplicated in at least one isolate
    rare_ids       : block ids that are too rare to serve as context anchors
    invertible_ids : block ids that appear in both orientations

    Returns
    -------
    deduplicated_paths : dict[isolate -> Path(DeduplicatedNode,...)]
    freq               : dict[DeduplicatedNode -> int]
    dedup_stats        : list[dict]
        One dict per isolate with keys: isolate, n_blocks, n_ambiguous_blocks.
        A block is counted as ambiguous when its context suffix count > 1,
        i.e. two or more copies of the same duplicated block share the same
        context anchor.
    """
    deduplicated_paths: dict = {}
    freq = Counter()
    dedup_stats = []

    for isolate, path in path_dict.items():
        last_non_dup: str | None = None
        context_counts: dict[str, int] = {}
        n_blocks = 0
        n_duplicated = 0
        n_ambiguous = 0

        dedup_nodes = []
        for n in path:
            n_blocks += 1
            if n.id in duplicated_ids:
                if last_non_dup is None:
                    raise ValueError("last_non_dup must not be None when encountering a duplicated node")

                n_duplicated += 1
                count = context_counts.get(n.id, 0) + 1
                context_counts[n.id] = count

                if count > 1:
                    n_ambiguous += 1

                context = f"{last_non_dup}_{count}"
                dn = pu.DeduplicatedNode(n.id, n.strand, context, n.nid)

            else:
                dn = pu.DeduplicatedNode(n.id, n.strand, "", n.nid)
                if n.id not in invertible_ids and n.id not in rare_ids:
                    last_non_dup = n.id
                    context_counts.clear()

            dedup_nodes.append(dn)
            freq[dn] += 1

        deduplicated_paths[isolate] = pu.Path(dedup_nodes)
        dedup_stats.append(dict(isolate=isolate, n_blocks=n_blocks, n_duplicated_blocks=n_duplicated, n_ambiguous_blocks=n_ambiguous))

    return deduplicated_paths, dict(freq), dedup_stats


def deduplicate_blocks_by_sequence(pangraph, path_dict, junction_name, length_threshold=0.001):
    """
    For each duplicated block, cluster isolate sequences by branch length on the
    block alignment tree and assign a new block id {block_id}_{cluster_id} to
    each node, effectively splitting the duplicated block into sequence-based
    sub-blocks.

    Nodes whose block id is not duplicated are left unchanged. Nodes for which
    no tree or no cluster assignment is found are also left unchanged.

    Parameters
    ----------
    pangraph : pp.Pangraph
    path_dict : dict[isolate -> Path(Node,...)]
        Plain paths with nids set. Modified in place.
    junction_name : str
        Used to locate block alignment trees under results/block_alignments/.
    length_threshold : float
        Branch-length threshold passed to cluster_tree_by_branch_length.

    Returns
    -------
    path_dict : dict[isolate -> Path(Node,...)]
        Same dict with node block ids updated for duplicated blocks.
    """
    results_dir = repo_root() / "results"
    blockstats_df = pangraph.to_blockstats_df()
    duplicated_block_ids = set(blockstats_df[blockstats_df["duplicated"] == True].index)

    for block_id in duplicated_block_ids:
        tree_path = results_dir / "block_alignments" / junction_name / f"block_{block_id}_aln.tree"
        if not tree_path.exists():
            continue

        cluster_map = cluster_tree_by_branch_length(str(tree_path), length_threshold)

        if len(set(cluster_map.values())) <= 1:
            continue

        # cluster_map keys are "{isolate}__{nid}" -> cluster_id
        iso_nid_to_cluster = {}
        for key, cluster_id in cluster_map.items():
            iso, nid = key.split("__", 1)
            iso_nid_to_cluster[(iso, nid)] = cluster_id

        for isolate, path in path_dict.items():
            for node in path.nodes:
                if node.id != block_id:
                    continue
                cluster_id = iso_nid_to_cluster.get((isolate, str(node.nid)))
                node.id = f"{block_id}_{cluster_id}"

    return path_dict


def make_deduplicated_paths(pangraph, path_dict, rare_context_thresh=0.1) -> dict:
    """
    Convert a dict[isolate -> Path(Node,...)] into dict[isolate -> Path(DeduplicatedNode, ...)],
    where duplicated blocks get a context = closest non-duplicated, never inverted, less rare than rare_context_thresh block (ID) to the left.
    For non-duplicated blocks, context is set to "".
    Nodes in path_dict must already have nid set before calling this function.
    Deduplication is not perfect if there are no suitable context anchors that are seperating to differing block copies because the blocks in between tham are inverted, duplicated or rare.
    @param rare_context_threshold: is a threshold how rare blocks can be to be allowed as context anchors, right now it is hard coded to 10
    Additionally also return deduplicated block count dictionary.
    """
    # compute duplicated_ids from path_dict directly so seq-dedup renamed ids (e.g. BLOCKID_0)
    # are correctly identified as duplicated when they still appear more than once per isolate
    isolate_counts: Counter = Counter()
    duplicated_ids: set = set()
    for path in path_dict.values():
        seen: Counter = Counter(n.id for n in path)
        for bid, cnt in seen.items():
            isolate_counts[bid] += 1
            if cnt > 1:
                duplicated_ids.add(bid)

    rare_context_thresh_abs = np.floor(rare_context_thresh * len(path_dict))
    rare_ids = {bid for bid, cnt in isolate_counts.items() if cnt < rare_context_thresh_abs}
    invertible_ids = find_invertible_ids(path_dict)

    deduplicated_paths, freq, dedup_stats = _deduplicate_path_dict(path_dict, duplicated_ids, rare_ids, invertible_ids)
    return deduplicated_paths, freq, dedup_stats


def remap_consensus_to_global_contexts(consensus_path, filtered_cluster_paths, path_dict_cluster_dedup, path_dict):
    """
    Replace per-cluster deduplication contexts in a consensus path with the
    corresponding global deduplication contexts from path_dict.

    The consensus path is the majority path among filtered_cluster_paths, so
    at least one isolate in filtered_cluster_paths has an identical path.
    For that isolate, we know the positional correspondence between
    path_dict_cluster_dedup (cluster contexts) and path_dict (global contexts)
    because both have the same block order — only the context strings differ.

    Returns a new pu.Path with global contexts.
    """
    # 1. Find an isolate whose filtered path matches the consensus
    majority_iso = next(
        (iso for iso, path in filtered_cluster_paths.items() if path == consensus_path),
        None,
    )
    if majority_iso is None:
        return consensus_path  # fallback: return unchanged

    # 2. Build positional mapping: cluster DeduplicatedNode → global DeduplicatedNode
    #    Both paths have the same block order; zip them to align by position.
    cluster_nodes = path_dict_cluster_dedup[majority_iso].nodes
    global_nodes  = path_dict[majority_iso].nodes
    cluster_to_global: dict = {}
    for c_node, g_node in zip(cluster_nodes, global_nodes):
        cluster_to_global[id(c_node)] = g_node

    # 3. The consensus path is a subsequence of cluster_nodes (filter removed some).
    #    Walk through cluster_nodes in order to find the matching global node for
    #    each consensus node.
    global_idx = 0
    remapped_nodes = []
    for c_node in consensus_path.nodes:
        # advance through cluster_nodes until we find this consensus node
        while global_idx < len(cluster_nodes) and id(cluster_nodes[global_idx]) != id(c_node):
            global_idx += 1
        if global_idx < len(cluster_nodes):
            g_node = global_nodes[global_idx]
            remapped_nodes.append(pu.DeduplicatedNode(g_node.id, g_node.strand, g_node.context, g_node.nid))
            global_idx += 1
        else:
            # fallback: keep original node if no match found
            remapped_nodes.append(c_node)

    return pu.Path(remapped_nodes)


def rededuplicate_cluster_paths(path_dict_cluster: dict, rare_context_thresh=0.1):
    """
    Redo deduplication for a subset of isolates (one cluster), recomputing which
    blocks are duplicated, rare, or invertible using only the isolates in the cluster.
    This typically yields better context anchors than the full-dataset deduplication
    because problematic blocks from other isolates no longer inflate the duplicated/
    invertible sets.

    The original path_dict (full deduplication) is left untouched; only a new
    path_dict restricted to the cluster isolates is returned.

    Parameters
    ----------
    path_dict_cluster : dict[isolate -> Path(Node,...)]
        Subset of isolates belonging to this cluster, using the *original* (non-
        deduplicated) node representation so that block ids and strands are intact.
        Nodes must already have nid set.
    rare_context_thresh : float
        Fraction threshold below which a block is considered too rare to anchor.

    Returns
    -------
    deduplicated_paths : dict[isolate -> Path(DeduplicatedNode,...)]
    freq               : dict[DeduplicatedNode -> int]
    """
    duplicated_ids, rare_ids, invertible_ids = _compute_path_dict_stats(
        path_dict_cluster, rare_context_thresh
    )
    deduplicated_paths, freq, dedup_stats = _deduplicate_path_dict(path_dict_cluster, duplicated_ids, rare_ids, invertible_ids)
    return deduplicated_paths, freq, dedup_stats

def count_edges(dedup_paths: dict) -> dict:
    """
    When edges are counted, context is not considered, deduplication is only done based on nodes which could lead to overcounting of edges within duplicated path segments.
    However, if path segments are duplicated incoming and outgoing edges to the path segments are rare which would still lead to filtering out these paths as consensus paths.
    dedup_paths: dict[isolate -> Path(DeduplicatedNode,...)]
    returns: dict[Edge, int]  (edge -> frequency across all paths)
    """
    counts = Counter()
    for path in dedup_paths.values():
        nodes = path.nodes  # or: list(path)
        if len(nodes) < 2:
            continue
        for u, v in zip(nodes, nodes[1:]):
            counts[pu.Edge(u, v)] += 1
    return dict(counts)

def find_unique_frequent_paths(paths_dict, edge_counts, flow_threshold = 10):

    unique_paths = set()

    for path in paths_dict.values():
        is_valid_path = True
        for idx in range(len(path.nodes)-1):
            edge = pu.Edge(path.nodes[idx], path.nodes[idx+1])
            if edge_counts[edge] < flow_threshold:
                is_valid_path = False
                break
        if is_valid_path:
            unique_paths.add(path)

    unique_paths = [p for p in unique_paths]
    print(f"Found {len(unique_paths)} unique paths.")

    return unique_paths

def filter_deduplicated_paths(paths, filter_set, strand_sensitive=True):
    """Removes blocks that are inside filter_set.
    If strand_sensitive=False, membership is checked by (id, context) only,
    so both strand orientations of a filtered block are removed.
    Return filtered paths dict."""

    filtered_paths = {}

    for iso, path in paths.items():
        if strand_sensitive:
            kept = [dnode for dnode in path.nodes if dnode not in filter_set]
        else:
            kept = [
                dnode for dnode in path.nodes
                if pu.DeduplicatedNode(dnode.id, True, dnode.context) not in filter_set
            ]
        filtered_paths[iso] = pu.Path(kept)

    return filtered_paths

def compute_edge_jaccard_matrix(deduplicated_paths: dict, consensus_paths: list) -> pd.DataFrame:
    """
    Compute a similarity matrix between each deduplicated path (rows)
    and each consensus path (columns) using edge-level Jaccard index.

    Returns:
        pd.DataFrame with isolates as rows and consensus indices as columns
    """

    # --- Precompute edge sets for consensus paths ---
    consensus_edge_sets = []
    for cons_path in consensus_paths:
        nodes = cons_path.nodes
        if len(nodes) < 2:
            consensus_edge_sets.append(set())
        else:
            consensus_edge_sets.append({pu.Edge(u, v) for u, v in zip(nodes, nodes[1:])})

    # --- Prepare matrix container ---
    isolates = list(deduplicated_paths.keys())
    n_cons = len(consensus_paths)
    data = []

    # --- Compute Jaccard similarities ---
    for iso in isolates:
        path = deduplicated_paths[iso]
        nodes = path.nodes
        if len(nodes) < 2:
            path_edges = set()
        else:
            path_edges = {pu.Edge(u, v) for u, v in zip(nodes, nodes[1:])}

        similarities = []
        for cons_edges in consensus_edge_sets:
            if not path_edges and not cons_edges:
                sim = 1.0  # both empty
            else:
                inter = len(path_edges & cons_edges)
                union = len(path_edges | cons_edges)
                sim = inter / union if union > 0 else 0.0
            similarities.append(sim)

        data.append(similarities)

    # --- Build DataFrame ---
    df = pd.DataFrame(data, index=isolates,
                      columns=[f"consensus_{i+1}" for i in range(n_cons)])
    return df

def assign_isolates_to_consensus(similarity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each isolate (row) to the consensus path with the highest
    edge-level Jaccard similarity.

    Parameters
    ----------
    similarity_df : pd.DataFrame
        Output from compute_edge_jaccard_matrix(), with isolates as rows
        and consensus paths as columns.

    Returns
    -------
    assignments : pd.DataFrame
        A DataFrame with columns:
            - 'best_consensus': consensus path name (column name)
            - 'similarity': maximum Jaccard value
    """

    best_consensus = similarity_df.idxmax(axis=1)
    best_similarity = similarity_df.max(axis=1)

    assignments = pd.DataFrame({
        "best_consensus": best_consensus,
        "similarity": best_similarity
    })

    return assignments

def remove_rare_consensus_paths(consensus_paths, deduplicated_paths, edge_ji_df, assignment_df, min_n_isolates_per_consensus=5):
    """
    Iteratively remove consensus paths that have less than min_n_isolates assigned to them.
    Returns filtered consensus paths and updated assignment_df.
    """

    while True:
        # Count how many isolates are assigned to each consensus
        isolates_per_consensus = assignment_df['best_consensus'].value_counts().sort_values()

        # Find consensus paths below threshold
        rare_consensus_paths = isolates_per_consensus[isolates_per_consensus < min_n_isolates_per_consensus].index.tolist()
        if not rare_consensus_paths:
            break  # all have enough isolates

        # Remove the rarest consensus path (the one with fewest isolates)
        rarest_consensus = rare_consensus_paths[0]
        print(f"Removing rare consensus path: {rarest_consensus} "
            f"({isolates_per_consensus[rarest_consensus]} isolates)")

        # Remove it from the consensus list
        consensus_paths = [p for i, p in enumerate(consensus_paths)
                        if f"consensus_{i+1}" != rarest_consensus]

        # Recompute Jaccard similarity & assignments
        edge_ji_df = compute_edge_jaccard_matrix(deduplicated_paths, consensus_paths)
        assignment_df = assign_isolates_to_consensus(edge_ji_df)

    return edge_ji_df, assignment_df, consensus_paths



def find_consensus_paths(pangraph, rare_block_threshold = 10, rare_edge_threshold = 10, min_n_isolates_per_consensus=5):
    """
    Finds consesus paths of junction pangraph. Paths are first deduplicated, then rare blocks are filtered out, then paths containing rare edges are filtered out.
    It is possible that the context of some blocks contains blocks that were filtered out, but this should not affect the consensus path finding too much.
    @param rare_block_threshold: blocks that are less frequent then this threshold are filtered out of the paths.
    @param rare_edge_threshold: paths containing an edge that is less frequent than this threshold are filtered out.
    @return consensus paths: unique paths that remain after filtering
    @return path_dict: original paths written as Path and Node objects
    """
    # create path dict and add nids
    path_dict_raw = pangraph.to_path_dictionary()
    path_dict_raw = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict_raw.items()}
    for isolate, path in path_dict_raw.items():
        for idx, node in enumerate(path.nodes):
            node.nid = pangraph.paths[isolate].nodes[idx]

    # deduplicate
    deduplicated_paths, deduplicated_blog_freq, _ = make_deduplicated_paths(pangraph, path_dict_raw)

    # refilter, after deduplication some blocks might now have a frequency below the threshold (now consider duplication and inversion)
    rare_deduplicated_blocks = {dnode for dnode, cnt in deduplicated_blog_freq.items() if cnt < rare_block_threshold}
    deduplicated_paths_filtered = filter_deduplicated_paths(deduplicated_paths, rare_deduplicated_blocks)

    # instead of edge matrix make dictionary with edge as key and count as value to store edge frequency
    edge_count = count_edges(deduplicated_paths_filtered)

    # filter paths with rare edges (inversed edged are considered the same)
    consensus_paths = find_unique_frequent_paths(deduplicated_paths_filtered, edge_count, flow_threshold = rare_edge_threshold)

    # to add paths to their consensus paths, compare the deduplicated paths (after rare node deletion) to the selected consensus paths and add it to the one which is most similar
    # TODO: what is a good similarity metric, since they should be quite similar to their consensus paths, edge jaccard index could be a good choice
    edge_ji_df = compute_edge_jaccard_matrix(deduplicated_paths_filtered, consensus_paths)
    assignment_df = assign_isolates_to_consensus(edge_ji_df)

    # refilter consensus paths such that each consensus path has at least n assigned isolate, e.g. n = 5
    edge_ji_df, assignment_df, consensus_paths = remove_rare_consensus_paths(consensus_paths, deduplicated_paths_filtered, edge_ji_df, assignment_df, min_n_isolates_per_consensus)

    return consensus_paths, deduplicated_paths, deduplicated_blog_freq, edge_ji_df, assignment_df


def filter_paths_by_block_freq(paths, freq_threshold = 0.5):
    """
    Filter blocks out of all paths that are in less than half of the isolates.

    - paths: dict {isolate_id: pu.Path}
    - freq_tresh: float, how frequent do blocks need to be to not be filtered out

    Returns
    -------
    dict {isolate_id: pu.Path} with filtered paths.
    """

    n_isos = len(paths)
    filtered_paths = {}

    # If only one isolate in the cluster, do not filter anything
    if n_isos <= 1:
        for iso in paths.keys():
            filtered_paths[iso] = paths[iso]

    # i.e. keep blocks with count >= ceil(n_isos * freq_threshold)
    threshold = math.ceil(n_isos * freq_threshold)

    # Count block presence per cluster (after deduplication max. once per isolate)
    block_counts = defaultdict(int)
    for iso in paths.keys():
        for node in paths[iso].nodes:
            block_counts[node] += 1

    # Blocks to filter in this cluster
    filter_set = {node for node, cnt in block_counts.items() if cnt < threshold}
    filtered_paths = filter_deduplicated_paths(paths, filter_set)

    return filtered_paths


def compute_majority_path(paths, verbose = True):
    """
    Find the majority path (consensus) among its isolates.

    - paths: dict {isolate_id: pu.Path}
    - cluster_map: dict {isolate_id: cluster_id}

    Returns
    -------
    pu.Path: consensus path
    """

    n_isos = len(paths)
    # If only one isolate in cluster, that path *is* the consensus
    if n_isos == 1:
        consensus_path = next(iter(paths.values()))
        if verbose:
            print(f"1 / 1 (single isolate)")
        return consensus_path

    # Count identical paths in this cluster
    path_counter = Counter(path for path in paths.values())

    # Pick the majority path
    # tie-breaker: higher count, then longer path
    majority_path, majority_count = max(
        path_counter.items(),
        key=lambda kv: (kv[1], len(kv[0].nodes))
    )

    if verbose:
        print(f"{majority_count} / {n_isos} isolates share the majority path")

    return majority_path


def consensus_paths_and_assignments(consensus_paths_by_cluster, cluster_map):
    """
    Build:
      1) an ordered list of consensus paths for plotting
      2) a DataFrame with best_consensus per isolate.

    Parameters
    ----------
    consensus_paths_by_cluster : dict
        {cluster_id: pu.Path}
    cluster_map : dict
        {isolate_id: cluster_id}

    Returns
    -------
    consensus_list : list of pu.Path
        Ordered list of consensus paths (for plotting).
    assignment_df : pd.DataFrame
        Index: isolate_id
        Columns:
          - 'cluster'        : cluster_id
          - 'best_consensus' : 'consensus_i' label for that cluster
    cluster_to_consensus_name : dict
        {cluster_id: 'consensus_i'}
    """
    # all clusters present
    clusters = sorted(set(cluster_map.values()))

    consensus_list = []
    cluster_to_consensus_name = {}

    # assign consensus_i names in a stable order of clusters
    i = 1
    for cl in clusters:
        if cl not in consensus_paths_by_cluster:
            continue
        consensus_list.append(consensus_paths_by_cluster[cl])
        cluster_to_consensus_name[cl] = f"consensus_{i}"
        i += 1

    # build assignment dataframe
    rows = []
    # preserve input order of isolates (if cluster_map is an OrderedDict) 
    # otherwise this will just be some deterministic order
    for iso, cl in cluster_map.items():
        best_cons = cluster_to_consensus_name.get(cl, None)
        rows.append(
            {"isolate": iso, "cluster": cl, "best_consensus": best_cons}
        )

    assignment_df = pd.DataFrame(rows).set_index("isolate")
    assignment_df = assignment_df.rename_axis(None)

    return consensus_list, assignment_df, cluster_to_consensus_name

def build_block_index(path_dict, strand_sensitive=True):
    """
    Returns:
      blocks: list of blocks in fixed order (index -> block)
      block2idx: dict mapping block -> index

    If strand_sensitive=False, blocks with the same (id, context) but different
    strands are treated as the same block. The stored representative node always
    has strand=True.
    """
    block2idx = OrderedDict()  # preserves insertion order deterministically
    for isolate, path in path_dict.items():
        for block in path.nodes:
            key = block if strand_sensitive else pu.DeduplicatedNode(block.id, True, block.context)
            if key not in block2idx:
                block2idx[key] = len(block2idx)
    blocks = list(block2idx.keys())
    return blocks, dict(block2idx)


def encode_paths_binary(path_dict, block2idx, strand_sensitive=True):
    """
    Returns:
      enc: dict isolate -> list[int] (0/1) with length = number of unique blocks

    If strand_sensitive=False, presence of a block is determined by (id, context)
    only, regardless of strand.
    """
    n_blocks = len(block2idx)
    enc = {}

    for isolate, path in path_dict.items():
        vec = [0] * n_blocks
        for block in path.nodes:
            key = block if strand_sensitive else pu.DeduplicatedNode(block.id, True, block.context)
            if key in block2idx:
                vec[block2idx[key]] = 1
        enc[isolate] = vec

    return enc

def build_edge_index(edge_path_dict):
    """
    Returns:
      edges: list of edges in fixed order (index -> edge)
      edge2idx: dict mapping edge -> index

    Edges are compared using Edge.__eq__ which handles reverse-complement
    symmetry and n_occurrence (both None treated as equal regardless of occurrence).
    """
    edge2idx = OrderedDict()
    for isolate, edge_path in edge_path_dict.items():
        for edge in edge_path:
            if edge not in edge2idx:
                edge2idx[edge] = len(edge2idx)
    edges = list(edge2idx.keys())
    return edges, dict(edge2idx)


def encode_edge_paths_binary(edge_path_dict, edge2idx):
    """
    Returns:
      enc: dict isolate -> list[int] (0/1) with length = number of unique edges
    """
    n_edges = len(edge2idx)
    enc = {}

    for isolate, edge_path in edge_path_dict.items():
        vec = [0] * n_edges
        for edge in edge_path:
            if edge in edge2idx:
                vec[edge2idx[edge]] = 1
        enc[isolate] = vec

    return enc


def fitch_ancestral_reconstruction(tree, binary_encodings):
    """
    Fitch parsimony for binary (0/1) sequences.
    Adds `clade.sequence` (list of 0/1) to every node.
    """

    # length of binary encoded sequences
    L = len(next(iter(binary_encodings.values())))

    # ---------- bottom-up (postorder): compute state sets ----------
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            seq = binary_encodings[clade.name]
            clade._state_sets = [{s} for s in seq]
        else:
            child_sets = [child._state_sets for child in clade.clades]
            clade._state_sets = []
            for i in range(L):
                intersection = set.intersection(*(cs[i] for cs in child_sets))
                if intersection:
                    clade._state_sets.append(intersection)
                else:
                    clade._state_sets.append(set.union(*(cs[i] for cs in child_sets)))
    
    # one could add a top down pass to refine ambiguities for lower level sequences but its not needed at the moment

def decide_ambiguities(root_states, isolate_list, junction_name, cl, blocks, block2idx, path_dict_cluster_dedup=None, individual_gain_thresh = 0.001, verbose = False):
    """
    Returns
    -------
    root_states : list[set]
        Updated root states after ambiguity resolution.
    stats : dict
        Keys:
          n_ambiguous        – total number of ambiguous blocks
          n_single_isolate   – ambiguous blocks with only one isolate (undecidable, defaulted to gain)
          n_decided_loss     – blocks decided as loss (low diversity)
          n_decided_gain     – blocks decided as gain (high diversity)
    """
    stats = dict(n_total_blocks=len(blocks), n_ambiguous=0, n_single_isolate=0, n_decided_loss=0, n_decided_gain=0, avg_pairwise_dists=[])
    ambiguous_blocks = [block for block, state in zip(blocks, root_states) if len(state) > 1]
    stats["n_ambiguous"] = len(ambiguous_blocks)

    # don't run following analysis if there are no ambiguous blocks
    if ambiguous_blocks == []:
        print("No ambiguous blocks.")
        return root_states, stats

    create_block_msas_for_cluster(junction_name, isolate_list, cl, ambiguous_blocks,
                                  path_dict_cluster_dedup=path_dict_cluster_dedup)
    df, pair_dists = summarize_block_msas(junction_name, cl, return_pairwise_dists=True, context_sensitive=True)

    # if all blocks only appear once → all undecidable, default to gain
    if df.empty:
        for block in ambiguous_blocks:
            pos = block2idx[block]
            root_states[pos] = {0}
        stats["n_single_isolate"] = len(ambiguous_blocks)
        return root_states, stats

    if verbose:
        display(df)
        plot_block_distance_distribution(pair_dists, [(block.id, block.context) for block in ambiguous_blocks], bins=70, cols=4, figsize=(14, 10), vline=individual_gain_thresh, vline_kwargs={"color": "black", "linestyle": "--"})

    for block in ambiguous_blocks:
        pos = block2idx[block]

        row = df.loc[(df["block_id"] == block.id) & (df["context"] == block.context), "avg_pairwise_dist"]

        # block not found in df, probably because it only exists once → default to gain (ancestral state 0)
        if row.empty:
            root_states[pos] = {0}
            stats["n_single_isolate"] += 1
            continue

        val = row.iloc[0]
        stats["avg_pairwise_dists"].append(val)

        # loss: low diversity → block present in ancestor
        if val < individual_gain_thresh:
            root_states[pos] = {1}
            stats["n_decided_loss"] += 1
        # gain: high diversity → block absent in ancestor
        else:
            root_states[pos] = {0}
            stats["n_decided_gain"] += 1

    return root_states, stats

def filter_paths_by_edge_presence(
    path_dict_cluster,
    edge_path_dict_cluster,
    edges,
    root_states,
    edge2idx,
    junction_name,
    cl,
    individual_gain_thresh=0.001,
    verbose=False,
):
    """
    Filter each isolate's path by deciding node presence/absence based on
    edge root states from Fitch ancestral reconstruction.

    For each internal node (all except first and last boundary nodes):
      - Both surrounding edges have root state {0} → remove node
      - Both have root state {1} → keep node
      - Otherwise (one or both ambiguous, or one {0} and one {1}) → ambiguous

    Ambiguous nodes are resolved by sequence similarity using block MSAs
    per block id (no deduplication): low avg pairwise distance → kept
    (loss event), high → removed (gain event). Nodes appearing in only one
    isolate default to removal. One decision per block id applies to all
    occurrences across all isolates.

    Parameters
    ----------
    path_dict_cluster : dict[isolate -> Path(Node,...)]
        Plain paths with nids set, pre-filtered to cluster isolates.
    edge_path_dict_cluster : dict[isolate -> EdgePath]
        Edge paths pre-filtered to cluster isolates.
    edges : list[Edge]
        Ordered edge list from build_edge_index.
    root_states : list[set]
        Fitch root state sets, aligned with edges.
    edge2idx : dict[Edge -> int]
    junction_name : str
    cl : cluster id
    individual_gain_thresh : float
        Avg pairwise distance threshold; below → loss (keep), above → gain (remove).
    verbose : bool

    Returns
    -------
    filtered_paths : dict[isolate -> Path(Node,...)]
    """
    edge_to_state = {e: s for e, s in zip(edges, root_states)}

    # --- Pass 1: determine keep / remove / ambiguous per (isolate, node_idx) ---
    node_decisions = {}
    ambiguous_block_ids = set()

    for isolate, path in path_dict_cluster.items():
        nodes = path.nodes
        edge_path = edge_path_dict_cluster[isolate]
        n = len(nodes)
        decisions = ['keep'] * n

        for i in range(1, n - 1):  # skip first and last boundary nodes
            in_state  = edge_to_state.get(edge_path.edges[i - 1], {1})
            out_state = edge_to_state.get(edge_path.edges[i],     {1})

            if in_state == {1} or out_state == {1}:
                decisions[i] = 'keep'
            elif in_state == {0} and out_state == {0}:
                decisions[i] = 'remove'
            else:
                decisions[i] = 'ambiguous'
                ambiguous_block_ids.add(nodes[i].id)

        node_decisions[isolate] = decisions

    # --- Pass 2: resolve ambiguous block ids via sequence similarity ---
    ambiguous_block_decision = {}  # block_id -> 'keep' | 'remove'

    if ambiguous_block_ids:
        # use DeduplicatedNode with empty context so create_block_msas_for_cluster
        # can look up block fasta files by id
        ambiguous_blocks = [pu.DeduplicatedNode(bid, True, "") for bid in ambiguous_block_ids]
        isolate_list = list(path_dict_cluster.keys())

        create_block_msas_for_cluster(junction_name, isolate_list, cl, ambiguous_blocks)
        df, pair_dists = summarize_block_msas(junction_name, cl, return_pairwise_dists=True, context_sensitive=False)

        if verbose and not df.empty:
            display(df)

        for block in ambiguous_blocks:
            if df.empty:
                ambiguous_block_decision[block.id] = 'remove'
                continue

            row = df.loc[df["block_id"] == block.id, "avg_pairwise_dist"]

            if row.empty:
                ambiguous_block_decision[block.id] = 'remove'
            elif row.iloc[0] < individual_gain_thresh:
                ambiguous_block_decision[block.id] = 'keep'
            else:
                ambiguous_block_decision[block.id] = 'remove'

    # --- Pass 3: build filtered paths ---
    filtered_paths = {}

    for isolate, path in path_dict_cluster.items():
        nodes = path.nodes
        decisions = node_decisions[isolate]

        kept = []
        for node, decision in zip(nodes, decisions):
            if decision == 'keep':
                kept.append(node)
            elif decision == 'ambiguous':
                if ambiguous_block_decision.get(node.id) == 'keep':
                    kept.append(node)

        filtered_paths[isolate] = pu.Path(kept)

    return filtered_paths


def cluster_isolates_by_core_blocks(pangraph, path_dict, junction_name, clustering_bl_thresh=0.001):
    """
    Cluster isolates by pairwise distance on core blocks.

    Builds a core-block alignment and tree for the given pangraph, then cuts
    the tree at clustering_bl_thresh to assign each isolate to a cluster.

    Parameters
    ----------
    pangraph : pp.Pangraph
    path_dict : dict[isolate -> Path(Node,...)]
        Paths with nids set, used to build the core-block alignment.
    junction_name : str
        Used to name the output directory under results/consensus_analysis/.
    clustering_bl_thresh : float
        Branch-length threshold for cutting the tree into clusters.

    Returns
    -------
    cluster_map : dict[isolate -> cluster_id]
    cluster_to_isos : dict[cluster_id -> list[isolate]]
    """
    root = repo_root()
    results_dir = root / "results"

    blockstats_df = pangraph.to_blockstats_df()
    core_block_ids = set(blockstats_df.loc[blockstats_df["core"] == True].index)
    _, any_path = next(iter(path_dict.items()))
    core_block_nodes = [node for node in any_path.nodes if node.id in core_block_ids]

    out_dir = results_dir / "consensus_analysis" / junction_name
    build_tree_from_block_list(pangraph, path_dict, core_block_nodes, list(path_dict.keys()), str(out_dir), "core")
    tree_path_core = out_dir / "core_blocks_aln.newick"
    cluster_map = cluster_tree_by_branch_length(tree_path_core, clustering_bl_thresh)

    cluster_to_isos = defaultdict(list)
    for iso, cl in cluster_map.items():
        if iso in path_dict:
            cluster_to_isos[cl].append(iso)

    return cluster_map, cluster_to_isos, tree_path_core, out_dir


def find_consensus_paths_core(junction_name, clustering_bl_thresh = 0.005, consensus_criterium = 'core_genome_tree', tree_path = None, block_freq_thresh = 0.5, rare_context_thresh = 0.2, min_anchor_fraction = 0.6, min_anchor_fraction_relaxed = 0.3, min_cluster_retention = 0.8, plot_consensus = False, plot_annotations = False, plot_pair_dist = False, plot_snp_dist = False, plot_ambiguities = False):
    """
    Determine consensus paths for each cluster of isolates at a junction.

    The function loads the pangraph for the given junction, clusters isolates by
    phylogenetic distance on core blocks, and then derives one consensus path per
    cluster using the chosen method.

    Parameters
    ----------
    junction_name : str
        Name of the junction (used to locate the pangraph JSON and output files).
    clustering_bl_thresh : float
        Branch-length threshold for cutting the core-block tree into clusters.
        Isolates with pairwise distance below this threshold end up in the same
        cluster.  Default 0.005.
    consensus_criterium : {'core_genome_tree', 'block_freq'}
        Method used to derive the consensus path for each cluster:
        - 'core_genome_tree': ancestral sequence reconstruction (Fitch parsimony)
          on the species tree.  For each cluster, blocks absent at the root are
          filtered out and the majority path of the remaining isolates is taken as
          the consensus.  Isolates with fewer than 70 % anchor blocks (non-
          duplicated, non-inverted) are excluded from consensus inference.
        - 'block_freq': blocks present in fewer than `block_freq_thresh` of the
          cluster isolates are filtered; the majority path is then taken as the
          consensus.
    tree_path : str or None
        Path to the species tree in Newick format.  If None, defaults to
        ``<repo_root>/config/polished_tree.nwk``.  Only used with
        ``consensus_criterium='core_genome_tree'``.
    block_freq_thresh : float
        Minimum fraction of cluster isolates a block must appear in to be kept
        when using ``consensus_criterium='block_freq'``.  Default 0.5.
    plot_consensus : bool
        If True, display an interactive pangraph plot with consensus paths and
        cluster assignments overlaid.
    plot_annotations : bool
        If True, display an interactive pangraph plot with MGE and integration/
        recombination annotations.
    plot_pair_dist : bool
        If True, plot the pairwise core-block distance distribution with the
        clustering threshold marked.
    plot_snp_dist : bool
        If True, plot the SNP position distribution across the core-block
        alignment.
    plot_ambiguities : bool
        If True, print verbose output during ancestral-state ambiguity resolution
        (only relevant for ``consensus_criterium='core_genome_tree'``).

    Returns
    -------
    cluster_map_core : dict[isolate -> cluster_id]
        Mapping of each isolate to its cluster.
    consensus_paths_core : dict[cluster_id -> Path]
        One deduplicated consensus path per cluster.
    path_dict : dict[isolate -> Path(DeduplicatedNode,...)]
        Deduplicated paths for all isolates (full dataset deduplication).
        Be careful, because deduplication in path_dict was done based on all isolates, while deduplication for consensus path was done per cluster only using isolates with enough anchors.
        So the context of some blocks in consensus paths might not be the same as in path_dict.
    consensus_paths_plotting : list[Path]
        Consensus paths renumbered and ordered for plotting.
    assignment_df_plotting : pd.DataFrame
        Per-isolate cluster assignments used for plotting.
    all_root_states : dict[cluster_id -> list[set]]
        Raw Fitch root state sets per block per cluster.
        Only returned when ``consensus_criterium='core_genome_tree'``.
    all_root_states_unique : dict[cluster_id -> list[set]]
        Root states after ambiguity resolution.
        Only returned when ``consensus_criterium='core_genome_tree'``.
    ambiguity_stats_df : pd.DataFrame
        One row per (junction, cluster) with columns:
        junction_name, cluster_id, n_ambiguous, n_single_isolate,
        n_decided_loss, n_decided_gain.
        Only returned when ``consensus_criterium='core_genome_tree'``.
    dedup_stats_df : pd.DataFrame
        One row per (junction, dedup_step, isolate) with columns:
        junction_name, dedup_step, isolate, n_blocks, n_ambiguous_blocks.
        dedup_step is "global" for the full-dataset deduplication and
        "cluster_<id>" for the per-cluster rededuplication.
    """
    root = repo_root()
    results_dir = root / "results"
    if tree_path is None:
        tree_path = str(root / "config" / "polished_tree.nwk")

    pangraph_path = results_dir / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    path_dict_raw = pangraph.to_path_dictionary()
    path_dict_raw = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict_raw.items()}
    for isolate, path in path_dict_raw.items():
        for idx, node in enumerate(path.nodes):
            node.nid = pangraph.paths[isolate].nodes[idx]

    cluster_map_core, cluster_to_isos, tree_path_core, out_dir = cluster_isolates_by_core_blocks(pangraph, path_dict_raw, junction_name, clustering_bl_thresh)

    path_dict, block_freq, global_dedup_stats = make_deduplicated_paths(pangraph, path_dict_raw)
    for row in global_dedup_stats:
        row["junction_name"] = junction_name
        row["dedup_step"] = "global"
    all_dedup_stats = list(global_dedup_stats)

    if consensus_criterium == 'block_freq':
        consensus_paths_core = {}
        for cl, isos in cluster_to_isos.items():
            cluster_paths = {iso: path_dict[iso] for iso in isos}
            # filter blocks that are in less than 50% of isolates in a cluster
            cluster_bf_filtered_paths = filter_paths_by_block_freq(cluster_paths, freq_threshold=block_freq_thresh)

            # choose majority path per cluster after filtering as consensus path
            consensus_path = compute_majority_path(cluster_bf_filtered_paths)
            consensus_paths_core[cl] = consensus_path

    elif consensus_criterium == 'core_genome_tree':
        tree = Phylo.read(tree_path, "newick")
        tree.rooted = True

        consensus_paths_core = {}
        all_root_states = {}
        all_root_states_unique = {}
        ambiguity_stats_rows = []

        # Pre-compute valid isolates at the strict threshold
        valid_strict = set(filter_isolates_by_anchor_fraction(
            path_dict, min_anchor_fraction=min_anchor_fraction, rare_context_thresh=rare_context_thresh
        ))

        for cl, isolate_list in cluster_to_isos.items():
            n_cluster = len(isolate_list)

            # Step 1: strict anchor threshold
            filtered = [iso for iso in isolate_list if iso in valid_strict]

            # Step 2: if fewer than min_cluster_retention of the cluster remains, retry with relaxed threshold
            if n_cluster > 0 and len(filtered) / n_cluster < min_cluster_retention:
                print(
                    f"Cluster {cl}: only {len(filtered)}/{n_cluster} isolates pass strict anchor filter "
                    f"({len(filtered)/n_cluster:.0%} < {min_cluster_retention:.0%}), retrying with relaxed threshold ({min_anchor_fraction_relaxed:.0%})."
                )
                valid_relaxed = set(filter_isolates_by_anchor_fraction(
                    path_dict, min_anchor_fraction=min_anchor_fraction_relaxed, rare_context_thresh=rare_context_thresh
                ))
                filtered = [iso for iso in isolate_list if iso in valid_relaxed]

                # Step 3: still not enough — skip filtering entirely for this cluster
                if n_cluster > 0 and len(filtered) / n_cluster < min_cluster_retention:
                    print(
                        f"Cluster {cl}: still only {len(filtered)}/{n_cluster} isolates pass relaxed anchor filter "
                        f"({len(filtered)/n_cluster:.0%} < {min_cluster_retention:.0%}), skipping isolate filtering for this cluster."
                    )
                    filtered = list(isolate_list)

            isolate_list = filtered
            if not isolate_list:
                print(f"Cluster {cl}: all isolates filtered out by anchor fraction, skipping.")
                continue
            path_dict_cluster = {iso: path_dict_raw[iso] for iso in isolate_list}
            path_dict_cluster_dedup, freq_cluster, cluster_dedup_stats = rededuplicate_cluster_paths(path_dict_cluster, rare_context_thresh=rare_context_thresh)
            for row in cluster_dedup_stats:
                row["junction_name"] = junction_name
                row["dedup_step"] = f"cluster_{cl}"
            all_dedup_stats.extend(cluster_dedup_stats)

            blocks, block2idx = build_block_index(path_dict_cluster_dedup, strand_sensitive=False)
            binary_encodings = encode_paths_binary(path_dict_cluster_dedup, block2idx, strand_sensitive=False)

            # build subtree for isolates of one cluster
            subtree = build_subtree(tree, isolate_list)
            #Phylo.draw_ascii(subtree)

            # do ancestral sequence reconstruction on subtree and read ancestral sequence from root
            fitch_ancestral_reconstruction(subtree, binary_encodings)
            root_states = subtree.root._state_sets  # list of {0}/{1}
            print(root_states)
            root_states_unique, amb_stats = decide_ambiguities(root_states, isolate_list, junction_name, cl, blocks, block2idx, path_dict_cluster_dedup=path_dict_cluster_dedup, individual_gain_thresh=0.001, verbose = plot_ambiguities)
            ambiguity_stats_rows.append(dict(junction_name=junction_name, cluster_id=cl, n_isolates_in_cluster=len(isolate_list), **amb_stats))

            filter_set = {block for block, state in zip(blocks, root_states_unique) if 0 in state}
            filtered_cluster_paths = filter_deduplicated_paths(path_dict_cluster_dedup, filter_set, strand_sensitive=False)
            consensus_path = compute_majority_path(filtered_cluster_paths)
            consensus_path = remap_consensus_to_global_contexts(
                consensus_path, filtered_cluster_paths, path_dict_cluster_dedup, path_dict
            )

            consensus_paths_core[cl] = consensus_path
            all_root_states[cl] = root_states
            all_root_states_unique[cl] = root_states_unique

    # visualization results
    consensus_paths_plotting, assignment_df_plotting, _ = consensus_paths_and_assignments(consensus_paths_core, cluster_map_core)

    if plot_consensus:
        fig = plot_junction_pangraph_interactive(
            pangraph,
            show_consensus=True,
            consensus_paths=consensus_paths_plotting,
            assignments=assignment_df_plotting,
            order="tree",
            cluster_map=cluster_map_core,
            title = "Junction Block Structure with Core Block Tree Clustering"
        )
        display(fig)

    if plot_annotations:
        fig = plot_junction_pangraph_interactive(
            pangraph,
            show_consensus=True,
            consensus_paths=consensus_paths_plotting,
            assignments=assignment_df_plotting,
            order="tree",
            cluster_map=cluster_map_core,
            add_cluster_annotation=False,
            title = "Junction Block Structure with Core Block Tree Clustering",
            show_mges_annotations=True,
            show_int_rec_annotations=True,
            show_cds_annotations=False,
            mges_gff_path=str(results_dir / "junction_mges" / f"{junction_name}.gff3"),
            annotations_gff_path=str(results_dir / "junction_annotations" / f"{junction_name}.gff"),
            annotation_alpha=0.7,
            cds_annotation_alpha=0.3,
        )
        display(fig)

    if plot_pair_dist:
        core_distances = compute_pairwise_distances(tree_path_core)
        plot_pairwise_distance_hist(core_distances, bins=100, vline=clustering_bl_thresh, vline_kwargs={"color": "black", "linestyle": "--"}, title="Core Blocks Pairwise Distance Distribution")

    if plot_snp_dist:
        example_isolate, example_path = next(iter(path_dict.items()))
        left_core_block_id = example_path.nodes[0].id
        left_core_block_length = get_block_length(str(results_dir / "block_alignments" / junction_name / f"block_{left_core_block_id}_aln.fa"))

        snp_pos = snp_positions(str(out_dir / "core_blocks_aln.fa"))
        plot_snp_pos_distribution(snp_pos, left_core_block_length, bins=70, title="SNP Position Distribution in Core Block Alignment")

    if consensus_criterium == 'core_genome_tree':
        ambiguity_stats_df = pd.DataFrame(ambiguity_stats_rows, columns=["junction_name", "cluster_id", "n_isolates_in_cluster", "n_total_blocks", "n_ambiguous", "n_single_isolate", "n_decided_loss", "n_decided_gain", "avg_pairwise_dists"])
        dedup_stats_df = pd.DataFrame(all_dedup_stats, columns=["junction_name", "dedup_step", "isolate", "n_blocks", "n_duplicated_blocks", "n_ambiguous_blocks"])
        return cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, all_root_states, all_root_states_unique, ambiguity_stats_df, dedup_stats_df

    dedup_stats_df = pd.DataFrame(all_dedup_stats, columns=["junction_name", "dedup_step", "isolate", "n_blocks", "n_duplicated_blocks", "n_ambiguous_blocks"])
    return cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, dedup_stats_df


def find_consensus_paths_refined(junction_name, clustering_bl_thresh=0.001, tree_path=None,
                                  rare_context_thresh=0.2, z_score_threshold=3.5,
                                  seq_dedup_length_threshold=0.002,
                                  plot_consensus=False, plot_pair_dist=False,
                                  plot_snp_dist=False, plot_ambiguities=False):
    """
    Like find_consensus_paths_core (core_genome_tree criterion only), but with two
    refinements before clustering:
      1. Duplicated blocks are pre-split by sequence cluster via
         deduplicate_blocks_by_sequence.
      2. Isolate filtering per cluster uses modified z-score outlier detection on
         anchor block counts (filter_isolates_by_anchor_count) rather than a fixed
         anchor fraction threshold.

    Parameters
    ----------
    junction_name : str
    clustering_bl_thresh : float
    tree_path : str or None
    rare_context_thresh : float
        Rarity threshold for anchor block computation and deduplication.
    z_score_threshold : float
        Modified z-score threshold for low-anchor outlier removal per cluster.
    seq_dedup_length_threshold : float
        Branch-length threshold for sequence-based block deduplication.
    plot_consensus, plot_pair_dist, plot_snp_dist, plot_ambiguities : bool

    Returns
    -------
    cluster_map_core, consensus_paths_core, path_dict,
    consensus_paths_plotting, assignment_df_plotting,
    all_root_states, all_root_states_unique, ambiguity_stats_df, dedup_stats_df
    """
    root = repo_root()
    results_dir = root / "results"
    if tree_path is None:
        tree_path = str(root / "config" / "polished_tree.nwk")

    pangraph_path = results_dir / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    path_dict_raw = pangraph.to_path_dictionary()
    path_dict_raw = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict_raw.items()}
    for isolate, path in path_dict_raw.items():
        for idx, node in enumerate(path.nodes):
            node.nid = pangraph.paths[isolate].nodes[idx]

    cluster_map_core, cluster_to_isos, tree_path_core, out_dir = cluster_isolates_by_core_blocks(pangraph, path_dict_raw, junction_name, clustering_bl_thresh)

    path_dict_raw = deduplicate_blocks_by_sequence(pangraph, path_dict_raw, junction_name, length_threshold=seq_dedup_length_threshold)

    path_dict, block_freq, global_dedup_stats = make_deduplicated_paths(pangraph, path_dict_raw)
    for row in global_dedup_stats:
        row["junction_name"] = junction_name
        row["dedup_step"] = "global"
    all_dedup_stats = list(global_dedup_stats)

    tree = Phylo.read(tree_path, "newick")
    tree.rooted = True

    consensus_paths_core = {}
    all_root_states = {}
    all_root_states_unique = {}
    ambiguity_stats_rows = []

    for cl, isolate_list in cluster_to_isos.items():
        path_dict_cluster_raw = {iso: path_dict_raw[iso] for iso in isolate_list}
        isolate_list = filter_isolates_by_anchor_count(path_dict_cluster_raw, rare_context_thresh=rare_context_thresh, z_score_threshold=z_score_threshold)

        if not isolate_list:
            raise ValueError(f"Cluster {cl}: all isolates filtered out by anchor count. Lower z_score_threshold to be less aggressive.")

        path_dict_cluster = {iso: path_dict_raw[iso] for iso in isolate_list}
        path_dict_cluster_dedup, freq_cluster, cluster_dedup_stats = rededuplicate_cluster_paths(path_dict_cluster, rare_context_thresh=rare_context_thresh)
        for row in cluster_dedup_stats:
            row["junction_name"] = junction_name
            row["dedup_step"] = f"cluster_{cl}"
        all_dedup_stats.extend(cluster_dedup_stats)

        blocks, block2idx = build_block_index(path_dict_cluster_dedup, strand_sensitive=False)
        binary_encodings = encode_paths_binary(path_dict_cluster_dedup, block2idx, strand_sensitive=False)

        subtree = build_subtree(tree, isolate_list)

        fitch_ancestral_reconstruction(subtree, binary_encodings)
        root_states = subtree.root._state_sets
        root_states_unique, amb_stats = decide_ambiguities(root_states, isolate_list, junction_name, cl, blocks, block2idx, path_dict_cluster_dedup=path_dict_cluster_dedup, individual_gain_thresh=0.001, verbose=plot_ambiguities)
        ambiguity_stats_rows.append(dict(junction_name=junction_name, cluster_id=cl, n_isolates_in_cluster=len(isolate_list), **amb_stats))

        filter_set = {block for block, state in zip(blocks, root_states_unique) if 0 in state}
        filtered_cluster_paths = filter_deduplicated_paths(path_dict_cluster_dedup, filter_set, strand_sensitive=False)
        consensus_path = compute_majority_path(filtered_cluster_paths)
        consensus_path = remap_consensus_to_global_contexts(
            consensus_path, filtered_cluster_paths, path_dict_cluster_dedup, path_dict
        )

        consensus_paths_core[cl] = consensus_path
        all_root_states[cl] = root_states
        all_root_states_unique[cl] = root_states_unique

    consensus_paths_plotting, assignment_df_plotting, _ = consensus_paths_and_assignments(consensus_paths_core, cluster_map_core)

    if plot_consensus:
        fig = plot_junction_pangraph_interactive(
            pangraph,
            show_consensus=True,
            consensus_paths=consensus_paths_plotting,
            assignments=assignment_df_plotting,
            order="tree",
            cluster_map=cluster_map_core,
            title="Junction Block Structure with Core Block Tree Clustering"
        )
        display(fig)

    if plot_pair_dist:
        core_distances = compute_pairwise_distances(tree_path_core)
        plot_pairwise_distance_hist(core_distances, bins=100, vline=clustering_bl_thresh, vline_kwargs={"color": "black", "linestyle": "--"}, title="Core Blocks Pairwise Distance Distribution")

    if plot_snp_dist:
        example_isolate, example_path = next(iter(path_dict.items()))
        left_core_block_id = example_path.nodes[0].id
        left_core_block_length = get_block_length(str(results_dir / "block_alignments" / junction_name / f"block_{left_core_block_id}_aln.fa"))
        snp_pos = snp_positions(str(out_dir / "core_blocks_aln.fa"))
        plot_snp_pos_distribution(snp_pos, left_core_block_length, bins=70, title="SNP Position Distribution in Core Block Alignment")

    ambiguity_stats_df = pd.DataFrame(ambiguity_stats_rows, columns=["junction_name", "cluster_id", "n_isolates_in_cluster", "n_total_blocks", "n_ambiguous", "n_single_isolate", "n_decided_loss", "n_decided_gain", "avg_pairwise_dists"])
    dedup_stats_df = pd.DataFrame(all_dedup_stats, columns=["junction_name", "dedup_step", "isolate", "n_blocks", "n_duplicated_blocks", "n_ambiguous_blocks"])
    return cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, all_root_states, all_root_states_unique, ambiguity_stats_df, dedup_stats_df


def find_consensus_paths_edge(junction_name, clustering_bl_thresh=0.001, tree_path=None, individual_gain_thresh=0.001):
    """
    Find consensus paths using edge-based Fitch ancestral reconstruction.

    For each cluster of isolates, edges are binary-encoded per isolate and
    ancestral edge presence/absence is reconstructed on the cluster subtree.
    Nodes are kept if at least one surrounding edge is ancestrally present,
    removed if both are absent, and resolved by sequence similarity if ambiguous.
    The majority path among the filtered isolate paths is taken as consensus.

    Parameters
    ----------
    junction_name : str
    clustering_bl_thresh : float
        Branch-length threshold for cutting the core-block tree into clusters.
    tree_path : str or None
        Path to the species tree in Newick format. Defaults to config/polished_tree.nwk.
    individual_gain_thresh : float
        Avg pairwise distance threshold for ambiguity resolution;
        below → loss (keep), above → gain (remove). Default 0.001.

    Returns
    -------
    cluster_map : dict[isolate -> cluster_id]
    consensus_paths : dict[cluster_id -> Path]
    path_dict : dict[isolate -> Path(Node,...)]
        Plain paths with nids set (for downstream insertion/deletion detection).
    consensus_paths_plotting : list[Path]
    assignment_df_plotting : pd.DataFrame
    """
    root = repo_root()
    results_dir = root / "results"
    if tree_path is None:
        tree_path = str(root / "config" / "polished_tree.nwk")

    pangraph_path = results_dir / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    path_dict = pangraph.to_path_dictionary()
    path_dict = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict.items()}
    for isolate, path in path_dict.items():
        for idx, node in enumerate(path.nodes):
            node.nid = pangraph.paths[isolate].nodes[idx]

    cluster_map, cluster_to_isos, _, _ = cluster_isolates_by_core_blocks(pangraph, path_dict, junction_name, clustering_bl_thresh)

    edge_path_dict = {isolate: pu.EdgePath.from_path(path) for isolate, path in path_dict.items()}

    tree = Phylo.read(tree_path, "newick")
    tree.rooted = True

    consensus_paths = {}

    for cl, isolate_list in cluster_to_isos.items():
        edge_path_dict_cluster = {iso: edge_path_dict[iso] for iso in isolate_list if iso in edge_path_dict}
        path_dict_cluster = {iso: path_dict[iso] for iso in isolate_list if iso in path_dict}

        edges, edge2idx = build_edge_index(edge_path_dict_cluster)
        binary_encodings = encode_edge_paths_binary(edge_path_dict_cluster, edge2idx)

        subtree = build_subtree(tree, isolate_list)
        fitch_ancestral_reconstruction(subtree, binary_encodings)
        root_states = subtree.root._state_sets

        filtered_paths = filter_paths_by_edge_presence(
            path_dict_cluster,
            edge_path_dict_cluster,
            edges,
            root_states,
            edge2idx,
            junction_name,
            cl,
            individual_gain_thresh=individual_gain_thresh,
        )

        consensus_path = compute_majority_path(filtered_paths)
        consensus_paths[cl] = consensus_path

    consensus_paths_plotting, assignment_df_plotting, _ = consensus_paths_and_assignments(consensus_paths, cluster_map)

    return cluster_map, consensus_paths, path_dict, consensus_paths_plotting, assignment_df_plotting


def save_consensus_cache(path, cluster_map_core, consensus_paths_plotting, assignment_df_plotting):
    """
    Save the three objects needed by the dash app to a compact JSON file.
    """
    import json
    data = {
        "cluster_map_core": cluster_map_core,
        "consensus_paths_plotting": [
            [n.to_str_id() for n in p.nodes]
            for p in consensus_paths_plotting
        ],
        "assignment_df_plotting": assignment_df_plotting.to_dict(orient="split"),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_consensus_cache(path):
    """
    Load cluster_map_core, consensus_paths_plotting, assignment_df_plotting from a JSON cache.
    """
    import json
    import pandas as pd
    with open(path) as f:
        data = json.load(f)

    cluster_map_core = data["cluster_map_core"]

    consensus_paths_plotting = [
        pu.Path([pu.DeduplicatedNode.from_str_id(s) for s in nodes])
        for nodes in data["consensus_paths_plotting"]
    ]

    assignment_df_plotting = pd.DataFrame(
        data["assignment_df_plotting"]["data"],
        index=data["assignment_df_plotting"]["index"],
        columns=data["assignment_df_plotting"]["columns"],
    )

    return cluster_map_core, consensus_paths_plotting, assignment_df_plotting


def cluster_map_to_dataframe(cluster_map: dict, save_path: str = None) -> pd.DataFrame:
    """
    Convert a cluster_map dict to a wide-form DataFrame where each row is one
    junction, the second column is the number of clusters, and the remaining
    columns are isolate names containing their cluster id (NaN if absent).

    Parameters
    ----------
    cluster_map : dict
        junction_name -> {isolate_name: cluster_id}
    save_path : str or None
        If provided, save the DataFrame as a CSV to this path.

    Returns
    -------
    pd.DataFrame with columns: junction_name, n_clusters, <isolate_1>, <isolate_2>, ...
    """
    rows = {}
    for junction, iso_map in cluster_map.items():
        row = {
            "junction_name": junction,
            "n_clusters": len(set(iso_map.values())),
            "n_isolates": len(iso_map),
        }
        row.update(iso_map)
        rows[junction] = row

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)

    # Cast isolate columns to nullable int (preserves NaN for absent isolates as <NA>)
    meta_cols = {"junction_name", "n_clusters", "n_isolates"}
    iso_cols = [c for c in df.columns if c not in meta_cols]
    for col in iso_cols:
        df[col] = df[col].astype("Int64")

    # Ensure junction_name, n_clusters, n_isolates are the first three columns
    df = df[["junction_name", "n_clusters", "n_isolates"] + iso_cols]

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved cluster map to {save_path}")

    return df


def collect_all_branch_lengths(results_dir, save_path=None):
    """
    Collect all individual branch lengths from the core genome tree of each junction.

    Parameters
    ----------
    results_dir : str or Path
        Base results directory; trees are expected at
        ``results_dir/consensus_analysis/<junction_name>/core_blocks_aln.newick``.
    save_path : str or Path or None
        If provided, save the resulting DataFrame as a CSV to this path.

    Returns
    -------
    pd.DataFrame with columns: junction_name, branch_length
    """
    results_dir = Path(results_dir)
    rows = []
    skipped = []

    for tree_path in sorted((results_dir / "consensus_analysis").glob("*/core_blocks_aln.newick")):
        jname = tree_path.parent.name
        try:
            tree = Phylo.read(str(tree_path), "newick")
            for clade in tree.find_clades():
                if clade.branch_length is not None:
                    rows.append({"junction_name": jname, "branch_length": clade.branch_length})
        except Exception as e:
            print(f"  {jname}: {e}")
            skipped.append(jname)

    if skipped:
        print(f"Skipped {len(skipped)} junctions (file not found or error).")

    df = pd.DataFrame(rows, columns=["junction_name", "branch_length"])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved branch lengths to {save_path}.")

    return df


def collect_all_block_branch_lengths(results_dir, save_path=None):
    """
    Collect branch lengths from all existing per-block FastTree trees.

    Globs for ``block_*_aln.tree`` files under
    ``results_dir/block_alignments/<junction_name>/`` and joins with the
    alignment stats CSV to attach ``core`` and ``duplicated`` flags.

    Returns
    -------
    pd.DataFrame with columns:
        junction_name, block_id, branch_length, core, duplicated
    """
    results_dir = Path(results_dir)
    rows = []

    for tree_path in sorted((results_dir / "block_alignments").glob("*/block_*_aln.tree")):
        if tree_path.stat().st_size == 0:
            continue
        jname = tree_path.parent.name
        block_id = int(tree_path.name.removeprefix("block_").removesuffix("_aln.tree"))
        tree = Phylo.read(str(tree_path), "newick")
        for clade in tree.find_clades():
            if clade.branch_length is not None:
                rows.append({"junction_name": jname, "block_id": block_id, "branch_length": clade.branch_length})

    df = pd.DataFrame(rows)

    # attach core/duplicated flags from stats CSVs
    stats_dfs = []
    for stats_csv in (results_dir / "block_alignments").glob("*/*_alignment_stats.csv"):
        stats_dfs.append(pd.read_csv(stats_csv)[["block_id", "core", "duplicated"]])
    if stats_dfs:
        stats = pd.concat(stats_dfs).drop_duplicates("block_id")
        df = df.merge(stats, on="block_id", how="left")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def collect_all_pairwise_distances(junction_names, results_dir,
                                    save_path=None, load_path=None):
    """
    Collect pairwise distances from the core genome alignment of each junction.

    Parameters
    ----------
    junction_names : list of str
        Junction names to process.
    results_dir : str or Path
        Base results directory; files are expected at
        ``results_dir/consensus_analysis/<junction_name>/core_blocks_aln.fa``.
    save_path : str or Path or None
        If provided, save the resulting DataFrame as a CSV to this path.
    load_path : str or Path or None
        If provided, load distances from this CSV instead of recomputing.

    Returns
    -------
    pd.DataFrame with columns: junction_name, distance
    """

    if load_path is not None:
        df = pd.read_csv(load_path)
        print(f"Loaded pairwise distances from {load_path} ({len(df)} rows).")
        return df

    results_dir = Path(results_dir)
    all_jnames = []
    all_dists = []
    all_iso1 = []
    all_iso2 = []
    skipped = []

    for jname in junction_names:
        target_path = results_dir / "consensus_analysis" / jname / "core_blocks_aln.fa"

        if not target_path.exists():
            skipped.append(jname)
            continue

        try:
            records = list(SeqIO.parse(str(target_path), "fasta"))
            names = [r.id for r in records]
            arr = np.array([list(str(r.seq).upper()) for r in records])
            n = len(names)
            iu, ju = np.triu_indices(n, k=1)
            _, dists = avg_pairwise_distance(arr)
            # iu/ju from avg_pairwise_distance may be masked; recompute to stay in sync
            valid = np.isin(arr, list("ACGT"))
            pair_valid = (valid[:, None, :] & valid[None, :, :]).sum(axis=2)
            mask = pair_valid[iu, ju] > 0
            n1 = [names[i] for i, m in zip(iu, mask) if m]
            n2 = [names[j] for j, m in zip(ju, mask) if m]

            all_jnames.append(np.full(len(dists), jname))
            all_dists.append(dists)
            all_iso1.extend(n1)
            all_iso2.extend(n2)
        except Exception as e:
            print(f"  {jname}: {e}")
            skipped.append(jname)
        print(f"Junction {jname} successfully processed.")

    if skipped:
        print(f"Skipped {len(skipped)} junctions (file not found or error).")

    df = pd.DataFrame({
        "junction_name": np.concatenate(all_jnames) if all_jnames else [],
        "isolate_1": all_iso1,
        "isolate_2": all_iso2,
        "distance": np.concatenate(all_dists) if all_dists else [],
    })

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved pairwise distances to {save_path}.")

    return df


def fitch_parsimony_steps_per_cluster(cluster_df, tree_path, junction_col="junction_name", save_path=None, plot_tree=False, plot_dir=None):
    """
    Count certain parsimonious gains per cluster for each junction using
    bottom-up multi-state Fitch only.

    A gain of cluster c is counted on edge parent -> child if:
        c not in Fitch_set(parent) and c in Fitch_set(child)

    This keeps ambiguity unresolved and only counts gains that are supported
    without forcing a specific ancestral reconstruction.
    """
    full_tree = Phylo.read(tree_path, "newick")
    full_tree_tips = {tip.name for tip in full_tree.get_terminals()}

    meta_cols = {junction_col, "n_clusters", "n_isolates"}
    isolate_cols = [c for c in cluster_df.columns if c not in meta_cols]
    isolate_cols = [c for c in isolate_cols if c in full_tree_tips]

    rows = []

    for _, row in cluster_df.iterrows():
        junction_name = row[junction_col]
        n_isolates = int(row["n_isolates"])
        n_clusters = int(row["n_clusters"])

        iso_to_cluster = {
            iso: int(row[iso])
            for iso in isolate_cols
            if pd.notna(row[iso])
        }

        present_isolates = set(iso_to_cluster.keys())
        if len(present_isolates) == 0 or n_clusters == 0:
            continue

        observed_clusters = sorted(set(iso_to_cluster.values()))

        tree = copy.deepcopy(full_tree)
        for tip in list(tree.get_terminals()):
            if tip.name not in present_isolates:
                tree.prune(tip)

        parent_map = {}
        for clade in tree.find_clades(order="preorder"):
            for child in clade.clades:
                parent_map[child] = clade

        # Bottom-up Fitch sets
        state_sets = {}
        for clade in tree.find_clades(order="postorder"):
            if clade.is_terminal():
                state_sets[clade] = {iso_to_cluster[clade.name]}
            else:
                child_sets = [state_sets[child] for child in clade.clades]
                inter = set.intersection(*child_sets)
                state_sets[clade] = inter if inter else set.union(*child_sets)

        # Optionally plot the tree colored by Fitch cluster assignment
        if plot_tree and (plot_tree is True or junction_name in plot_tree):
            tree_save = None
            if plot_dir is not None:
                Path(plot_dir).mkdir(parents=True, exist_ok=True)
                tree_save = str(Path(plot_dir) / f"{junction_name}_fitch_tree.png")
            plot_fitch_tree(tree, iso_to_cluster, state_sets, junction_name, save_path=tree_save)

        # Count certain gains only
        cluster_steps = {cl: 0 for cl in observed_clusters}

        for clade in tree.find_clades(order="level"):
            if clade is tree.root:
                continue

            parent = parent_map[clade]
            parent_state = state_sets[parent]
            child_state = state_sets[clade]

            for cl in child_state:
                if cl not in parent_state:
                    cluster_steps[cl] += 1

        cluster_sizes = {}
        for cl in observed_clusters:
            cluster_sizes[cl] = sum(1 for c in iso_to_cluster.values() if c == cl)

        for cluster_id in observed_clusters:
            rows.append({
                "junction_name": junction_name,
                "n_isolates": n_isolates,
                "n_clusters": n_clusters,
                "cluster_id": cluster_id,
                "n_isolates_in_cluster": cluster_sizes[cluster_id],
                "n_parsimony_steps": cluster_steps[cluster_id],
            })

    result = pd.DataFrame(
        rows,
        columns=["junction_name", "n_isolates", "n_clusters", "cluster_id", "n_isolates_in_cluster", "n_parsimony_steps"]
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        print(f"Saved parsimony steps to {save_path}")

    return result

