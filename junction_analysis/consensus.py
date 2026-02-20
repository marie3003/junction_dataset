import pandas as pd
import numpy as np
import math
import re

from pathlib import Path
from IPython.display import display

from itertools import combinations
from collections import Counter, defaultdict, OrderedDict

from Bio import Phylo
import pypangraph as pp
import junction_analysis.pangraph_utils as pu
from junction_analysis.plotting import plot_junction_pangraph_interactive, plot_pairwise_distance_hist, plot_snp_pos_distribution, plot_block_distance_distribution
from junction_analysis.junction_trees import build_tree_from_block_list, cluster_tree_by_branch_length, compute_pairwise_distances
from junction_analysis.helpers import get_block_length, snp_positions, build_subtree, repo_root
from junction_analysis.block_alignment import create_block_msas_for_cluster, summarize_block_msas


def find_invertible_ids(paths: dict) -> set:
    """
    Returns a set of block ids that are inverted in any path. This is needed for context definition of inverted blocks.
    We need to make sure, that the context block is never invertible, such that when comparing invertible edges, their context is the same.
    """

    invertible_ids = set()

    for path in paths.values():
        for node in path:
            if node.strand == False:
                invertible_ids.add(node.id)

    return invertible_ids

def make_deduplicated_paths(pangraph, rare_context_thresh=0.1) -> dict:
    """
    Convert a dict[isolate -> Path(Node,...)] into dict[isolate -> Path(DeduplicatedNode, ...)],
    where duplicated blocks get a context = closest non-duplicated, never inverted, less rare than rare_context_thresh block (ID) to the left.
    For non-duplicated blocks, context is set to "".
    Deduplication is not perfect if there are no suitable context anchors that are seperating to differing block copies because the blocks in between tham are inverted, duplicated or rare.
    @param rare_context_threshold: is a threshold how rare blocks can be to be allowed as context anchors, right now it is hard coded to 10
    Additionally also return deduplicated block count dictionary.
    """
    # create path dictionary of Path objects
    path_dict = pangraph.to_path_dictionary()
    path_dict = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict.items()}

    blockstats_df = pangraph.to_blockstats_df()
    duplicated_ids = set(blockstats_df.loc[blockstats_df['duplicated'] == True].index)
    rare_context_thresh = np.floor(rare_context_thresh * len(path_dict))
    rare_ids = set(blockstats_df.loc[blockstats_df['count'] < rare_context_thresh].index)
    invertible_ids = find_invertible_ids(path_dict)

    deduplicated_paths: dict = {}
    freq = Counter()

    for isolate, path in path_dict.items():
        last_non_dup: str | None = None # context anchor = last suitable non-dup block id
        context_counts: dict[str, int] = {}  # per-context counts for duplicated node ids

        dedup_nodes = []
        for idx, n in enumerate(path):  # path is iterable over its nodes
            block_nid = pangraph.paths[isolate].nodes[idx]
            if n.id in duplicated_ids:
                if last_non_dup is None:
                    raise ValueError("last_non_dup must not be None when encountering a duplicated node")
                
                # per-node count within the current context
                count = context_counts.get(n.id, 0) + 1 # default is 0 if not inside dict yet
                context_counts[n.id] = count

                context = f"{last_non_dup}_{count}"
                dn = pu.DeduplicatedNode(n.id, n.strand, context, block_nid)
                
            else:
                dn = pu.DeduplicatedNode(n.id, n.strand, "", block_nid)
                if n.id not in invertible_ids and n.id not in rare_ids:
                    last_non_dup = n.id
                    context_counts.clear()

            dedup_nodes.append(dn)
            freq[dn] += 1

        deduplicated_paths[isolate] = pu.Path(dedup_nodes)

    return deduplicated_paths, dict(freq)

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

def filter_deduplicated_paths(paths, filter_set):
    """Removes blocks (id, strand, context) that are inside a given set to be filtered.
    Return filtered paths list."""
    
    filtered_paths = {}
    
    for iso, path in paths.items():
        filtered_path = pu.Path([dnode for dnode in path.nodes if dnode not in filter_set])
        filtered_paths[iso] = filtered_path
    
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
    # deduplicate
    deduplicated_paths, deduplicated_blog_freq = make_deduplicated_paths(pangraph)

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

def build_block_index(path_dict):
    """
    Returns:
      blocks: list of blocks in fixed order (index -> block)
      block2idx: dict mapping block -> index
    """
    block2idx = OrderedDict()  # preserves insertion order deterministically
    for isolate, path in path_dict.items():
        for block in path.nodes:          # block supports == (and should be hashable)
            if block not in block2idx:
                block2idx[block] = len(block2idx)
    blocks = list(block2idx.keys())
    return blocks, dict(block2idx)


def encode_paths_binary(path_dict, block2idx):
    """
    Returns:
      enc: dict isolate -> list[int] (0/1) with length = number of unique blocks
    """
    n_blocks = len(block2idx)
    enc = {}

    for isolate, path in path_dict.items():
        vec = [0] * n_blocks
        for block in path.nodes:
            vec[block2idx[block]] = 1
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

def decide_ambiguities(root_states, isolate_list, junction_name, cl, blocks, block2idx, individual_gain_thresh = 0.01, verbose = False):
    ambiguous_blocks = [block for block, state in zip(blocks, root_states) if len(state) > 1]

    # don't run followig analysis if there are no ambiguous blocks
    if ambiguous_blocks == []:
        print("No ambiguous blocks.")
        return root_states
    
    create_block_msas_for_cluster(junction_name, isolate_list, cl, ambiguous_blocks)
    df, pair_dists = summarize_block_msas(junction_name, cl, return_pairwise_dists=True)

    # if all blocks only appear once
    if df.empty:
        for block in ambiguous_blocks:
            pos = block2idx[block]
            root_states[pos] = {0}
        return root_states

    if verbose:
        display(df)
        plot_block_distance_distribution(pair_dists, [block.id for block in ambiguous_blocks], bins=70, cols=4, figsize=(14, 10), vline=0.01, vline_kwargs={"color": "black", "linestyle": "--"})

    for block in ambiguous_blocks:
        pos = block2idx[block]

        row = df.loc[df["block_id"] == block.id, "avg_pairwise_dist"]

        # block not found in df, probably because it only exists once → default to gain (ancestral state 0)
        if row.empty:
            root_states[pos] = {0}
            continue

        # loss: low diversity → block present in ancestor
        if row.iloc[0] < individual_gain_thresh:
            root_states[pos] = {1}
        # gain: high diversity → block absent in ancestor
        else:
            root_states[pos] = {0}

    return root_states

def find_consensus_paths_core(junction_name, clustering_bl_thresh = 0.005, consensus_criterium = 'core_genome_tree', tree_path = None, block_freq_thresh = 0.5, plot_consensus = False, plot_annotations = False, plot_pair_dist = False, plot_snp_dist = False, plot_ambiguities = False):
    # create deduplicated paths dict
    root = repo_root()
    results_dir = root / "results"
    if tree_path is None:
        tree_path = str(root / "config" / "polished_tree.nwk")

    pangraph_path = results_dir / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))
    path_dict, block_freq = make_deduplicated_paths(pangraph)

    # create tree and do clustering based on core blocks
    blockstats_df = pangraph.to_blockstats_df()
    core_block_ids = set(blockstats_df.loc[blockstats_df["core"] == True].index)
    _, any_path = next(iter(path_dict.items()))
    # makes sure that core blocks are also in the order of their appearance within a path
    core_block_nodes = [node for node in any_path.nodes if node.id in core_block_ids]
    
    out_dir = results_dir / "consensus_analysis" / junction_name
    build_tree_from_block_list(pangraph, path_dict, core_block_nodes, list(path_dict.keys()), str(out_dir), "core")
    tree_path_core = out_dir / "core_blocks_aln.newick"
    cluster_map_core = cluster_tree_by_branch_length(tree_path_core, clustering_bl_thresh)

    # Group isolates by cluster
    cluster_to_isos = defaultdict(list)
    for iso, cl in cluster_map_core.items():
        if iso in path_dict:  # ignore isolates not in paths
            cluster_to_isos[cl].append(iso)

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

        for cl, isolate_list in cluster_to_isos.items():

            path_dict_cluster = {iso: path_dict[iso] for iso in isolate_list}

            blocks, block2idx = build_block_index(path_dict_cluster)
            binary_encodings = encode_paths_binary(path_dict_cluster, block2idx)

            # build subtree for isolates of one cluster
            subtree = build_subtree(tree, isolate_list)
            #Phylo.draw_ascii(subtree)

            # do ancestral sequence reconstruction on subtree and read ancestral sequence from root
            fitch_ancestral_reconstruction(subtree, binary_encodings)
            root_states = subtree.root._state_sets  # list of {0}/{1}
            print(root_states)
            root_states_unique = decide_ambiguities(root_states, isolate_list, junction_name, cl, blocks, block2idx, individual_gain_thresh=0.01, verbose = plot_ambiguities)
            
            filter_set = {block for block, state in zip(blocks, root_states_unique) if 0 in state}
            filtered_cluster_paths = filter_deduplicated_paths(path_dict_cluster, filter_set)
            consensus_path = compute_majority_path(filtered_cluster_paths)

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
        return cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, all_root_states, all_root_states_unique

    return cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting


def renumber_context_numbering(consensus_paths, path_dict, assignment_df):
    """
    Renumber context suffixes (_N) in consensus paths and isolate paths consistently.

    Step 1 — Consensus: For each consensus path, renumber duplicated blocks
    (context != "") left-to-right as _1, _2, _3, ... per (block_id, ctx_base) group.

    Step 2 — Isolates: For each isolate, match its duplicated blocks to the
    corresponding consensus blocks using the closest non-duplicated anchor to the
    left. Anchors are non-duplicated blocks (context == "") shared by both the
    consensus and the isolate (same block id and strand).

    Matching rules per (consensus duplicated block, isolate):
      - Exactly one isolate candidate with same block id, strand, ctx_base and
        same left anchor → assign the same number as the consensus block.
      - No candidate → block absent in isolate, leave untouched.
      - Multiple candidates → warn and pick the first (leftmost).

    Unmatched isolate duplicated blocks are renumbered starting after the
    highest consensus number for that (block_id, ctx_base) group (or from 1 if
    the group does not appear in the consensus), in left-to-right order.

    Parameters
    ----------
    consensus_paths : list of pu.Path
        Typically consensus_paths_plotting from find_consensus_paths_core.
    path_dict : dict {isolate_name: pu.Path}
        Deduplicated paths for all isolates.
    assignment_df : pd.DataFrame
        Index: isolate name. Must have column 'best_consensus' with values
        like 'consensus_1', 'consensus_2', etc.

    Returns
    -------
    new_consensus_paths : list of pu.Path
    new_path_dict : dict {isolate_name: pu.Path}
    """

    def _ctx_base(ctx):
        """Return (ctx_base, number) if context has _N suffix, else None."""
        m = re.match(r'^(.+)_(\d+)$', ctx)
        if m:
            return m.group(1), int(m.group(2))
        return None

    def _find_left_anchor(nodes, pos, anchor_set):
        """Return block_id of the closest anchor to the left of pos."""
        for i in range(pos - 1, -1, -1):
            n = nodes[i]
            if n.id in anchor_set:
                return n.id
        return None

    # ------------------------------------------------------------------ #
    # Step 1: renumber consensus paths sequentially, record max per group #
    # ------------------------------------------------------------------ #
    consensus_label_to_idx = {f"consensus_{i + 1}": i for i in range(len(consensus_paths))}

    new_consensus_paths = []
    consensus_ctx_max = []  # list of dicts: {(block_id, ctx_base): max_number}

    for path in consensus_paths:
        counter = defaultdict(int)
        new_nodes = []
        for node in path.nodes:
            parsed = _ctx_base(node.context)
            if parsed:
                ctx_base, _ = parsed
                key = (node.id, ctx_base)
                counter[key] += 1
                new_nodes.append(pu.DeduplicatedNode(node.id, node.strand, f"{ctx_base}_{counter[key]}", node.nid))
            else:
                new_nodes.append(node)
        new_consensus_paths.append(pu.Path(new_nodes))
        consensus_ctx_max.append(dict(counter))

    # ------------------------------------------------------------------ #
    # Step 2: renumber isolate paths to match consensus numbering         #
    # ------------------------------------------------------------------ #
    new_path_dict = {}

    for iso, row in assignment_df.iterrows():
        consensus_label = row.get('best_consensus')
        if consensus_label not in consensus_label_to_idx:
            new_path_dict[iso] = path_dict[iso]
            continue

        c_idx = consensus_label_to_idx[consensus_label]
        c_path = new_consensus_paths[c_idx]
        ctx_max = consensus_ctx_max[c_idx]  # {(block_id, ctx_base): max consensus number}

        if iso not in path_dict:
            continue
        iso_path = path_dict[iso]
        c_nodes = c_path.nodes
        iso_nodes = iso_path.nodes

        # Anchors: blocks that appear exactly once (with any strand) in BOTH the
        # consensus and the isolate path. Recomputed fresh per isolate.
        # Counting by id only (strand-agnostic) so that blocks appearing once
        # but on different strands still qualify as anchors.
        c_counts   = Counter(n.id for n in c_nodes)
        iso_counts = Counter(n.id for n in iso_nodes)
        anchor_set = {bid for bid in c_counts if c_counts[bid] == 1 and iso_counts.get(bid, 0) == 1}

        # Collect info on every duplicated node in the isolate
        # iso_dup[pos] = (block_id, strand, ctx_base)
        iso_dup = {}
        for pos, node in enumerate(iso_nodes):
            parsed = _ctx_base(node.context)
            if parsed:
                ctx_base, _ = parsed
                iso_dup[pos] = (node.id, node.strand, ctx_base)

        iso_left_anchors = {pos: _find_left_anchor(iso_nodes, pos, anchor_set)
                            for pos in iso_dup}

        # Match each duplicated consensus block to isolate candidates
        new_ctx_for_pos = {}   # {iso_pos: new_context_string}
        assigned_positions = set()

        for c_pos, c_node in enumerate(c_nodes):
            parsed = _ctx_base(c_node.context)
            if not parsed:
                continue
            ctx_base, c_number = parsed
            c_anchor = _find_left_anchor(c_nodes, c_pos, anchor_set)

            print
            candidates = [
                pos for pos, (bid, strand, cb) in iso_dup.items()
                if bid == c_node.id
                and strand == c_node.strand
                and cb == ctx_base
                and iso_left_anchors[pos] == c_anchor
                and pos not in assigned_positions
            ]

            if len(candidates) == 0:
                pass  # block absent in isolate
            elif len(candidates) == 1:
                pos = candidates[0]
                new_ctx_for_pos[pos] = f"{ctx_base}_{c_number}"
                assigned_positions.add(pos)
            else:
                print(f"Warning: {len(candidates)} matches for block {c_node.id} "
                      f"(ctx_base={ctx_base}, anchor={c_anchor}) in isolate {iso}. "
                      f"Picking first (leftmost).")
                pos = candidates[0]
                new_ctx_for_pos[pos] = f"{ctx_base}_{c_number}"
                assigned_positions.add(pos)

        # Renumber unmatched isolate duplicated blocks left-to-right
        next_number = {}  # (block_id, ctx_base) -> next free number
        for pos in sorted(iso_dup):
            if pos in assigned_positions:
                continue
            bid, strand, ctx_base = iso_dup[pos]
            key = (bid, ctx_base)
            if key not in next_number:
                next_number[key] = ctx_max.get(key, 0) + 1
            new_ctx_for_pos[pos] = f"{ctx_base}_{next_number[key]}"
            next_number[key] += 1

        # Reconstruct isolate path
        new_iso_nodes = []
        for pos, node in enumerate(iso_nodes):
            if pos in new_ctx_for_pos:
                new_iso_nodes.append(pu.DeduplicatedNode(node.id, node.strand, new_ctx_for_pos[pos], node.nid))
            else:
                new_iso_nodes.append(node)
        new_path_dict[iso] = pu.Path(new_iso_nodes)

    # Isolates not in assignment_df are passed through unchanged
    for iso in path_dict:
        if iso not in new_path_dict:
            new_path_dict[iso] = path_dict[iso]

    return new_consensus_paths, new_path_dict