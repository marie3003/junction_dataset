import pandas as pd
import numpy as np

from itertools import combinations
from collections import Counter

import pypangraph as pp
import junction_analyis.pangraph_utils as pu


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

def make_deduplicated_paths(blockstats_df, paths: dict) -> dict:
    """
    Convert a dict[isolate -> Path(Node,...)] into dict[isolate -> Path(DeduplicatedNode, ...)],
    where duplicated blocks get a context = closest non-duplicated, never inverted block (ID) to the left.
    For non-duplicated blocks, context is set to "".
    Additionally also return deduplicated block count dictionary.
    """
    duplicated_ids = set(blockstats_df.loc[blockstats_df['duplicated'] == True].index)
    invertible_ids = find_invertible_ids(paths)

    deduplicated_paths: dict = {}
    freq = Counter()

    for isolate, path in paths.items():
        last_non_dup: str | None = None
        dedup_nodes = []
        for n in path:  # path is iterable over its nodes
            if n.id in duplicated_ids:
                context = last_non_dup if last_non_dup is not None else ""
                dn = pu.DeduplicatedNode(n.id, n.strand, context)
                
            else:
                dn = pu.DeduplicatedNode(n.id, n.strand, "")
                if n.id not in invertible_ids:
                    last_non_dup = n.id

            dedup_nodes.append(dn)
            freq[dn] += 1

        deduplicated_paths[isolate] = pu.Path(dedup_nodes)

    return deduplicated_paths, dict(freq)

def count_edges(dedup_paths: dict) -> dict:
    """
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
    Finds consesus paths of junction pangraph.
    Paths are first filtered by rare blocks and then deduplicated. Block filtering is repeated on deduplicated blocks also taking into account strandedness and block context.
    A third filtering step filters out paths with rare edges.
    @param rare_block_threshold: blocks that are less frequent then this threshold are filtered out of the paths.
    @param rare_edge_threshold: paths containing an edge that is less frequent than this threshold are filtered out.
    @return consensus paths: unique paths that remain after filtering
    @return path_dict: original paths written as Path and Node objects
    """

    # transform Node, Path structure from path_dict
    path_dict = pangraph.to_path_dictionary()
    blockstats_df = pangraph.to_blockstats_df()

    path_dict = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict.items()}

    # removal of rare blocks (ignore duplication and strandedness)
    rare_blocks = set(blockstats_df.loc[blockstats_df['count'] < rare_block_threshold].index)
    keep_f = lambda bid: bid not in rare_blocks

    filtered_paths = pu.filter_paths(path_dict, keep_f)

    # deduplicate
    deduplicated_paths, deduplicated_blog_freq = make_deduplicated_paths(blockstats_df, filtered_paths)

    # refilter, after deduplication some blocks might now have a frequency below the threshold (now consider duplication and inversion)
    rare_deduplicated_blocks = {dnode for dnode, cnt in deduplicated_blog_freq.items() if cnt < rare_block_threshold}
    deduplicated_paths = filter_deduplicated_paths(deduplicated_paths, rare_deduplicated_blocks)

    # instead of edge matrix make dictionary with edge as key and count as value to store edge frequency
    edge_count = count_edges(deduplicated_paths)

    # filter paths with rare edges (inversed edged are considered the same)
    consensus_paths = find_unique_frequent_paths(deduplicated_paths, edge_count, flow_threshold = rare_edge_threshold)

    # to add paths to their consensus paths, compare the deduplicated paths (after rare node deletion) to the selected consensus paths and add it to the one which is most similar
    # TODO: what is a good similarity metric, since they should be quite similar to their consensus paths, edge jaccard index could be a good choice
    edge_ji_df = compute_edge_jaccard_matrix(deduplicated_paths, consensus_paths)
    assignment_df = assign_isolates_to_consensus(edge_ji_df)

    # refilter consensus paths such that each consensus path has at least n assigned isolate, e.g. n = 5
    edge_ji_df, assignment_df, consensus_paths = remove_rare_consensus_paths(consensus_paths, deduplicated_paths, edge_ji_df, assignment_df, min_n_isolates_per_consensus)

    return consensus_paths, path_dict, edge_ji_df, assignment_df