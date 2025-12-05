import pandas as pd
import numpy as np
import math

from itertools import combinations
from collections import Counter, defaultdict
import pypangraph as pp
import junction_analysis.pangraph_utils as pu


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

def make_deduplicated_paths(pangraph, rare_context_thresh=10, include_isolates=None) -> dict:
    """
    Convert a dict[isolate -> Path(Node,...)] into dict[isolate -> Path(DeduplicatedNode, ...)],
    where duplicated blocks get a context = closest non-duplicated, never inverted, less rare than rare_context_thresh block (ID) to the left.
    For non-duplicated blocks, context is set to "".
    @param rare_context_threshold: is a threshold how rare blocks can be to be allowed as context anchors, right now it is hard coded to 10
    Additionally also return deduplicated block count dictionary.
    """
    # create path dictionary of Path objects
    path_dict = pangraph.to_path_dictionary()
    # filter isolates if requested
    if include_isolates is not None:
        include_isolates = set(include_isolates)
        path_dict = {iso: path for iso, path in path_dict.items() if iso in include_isolates}
    path_dict = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict.items()}

    blockstats_df = pangraph.to_blockstats_df()
    duplicated_ids = set(blockstats_df.loc[blockstats_df['duplicated'] == True].index)
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


def filter_cluster_paths_by_block_freq(paths, cluster_map, freq_threshold = 0.5):
    """
    Filter blocks out of all paths that are in less than half of the isolates
    of their cluster.

    - paths: dict {isolate_id: pu.Path}
    - cluster_map_core: dict {isolate_id: cluster_id}
    - freq_tresh: float, how frequent do blocks need to be per cluster to not be filtered out

    Returns
    -------
    dict {isolate_id: pu.Path} with filtered paths.
    """
    # Group isolates by cluster
    cluster_to_isos = defaultdict(list)
    for iso, cl in cluster_map.items():
        if iso in paths:  # ignore isolates not in paths
            cluster_to_isos[cl].append(iso)

    filtered_paths = {}

    for cl, isos in cluster_to_isos.items():
        n_isos = len(isos)

        # If only one isolate in the cluster, do not filter anything
        if n_isos <= 1:
            for iso in isos:
                filtered_paths[iso] = paths[iso]
            continue

        # i.e. keep blocks with count >= ceil(n_isos * freq_threshold)
        threshold = math.ceil(n_isos * freq_threshold)

        # Count block presence per cluster (after deduplication max. once per isolate)
        block_counts = defaultdict(int)
        for iso in isos:
            for node in paths[iso].nodes:
                block_counts[node] += 1

        # Blocks to filter in this cluster
        filter_set = {node for node, cnt in block_counts.items() if cnt < threshold}

        # Apply your helper function on this cluster only
        cluster_paths = {iso: paths[iso] for iso in isos}
        filtered_cluster_paths = filter_deduplicated_paths(cluster_paths, filter_set)

        # Collect results
        filtered_paths.update(filtered_cluster_paths)

    # If there are isolates in `paths` not in cluster_map_core, copy as-is
    for iso, path in paths.items():
        if iso not in filtered_paths:
            filtered_paths[iso] = path

    return filtered_paths


def compute_cluster_consensus_paths(paths, cluster_map):
    """
    For each cluster, find the majority path (consensus) among its isolates.

    - paths: dict {isolate_id: pu.Path}
    - cluster_map: dict {isolate_id: cluster_id}

    Returns
    -------
    dict {cluster_id: pu.Path}  # consensus path per cluster
    """
    # group isolates by cluster
    cluster_to_isos = defaultdict(list)
    for iso, cl in cluster_map.items():
        if iso in paths:  # only consider isolates that have a path
            cluster_to_isos[cl].append(iso)

    consensus_paths = {}

    for cl, isos in cluster_to_isos.items():
        # If only one isolate in cluster, that path *is* the consensus
        if len(isos) == 1:
            iso = isos[0]
            consensus_paths[cl] = paths[iso]
            continue

        # Count identical paths in this cluster
        path_counter = Counter(paths[iso] for iso in isos)

        # Pick the majority path
        # tie-breaker: higher count, then longer path
        majority_path, _ = max(
            path_counter.items(),
            key=lambda kv: (kv[1], len(kv[0].nodes))
        )
        consensus_paths[cl] = majority_path

    return consensus_paths


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