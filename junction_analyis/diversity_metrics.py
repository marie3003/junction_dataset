import numpy as np
import pandas as pd

import string
from itertools import combinations

from Bio import Align

from junction_analyis.helpers import get_hierarchical_order, get_tree_order


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