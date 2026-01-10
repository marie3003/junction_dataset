import os
import subprocess

from junction_analysis.helpers import write_shared_nodes_fasta
from Bio import Phylo

from itertools import combinations
from collections import defaultdict, deque

def build_tree_from_block_list(pangraph, path_dict, block_list, isolate_list, parent_dir, file_name_prefix,
                               mafft_bin="mafft", fasttree_bin="fasttree", no_tree_gen=False):
    """
    Build block fasta, alignment, and tree for the given list of blocks. If these files already exist, the code is not rerun.
    block_list: list of blocks (potentially including context)
    isolate_list: list of isolate names to include
    """

    os.makedirs(parent_dir, exist_ok=True)

    fasta_file = os.path.join(parent_dir, f"{file_name_prefix}_blocks.fa")
    aln_file = os.path.join(parent_dir, f"{file_name_prefix}_blocks_aln.fa")
    tree_file = os.path.join(parent_dir, f"{file_name_prefix}_blocks_aln.newick")

    # skip everything if all required outputs already exist
    fasta_exists = os.path.exists(fasta_file)
    aln_exists = os.path.exists(aln_file)
    tree_exists = os.path.exists(tree_file)

    if fasta_exists and aln_exists and (no_tree_gen or tree_exists):
        print(
            f"Skipping {file_name_prefix}: FASTA, alignment and tree already exist in {parent_dir}"
        )
        return

    # Write FASTA file of shared blocks for this consensus
    write_shared_nodes_fasta(
        pangraph,
        path_dict,
        block_list,
        isolate_list,
        fasta_file,
    )
    print(f"Wrote FASTA: {fasta_file}")

    # Run MAFFT alignment
    with open(aln_file, "w") as aln_out:
        subprocess.run(
            [mafft_bin, "--quiet", fasta_file],
            stdout=aln_out,
            check=True,
        )
    print(f"Wrote alignment: {aln_file}")

    # Run FastTree to build tree (gaps treated as missing data)
    if no_tree_gen == False:
        with open(tree_file, "w") as tree_out:
            subprocess.run(
                [fasttree_bin, "-nt", "-gtr", aln_file],
                stdout=tree_out,
                check=True,
            )
        print(f"Wrote tree: {tree_file}")

def build_trees_for_all_consensus(
    consensus_paths,
    consensus_shared_nodes,
    consensus_isolates_with_all_shared,
    pangraph,
    path_dict,
    parent_dir,
    mafft_bin="mafft",
    fasttree_bin="fasttree",
):
    """
    For each consensus path:
      1) write a FASTA of shared blocks for isolates that contain all shared blocks
      2) create a MAFFT alignment
      3) build a FastTree tree

    Files created per consensus_k:
      - {parent_dir}/consensus_k_shared_blocks.fa
      - {parent_dir}/consensus_k_shared_blocks_aln.fa
      - {parent_dir}/consensus_k_shared_blocks_aln.tree
    """

    os.makedirs(parent_dir, exist_ok=True)

    for idx, consensus_path in enumerate(consensus_paths):
        consensus_label = f"consensus_{idx+1}"

        shared_blocks = consensus_shared_nodes.get(consensus_label, [])
        isolates_with_all = consensus_isolates_with_all_shared.get(consensus_label, [])

        if not shared_blocks:
            print(f"[{consensus_label}] No shared blocks found, skipping.")
            continue
        if not isolates_with_all:
            print(f"[{consensus_label}] No isolates contain all shared blocks, skipping.")
            continue

        # Write FASTA file of shared blocks for this consensus
        print(consensus_label)
        build_tree_from_block_list(pangraph, path_dict, shared_blocks, isolates_with_all, parent_dir, f"{consensus_label}_shared", mafft_bin, fasttree_bin)

    print("Done building trees for all consensus paths.")

def compute_pairwise_distances(tree_path):
    tree = Phylo.read(tree_path, "newick")

    # Get all terminal nodes (tips)
    tips = tree.get_terminals()

    # Compute pairwise distances
    pairwise_distances = []
    for t1, t2 in combinations(tips, 2):
        dist = tree.distance(t1, t2)
        pairwise_distances.append(dist)

    return pairwise_distances

def cluster_tree_by_branch_length(tree_path, length_threshold, fmt="newick"):
    tree = Phylo.read(tree_path, fmt)

    adj = defaultdict(list) #missing key would return empty list

    def traverse(clade, parent=None):
        adj[clade]  # ensure key

        if parent is not None:
            bl = clade.branch_length
            if bl is None or bl <= length_threshold:
                adj[clade].append(parent)
                adj[parent].append(clade)

        for child in clade.clades:
            traverse(child, clade)

    # Use any clade as a start node (works for rooted and unrooted trees)
    start_clade = next(tree.find_clades())
    traverse(start_clade)

    # Connected components → clusters
    cluster_map = {}
    visited = set()
    cluster_id = 0

    for node in list(adj.keys()):
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        component_nodes = []

        while queue:
            cur = queue.popleft()
            component_nodes.append(cur)
            for neigh in adj[cur]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)

        # assign cluster id to all named clades in this component
        for clade in component_nodes:
            if clade.name:
                cluster_map[clade.name] = cluster_id

        cluster_id += 1

    return cluster_map