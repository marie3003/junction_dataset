import numpy as np
import pandas as pd

import pypangraph as pp
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pathlib import Path

from junction_analysis.consensus import make_deduplicated_paths
from junction_analysis import pangraph_utils as pu


def write_block_fasta(example_pangraph, example_junction, isolate_name, block_id, single_sequence = True):
    if single_sequence:
        sequence = Seq(example_pangraph.blocks[block_id].to_biopython_records()[0].seq)
    else:
        sequence = Seq(example_pangraph.blocks[block_id].consensus())
    record = SeqRecord(
        Seq(example_pangraph.blocks[block_id].to_biopython_records()[0].seq),
        id=f"{isolate_name}|block_{block_id}",
        description=f"block {block_id} from isolate {isolate_name}"
    )
    output_path = f"../results/atb_lookup/{example_junction}/{isolate_name}_block_{block_id}.fasta"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    SeqIO.write(record, output_path, "fasta")

def get_insertions_deletions(deduplicated_paths, consensus_path):
    """Identify insertions and deletions in each isolate's path compared to the matching consensus path.
    Arguments:
        deduplicated_paths: dict of isolate -> Path object (prefiltered to only contain isolates matching the consensus path)
        consensus_path: Path object representing the consensus path
    Returns:
        insertions: dict of isolate -> list of inserted segments (each is a Path object)
        deletions: dict of isolate -> list of deleted segments (each is a Path object)
    """

    insertions = {}  # isolate -> list of inserted segments (each is a list of nodes)
    deletions = {}   # isolate -> list of deleted segments (each is a list of nodes)

    consensus_nodes = set(consensus_path.nodes)
    for isolate, path in deduplicated_paths.items():

        # --- Find insertions ---
        current_insertion = []
        strand = None

        def flush():
            nonlocal current_insertion
            if current_insertion:
                insertions.setdefault(isolate, []).append(pu.Path(current_insertion))
                current_insertion = []

        for node in path.nodes:
            if node in consensus_nodes:
                # Finish any ongoing insertion at a consensus boundary
                flush()
                strand = None
                continue

            # Node is part of an insertion region
            if strand is None:
                # Start a new insertion block
                strand = node.strand
                current_insertion.append(node)
            elif node.strand == strand:
                # Keep the same insertion block
                current_insertion.append(node)
            else:
                # Strand changed → split
                flush()
                strand = node.strand
                current_insertion.append(node)

        # Handle trailing insertion
        flush()

        # --- Find deletions ---
        current_deletion = []
        path_nodes_set = set(path.nodes)
        for node in consensus_path.nodes:
            if node not in path_nodes_set:
                current_deletion.append(node)
            else:
                if current_deletion:
                    deletions.setdefault(isolate, []).append(pu.Path(current_deletion))
                    current_deletion = []
        if current_deletion:  # handle trailing deletion
            deletions.setdefault(isolate, []).append(pu.Path(current_deletion))
    
    return insertions, deletions

def get_isolate_sequence_from_fasta(fasta_path, isolate_name):
    """
    Reads a FASTA file and returns the sequence for the given isolate name.
    """
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == isolate_name:
            return str(record.seq)
    return None

def write_segment_fasta(example_junction, isolate_name, segment_name, consensus, sequence, path):
    record = SeqRecord(
        Seq(sequence),
        id=f"{isolate_name}|{segment_name}",
        description=f"path{path} length{len(sequence)}"
    )
    output_path = f"../results/atb_lookup/{example_junction}/consensus{consensus}/{isolate_name}_{segment_name}.fasta"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    SeqIO.write(record, output_path, "fasta")

def write_insertions_fasta(example_junction, insertions, consensus = 1):
    """
    retrieve sequence of insertion from isolate's block
    insertion sequence should either be all inverted or all non-inverted, otherwise split in two
    for inverted sequences write the last block first and then go to the front (result will be the correct blocks just the other way around)
    """
    for isolate, inserted_paths in insertions.items():
        for idx, inserted_path in enumerate(inserted_paths):
            seq = ""
            # switch block order if all - strand
            if inserted_path.nodes[0].strand == False:
                inserted_path.nodes.reverse()
            for block in inserted_path.nodes:
                seq = seq + get_isolate_sequence_from_fasta(f"../results/block_fastas/{example_junction}/block_{block.id}_sequences.fa", isolate)
            write_segment_fasta(example_junction, isolate, f"segment_{idx}", consensus, seq, inserted_path)

def print_insertions_deletions(insertions, deletions):
    print("Insertions:")
    for isolate, segs in insertions.items():
        for seg in segs:
            print(isolate, "INSERTED:", seg)

    print("\nDeletions:")
    for isolate, segs in deletions.items():
        for seg in segs:
            print(isolate, "DELETED:", seg)

def get_insertions_deletions_from_consensus(example_pangraph, assignment_df, consensus_paths, consensus = 1, verbose = True):
    # get isolates belonging to consensus 1
    isolates_1 = assignment_df[assignment_df['best_consensus'] == f"consensus_{consensus}"].index.tolist()
    isolates_1_set = set(isolates_1)

    # make path dict
    path_dict = example_pangraph.to_path_dictionary()
    path_dict = {isolate: pu.Path.from_tuple_list(path, 'node') for isolate, path in path_dict.items() if isolate in isolates_1_set}

    # deduplicate blocks
    blockstats_df = example_pangraph.to_blockstats_df()
    deduplicated_paths, deduplicated_blog_freq = make_deduplicated_paths(blockstats_df, path_dict)

    # compare deduplicated paths to consensus paths to find deviations, consensus paths are already deduplicated
    insertions, deletions = get_insertions_deletions(deduplicated_paths, consensus_paths[consensus - 1])

    # Print results
    if verbose:
        print_insertions_deletions(insertions, deletions)

    return insertions, deletions

def write_sgenome_ids(atb_hits_df, output_file):
    sgenome_ids = atb_hits_df.sgenome.to_list()
    with open(output_file, "w") as f:
        for sid in sgenome_ids:
            f.write(str(sid) + "\n")

def retrieve_SAMids_txt(parent_dir):
    parent_dir = Path(parent_dir)

    for file_path in parent_dir.glob("*.lexicmap.tsv"):
        hits_df = pd.read_csv(file_path, sep="\t")
        output_path = file_path.with_name(file_path.name.replace(".lexicmap.tsv", ".ids.txt"))
        write_sgenome_ids(hits_df, output_path)

def combine_NCBI_atb_results(parent_dir):
    parent_dir = Path(parent_dir)

    for file_path in parent_dir.glob("*.ncbi_results.tsv"):
        ncbi_res_df = pd.read_csv(file_path, sep="\t")
        hits_df = pd.read_csv(file_path.with_name(file_path.name.replace(".ncbi_results.tsv", ".lexicmap.tsv")), sep="\t")
        merged_df = pd.merge(hits_df, ncbi_res_df, on="sgenome", how="left")

        output_path = file_path.with_name(file_path.name.replace(".ncbi_results.tsv", ".hits_info.tsv"))
        merged_df.to_csv(output_path, index = False, sep="\t")