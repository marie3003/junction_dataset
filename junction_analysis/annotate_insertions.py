import numpy as np
import pandas as pd

import pypangraph as pp

import os
import subprocess
import re

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
        deletions: dict of isolate -> list of deleted segments (each is a list of nodes)
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

def get_isolate_sequence(pangraph, block_id, node_id):
    sequence = pangraph.blocks[block_id].alignment.generate_alignment()[str(node_id)]
    return sequence

def write_segment_fasta(example_junction, isolate_name, segment_name, consensus, sequence, path):
    record = SeqRecord(
        Seq(sequence),
        id=f"{isolate_name}|{segment_name}",
        description=f"path{path} length{len(sequence)}"
    )
    output_path = f"../results/atb_lookup/{example_junction}/consensus{consensus}/{isolate_name}_{segment_name}.fasta"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    SeqIO.write(record, output_path, "fasta")

def write_insertions_fasta(example_junction, pangraph, insertions, consensus = 1):
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
                seq = seq + get_isolate_sequence(pangraph, block.id, block.nid)
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
    deduplicated_paths, deduplicated_blog_freq = make_deduplicated_paths(example_pangraph, path_dict)

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


def find_insertion_hits_own_genome(genome_root, insertions_seq_dir):
    """
    Look up insertion sequences in an isolates own genome.
    @param genome_root: directory of genome files
    @param insertion_seq_dir: directory with subfolder structure junction_name/consensus_n/insertion_file.fasta
    """
    results = []

    for dirpath, dirnames, filenames in os.walk(insertions_seq_dir):

        for fname in filenames:
            if not fname.endswith(".fasta"):
                continue

            insertion_seq_path = os.path.join(dirpath, fname)

            # Read header and sequence from the consensus fasta
            with open(insertion_seq_path) as f:
                header = f.readline().strip()

            # Extract genome name from header, e.g.
            # >NZ_AP022044.1|segment_0 ...  -> "NZ_AP022044.1"
            header_main = header.lstrip(">")
            genome_name, segment = header_main.split()[0].split("|")  

            path_match = re.search(r"path\[(.*)\]\s+length", header)
            path_string = path_match.group(1) if path_match else None

            # Extract length (integer after 'length')
            length_match = re.search(r"length(\d+)", header)
            seq_length = int(length_match.group(1)) if length_match else None


            genome_fasta = os.path.join(genome_root, f"{genome_name}.fasta")

            # Skip if genome fasta doesn't exist (optional safety)
            if not os.path.exists(genome_fasta):
                print(f"WARNING: genome fasta not found: {genome_fasta}")
                continue

            # Define output PAF path
            paf_path = insertion_seq_path.replace(".fasta", ".paf")

            # Run minimap2: consensus (query) vs genome (target)
            subprocess.run(
                [
                    "minimap2", "-x", "asm5", "-N", "50", "-p", "0.9", "-k", "19", "--eqx",
                    genome_fasta,         # target / reference
                    insertion_seq_path    # query
                ],
                stdout=open(paf_path, "w")
            )

            # Count hits with <1% divergence and >= 90% coverage
            count = 0
            with open(paf_path) as paf_file:
                for line in paf_file:
                    if "\tdv:f:" not in line:
                        continue

                    fields = line.split("\t")
                    # Query length, start and end
                    query_len = int(fields[1])
                    query_start = int(fields[2])
                    query_end = int(fields[3])

                    dv = float(line.split("dv:f:")[1].split("\t")[0])
                    coverage = (query_end - query_start) / query_len if query_len > 0 else 0

                    if dv < 0.01 and coverage >= 0.9:
                        count += 1

            results.append(
                {
                    "junction_name": os.path.basename(os.path.dirname(dirpath)),
                    "consensus": os.path.basename(dirpath),
                    "genome_name": genome_name,
                    "insertion_path": path_string,
                    "insertion_length": seq_length,
                    "segment": segment,
                    "hits_in_genome": count,
                }
            )

    # Build final DataFrame
    insertions_df = pd.DataFrame(results)
    return insertions_df


def find_insertion_hits_in_plasmids(plasmid_fasta_root, insertions_seq_dir):
    """
    Looks up insertion sequences in all plasmid sequences that match the isolate of the insertion.
    For each insertion and each corresponding plasmid, it runs minimap2 and saves the output
    to a PAF file named after the insertion and plasmid.

    Args:
        plasmid_fasta_root (str): Directory of plasmid FASTA files.
                                  Expected structure is .../{isolate_name}/{plasmid}.fasta
        insertions_seq_dir (str): Directory with insertion sequences.
                                  Expected structure is .../{junction}/{consensus}/{isolate_name}_segment_*.fasta
    Returns:
        pandas.DataFrame: A DataFrame with the results, where each row corresponds to an
                          insertion-plasmid pair and includes the count of significant hits.
    """
    results = []

    # Walk through the insertion sequences directory
    for dirpath, _, filenames in os.walk(insertions_seq_dir):
        for fname in filenames:
            if not fname.endswith(".fasta"):
                continue

            insertion_seq_path = os.path.join(dirpath, fname)

            # Extract isolate name from the insertion FASTA header
            with open(insertion_seq_path) as f:
                header = f.readline().strip()
            if not header.startswith(">"):
                continue

            header_main = header.lstrip(">")
            isolate_name, segment_name = header_main.split()[0].split("|")

            # Find all plasmid fasta files for this isolate
            isolate_plasmid_dir = os.path.join(plasmid_fasta_root, isolate_name)
            if not os.path.isdir(isolate_plasmid_dir):
                continue

            plasmid_files = [os.path.join(isolate_plasmid_dir, f) for f in os.listdir(isolate_plasmid_dir) if f.endswith((".fasta", ".fa"))]
            if not plasmid_files:
                continue

            # Extract metadata from header once per insertion
            path_match = re.search(r"path\[(.*)\]\s+length", header)
            path_string = path_match.group(1) if path_match else None
            length_match = re.search(r"length(\d+)", header)
            seq_length = int(length_match.group(1)) if length_match else None

            # Iterate over each plasmid file
            for plasmid_file in plasmid_files:
                plasmid_name = os.path.splitext(os.path.basename(plasmid_file))[0]

                # Define a unique PAF output path for each plasmid
                paf_path = insertion_seq_path.replace(".fasta", f"_{plasmid_name}.paf")

                # Run minimap2 for each plasmid separately
                subprocess.run(
                    [
                        "minimap2", "-x", "asm5", "-N", "50", "-p", "0.9", "-k", "19", "--eqx",
                        plasmid_file,
                        insertion_seq_path
                    ],
                    stdout=open(paf_path, "w")
                )

                # Process PAF output to count significant hits
                count = 0
                with open(paf_path) as paf_file:
                    for line in paf_file:
                        if not line or "\tdv:f:" not in line:
                            continue
                        fields = line.split("\t")
                        query_len = int(fields[1])
                        query_start = int(fields[2])
                        query_end = int(fields[3])

                        dv = float(line.split("dv:f:")[1].split("\t")[0])
                        coverage = (query_end - query_start) / query_len if query_len > 0 else 0

                        if dv < 0.01 and coverage >= 0.9:
                            count += 1
                results.append({
                    "junction_name": os.path.basename(os.path.dirname(os.path.dirname(dirpath))),
                    "consensus": os.path.basename(dirpath),
                    "isolate_name": isolate_name,
                    "plasmid_name": plasmid_name,
                    "insertion_path": path_string,
                    "insertion_length": seq_length,
                    "segment": segment_name,
                    "hits_in_plasmid": count,
                })

    return pd.DataFrame(results)

