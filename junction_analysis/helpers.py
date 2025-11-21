import scipy.cluster.hierarchy as sch
from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord
import os

def get_hierarchical_order(distance_df):
    linkage_matrix = sch.linkage(distance_df, method="ward")
    ordered_indices = sch.leaves_list(linkage_matrix)
    order = distance_df.index[ordered_indices]
    return order

def get_tree_order():
    _fname = f"../config/polished_tree.nwk"
    _tree = Phylo.read(_fname, "newick")
    _tree.root_at_midpoint()
    _tree.ladderize()
    # extract the order of isolates from the tree
    leaf_order = [leaf.name for leaf in _tree.get_terminals()]
    return leaf_order


def get_isolate_name_from_node_id(example_pangraph, node_id):
    return example_pangraph.paths.idx_to_name[example_pangraph.nodes[node_id].path_id]

def write_isolate_fasta(example_pangraph, block, output_path):
    """
    Write sequences to a FASTA file, using isolate names derived from node IDs.

    Parameters
    ----------
    example_pangraph : Pangraph object
        Object containing mapping from node IDs to isolate names.
    block : Block object
    output_path : str or Path
        Path to the FASTA file to write.
    """
    records = block.to_biopython_records()
    fasta_records = []
    for record in records:
        node_id = record.id
        isolate_name = get_isolate_name_from_node_id(example_pangraph, node_id)

        # Create a new SeqRecord with a readable isolate name
        new_record = SeqRecord(
            record.seq,
            id=isolate_name,        # what appears after ">" in FASTA
            description=""          # no extra text
        )
        fasta_records.append(new_record)

    # Write all sequences to a FASTA file
    SeqIO.write(fasta_records, output_path, "fasta")


def convert_gbk_fasta(gbk_folder, fasta_folder):

    os.makedirs(fasta_folder, exist_ok=True)

    # List all .gbk files
    for fname in os.listdir(gbk_folder):
        if not fname.endswith(".gbk"):
            continue

        input_gbk = os.path.join(gbk_folder, fname)
        genome_fasta = os.path.join(fasta_folder, fname.replace(".gbk", ".fasta"))

        print(f"Converting: {input_gbk} → {genome_fasta}")

        with open(genome_fasta, "w") as out_f:
            for record in SeqIO.parse(input_gbk, "genbank"):
                SeqIO.write(record, out_f, "fasta")


def convert_gbk_fasta_plasmids(gbk_folder, fasta_folder):
    """
    Converts plasmid GenBank files from a nested directory structure into FASTA format,
    preserving the directory structure.

    For each plasmid, a FASTA record is created with a header in the format:
    >{plasmid_name}|{isolate_name}

    The plasmid name is extracted from the GenBank record's source feature if available,
    otherwise it defaults to the filename. The isolate name is derived from the
    directory path relative to the input folder.

    Parameters
    ----------
    gbk_folder : str
        The root directory containing the plasmid GenBank files (.gbk or .gb).
    fasta_folder : str
        The root directory where the output FASTA files will be saved.
    """
    os.makedirs(fasta_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(gbk_folder):
        for filename in filenames:
            if not (filename.endswith(".gbk") or filename.endswith(".gb")):
                continue

            input_gbk = os.path.join(dirpath, filename)

            # Determine output path while preserving structure
            relative_dir = os.path.relpath(dirpath, gbk_folder)
            output_dir = os.path.join(fasta_folder, relative_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_fasta = os.path.join(output_dir, os.path.splitext(filename)[0] + ".fasta")

            print(f"Converting: {input_gbk} -> {output_fasta}")

            records_to_write = []
            for record in SeqIO.parse(input_gbk, "genbank"):
                # Determine isolate name from directory structure
                isolate_name = "unknown_isolate"
                if relative_dir and relative_dir != ".":
                    isolate_name = relative_dir.replace(os.path.sep, "_")

                # Determine plasmid name from GenBank record
                plasmid_name = os.path.splitext(filename)[0]  # Default to filename
                if record.features:
                    for feature in record.features:
                        if feature.type == "source":
                            if "plasmid" in feature.qualifiers:
                                plasmid_name = feature.qualifiers["plasmid"][0]
                                break
                
                # Clean up names to be filesystem-friendly and header-friendly
                plasmid_name = plasmid_name.replace(" ", "_").replace("|", "-")
                isolate_name = isolate_name.replace(" ", "_").replace("|", "-")

                # Create a new record with the desired header format
                new_record = SeqRecord(
                    record.seq,
                    id=f"{plasmid_name}|{isolate_name}",
                    description=""
                )
                records_to_write.append(new_record)

            # Write the new record(s) to the FASTA file
            if records_to_write:
                with open(output_fasta, "w") as out_f:
                    SeqIO.write(records_to_write, out_f, "fasta")

