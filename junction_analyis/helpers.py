import scipy.cluster.hierarchy as sch
from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord

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
