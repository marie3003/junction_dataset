import scipy.cluster.hierarchy as sch
from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os, re
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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
            id=f"{isolate_name}__{node_id}",        # what appears after ">" in FASTA
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


def cluster_by_tree(
    tree_file,
    n_clusters=None,
    distance_threshold=None,
    frac_of_max=None,
    verbose=True,
):
    """
    Cluster isolates in a Newick tree using:
    - A) fixed number of clusters (n_clusters)
    - B) absolute distance threshold (distance_threshold)
    - C) relative distance threshold (frac_of_max * max_distance)

    Returns:
        dict {isolate_name: cluster_id}
    """

    modes = [n_clusters is not None,
             distance_threshold is not None,
             frac_of_max is not None]

    if sum(modes) != 1:
        raise ValueError(
            "Specify exactly ONE of the following:\n"
            " - n_clusters\n"
            " - distance_threshold\n"
            " - frac_of_max"
        )

    # Read tree 
    tree = Phylo.read(tree_file, "newick")
    terminals = tree.get_terminals()
    n = len(terminals)

    # Build patristic distance matrix 
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(terminals[i], terminals[j])
            dist[i, j] = dist[j, i] = d

    # Determine distance threshold if using frac_of_max 
    if frac_of_max is not None:
        max_d = dist.max()
        distance_threshold = frac_of_max * max_d
        if verbose:
            print(
                f"Using distance threshold = {distance_threshold:.6f} "
                f"(= {frac_of_max*100:.1f}% of max distance {max_d:.6f})"
            )

    # Configure clustering
    if n_clusters is not None:
        # Mode A: fixed number of clusters
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

    else:
        # Mode B/C: distance-based clustering
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

    # Remap labels to isolate names
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique, start=1)}

    cluster_map = {
        tip.name: int(remap[label]) for tip, label in zip(terminals, labels)
    }

    if verbose:
        print(f"Formed {len(unique)} clusters")

    return cluster_map


def strip_newick_suffixes(input_path, output_path):
    """
    Remove everything after '__' in all node names inside a Newick tree string.
    Example: 'A__123' -> 'A'
    """
    with open(input_path, "r") as f:
        newick_str = f.read()
    pattern = re.compile(r'([^,():;]+)__[^,():;]+')
    cleaned = pattern.sub(r'\1', newick_str)

    with open(output_path, "w") as f:
        f.write(cleaned)

def get_isolate_sequence(pangraph, block_id, node_id):
    sequence = pangraph.blocks[block_id].alignment.generate_alignment()[str(node_id)]
    return sequence

def write_shared_nodes_fasta(pangraph, path_dict, shared_nodes, shared_node_isolates, output_path):

    shared_id_set = {node for node in shared_nodes}

    fasta_records = []
    for isolate in shared_node_isolates:
        seq = ""
        for block in path_dict[isolate].nodes:
            if block.id in shared_id_set:
                seq = seq + get_isolate_sequence(pangraph, block.id, block.nid)
        record = SeqRecord(Seq(seq), id = isolate, description = "")
        fasta_records.append(record)

    SeqIO.write(fasta_records, output_path, "fasta")

def simplify_cluster_keys(cluster_map):
    """
    Return a new cluster_map where keys are truncated at the first '__'.
    
    Example:
    'NZ_CP080117.1__5727886839925671655' -> 'NZ_CP080117.1'
    """
    new_map = {}
    for key, value in cluster_map.items():
        simple_key = key.split("__")[0]
        new_map[simple_key] = value
    return new_map