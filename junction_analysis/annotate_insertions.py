from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import pypangraph as pp

import os
import subprocess
import re
import shutil

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pathlib import Path

from junction_analysis.consensus import make_deduplicated_paths
from junction_analysis.helpers import get_isolate_sequence, get_consensus_seq_from_alignment
from junction_analysis import pangraph_utils as pu
from junction_analysis.junction_trees import build_tree_from_block_list


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
    """Identify insertions, deletions, and inversions in each isolate's path compared to the matching consensus path.
    Arguments:
        deduplicated_paths: dict of isolate -> Path object (prefiltered to only contain isolates matching the consensus path)
        consensus_path: Path object representing the consensus path
    Returns:
        insertions: dict of isolate -> list of inserted segments (each is a Path object)
            insertions are defined as blocks that do not exist in the consensus path but in an isolate path dependent on block id and in case of duplicated blocks on context (regardless of strand)
        deletions: dict of isolate -> list of deleted segments (each is a list of nodes)
        inversions: dict of isolate -> list of inverted segments (each is a Path object with isolate nodes),
            inversions are defined as blocks that exist in both consensus and isolate but have different strand,
            this means that inverted insertions will not be reported as inversions but as insertions, and inverted deletions will not be reported as inversions but as deletions
        translocations: dict of isolate -> list of pu.Path objects (each Path contains isolate nodes),
            translocations are blocks present in both consensus and isolate but whose left anchor differs.
            Anchors are blocks matching by (id, strand, context) in both consensus and isolate.
            Each block is assigned its rightmost anchor to the left; mismatched anchors indicate translocation.
    """

    insertions = {}  # isolate -> list of inserted segments (each is a list of nodes)
    deletions = {}   # isolate -> list of deleted segments (each is a list of nodes)
    inversions = {}  # isolate -> list of inverted segments (each is a Path of isolate nodes)
    translocations = {}  # isolate -> list of translocated segments (each is a Path of isolate nodes)

    # Strand-agnostic lookup: (id, context) for consensus membership
    consensus_id_context = {(n.id, n.context) for n in consensus_path.nodes}

    for isolate, path in deduplicated_paths.items():

        # --- Find insertions (strand-agnostic consensus check) ---
        current_insertion = []
        strand = None

        def flush():
            nonlocal current_insertion
            if current_insertion:
                insertions.setdefault(isolate, []).append(pu.Path(current_insertion))
                current_insertion = []

        for node in path.nodes:
            if (node.id, node.context) in consensus_id_context:
                # Block exists in consensus (regardless of strand) — not an insertion
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

        # --- Find deletions (strand-agnostic isolate check) ---
        current_deletion = []
        del_strand = None
        last_existing_isolate_node = None
        # Strand-agnostic lookup: (id, context) -> isolate node
        path_node_lookup = {(n.id, n.context): n for n in path.nodes}

        def flush_del():
            nonlocal current_deletion, del_strand, last_existing_isolate_node
            if current_deletion:
                deletions.setdefault(isolate, []).append({
                    "path": pu.Path(current_deletion),
                    "left_nid": last_existing_isolate_node.nid if last_existing_isolate_node else None,})
                current_deletion = []
            del_strand = None

        for cnode in consensus_path.nodes:
            isolate_node = path_node_lookup.get((cnode.id, cnode.context))

            if isolate_node is None:
                # node is truly absent from isolate — part of a deletion region
                if del_strand is None:
                    del_strand = cnode.strand
                    current_deletion.append(cnode)
                elif cnode.strand == del_strand:
                    current_deletion.append(cnode)
                else:
                    # strand changed inside the deletion → split
                    flush_del()
                    del_strand = cnode.strand
                    current_deletion.append(cnode)
            else:
                flush_del()
                last_existing_isolate_node = isolate_node # should always be set because paths start with core block

        # trailing deletion even though it should technically not happen
        flush_del()

        # --- Find inversions ---
        # Build lookup: (id, context) -> consensus node strand
        consensus_strand_lookup = {}
        for cnode in consensus_path.nodes:
            consensus_strand_lookup[(cnode.id, cnode.context)] = cnode.strand

        current_inversion = []

        def flush_inv():
            nonlocal current_inversion
            if current_inversion:
                inversions.setdefault(isolate, []).append(pu.Path(current_inversion))
                current_inversion = []

        for node in path.nodes:
            cons_strand = consensus_strand_lookup.get((node.id, node.context))
            if cons_strand is None:
                # Insertion node (not in consensus) — don't interrupt an ongoing inversion
                continue
            elif node.strand != cons_strand:
                # Block exists in both but with different strand — inverted
                current_inversion.append(node)
            else:
                # Block exists with correct strand — end any ongoing inversion
                flush_inv()

        flush_inv()

        # --- Find translocations ---
        # Anchors are blocks present in both consensus and isolate with matching
        # (id, strand, context) — i.e. full node equality. Only anchors can serve
        # as left context for translocation detection.
        #
        # Assign each block a "left anchor" = the rightmost anchor to the left of
        # the current position. Then for each block present in both consensus and
        # isolate (by id and context, strand-agnostic), check if left anchors match.
        # If not, that block is part of a translocation.

        # Anchors: blocks that (1) appear exactly once by id in both consensus and
        # isolate (strand-agnostic count) AND (2) have an exact (id, strand, context)
        # match between the two paths. Condition (1) prevents duplicated blocks from
        # serving as anchors; condition (2) ensures the anchor position is unambiguous.
        c_id_counts   = Counter(n.id for n in consensus_path.nodes)
        iso_id_counts = Counter(n.id for n in path.nodes)
        unique_ids = {bid for bid in c_id_counts
                      if c_id_counts[bid] == 1 and iso_id_counts.get(bid, 0) == 1}
        consensus_node_set = set(consensus_path.nodes)  # __hash__ = (id, strand, context)
        isolate_node_set   = set(path.nodes)
        exact_match_ids = {n.id for n in consensus_node_set & isolate_node_set}
        anchor_ids = unique_ids & exact_match_ids

        # Assign left anchor (block id) to each block in consensus (keyed by id, context)
        consensus_left_anchor = {}
        current_anchor = None
        for cnode in consensus_path.nodes:
            if cnode.id in anchor_ids:
                current_anchor = cnode.id
            consensus_left_anchor[(cnode.id, cnode.context)] = current_anchor

        # Assign left anchor (block id) to each block in isolate (keyed by id, context)
        isolate_left_anchor = {}
        current_anchor = None
        for node in path.nodes:
            if node.id in anchor_ids:
                current_anchor = node.id
            isolate_left_anchor[(node.id, node.context)] = current_anchor

        # Walk through isolate left to right, detect translocations among shared blocks
        current_translocation = []

        def flush_trans():
            nonlocal current_translocation
            if current_translocation:
                translocations.setdefault(isolate, []).append(pu.Path(current_translocation))
                current_translocation = []

        for node in path.nodes:
            if (node.id, node.context) not in consensus_id_context:
                # Insertion — skip, don't interrupt ongoing translocation
                continue

            iso_anchor = isolate_left_anchor[(node.id, node.context)]
            cons_anchor = consensus_left_anchor[(node.id, node.context)]

            if iso_anchor != cons_anchor:
                print(f"Isolate {isolate} block {node} has mismatched anchors: iso {iso_anchor} vs cons {cons_anchor}")
                # Mismatched left anchor — part of a translocation
                current_translocation.append(node)
            else:
                # Matching left anchor — end any ongoing translocation
                flush_trans()

        flush_trans()

    return insertions, deletions, inversions, translocations

def get_insertions_deletions_v2(deduplicated_paths, consensus_path):  # noqa: C901
    """Identify insertions, deletions, inversions, and translocations.

    For each (consensus, isolate) pair, context is recomputed locally:
    each block gets the id of the nearest anchor block to its left as context.
    Anchors are block ids that appear exactly once in both paths with the same
    strand. If an anchor id equals the block id being assigned context, "_1" is
    appended to keep (block_id, context) unique. The first block always gets
    empty context.

    Detection (three passes):
      Pass 1 (isolate):   insertions and translocations
      Pass 2 (consensus): deletions
      Pass 3 (isolate):   inversions

    Arguments:
        deduplicated_paths: dict of isolate -> Path object
        consensus_path: Path object representing the consensus path

    Returns:
        insertions:     dict of isolate -> list of pu.Path objects
        deletions:      dict of isolate -> list of dicts {"path": pu.Path, "left_nid": ...}
        inversions:     dict of isolate -> list of pu.Path objects
        translocations: dict of isolate -> list of pu.Path objects
    """
    insertions            = {}
    ambiguous_insertions  = {}
    deletions             = {}
    inversions            = {}
    translocations        = {}
    context_stats         = []  # one dict per isolate

    for isolate, path in deduplicated_paths.items():
        c_nodes   = list(consensus_path.nodes)
        iso_nodes = list(path.nodes)

        # ------------------------------------------------------------------ #
        # Compute candidate anchor ids (blocks appearing exactly once in both #
        # paths with the same strand). Translocated block ids will be removed #
        # after a preliminary detection run.                                  #
        # ------------------------------------------------------------------ #
        c_id_counts   = Counter(n.id for n in c_nodes)
        iso_id_counts = Counter(n.id for n in iso_nodes)
        c_id_to_strand   = {n.id: n.strand for n in c_nodes   if c_id_counts[n.id]   == 1}
        iso_id_to_strand = {n.id: n.strand for n in iso_nodes if iso_id_counts[n.id] == 1}

        def _candidate_anchor_ids(excluded=frozenset()):
            return {
                bid for bid in c_id_counts
                if c_id_counts[bid] == 1
                and iso_id_counts.get(bid, 0) == 1
                and c_id_to_strand.get(bid) == iso_id_to_strand.get(bid)
                and bid not in excluded
            }

        def _assign_ctx(nodes, anchor_ids):
            cur = None
            ctxs = []
            region_counts = Counter()
            for i, n in enumerate(nodes):
                base_ctx = "" if (i == 0 or cur is None) else str(cur)
                region_counts[n.id] += 1
                count = region_counts[n.id]
                ctx = base_ctx if count == 1 else f"{base_ctx}_{count}"
                ctxs.append(ctx)
                if n.id in anchor_ids:
                    if cur != n.id:
                        region_counts = Counter()
                    cur = n.id
            return ctxs

        def _count_ambiguous(nodes, ctxs):
            """Count blocks that received a count suffix > 1 (ambiguous context)."""
            n_ambiguous = sum(1 for ctx in ctxs if "_" in ctx and ctx.rsplit("_", 1)[-1].isdigit() and int(ctx.rsplit("_", 1)[-1]) > 1)
            return len(nodes), n_ambiguous

        def _detect_translocations_iterative(initial_anchor_ids):
            """Iterative preliminary pass: each time a new block id is identified as
            translocated it is immediately removed from the anchor set and contexts
            are recomputed, so that subsequent blocks in the same run are exposed
            with different contexts and can be detected in the same pass."""
            translocated_ids = set()
            excluded = set()

            while True:
                anchor_ids = initial_anchor_ids - excluded
                c_ctxs   = _assign_ctx(c_nodes,   anchor_ids)
                iso_ctxs = _assign_ctx(iso_nodes, anchor_ids)

                c_id_ctx_strand   = {(n.id, ctx): n.strand for n, ctx in zip(c_nodes,   c_ctxs)}
                iso_id_ctx_strand = {(n.id, ctx): n.strand for n, ctx in zip(iso_nodes, iso_ctxs)}
                matched = {key for key in c_id_ctx_strand if key in iso_id_ctx_strand}

                c_ctx_by_id = defaultdict(list)
                for n, ctx in zip(c_nodes, c_ctxs):
                    if (n.id, ctx) not in matched:
                        c_ctx_by_id[n.id].append(ctx)

                new_id = None
                for n, ctx in zip(iso_nodes, iso_ctxs):
                    if (n.id, ctx) in matched:
                        continue
                    if c_ctx_by_id.get(n.id) and n.id not in translocated_ids:
                        c_ctx_by_id[n.id].pop(0)
                        new_id = n.id
                        break  # add only one id per iteration

                if new_id is None:
                    break

                translocated_ids.add(new_id)
                excluded.add(new_id)

            return translocated_ids

        # Iterative preliminary run: recompute contexts each time a new translocated
        # id is found, so that all blocks in a translocated run are detected.
        preliminary_trans_ids = _detect_translocations_iterative(_candidate_anchor_ids())
        anchor_ids = _candidate_anchor_ids(excluded=preliminary_trans_ids)

        # ------------------------------------------------------------------ #
        # Build final contexts with translocated ids excluded from anchors    #
        # ------------------------------------------------------------------ #
        c_ctxs   = _assign_ctx(c_nodes,   anchor_ids)
        iso_ctxs = _assign_ctx(iso_nodes, anchor_ids)

        c_id_ctx_strand   = {(n.id, ctx): n.strand for n, ctx in zip(c_nodes,   c_ctxs)}
        iso_id_ctx_strand = {(n.id, ctx): n.strand for n, ctx in zip(iso_nodes, iso_ctxs)}
        matched = {key for key in c_id_ctx_strand if key in iso_id_ctx_strand}

        iso_n_blocks, iso_n_ambiguous = _count_ambiguous(iso_nodes, iso_ctxs)
        iso_n_duplicated = sum(1 for n in iso_nodes if iso_id_counts[n.id] > 1)

        c_ctx_by_id = defaultdict(list)
        for n, ctx in zip(c_nodes, c_ctxs):
            if (n.id, ctx) not in matched:
                c_ctx_by_id[n.id].append(ctx)

        def _base_ctx(ctx):
            parts = ctx.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return ctx

        c_id_to_base_ctxs = defaultdict(set)
        for n, ctx in zip(c_nodes, c_ctxs):
            c_id_to_base_ctxs[n.id].add(_base_ctx(ctx))

        iso_id_ctx_to_node = {(n.id, ctx): n for n, ctx in zip(iso_nodes, iso_ctxs)}

        # ------------------------------------------------------------------ #
        # Pass 1: walk isolate — insertions and translocations               #
        # ------------------------------------------------------------------ #
        current_insertion     = []
        current_insertion_first_ctx = None
        current_ins_has_dup_ambiguous = False
        current_translocation = []
        current_translocation_first_ctx = None
        current_trans_has_ambiguous = False
        n_insertions_isolate = 0
        n_ambiguous_ins_isolate = 0
        n_trans_isolate = 0
        n_ambiguous_trans_isolate = 0

        def flush_ins():
            nonlocal current_insertion, current_insertion_first_ctx
            nonlocal n_insertions_isolate, n_ambiguous_ins_isolate, current_ins_has_dup_ambiguous
            if current_insertion:
                insertions.setdefault(isolate, []).append({
                    "path": pu.Path(list(current_insertion)),
                    "ctx": current_insertion_first_ctx,
                })
                n_insertions_isolate += 1
                if current_ins_has_dup_ambiguous:
                    n_ambiguous_ins_isolate += 1
                    ambiguous_insertions.setdefault(isolate, []).append({
                        "path": pu.Path(list(current_insertion)),
                        "ctx": current_insertion_first_ctx,
                    })
                current_insertion = []
                current_insertion_first_ctx = None
                current_ins_has_dup_ambiguous = False

        def flush_trans():
            nonlocal current_translocation, current_translocation_first_ctx
            nonlocal n_trans_isolate, n_ambiguous_trans_isolate, current_trans_has_ambiguous
            if current_translocation:
                translocations.setdefault(isolate, []).append({
                    "path": pu.Path(list(current_translocation)),
                    "ctx": current_translocation_first_ctx,
                })
                n_trans_isolate += 1
                if current_trans_has_ambiguous:
                    n_ambiguous_trans_isolate += 1
                current_translocation = []
                current_translocation_first_ctx = None
                current_trans_has_ambiguous = False

        for n, ctx in zip(iso_nodes, iso_ctxs):
            if (n.id, ctx) in matched:
                flush_ins()
                flush_trans()
                continue

            avail = c_ctx_by_id.get(n.id, [])
            if avail:
                if len(avail) > 1:
                    current_trans_has_ambiguous = True
                c_ctx = avail.pop(0)
                matched.add((n.id, c_ctx))
                matched.add((n.id, ctx))
                flush_ins()
                if not current_translocation:
                    current_translocation_first_ctx = ctx
                current_translocation.append(n)
            else:
                matched.add((n.id, ctx))
                flush_trans()
                if not current_insertion:
                    current_insertion_first_ctx = ctx
                if _base_ctx(ctx) in c_id_to_base_ctxs.get(n.id, set()):
                    current_ins_has_dup_ambiguous = True
                current_insertion.append(n)

        flush_ins()
        flush_trans()

        context_stats.append(dict(
            isolate=isolate,
            n_blocks=iso_n_blocks,
            n_duplicated_blocks=iso_n_duplicated,
            n_ambiguous_blocks=iso_n_ambiguous,
            n_insertions=n_insertions_isolate,
            n_ambiguous_insertions=n_ambiguous_ins_isolate,
            n_translocations=n_trans_isolate,
            n_ambiguous_translocations=n_ambiguous_trans_isolate,
        ))

        # ------------------------------------------------------------------ #
        # Pass 2: walk consensus — deletions                                  #
        # ------------------------------------------------------------------ #
        last_iso_node    = None
        current_deletion = []
        current_deletion_first_ctx = None

        def flush_del():
            nonlocal current_deletion, last_iso_node, current_deletion_first_ctx
            if current_deletion:
                deletions.setdefault(isolate, []).append({
                    "path": pu.Path(list(current_deletion)),
                    "left_nid": last_iso_node.nid if last_iso_node else None,
                    "ctx": current_deletion_first_ctx,
                })
                current_deletion = []
                current_deletion_first_ctx = None

        for n, ctx in zip(c_nodes, c_ctxs):
            if (n.id, ctx) in matched:
                iso_n = iso_id_ctx_to_node.get((n.id, ctx))
                if iso_n is not None:
                    flush_del()
                    last_iso_node = iso_n
                continue

            if not current_deletion:
                current_deletion_first_ctx = ctx
            current_deletion.append(n)

        flush_del()

        # ------------------------------------------------------------------ #
        # Pass 3: walk isolate again — inversions                             #
        # Inversion: exact (id, ctx) match in consensus but opposite strand. #
        # ------------------------------------------------------------------ #
        current_inversion = []
        current_inversion_first_ctx = None

        def flush_inv():
            nonlocal current_inversion, current_inversion_first_ctx
            if current_inversion:
                inversions.setdefault(isolate, []).append({
                    "path": pu.Path(list(current_inversion)),
                    "ctx": current_inversion_first_ctx,
                })
                current_inversion = []
                current_inversion_first_ctx = None

        insertion_id_ctxs = {
            (n.id, n.context)
            for ins_entry in insertions.get(isolate, [])
            for n in ins_entry["path"].nodes
        } | {
            (n.id, n.context)
            for te in translocations.get(isolate, [])
            for n in te["path"].nodes
        }

        for n, ctx in zip(iso_nodes, iso_ctxs):
            if (n.id, n.context) in insertion_id_ctxs:
                # Part of an insertion or translocation— skip without flushing the inversion to not interrupt it
                continue
            c_strand = c_id_ctx_strand.get((n.id, ctx))
            if c_strand is not None and n.strand != c_strand:
                if not current_inversion:
                    current_inversion_first_ctx = ctx
                current_inversion.append(n)
            else:
                flush_inv()

        flush_inv()

    return insertions, ambiguous_insertions, deletions, inversions, translocations, context_stats


def get_isolate_sequence_from_fasta(fasta_path, isolate_name):
    """
    Reads a FASTA file and returns the sequence for the given isolate name.
    """
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == isolate_name:
            return str(record.seq)
    return None

def write_segment_fasta(example_junction, isolate_name, segment_name, consensus, sequence, path, parent_dir):
    record = SeqRecord(
        Seq(sequence),
        id=f"{isolate_name}|{segment_name}",
        description=f"path{path} length{len(sequence)}"
    )
    output_path = f"{parent_dir}/{example_junction}/consensus{consensus}/{isolate_name}_{segment_name}.fasta"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    SeqIO.write(record, output_path, "fasta")
    return output_path

def write_insertions_fasta(example_junction, pangraph, insertions, consensus = 1, parent_dir = "../results/atb_lookup/insertions", save_df = False):
    """
    Retrieve sequence of insertion from isolate's blocks.
    Blocks are kept in their original order; minus-strand blocks have their
    sequence reverse-complemented because they are stored as positive strand.
    """
    results = []

    out_dir = f"{parent_dir}/{example_junction}/consensus{consensus}"
    os.makedirs(out_dir, exist_ok=True)

    for isolate, inserted_paths in insertions.items():
        for idx, ins_entry in enumerate(inserted_paths):
            inserted_path = ins_entry["path"]
            ctx = ins_entry["ctx"]
            start_pos = pangraph.nodes[inserted_path.nodes[0].nid].start
            end_pos = pangraph.nodes[inserted_path.nodes[-1].nid].end

            parts = []
            for block in inserted_path.nodes:
                block_seq = get_isolate_sequence(pangraph, block.id, block.nid)
                if not block.strand:
                    block_seq = str(Seq(block_seq).reverse_complement())
                parts.append(block_seq)
            seq = "".join(parts)
            fasta_path = write_segment_fasta(example_junction, isolate, f"segment_{idx}", consensus, seq, inserted_path, parent_dir)

            results.append(
                {
                    "junction_name": example_junction,
                    "consensus": f"consensus_{consensus}",
                    "genome_name": isolate,
                    "path": str(inserted_path),
                    "insertion": f"segment_{idx}",
                    "fasta_path": fasta_path,
                    "length": len(seq),
                    "strand": "+" if inserted_path.nodes[0].strand else "-",
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "ctx": ctx,
                }
            )

    if not results:
        print(f"No insertions found for consensus_{consensus}, nothing saved.")
        return None

    insertions_df = pd.DataFrame(results)

    if save_df:
        insertions_df.to_csv(os.path.join(out_dir, "insertions_summary.csv"), index=False)

    return insertions_df


def summarize_ambiguous_insertions(example_junction, pangraph, ambiguous_insertions, consensus=1, parent_dir="../results/atb_lookup/insertions", save_df=False):
    """
    Build a summary DataFrame for ambiguous insertions (potential duplications).
    No sequences are extracted or FASTA files written.
    """
    results = []

    for isolate, inserted_paths in ambiguous_insertions.items():
        for idx, ins_entry in enumerate(inserted_paths):
            inserted_path = ins_entry["path"]
            ctx = ins_entry["ctx"]
            start_pos = pangraph.nodes[inserted_path.nodes[0].nid].start
            end_pos = pangraph.nodes[inserted_path.nodes[-1].nid].end

            results.append({
                "junction_name": example_junction,
                "consensus": f"consensus_{consensus}",
                "genome_name": isolate,
                "path": str(inserted_path),
                "insertion": f"segment_{idx}",
                "length": abs(end_pos - start_pos),
                "strand": "+" if inserted_path.nodes[0].strand else "-",
                "start_pos": start_pos,
                "end_pos": end_pos,
                "ctx": ctx,
            })

    if not results:
        return None

    ambiguous_insertions_df = pd.DataFrame(results)

    if save_df:
        out_dir = f"{parent_dir}/{example_junction}/consensus{consensus}"
        os.makedirs(out_dir, exist_ok=True)
        ambiguous_insertions_df.to_csv(os.path.join(out_dir, "ambiguous_insertions_summary.csv"), index=False)

    return ambiguous_insertions_df


def print_insertions_deletions(insertions, deletions, inversions=None, translocations=None):
    def _path(entry):
        return entry["path"] if isinstance(entry, dict) else entry

    if insertions:
        print("Insertions:")
        for isolate, segs in insertions.items():
            for seg in segs:
                print(isolate, "INSERTED:", _path(seg))

    if deletions:
        print("\nDeletions:")
        for isolate, segs in deletions.items():
            for seg in segs:
                print(isolate, "DELETED:", _path(seg))

    if inversions:
        print("\nInversions:")
        for isolate, segs in inversions.items():
            for seg in segs:
                print(isolate, "INVERTED:", _path(seg))

    if translocations:
        print("\nTranslocations:")
        for isolate, segs in translocations.items():
            for seg in segs:
                print(isolate, "TRANSLOCATED:", _path(seg))

def get_insertions_deletions_from_consensus(assignment_df, consensus_paths, deduplicated_paths, consensus=1, verbose=True, junction_name=None):
    # get isolates belonging to this consensus
    isolates_1 = assignment_df[assignment_df['best_consensus'] == f"consensus_{consensus}"].index.tolist()
    # only keep deduplicated paths for these isolates
    deduplicated_paths = {iso: path for iso, path in deduplicated_paths.items() if iso in isolates_1}

    # compare deduplicated paths to consensus paths to find deviations
    insertions, ambiguous_insertions, deletions, inversions, translocations, context_stats = get_insertions_deletions_v2(deduplicated_paths, consensus_paths[consensus - 1])

    # annotate stats with junction name and consensus id
    for row in context_stats:
        row["junction_name"] = junction_name
        row["consensus"] = consensus

    context_stats_df = pd.DataFrame(context_stats, columns=["junction_name", "consensus", "isolate", "n_blocks", "n_duplicated_blocks", "n_ambiguous_blocks", "n_insertions", "n_ambiguous_insertions", "n_translocations", "n_ambiguous_translocations"])

    # Print results
    if verbose:
        print_insertions_deletions(insertions, deletions, inversions, translocations)

    return insertions, ambiguous_insertions, deletions, inversions, translocations, context_stats_df

def write_sgenome_ids(atb_hits_df, output_file):
    sgenome_ids = atb_hits_df.sgenome.to_list()
    with open(output_file, "w") as f:
        for sid in sgenome_ids:
            f.write(str(sid) + "\n")

def retrieve_SAMids_txt(parent_dir):
    parent_dir = Path(parent_dir)

    for file_path in parent_dir.rglob("*.lexicmap.tsv"):
        hits_df = pd.read_csv(file_path, sep="\t")

        output_path = file_path.with_name(
            file_path.name.replace(".lexicmap.tsv", ".ids.txt")
        )

        write_sgenome_ids(hits_df, output_path)

def combine_NCBI_atb_results(parent_dir):
    parent_dir = Path(parent_dir)

    for file_path in parent_dir.rglob("*.ncbi_results.tsv"):
        lexicmap_path = file_path.with_name(file_path.name.replace(".ncbi_results.tsv", ".lexicmap.tsv"))
        output_path = file_path.with_name(file_path.name.replace(".ncbi_results.tsv", ".hits_info.tsv"))
        try:
            ncbi_res_df = pd.read_csv(file_path, sep="\t")
            hits_df = pd.read_csv(lexicmap_path, sep="\t")
            ncbi_res_df = ncbi_res_df.drop_duplicates(subset="sgenome", keep="first")
            merged_df = pd.merge(hits_df, ncbi_res_df, on="sgenome", how="left")
            merged_df.to_csv(output_path, index=False, sep="\t")
        except pd.errors.EmptyDataError as e:
            print(f"Empty file, skipping merge ({file_path}): {e}")
            open(output_path, "w").close()


def combine_NCBI_atb_results_centralized(parent_dir):
    """
    Like combine_NCBI_atb_results, but uses a single all_ncbi_results.tsv
    in parent_dir as the lookup table instead of per-segment ncbi_results files.
    """
    parent_dir = Path(parent_dir)
    ncbi_results_path = parent_dir / "all_ncbi_results.tsv"

    ncbi_res_df = pd.read_csv(ncbi_results_path, sep="\t")
    ncbi_res_df = ncbi_res_df.drop_duplicates(subset="sgenome", keep="first")

    for lexicmap_path in parent_dir.rglob("*.lexicmap.tsv"):
        output_path = lexicmap_path.with_name(
            lexicmap_path.name.replace(".lexicmap.tsv", ".hits_info.tsv")
        )
        try:
            hits_df = pd.read_csv(lexicmap_path, sep="\t")
            merged_df = pd.merge(hits_df, ncbi_res_df, on="sgenome", how="left")
            merged_df.to_csv(output_path, index=False, sep="\t")
        except pd.errors.EmptyDataError as e:
            print(f"Empty file, skipping merge ({lexicmap_path}): {e}")
            open(output_path, "w").close()


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

            path_match = re.search(r"path(\[.*])\s+length", header)
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
            path_match = re.search(r"path(\[.*])\s+length", header)
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


#### Deletions

def summarize_deletions_consensus(
    deletions,
    junction_name,
    pangraph,
    path_dict,
    assignment_df,
    consensus_id,
    parent_dir,
    rerun_alignment=True,
    save_df=False,
):
    """
    Summarize information about deletions and find consensus sequence for each deletion.
    There might be a situation in which the previous deduplication of paths failed and some identical blocks get different contexts.
    In this case only the isolates with the same context as the consensus path will be used to build the consensus sequence for the deletion. If no isolate matches the context definition, no consensus sequence will be found and a value error is raised.
    """

    isolates = assignment_df[assignment_df['best_consensus'] == f"consensus_{consensus_id}"].index.tolist()
    out_dir = f"{parent_dir}/{junction_name}/consensus_{consensus_id}"
    os.makedirs(out_dir, exist_ok=True)

    # key: Path -> file_name_prefix used for the first occurrence
    seen_paths: dict[Path, str] = {}

    if rerun_alignment:
        for iso, entries in deletions.items():
            for idx, entry in enumerate(entries):
                path = entry["path"]
                left_nid = entry["left_nid"]
                file_prefix = f"{iso}_deletion{idx}"

                if path in seen_paths:
                    # Reuse files from the first time we saw this exact path
                    src_prefix = seen_paths[path]
                    print(
                        f"Reusing alignment/tree for {iso} deletion{idx} "
                        f"from {src_prefix}"
                    )
                    for suffix in ("_blocks.fa", "_blocks_aln.fa", "_blocks_aln.newick"):
                        src = os.path.join(out_dir, f"{src_prefix}{suffix}")
                        dst = os.path.join(out_dir, f"{file_prefix}{suffix}")
                        if os.path.exists(src) and not os.path.exists(dst):
                            shutil.copyfile(src, dst)
                    continue

                # First time we see this path → build everything
                print(f"Processing {iso} with {path}")

                
                build_tree_from_block_list(
                    pangraph,
                    path_dict,
                    path.nodes,
                    isolates,
                    out_dir,
                    file_prefix,
                )
                seen_paths[path] = file_prefix

    # --- collect consensus sequences ---
    results = []
    for iso, entries in deletions.items():
        for idx, entry in enumerate(entries):
            path = entry["path"]
            left_nid = entry["left_nid"]
            aln_path = os.path.join(out_dir, f"{iso}_deletion{idx}_blocks_aln.fa")
            if os.path.exists(aln_path):
                consensus_seq = get_consensus_seq_from_alignment(aln_path)
                consensus_sequence = str(consensus_seq)
                length = len(consensus_seq)
            else:
                print(f"Warning: alignment file not found, consensus_sequence will be None: {aln_path}")
                consensus_sequence = None
                length = sum(
                    int(pangraph.nodes[n.nid].end - pangraph.nodes[n.nid].start)
                    for n in path.nodes
                )

            results.append(
                {
                    "junction_name": junction_name,
                    "consensus": f"consensus_{consensus_id}",
                    "genome_name": iso,
                    "path": str(path),
                    "deletion": f"deletion{idx}",
                    "consensus_sequence": consensus_sequence,
                    "length": length,
                    "position": pangraph.nodes[left_nid].end if left_nid is not None else None,
                    "strand": "+" if path.nodes[0].strand else "-",
                    "ctx": entry.get("ctx"),
                }
            )

    if not results:
        print(f"No deletions found for consensus_{consensus_id}, nothing saved.")
        return None

    deletions_df = pd.DataFrame(results)

    if save_df:
        deletions_df.to_csv(os.path.join(out_dir, "deletions_summary.csv"), index=False)

    return deletions_df


def summarize_inversions_consensus(
    inversions,
    junction_name,
    pangraph,
    consensus_id,
    parent_dir,
    save_df=False,
):
    """
    Summarize information about inversions for one consensus path.
    Each inversion is a contiguous run of blocks that exist in both
    consensus and isolate but with flipped strand.

    Parameters
    ----------
    inversions : dict
        isolate -> list of pu.Path objects (each Path contains isolate nodes with .nid)
    junction_name : str
    pangraph : pp.Pangraph
    consensus_id : int
    parent_dir : str
        Base output directory (e.g. results/atb_lookup/inversions)
    save_df : bool
        Whether to write the summary CSV to disk.

    Returns
    -------
    pd.DataFrame or None
    """
    out_dir = os.path.join(parent_dir, junction_name, f"consensus_{consensus_id}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for iso, inv_paths in inversions.items():
        for idx, inv_entry in enumerate(inv_paths):
            inv_path = inv_entry["path"]
            ctx = inv_entry["ctx"]
            nodes = inv_path.nodes
            if not nodes:
                continue

            # Compute total length from pangraph node coordinates
            total_length = sum(
                int(pangraph.nodes[n.nid].end - pangraph.nodes[n.nid].start)
                for n in nodes
            )

            # Start and end positions from the first and last node
            starts = [int(pangraph.nodes[n.nid].start) for n in nodes]
            ends = [int(pangraph.nodes[n.nid].end) for n in nodes]
            start_pos = min(starts)
            end_pos = max(ends)

            # Majority strand among the isolate's inverted nodes
            plus_count = sum(1 for n in nodes if n.strand)
            majority_strand = "+" if plus_count >= len(nodes) / 2 else "-"

            results.append({
                "junction_name": junction_name,
                "consensus": f"consensus_{consensus_id}",
                "genome_name": iso,
                "path": str(inv_path),
                "inversion": f"inversion{idx}",
                "length": total_length,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "strand": majority_strand,
                "ctx": ctx,
            })

    if not results:
        print(f"No inversions found for consensus_{consensus_id}, nothing saved.")
        return None

    inversions_df = pd.DataFrame(results)

    if save_df:
        inversions_df.to_csv(os.path.join(out_dir, "inversions_summary.csv"), index=False)

    return inversions_df


def load_all_deletions_summaries(parent_dir, junction_name, save_df=False):
    """
    Read and combine all deletions_summary.csv files from
    {parent_dir}/{junction_name}/consensus_*/deletions_summary.csv

    Returns
    -------
    pd.DataFrame
        One long DataFrame with all deletions combined.
    """
    base_dir = os.path.join(parent_dir, junction_name)
    all_dfs = []

    if not os.path.isdir(base_dir):
        print(f"No deletions directory found for {junction_name}, skipping.")
        return pd.DataFrame(columns=["junction_name", "consensus", "genome_name", "path",
                                     "deletion", "consensus_sequence", "length", "position", "strand"])

    for subdir in sorted(os.listdir(base_dir)):
        if not subdir.startswith("consensus_"):
            continue

        csv_path = os.path.join(base_dir, subdir, "deletions_summary.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["junction_name", "consensus", "genome_name", "path",
                                     "deletion", "consensus_sequence", "length", "position", "strand"])

    complete_df = pd.concat(all_dfs, ignore_index=True)
    if save_df:
        complete_df.to_csv(os.path.join(base_dir, "all_deletions_summary.csv"), index=False)

    return complete_df

def load_all_insertions_summaries(parent_dir, junction_name, save_df=False):
    """
    Read and combine all insertions_summary.csv files from
    {parent_dir}/{junction_name}/consensus*/insertions_summary.csv

    Returns
    -------
    pd.DataFrame
        One long DataFrame with all insertions combined.
    """
    base_dir = os.path.join(parent_dir, junction_name)
    all_dfs = []

    if not os.path.isdir(base_dir):
        print(f"No insertions directory found for {junction_name}, skipping.")
        return pd.DataFrame()

    for subdir in sorted(os.listdir(base_dir)):
        if not subdir.startswith("consensus"):
            continue

        csv_path = os.path.join(base_dir, subdir, "insertions_summary.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    complete_df = pd.concat(all_dfs, ignore_index=True)

    if save_df:
        complete_df.to_csv(
            os.path.join(base_dir, "all_insertions_summary.csv"),
            index=False,
        )

    return complete_df


def load_all_ambiguous_insertions_summaries(parent_dir, junction_name, save_df=False):
    """
    Read and combine all ambiguous_insertions_summary.csv files from
    {parent_dir}/{junction_name}/consensus*/ambiguous_insertions_summary.csv

    Returns
    -------
    pd.DataFrame
        One long DataFrame with all ambiguous insertions combined.
    """
    base_dir = os.path.join(parent_dir, junction_name)
    all_dfs = []

    if not os.path.isdir(base_dir):
        print(f"No insertions directory found for {junction_name}, skipping.")
        return pd.DataFrame()

    for subdir in sorted(os.listdir(base_dir)):
        if not subdir.startswith("consensus"):
            continue

        csv_path = os.path.join(base_dir, subdir, "ambiguous_insertions_summary.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    complete_df = pd.concat(all_dfs, ignore_index=True)

    if save_df:
        complete_df.to_csv(
            os.path.join(base_dir, "all_ambiguous_insertions_summary.csv"),
            index=False,
        )

    return complete_df


def summarize_translocations_consensus(
    translocations,
    junction_name,
    pangraph,
    consensus_id,
    parent_dir,
    save_df=False,
):
    """
    Summarize information about translocations for one consensus path.

    Parameters
    ----------
    translocations : dict
        isolate -> list of pu.Path objects (each Path contains isolate nodes with .nid)
    junction_name : str
    pangraph : pp.Pangraph
    consensus_id : int
    parent_dir : str
        Base output directory (e.g. results/atb_lookup/translocations)
    save_df : bool

    Returns
    -------
    pd.DataFrame or None
    """
    out_dir = os.path.join(parent_dir, junction_name, f"consensus_{consensus_id}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for iso, trans_paths in translocations.items():
        for idx, trans_entry in enumerate(trans_paths):
            trans_path = trans_entry["path"]
            ctx = trans_entry["ctx"]
            nodes = trans_path.nodes
            if not nodes:
                continue

            starts = [int(pangraph.nodes[n.nid].start) for n in nodes]
            ends = [int(pangraph.nodes[n.nid].end) for n in nodes]
            start_pos = min(starts)
            end_pos = max(ends)
            total_length = end_pos - start_pos

            results.append({
                "junction_name": junction_name,
                "consensus": f"consensus_{consensus_id}",
                "genome_name": iso,
                "path": str(trans_path),
                "translocation": f"translocation{idx}",
                "length": total_length,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "ctx": ctx,
            })

    if not results:
        print(f"No translocations found for consensus_{consensus_id}, nothing saved.")
        return None

    translocations_df = pd.DataFrame(results)

    if save_df:
        translocations_df.to_csv(os.path.join(out_dir, "translocations_summary.csv"), index=False)

    return translocations_df


def load_all_translocations_summaries(parent_dir, junction_name, save_df=False):
    """
    Read and combine all translocations_summary.csv files from
    {parent_dir}/{junction_name}/consensus_*/translocations_summary.csv
    """
    base_dir = os.path.join(parent_dir, junction_name)
    all_dfs = []

    if not os.path.isdir(base_dir):
        print(f"No translocations directory found for {junction_name}, skipping.")
        return pd.DataFrame()

    for subdir in sorted(os.listdir(base_dir)):
        if not subdir.startswith("consensus_"):
            continue

        csv_path = os.path.join(base_dir, subdir, "translocations_summary.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    complete_df = pd.concat(all_dfs, ignore_index=True)

    if save_df:
        complete_df.to_csv(
            os.path.join(base_dir, "all_translocations_summary.csv"),
            index=False,
        )

    return complete_df


def load_all_inversions_summaries(parent_dir, junction_name, save_df=False):
    """
    Read and combine all inversions_summary.csv files from
    {parent_dir}/{junction_name}/consensus_*/inversions_summary.csv

    Returns
    -------
    pd.DataFrame
        One long DataFrame with all inversions combined.
    """
    base_dir = os.path.join(parent_dir, junction_name)
    all_dfs = []

    if not os.path.isdir(base_dir):
        print(f"No inversions directory found for {junction_name}, skipping.")
        return pd.DataFrame()

    for subdir in sorted(os.listdir(base_dir)):
        if not subdir.startswith("consensus_"):
            continue

        csv_path = os.path.join(base_dir, subdir, "inversions_summary.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    complete_df = pd.concat(all_dfs, ignore_index=True)

    if save_df:
        complete_df.to_csv(
            os.path.join(base_dir, "all_inversions_summary.csv"),
            index=False,
        )

    return complete_df


def combine_all_junctions_summaries(parent_dir, save_df=False):
    """
    Combine per-junction summary CSVs for insertions, deletions, translocations,
    and inversions across all junctions found under parent_dir.

    Expected directory structure:
        parent_dir/
            insertions/{junction_name}/consensus{n}/insertions_summary.csv
            deletions/{junction_name}/consensus_{n}/deletions_summary.csv
            translocations/{junction_name}/consensus_{n}/translocations_summary.csv
            inversions/{junction_name}/consensus_{n}/inversions_summary.csv

    Parameters
    ----------
    parent_dir : str
        Root results directory, e.g. "../results/atb_lookup".
    save_df : bool
        If True, write one combined CSV per type into the respective type
        directory, e.g. parent_dir/insertions/all_junctions_insertions.csv.

    Returns
    -------
    dict with keys "insertions", "deletions", "translocations", "inversions",
    each mapping to a pd.DataFrame (empty if nothing was found).
    """
    type_cfg = {
        "insertions":            (load_all_insertions_summaries,            "insertions"),
        "ambiguous_insertions":  (load_all_ambiguous_insertions_summaries,  "insertions"),
        "deletions":             (load_all_deletions_summaries,             "deletions"),
        "translocations":        (load_all_translocations_summaries,        "translocations"),
        "inversions":            (load_all_inversions_summaries,            "inversions"),
    }

    results = {}

    for dtype, (loader, subdir_name) in type_cfg.items():
        type_dir = os.path.join(parent_dir, subdir_name)
        all_dfs = []

        if not os.path.isdir(type_dir):
            print(f"No {subdir_name} directory found at {type_dir}, skipping.")
            results[dtype] = pd.DataFrame()
            continue

        for junction_name in sorted(os.listdir(type_dir)):
            junction_dir = os.path.join(type_dir, junction_name)
            if not os.path.isdir(junction_dir):
                continue

            df = loader(type_dir, junction_name, save_df=False)
            if df is not None and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            print(f"No {dtype} found across any junction.")
            results[dtype] = pd.DataFrame()
            continue

        combined = pd.concat(all_dfs, ignore_index=True)

        if save_df:
            out_path = os.path.join(type_dir, f"all_junctions_{dtype}.csv")
            combined.to_csv(out_path, index=False)
            print(f"Saved {dtype} summary to {out_path}")

        results[dtype] = combined

    return results


def deduplicate_events(summaries, save_path=None):
    """
    Deduplicate all event summary DataFrames and combine into one DataFrame.

    For each event type:
      - Deduplicate by (junction_name, path, ctx) so each unique event is
        represented exactly once regardless of which isolate carried it.
      - Drop the isolate column (genome_name) since events are no longer
        tied to a specific isolate after deduplication.
      - Add an 'event_type' column with the event type string.

    No length filtering is applied here — pass the returned DataFrame to
    ``count_events_per_junction`` with a ``min_length`` argument to filter
    without mutating this DataFrame.

    Parameters
    ----------
    summaries : dict
        Output of combine_all_junctions_summaries(), with keys
        "insertions", "deletions", "translocations", "inversions".
    save_path : str or None
        If provided, save the combined deduplicated DataFrame as a CSV.

    Returns
    -------
    pd.DataFrame
        One row per unique event with columns: event_type, junction_name,
        path, ctx, length, ... (genome_name dropped).
    """
    # Maps summaries dict keys (plural, match directory names) to singular event_type labels
    event_type_map = {
        "insertions":           "insertion",
        "ambiguous_insertions": "ambiguous_insertion",
        "deletions":            "deletion",
        "translocations":       "translocation",
        "inversions":           "inversion",
    }
    parts = []

    for dtype, etype_label in event_type_map.items():
        df = summaries.get(dtype)
        if df is None or df.empty:
            continue

        df = df.copy()

        # Deduplicate by (junction_name, path, ctx) or (junction_name, path)
        if "ctx" in df.columns:
            dedup_cols = ["junction_name", "path", "ctx"]
        else:
            dedup_cols = ["junction_name", "path"]

        df = df.drop_duplicates(subset=dedup_cols)

        # Drop isolate column — events are no longer isolate-specific
        df = df.drop(columns=["genome_name"], errors="ignore")

        df = df.reset_index(drop=True)
        df.insert(0, "event_type", etype_label)
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        combined.to_csv(save_path, index=False)
        print(f"Saved deduplicated events to {save_path}")

    return combined


def count_events_per_junction(deduped_df, min_length=200, save_path=None):
    """
    Count the number of unique events per junction for each event type.

    Parameters
    ----------
    deduped_df : pd.DataFrame
        Output of deduplicate_events(). Must have columns 'event_type',
        'junction_name', and optionally 'length'.
    min_length : int or None
        Minimum event length in bp (inclusive). Events below this threshold
        are excluded. The original DataFrame is never modified. Default: 200.
    save_path : str or None
        If provided, the counts DataFrame is written as a CSV to this path.

    Returns
    -------
    pd.DataFrame
        One row per junction_name with columns:
        junction_name, n_insertions, n_deletions, n_translocations, n_inversions
    """
    event_types = ["insertion", "ambiguous_insertion", "deletion", "translocation", "inversion"]

    # Apply length filter on a view — original df is untouched
    if min_length is not None and "length" in deduped_df.columns:
        df = deduped_df[deduped_df["length"] >= min_length]
    else:
        df = deduped_df

    if df.empty:
        empty_cols = (
            ["junction_name"]
            + [f"n_{t}" for t in event_types]
            + ["n_events"]
            + ([f"mean_length_{t}" for t in event_types] if "length" in deduped_df.columns else [])
            + (["mean_length"] if "length" in deduped_df.columns else [])
        )
        return pd.DataFrame(columns=empty_cols)

    # All junctions present in the original (pre-filter) df
    all_junctions = pd.DataFrame({"junction_name": deduped_df["junction_name"].unique()})

    grp = df.groupby(["junction_name", "event_type"])

    # Counts per (junction, event_type)
    counts = grp.size().unstack(fill_value=0).reset_index()
    # Include junctions with zero events after filtering
    counts = all_junctions.merge(counts, on="junction_name", how="left").fillna(0)
    for dtype in event_types:
        if dtype not in counts.columns:
            counts[dtype] = 0
        counts = counts.rename(columns={dtype: f"n_{dtype}"})
    count_cols = [f"n_{t}" for t in event_types]
    counts = counts.reindex(columns=["junction_name"] + count_cols, fill_value=0)

    # Total events per junction (ambiguous_insertions excluded)
    sum_cols = [c for c in count_cols if c != "n_ambiguous_insertion"]
    counts["n_events"] = counts[sum_cols].sum(axis=1)

    # Mean length per (junction, event_type) and overall
    if "length" in df.columns:
        mean_len = grp["length"].mean().unstack().reset_index()
        for dtype in event_types:
            col = f"mean_length_{dtype}"
            if dtype in mean_len.columns:
                mean_len = mean_len.rename(columns={dtype: col})
            else:
                mean_len[col] = float("nan")
        mean_len_cols = [f"mean_length_{t}" for t in event_types]
        mean_len = mean_len.reindex(columns=["junction_name"] + mean_len_cols)

        overall_mean = df.groupby("junction_name")["length"].mean().rename("mean_event_length").reset_index()

        counts_df = counts.merge(mean_len, on="junction_name", how="left")
        counts_df = counts_df.merge(overall_mean, on="junction_name", how="left")
    else:
        counts_df = counts

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        counts_df.to_csv(save_path, index=False)
        print(f"Saved event counts to {save_path}")

    return counts_df


def collect_hits_info_counts(search_root: str) -> pd.DataFrame:
    """
    Iterate over all *.lexicmap.tsv files under `search_root` and collect
    the number of hits (lines excluding header) and number of genomes per file.

    Parses junction_name, consensus, genome_name, and segment from the path:
      .../insertions/{junction_name}/{consensus}/{genome_name}_segment_{n}.lexicmap.tsv

    Returns a DataFrame with columns:
        junction_name, consensus, genome_name, segment, n_hits, n_genomes, file_path
    """
    rows = []
    for p in sorted(Path(search_root).rglob("*.lexicmap.tsv")):
        # path structure: .../insertions/<junction>/<consensus>/<genome>_segment_<n>.hits_info.tsv
        parts = p.parts
        try:
            ins_idx = [i for i, x in enumerate(parts) if x == "insertions"][-1]
            junction_name = parts[ins_idx + 1]
            consensus = parts[ins_idx + 2]
        except (IndexError, ValueError):
            junction_name = consensus = None

        stem = p.name.replace(".lexicmap.tsv", "")   # e.g. NZ_AP022044.1_segment_1
        seg_match = re.search(r"_segment_(\d+)$", stem)
        if seg_match:
            segment = f"segment_{seg_match.group(1)}"
            genome_name = stem[: seg_match.start()]
        else:
            segment = None
            genome_name = stem

        n_hits = 0
        n_genomes = None
        with open(p) as f:
            header = f.readline()
            if header:
                try:
                    hits_col = header.strip().split("\t").index("hits")
                except ValueError:
                    hits_col = None
                for line in f:
                    n_hits += 1
                    if n_genomes is None and hits_col is not None:
                        fields = line.strip().split("\t")
                        if len(fields) > hits_col:
                            try:
                                n_genomes = int(fields[hits_col])
                            except ValueError:
                                pass

        rows.append(dict(
            junction_name=junction_name,
            consensus=consensus,
            genome_name=genome_name,
            segment=segment,
            n_hits=n_hits,
            n_genomes=n_genomes,
            file_path=str(p),
        ))

    return pd.DataFrame(rows)
