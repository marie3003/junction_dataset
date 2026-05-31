from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from Bio import Phylo

import os
import subprocess
import re
import shutil

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pathlib import Path

from junction_analysis.helpers import get_isolate_sequence, get_consensus_seq_from_alignment, read_gff3_annotations
from junction_analysis.consensus import build_subtree
from junction_analysis import pangraph_utils as pu
from junction_analysis.junction_trees import build_tree_from_block_list

def detect_event_blocks(deduplicated_paths, consensus_path):  # noqa: C901
    """
    Detect insertions, deletions, and inversions block by block for each
    isolate vs. the consensus path.

    Uses the same anchor/context matching as get_insertions_deletions_v2 to
    identify which (block_id, context) pairs correspond across paths.

    Each consensus block at position i gets a position-unique ID
    ``"{block.id}_{i}"`` used as context anchor.

    Context assignment per event type
    ----------------------------------
    deletions / inversions
        Context = uid of the consensus block immediately to the left.
    insertions
        Context is derived from the nearest non-inserted block to the left
        in the isolate path:
          - Left block is *matched*: use its consensus position, then walk
            right over any contiguous deleted blocks; use the furthest
            deleted block's uid (or the matched block's uid if none).
          - Left block is *inverted*, right neighbour also inverted
            (within the inversion): use the right inverted block's
            consensus position (inversion flips ordering) and apply the
            same rightward deletion scan.
          - Left block is *inverted*, right neighbour matched or absent
            (at the end of the inversion): walk further left through the
            inversion run to find its first block, then apply the
            rightward deletion scan.

    Annotated path outputs
    ----------------------
    Every node in both the isolate and consensus paths is represented as a
    ``DeduplicatedNode(id, strand, context, nid)`` and collected into
    ``Path`` objects.  For isolate nodes, the context encodes position
    relative to the consensus (matched blocks get the uid of the consensus
    block to their left; event blocks get the context computed above).
    These annotated paths are the primary input to ``merge_event_blocks``.

    Parameters
    ----------
    deduplicated_paths : dict[str, pu.Path]
    consensus_path     : pu.Path

    Returns
    -------
    events_df : pd.DataFrame
        Columns: event_type, isolate, block_id, context, nid.
        One row per event block per isolate.
    iso_annotated_paths : dict[str, pu.Path]
        Per-isolate ``Path`` of ``DeduplicatedNode`` covering every block
        in the isolate path (matched, inverted, and inserted).
    consensus_annotated : pu.Path
        ``Path`` of ``DeduplicatedNode`` for the consensus, with each
        node's context set to the uid of its left neighbour.
    """
    c_nodes = list(consensus_path.nodes)
    # unique position IDs for every consensus slot
    c_uid = [f"{n.id}_{i}" for i, n in enumerate(c_nodes)]

    # annotated consensus path — each block as DeduplicatedNode with left-neighbour as context
    consensus_annotated = pu.Path([
        pu.DeduplicatedNode(n.id, n.strand, c_uid[j - 1] if j > 0 else "", n.nid)
        for j, n in enumerate(c_nodes)
    ])

    rows = []
    iso_annotated_paths = {}   # isolate -> Path of DeduplicatedNode

    for isolate, path in deduplicated_paths.items():
        iso_nodes = list(path.nodes)

        # ------------------------------------------------------------------ #
        # Anchor / context assignment (identical to get_insertions_deletions_v2)
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

        def _detect_translocations_iterative(initial_anchor_ids):
            translocated_ids = set()
            excluded = set()
            while True:
                anchor_ids = initial_anchor_ids - excluded
                c_ctxs_tmp   = _assign_ctx(c_nodes,   anchor_ids)
                iso_ctxs_tmp = _assign_ctx(iso_nodes, anchor_ids)
                c_ic   = {(n.id, ctx): n.strand for n, ctx in zip(c_nodes,   c_ctxs_tmp)}
                iso_ic = {(n.id, ctx): n.strand for n, ctx in zip(iso_nodes, iso_ctxs_tmp)}
                matched_tmp = {k for k in c_ic if k in iso_ic}
                c_ctx_by_id = defaultdict(list)
                for n, ctx in zip(c_nodes, c_ctxs_tmp):
                    if (n.id, ctx) not in matched_tmp:
                        c_ctx_by_id[n.id].append(ctx)
                new_id = None
                for n, ctx in zip(iso_nodes, iso_ctxs_tmp):
                    if (n.id, ctx) in matched_tmp:
                        continue
                    if c_ctx_by_id.get(n.id) and n.id not in translocated_ids:
                        c_ctx_by_id[n.id].pop(0)
                        new_id = n.id
                        break
                if new_id is None:
                    break
                translocated_ids.add(new_id)
                excluded.add(new_id)
            return translocated_ids

        preliminary_trans_ids = _detect_translocations_iterative(_candidate_anchor_ids())
        anchor_ids = _candidate_anchor_ids(excluded=preliminary_trans_ids)
        c_ctxs   = _assign_ctx(c_nodes,   anchor_ids)
        iso_ctxs = _assign_ctx(iso_nodes, anchor_ids)

        c_key_to_strand   = {(n.id, ctx): n.strand for n, ctx in zip(c_nodes,   c_ctxs)}
        iso_key_to_strand = {(n.id, ctx): n.strand for n, ctx in zip(iso_nodes, iso_ctxs)}
        iso_key_to_nid    = {(n.id, ctx): n.nid    for n, ctx in zip(iso_nodes, iso_ctxs)}
        c_key_to_pos      = {(n.id, ctx): j        for j, (n, ctx) in enumerate(zip(c_nodes, c_ctxs))}

        # ------------------------------------------------------------------ #
        # Pass 1: walk consensus → deletions and inversions
        # ------------------------------------------------------------------ #
        inverted_keys         = set()   # (id, ctx) keys that are inverted in isolate
        deleted_c_positions   = set()   # consensus positions that are deleted in isolate
        inv_key_to_ctx        = {}      # (id, ctx) -> left_ctx for inverted keys
        predecessor_of_deleted = defaultdict(list)  # predecessor_key -> [c_pos j, ...]
        leading_deleted        = []     # deleted positions with no non-deleted predecessor
        last_non_del_key       = None   # most recent matched or inverted key

        for j, (n, ctx) in enumerate(zip(c_nodes, c_ctxs)):
            key       = (n.id, ctx)
            left_ctx  = c_uid[j - 1] if j > 0 else ""
            iso_strand = iso_key_to_strand.get(key)

            # doesn't exist in isolate path means lookup gives None
            if iso_strand is None:
                deleted_c_positions.add(j)
                rows.append({"event_type": "deletion",  "isolate": isolate, "block_id": n.id, "strand": n.strand, "context": left_ctx, "nid": pd.NA})
                if last_non_del_key is not None:
                    predecessor_of_deleted[last_non_del_key].append(j)
                else:
                    leading_deleted.append(j)
            elif iso_strand != n.strand:
                inverted_keys.add(key)
                inv_key_to_ctx[key] = left_ctx
                rows.append({"event_type": "inversion", "isolate": isolate, "block_id": n.id, "strand": n.strand, "context": left_ctx, "nid": iso_key_to_nid.get(key)})
                last_non_del_key = key
            else:
                last_non_del_key = key

        # ------------------------------------------------------------------ #
        # Pass 2: walk isolate → build DeduplicatedNode path + insertions   #
        # Classify each node as matched / inverted / inserted, compute its  #
        # context, and emit insertion rows on the fly.                       #
        # ------------------------------------------------------------------ #
        def _ctx_from_c_pos(c_pos):
            """Consensus uid at c_pos, walking right over contiguous deletions."""
            furthest = None
            for p in range(c_pos + 1, len(c_nodes)):
                if p in deleted_c_positions:
                    furthest = p
                else:
                    break
            return c_uid[furthest] if furthest is not None else c_uid[c_pos]

        # first pass: classify every node so left/right neighbour lookups work
        iso_status = []
        for n, ctx in zip(iso_nodes, iso_ctxs):
            key = (n.id, ctx)
            if key in inverted_keys:
                iso_status.append("inversion")
            elif key in c_key_to_strand and c_key_to_strand[key] == n.strand:
                iso_status.append("matched")
            else:
                iso_status.append("insertion")

        def _del_node(j):
            """Build a deleted DeduplicatedNode from consensus position j."""
            cn = c_nodes[j]
            left_ctx = c_uid[j - 1] if j > 0 else ""
            return pu.DeduplicatedNode(cn.id, cn.strand, left_ctx, cn.nid, type="deletion")

        # prepend any deleted blocks that have no non-deleted predecessor
        dedup_nodes = [_del_node(j) for j in leading_deleted]

        for i, (n, ctx) in enumerate(zip(iso_nodes, iso_ctxs)):
            key       = (n.id, ctx)
            node_type = iso_status[i]

            if node_type == "inversion":
                node_ctx = inv_key_to_ctx[key]
                # deleted blocks whose predecessor is this inverted block go BEFORE it
                pre_deleted = [_del_node(j) for j in predecessor_of_deleted.get(key, [])]
                dedup_nodes.extend(pre_deleted)
                dedup_nodes.append(pu.DeduplicatedNode(n.id, n.strand, node_ctx, n.nid, type=node_type))

            elif node_type == "insertion":
                left_idx  = next((k for k in range(i - 1, -1,            -1) if iso_status[k] != "insertion"), None)
                right_idx = next((k for k in range(i + 1, len(iso_nodes))    if iso_status[k] != "insertion"), None)

                if left_idx is None:
                    node_ctx = ""
                elif iso_status[left_idx] == "matched":
                    c_pos_   = c_key_to_pos.get((iso_nodes[left_idx].id, iso_ctxs[left_idx]))
                    node_ctx = _ctx_from_c_pos(c_pos_) if c_pos_ is not None else ""
                else:
                    # left is inverted
                    if right_idx is not None and iso_status[right_idx] == "inversion":
                        c_pos_   = c_key_to_pos.get((iso_nodes[right_idx].id, iso_ctxs[right_idx]))
                        node_ctx = _ctx_from_c_pos(c_pos_) if c_pos_ is not None else ""
                    else:
                        first_inv = left_idx
                        for k in range(left_idx - 1, -1, -1):
                            if iso_status[k] == "inversion":
                                first_inv = k
                            elif iso_status[k] == "matched":
                                break
                        c_pos_   = c_key_to_pos.get((iso_nodes[first_inv].id, iso_ctxs[first_inv]))
                        node_ctx = _ctx_from_c_pos(c_pos_) if c_pos_ is not None else ""

                rows.append({"event_type": "insertion", "isolate": isolate, "block_id": n.id, "strand": n.strand, "context": node_ctx, "nid": n.nid})
                dedup_nodes.append(pu.DeduplicatedNode(n.id, n.strand, node_ctx, n.nid, type=node_type))

            else:  # matched
                c_pos_   = c_key_to_pos.get(key)
                node_ctx = c_uid[c_pos_ - 1] if (c_pos_ is not None and c_pos_ > 0) else ""
                dedup_nodes.append(pu.DeduplicatedNode(n.id, n.strand, node_ctx, n.nid, type=node_type))
                # deleted blocks whose predecessor is this matched block go AFTER it
                post_deleted = [_del_node(j) for j in predecessor_of_deleted.get(key, [])]
                dedup_nodes.extend(post_deleted)

        iso_annotated_paths[isolate] = pu.Path(dedup_nodes)

    cols = ["event_type", "isolate", "block_id", "strand", "context", "nid"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return df, iso_annotated_paths, consensus_annotated


def group_events_by_clade(events_df, iso_annotated_paths, isolate_list,
                          tree_path="../config/polished_tree.nwk",
                          method="fitch"):
    """
    Group detected events into clades according to the phylogeny, and annotate
    the isolate paths with the resolved ``parent_clade_id`` for every
    non-matched node.

    For each unique (event_type, block_id, context) combination, the isolates
    carrying the event are mapped onto the subtree and clades are identified
    using one of two methods:

    method="maximal_clades"
        Strict: finds maximally connected subtrees where ALL tips carry the
        event. Any absent tip breaks the clade into smaller pieces. Can
        over-split when losses occur within an otherwise positive clade.

    method="fitch" (default)
        Runs binary Fitch parsimony (0=absent, 1=present) with a bottom-up
        pass to assign state sets and a top-down pass to resolve ambiguous
        nodes. Each 0→1 gain edge defines one independent event acquisition;
        the clade below that edge (restricted to tips that actually carry the
        event) is returned as one row. Allows losses within a gain clade, so
        it merges clades that would be split by the maximal_clades approach.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of detect_event_blocks().
    iso_annotated_paths : dict[str, pu.Path]
        Output of detect_event_blocks(). Nodes will be updated in-place with
        their resolved ``parent_clade_id``.
    isolate_list : list[str]
        All isolates in the cluster (used to build the subtree).
    tree_path : str
        Path to the full Newick phylogeny.
    method : {"fitch", "maximal_clades"}

    Returns
    -------
    clade_df : pd.DataFrame
        Columns: event_type, block_id, strand, context, clade_isolates,
        clade_branch_length, parent_clade_id.
        One row per independent clade/gain per event.
    iso_annotated_paths : dict[str, pu.Path]
        Same dict as the input, with ``parent_clade_id`` set on every
        inserted, deleted, and inverted node.
    """
    tree = Phylo.read(tree_path, "newick")
    subtree = build_subtree(tree, isolate_list)

    def _total_branch_length(clade):
        return sum(
            c.branch_length for c in clade.find_clades()
            if c.branch_length is not None
        )

    # assign a unique integer index to every internal node (preorder)
    node_index = {}
    idx = 0
    for clade in subtree.find_clades(order="preorder"):
        if not clade.is_terminal():
            node_index[clade] = idx
            idx += 1

    # build parent map once, shared by both methods
    parent_map = {}
    for clade in subtree.find_clades(order="preorder"):
        for child in clade.clades:
            parent_map[child] = clade

    def _maximal_clades(event_isolates):
        event_set = set(event_isolates)
        result = []

        def _walk(clade):
            if all(tip.name in event_set for tip in clade.get_terminals()):
                tips = [tip.name for tip in clade.get_terminals()]
                result.append((
                    tips,
                    _total_branch_length(clade),
                    clade.name,
                ))
            else:
                for child in clade.clades:
                    _walk(child)

        _walk(subtree.root)
        return result

    def _fitch_clades(event_isolates):
        event_set = set(event_isolates)

        # bottom-up: assign Fitch state sets
        state = {}
        for clade in subtree.find_clades(order="postorder"):
            if clade.is_terminal():
                state[clade] = {1} if clade.name in event_set else {0}
            else:
                child_sets = [state[c] for c in clade.clades]
                inter = set.intersection(*child_sets)
                state[clade] = inter if inter else set.union(*child_sets)

        # precompute fraction of event-carrying tips below each node
        tip_fraction = {}
        for clade in subtree.find_clades(order="postorder"):
            tips = clade.get_terminals()
            tip_fraction[clade] = sum(1 for t in tips if t.name in event_set) / len(tips)

        # top-down: resolve ambiguous nodes
        # unambiguous nodes take their Fitch state; ambiguous nodes resolve
        # towards 1 (present) if the majority of tips below carry the event,
        # otherwise follow the parent state.

        resolved = {subtree.root: 0 if 0 in state[subtree.root] else 1}
        for clade in subtree.find_clades(order="preorder"):
            if clade is subtree.root:
                continue
            if len(state[clade]) == 1:
                resolved[clade] = next(iter(state[clade]))  # unambiguous
            else:
                # ambiguous: majority vote on tips below
                resolved[clade] = 1 if tip_fraction[clade] > 0.5 else resolved[parent_map[clade]]

        # collect all gain edges (parent=0 → child=1)
        gain_clades = [
            clade for clade in subtree.find_clades(order="preorder")
            if clade is not subtree.root
            and resolved[parent_map[clade]] == 0
            and resolved[clade] == 1
        ]
        gain_clade_set = {id(c) for c in gain_clades}

        def _exclusive_tips(clade, top=True):
            """Tips under clade that are not under any nested gain clade."""
            if not top and id(clade) in gain_clade_set:
                return []
            if clade.is_terminal():
                return [clade.name] if clade.name in event_set else []
            tips = []
            for child in clade.clades:
                tips.extend(_exclusive_tips(child, top=False))
            return tips

        result = []
        for clade in gain_clades:
            tips = _exclusive_tips(clade)
            if tips:
                mrca = subtree.common_ancestor(tips)
                result.append((tips, _total_branch_length(clade), mrca.name))

        return result

    _group_fn = _fitch_clades if method == "fitch" else _maximal_clades

    rows = []
    # (event_type, block_id, strand, context, isolate) -> parent_clade_id
    evt_to_clade = {}

    for (event_type, block_id, strand, context), group in events_df.groupby(
        ["event_type", "block_id", "strand", "context"]
    ):
        event_isolates = group["isolate"].tolist()
        for clade_isolates, clade_bl, parent_id in _group_fn(event_isolates):
            rows.append({
                "event_type":          event_type,
                "block_id":            block_id,
                "strand":              strand,
                "context":             context,
                "clade_isolates":      clade_isolates,
                "clade_branch_length": clade_bl,
                "parent_clade_id":     parent_id,
            })
            for iso in clade_isolates:
                evt_to_clade[(event_type, block_id, strand, context, iso)] = parent_id

    clade_df = pd.DataFrame(rows, columns=["event_type", "block_id", "strand", "context", "clade_isolates", "clade_branch_length", "parent_clade_id"])

    # annotate iso_annotated_paths in-place
    for isolate, path in iso_annotated_paths.items():
        for dn in path.nodes:
            if dn.type in ("insertion", "deletion", "inversion"):
                dn.parent_clade_id = evt_to_clade.get((dn.type, dn.id, dn.strand, dn.context, isolate))

    return clade_df, iso_annotated_paths


def find_combined_events(  # noqa: C901
    iso_annotated_paths,
    consensus_annotated,
    tree_path,
    isolate_list,
):
    """
    Walk each annotated isolate path and combine adjacent per-block events of
    the same type and ``parent_clade_id`` into longer events.

    Requires that ``group_events_by_clade`` has already been called so that
    every non-matched node in ``iso_annotated_paths`` carries a
    ``parent_clade_id``.

    The tree is read to resolve ancestor/descendant relationships between
    ``parent_clade_id`` values, which governs which nodes may be skipped over
    without interrupting a running event group.

    Combination rules
    -----------------
    deletions  (walk isolate path left-to-right; deleted nodes are consecutive
                in the consensus because ``detect_event_blocks`` places them
                after/before their predecessor)
        Continue : same ``parent_clade_id`` and type "deletion".
        Jump     : a deleted node whose ``parent_clade_id`` is an *ancestor* of
                   the current group's clade (higher in the tree, i.e. the
                   current deletion is a refinement of a broader deletion).
        Interrupt: any matched, inverted, or inserted node.

    inversions  (walk isolate path left-to-right)
        Continue : same ``parent_clade_id`` and type "inversion".
        Jump     : deleted or inserted nodes (regardless of clade).
        Interrupt: any matched node.

    insertions  (walk isolate path left-to-right)
        Continue : same ``parent_clade_id`` and type "insertion".
        Jump     : inserted nodes whose ``parent_clade_id`` is a *descendant*
                   of the current group's clade (closer to the tips).
        Interrupt: matched or inverted nodes, or an inserted node whose
                   ``parent_clade_id`` is an ancestor of the current group's
                   clade.

    Parameters
    ----------
    iso_annotated_paths : dict[str, pu.Path]
        Output of ``group_events_by_clade`` — every node carries ``.type``
        and, for event nodes, ``.parent_clade_id``.
    consensus_annotated : pu.Path
        Output of ``detect_event_blocks`` (used only to look up consensus
        positions for deletion ordering).
    tree_path : str
        Path to the full Newick phylogeny.
    isolate_list : list[str]
        All isolates in the cluster (used to build the subtree).

    Returns
    -------
    insertions : dict[str, list[pu.Path]]
    deletions  : dict[str, list[pu.Path]]
    inversions : dict[str, list[pu.Path]]
    """
    # ---------------------------------------------------------------------- #
    # Build ancestor / descendant helpers from the subtree                   #
    # ---------------------------------------------------------------------- #
    tree = Phylo.read(tree_path, "newick")
    subtree = build_subtree(tree, isolate_list)

    parent_name = {
        child.name: clade.name
        for clade in subtree.find_clades(order="preorder")
        for child in clade.clades
        if child.name
    }
    ancestor_map = {}   # node_name -> frozenset of all ancestor names (inclusive)
    for clade in subtree.find_clades():
        if not clade.name:
            continue
        ancs, cur = set(), clade.name
        while cur:
            ancs.add(cur)
            cur = parent_name.get(cur)
        ancestor_map[clade.name] = frozenset(ancs)

    def is_ancestor(a, b):
        """True if a is an ancestor of (or equal to) b."""
        return bool(a and b and a in ancestor_map.get(b, frozenset()))

    def is_descendant(a, b):
        """True if a is a descendant of (or equal to) b."""
        return bool(a and b and b in ancestor_map.get(a, frozenset()))

    # ---------------------------------------------------------------------- #
    # Walk each isolate path                                                  #
    # ---------------------------------------------------------------------- #
    insertions: dict = {}
    deletions:  dict = {}
    inversions: dict = {}

    for isolate, iso_path in iso_annotated_paths.items():
        nodes = iso_path.nodes

        def _flush(open_groups, result):
            """Emit all open groups into result and clear them."""
            for cl, grp in open_groups.items():
                if grp:
                    result.setdefault(isolate, []).append(pu.Path(grp))
            open_groups.clear()

        # ---- deletions --------------------------------------------------- #
        # open_del: clade_id -> list[dn]  (one open group per active clade)
        # A node with an ancestor clade is a "jump" for the current group but
        # continues its own open group — so we keep multiple groups alive.
        # Any matched/inverted/inserted node interrupts ALL open groups.
        open_del: dict = {}
        for dn in nodes:
            t = dn.type or "matched"
            if t == "deletion":
                cl = dn.parent_clade_id
                # close any open group whose clade is a descendant of cl
                # (a broader deletion arrived — the narrower one is done)
                to_close = [c for c in open_del if is_descendant(c, cl) and c != cl]
                for c in to_close:
                    deletions.setdefault(isolate, []).append(pu.Path(open_del.pop(c)))
                open_del.setdefault(cl, []).append(dn)
            else:
                # matched / inverted / inserted — interrupt all
                _flush(open_del, deletions)
        _flush(open_del, deletions)

        # ---- inversions -------------------------------------------------- #
        # deleted and inserted nodes are jumped: they don't belong to the
        # inversion group but don't break it either.  They are handled by
        # their own deletion/insertion passes above/below.
        # Multiple simultaneous inversion groups at different clades are
        # kept alive in open_inv.
        open_inv: dict = {}
        for dn in nodes:
            t = dn.type or "matched"
            if t == "inversion":
                cl = dn.parent_clade_id
                open_inv.setdefault(cl, []).append(dn)
            elif t in ("deletion", "insertion"):
                pass  # jump — each is handled by its own pass
            else:
                # matched — interrupt all open inversion groups
                _flush(open_inv, inversions)
        _flush(open_inv, inversions)

        # ---- insertions -------------------------------------------------- #
        # A descendant-clade insertion is a "jump" for the current group but
        # continues its own open group.  An ancestor-clade insertion closes
        # descendant groups and continues/starts the ancestor group.
        # matched / inverted interrupts all.
        #
        # Nested-context rule: when a new lower-level (descendant-clade) group
        # starts, scan back through the current contiguous insertion-only
        # streak (any matched/deleted/inverted node resets the streak).  If a
        # higher-level (ancestor-clade) insertion is found within that streak,
        # every node in the new lower-level group gets its context set to that
        # node's block id — placing the nested insertion relative to its
        # surrounding insertion rather than the original consensus position.
        open_ins: dict = {}
        nested_ctx: dict = {}   # cl -> context string to apply to all nodes in group
        ins_streak: list = []   # insertion nodes seen since last non-insertion node
        for dn in nodes:
            t = dn.type or "matched"
            if t == "insertion":
                cl = dn.parent_clade_id
                # close any open group whose clade is a descendant of cl
                to_close = [c for c in open_ins if is_descendant(c, cl) and c != cl]
                for c in to_close:
                    insertions.setdefault(isolate, []).append(pu.Path(open_ins.pop(c)))
                    nested_ctx.pop(c, None)
                # when a new group starts, look back through the unbroken
                # insertion streak for the last ancestor-clade node
                if cl not in open_ins:
                    last_higher = next(
                        (prev for prev in reversed(ins_streak)
                         if prev.parent_clade_id and prev.parent_clade_id != cl
                         and is_ancestor(prev.parent_clade_id, cl)),
                        None,
                    )
                    if last_higher is not None:
                        nested_ctx[cl] = last_higher.id
                # apply nested context to every node in a nested group
                if cl in nested_ctx:
                    dn.context = nested_ctx[cl]
                open_ins.setdefault(cl, []).append(dn)
                ins_streak.append(dn)
            elif t == "deletion":
                ins_streak = []  # deletion breaks the contiguous insertion streak
            else:
                # matched / inverted — interrupt all
                _flush(open_ins, insertions)
                nested_ctx.clear()
                ins_streak = []
        _flush(open_ins, insertions)

    return insertions, deletions, inversions


def get_insertions_deletions_v2(deduplicated_paths, consensus_path):  # noqa: C901
    """Identify insertions, deletions, and inversions.

    For each (consensus, isolate) pair, context is recomputed locally:
    each block gets the id of the nearest anchor block to its left as context.
    Anchors are block ids that appear exactly once in both paths with the same
    strand. Translocated block ids are excluded from anchors so their context
    differs, causing them to be detected as insertions (isolate side) and
    deletions (consensus side) rather than as a separate event type.

    Detection (three passes):
      Pass 1 (isolate):   insertions (including translocated blocks)
      Pass 2 (consensus): deletions (including consensus positions of translocated blocks)
      Pass 3 (isolate):   inversions

    Arguments:
        deduplicated_paths: dict of isolate -> Path object
        consensus_path: Path object representing the consensus path

    Returns:
        insertions:          dict of isolate -> list of dicts {"path": pu.Path, "ctx": ...}
        ambiguous_insertions: dict of isolate -> list of dicts
        deletions:           dict of isolate -> list of dicts {"path": pu.Path, "left_nid": ...}
        inversions:          dict of isolate -> list of dicts {"path": pu.Path, "ctx": ...}
        context_stats:       list of per-isolate stat dicts
    """
    insertions            = {}
    ambiguous_insertions  = {}
    deletions             = {}
    inversions            = {}
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
        # Pass 1: walk isolate — insertions                                  #
        # Translocated blocks are excluded from anchors (above) so their    #
        # context differs from the consensus; they are detected as           #
        # insertions here and their consensus counterpart as a deletion      #
        # in Pass 2.                                                         #
        # ------------------------------------------------------------------ #
        current_insertion     = []
        current_insertion_first_ctx = None
        current_ins_has_dup_ambiguous = False
        non_consensus_id_ctxs = set()

        n_insertions_isolate = 0
        n_ambiguous_ins_isolate = 0

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

        for n, ctx in zip(iso_nodes, iso_ctxs):
            if (n.id, ctx) in matched:
                flush_ins()
                continue

            matched.add((n.id, ctx))

            if not current_insertion:
                current_insertion_first_ctx = ctx

            # ambiguous if this block exists in the consensus under the same anchor
            # region (same base context, different count) — likely a duplication of
            # a consensus block rather than a genuine insertion
            if _base_ctx(ctx) in c_id_to_base_ctxs.get(n.id, set()):
                current_ins_has_dup_ambiguous = True

            current_insertion.append(n)
            non_consensus_id_ctxs.add((n.id, ctx))

        flush_ins()

        context_stats.append(dict(
            isolate=isolate,
            n_blocks=iso_n_blocks,
            n_duplicated_blocks=iso_n_duplicated,
            n_ambiguous_blocks=iso_n_ambiguous,
            n_insertions=n_insertions_isolate,
            n_ambiguous_insertions=n_ambiguous_ins_isolate,
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

        for n, ctx in zip(iso_nodes, iso_ctxs):
            if (n.id, ctx) in non_consensus_id_ctxs:
                # Part of an insertion — skip without flushing
                # so it does not interrupt a surrounding inversion.
                continue

            c_strand = c_id_ctx_strand.get((n.id, ctx))

            if c_strand is not None and n.strand != c_strand:
                if not current_inversion:
                    current_inversion_first_ctx = ctx

                current_inversion.append(n)
            else:
                flush_inv()

        flush_inv()

    return insertions, ambiguous_insertions, deletions, inversions, context_stats


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
        for idx, inserted_path in enumerate(inserted_paths):
            parent_clade_id = inserted_path.nodes[0].parent_clade_id if inserted_path.nodes else None
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
                    "parent_clade_id": parent_clade_id,
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


def print_insertions_deletions(insertions, deletions, inversions=None):
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

def get_insertions_deletions_from_consensus(assignment_df, consensus_paths, deduplicated_paths, consensus=1, verbose=True, junction_name=None):
    # get isolates belonging to this consensus
    isolates_1 = assignment_df[assignment_df['best_consensus'] == f"consensus_{consensus}"].index.tolist()
    # only keep deduplicated paths for these isolates
    deduplicated_paths = {iso: path for iso, path in deduplicated_paths.items() if iso in isolates_1}

    # compare deduplicated paths to consensus paths to find deviations
    insertions, ambiguous_insertions, deletions, inversions, context_stats = get_insertions_deletions_v2(deduplicated_paths, consensus_paths[consensus - 1])

    # annotate stats with junction name and consensus id
    for row in context_stats:
        row["junction_name"] = junction_name
        row["consensus"] = consensus

    context_stats_df = pd.DataFrame(context_stats, columns=["junction_name", "consensus", "isolate", "n_blocks", "n_duplicated_blocks", "n_ambiguous_blocks", "n_insertions", "n_ambiguous_insertions"])

    # Print results
    if verbose:
        print_insertions_deletions(insertions, deletions, inversions)

    return insertions, ambiguous_insertions, deletions, inversions, context_stats_df

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


def find_insertion_hits_own_genome(genome_root, insertions_seq_dir, rerun_minimap=True):
    """
    Look up insertion sequences in the isolate's own chromosome using minimap2 (asm5 preset).

    For each .fasta file under insertions_seq_dir, the isolate name is extracted from the
    FASTA header and the insertion is mapped to the corresponding genome FASTA in genome_root.

    Hit counting
    ------------
    A hit is counted if the query is sufficiently covered by low-divergence alignments
    (dv < 1%). Sufficiency is defined as: uncovered query bases <= max(50, 10% of query
    length). This relaxed threshold handles short insertions (200-250 bp) where block
    boundary artefacts prevent perfect end-to-end alignment.

    Insertions spanning the linearisation point of a circular chromosome map as two
    complementary alignments. These are detected and counted as a single hit when:
      - Both alignments have dv < 1%
      - Their target intervals are ≤50 bp apart on the circular genome
      - Their combined query coverage is ≤100% (no spurious overlap)
      - The combined query coverage meets the sufficiency threshold above
      - The larger of the two individual coverages is ≥50%
    Full single-hit alignments are matched first; only the remaining partial alignments
    enter the pairing step. Pairs are selected greedily in order of combined coverage
    closest to 100%.

    Parameters
    ----------
    genome_root : str
        Directory containing one FASTA per isolate, named {isolate_name}.fasta.
    insertions_seq_dir : str
        Directory with insertion FASTA files, structured as
        {junction_name}/consensus{n}/{isolate_name}_segment_{k}.fasta.
    rerun_minimap : bool
        If True (default), always run minimap2. If False, skip minimap2 when the
        output PAF file already exists.

    Returns
    -------
    pd.DataFrame
        One row per insertion segment with columns: junction_name, consensus,
        genome_name, insertion_path, insertion_length, segment, hits_in_genome.
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
            if rerun_minimap or not os.path.exists(paf_path):
                subprocess.run(
                    [
                        "minimap2", "-x", "asm5", "-N", "50", "-p", "0.9", "-k", "19", "--eqx",
                        genome_fasta,         # target / reference
                        insertion_seq_path    # query
                    ],
                    stdout=open(paf_path, "w")
                )

            # Count hits with <1% divergence and >= 90% query coverage.
            # An insertion spanning the linearisation point of a circular chromosome
            # produces two complementary alignments. We pair such alignments when:
            #   - both have dv < 1%
            #   - their target intervals are ≤50 bp apart on the circular genome
            #   - their combined query coverage is in [90%, 100%]
            #   - the larger of the two individual coverages is ≥50%
            # Pairs are selected greedily in order of combined coverage closest to
            # 100%. Remaining unmatched alignments are counted if single coverage ≥90%.

            _target_alns = defaultdict(list)
            with open(paf_path) as paf_file:
                for line in paf_file:
                    if "\tdv:f:" not in line:
                        continue
                    fields = line.split("\t")
                    dv = float(line.split("dv:f:")[1].split("\t")[0])
                    if dv >= 0.01:
                        continue
                    qlen  = int(fields[1])
                    qs, qe = int(fields[2]), int(fields[3])
                    tname = fields[5]
                    tlen  = int(fields[6])
                    ts, te = int(fields[7]), int(fields[8])
                    _target_alns[tname].append({"qs": qs, "qe": qe, "qlen": qlen,
                                                "ts": ts, "te": te, "tlen": tlen})

            count = 0
            for tname, alns in _target_alns.items():
                qlen   = alns[0]["qlen"]
                tlen   = alns[0]["tlen"]
                for a in alns:
                    a["cov"] = (a["qe"] - a["qs"]) / qlen

                # A hit is sufficient if uncovered bases <= max(50, 10% of query length)
                max_uncovered = max(50, 0.1 * qlen)

                # Mark full single hits immediately; only partial hits need pairing
                matched = [(qlen - (a["qe"] - a["qs"])) <= max_uncovered for a in alns]
                count += sum(matched)

                # Find candidate pairs among the remaining partial hits
                partial = [i for i, m in enumerate(matched) if not m]
                pairs = []
                for ii in range(len(partial)):
                    for jj in range(ii + 1, len(partial)):
                        i, j = partial[ii], partial[jj]
                        a, b = alns[i], alns[j]
                        # Circular gap between the two target intervals
                        target_gap = min((b["ts"] - a["te"]) % tlen,
                                         (a["ts"] - b["te"]) % tlen)
                        if target_gap > 50:
                            continue
                        # Combined query coverage (subtract any overlap)
                        overlap = max(0, min(a["qe"], b["qe"]) - max(a["qs"], b["qs"]))
                        combined_bases = a["qe"] - a["qs"] + b["qe"] - b["qs"] - overlap
                        combined_cov = combined_bases / qlen
                        if combined_cov > 1.0 or (qlen - combined_bases) > max_uncovered:
                            continue
                        if max(a["cov"], b["cov"]) < 0.5:
                            continue
                        pairs.append((abs(combined_cov - 1.0), i, j))

                # Greedy assignment: best pairs (closest to 100%) first
                pairs.sort(key=lambda x: x[0])
                for _, i, j in pairs:
                    if not matched[i] and not matched[j]:
                        matched[i] = matched[j] = True
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


def find_insertion_hits_in_plasmids(plasmid_fasta_root, insertions_seq_dir, rerun_minimap=True):
    """
    Look up insertion sequences in all plasmids belonging to the same isolate using
    minimap2 (asm5 preset). A separate PAF file is written per insertion-plasmid pair.

    Hit counting
    ------------
    Identical logic to find_insertion_hits_own_genome: a hit is counted when the query
    is sufficiently covered by low-divergence alignments (dv < 1%), where sufficiency
    means uncovered query bases <= max(50, 10% of query length).

    Plasmids are circular, so insertions spanning the linearisation point map as two
    complementary alignments. These are detected and counted as a single hit when:
      - Both alignments have dv < 1%
      - Their target intervals are ≤50 bp apart on the circular plasmid
      - Their combined query coverage is ≤100% (no spurious overlap)
      - The combined query coverage meets the sufficiency threshold above
      - The larger of the two individual coverages is ≥50%
    Full single-hit alignments are matched first; only the remaining partial alignments
    enter the pairing step. Pairs are selected greedily in order of combined coverage
    closest to 100%.

    Parameters
    ----------
    plasmid_fasta_root : str
        Directory of plasmid FASTA files, structured as {isolate_name}/{plasmid}.fasta.
    insertions_seq_dir : str
        Directory with insertion FASTA files, structured as
        {junction_name}/consensus{n}/{isolate_name}_segment_{k}.fasta.
    rerun_minimap : bool
        If True (default), always run minimap2. If False, skip minimap2 when the
        output PAF file already exists.

    Returns
    -------
    pd.DataFrame
        One row per insertion-plasmid pair with columns: junction_name, consensus,
        isolate_name, plasmid_name, insertion_path, insertion_length, segment,
        hits_in_plasmid.
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
                if rerun_minimap or not os.path.exists(paf_path):
                    subprocess.run(
                        [
                            "minimap2", "-x", "asm5", "-N", "50", "-p", "0.9", "-k", "19", "--eqx",
                            plasmid_file,
                            insertion_seq_path
                        ],
                        stdout=open(paf_path, "w")
                    )

                # Count hits with <1% divergence and >= 90% query coverage.
                # Plasmids are circular, so an insertion spanning the linearisation
                # point produces two complementary alignments — handled identically
                # to the chromosome case: full single hits are counted first, then
                # partial hits are paired if their target gap is ≤50 bp on the
                # circular plasmid and their combined query coverage is in [90%, 100%].
                _target_alns = defaultdict(list)
                with open(paf_path) as paf_file:
                    for line in paf_file:
                        if not line or "\tdv:f:" not in line:
                            continue
                        fields = line.split("\t")
                        dv = float(line.split("dv:f:")[1].split("\t")[0])
                        if dv >= 0.01:
                            continue
                        qlen = int(fields[1])
                        qs, qe = int(fields[2]), int(fields[3])
                        tname = fields[5]
                        tlen  = int(fields[6])
                        ts, te = int(fields[7]), int(fields[8])
                        _target_alns[tname].append({"qs": qs, "qe": qe, "qlen": qlen,
                                                    "ts": ts, "te": te, "tlen": tlen})

                count = 0
                for tname, alns in _target_alns.items():
                    qlen = alns[0]["qlen"]
                    tlen = alns[0]["tlen"]
                    for a in alns:
                        a["cov"] = (a["qe"] - a["qs"]) / qlen

                    # A hit is sufficient if uncovered bases <= max(50, 10% of query length)
                    max_uncovered = max(50, 0.1 * qlen)

                    # Mark full single hits immediately; only partial hits need pairing
                    matched = [(qlen - (a["qe"] - a["qs"])) <= max_uncovered for a in alns]
                    count += sum(matched)

                    # Find candidate pairs among the remaining partial hits
                    partial = [i for i, m in enumerate(matched) if not m]
                    pairs = []
                    for ii in range(len(partial)):
                        for jj in range(ii + 1, len(partial)):
                            i, j = partial[ii], partial[jj]
                            a, b = alns[i], alns[j]
                            target_gap = min((b["ts"] - a["te"]) % tlen,
                                             (a["ts"] - b["te"]) % tlen)
                            if target_gap > 50:
                                continue
                            overlap = max(0, min(a["qe"], b["qe"]) - max(a["qs"], b["qs"]))
                            combined_bases = a["qe"] - a["qs"] + b["qe"] - b["qs"] - overlap
                            combined_cov = combined_bases / qlen
                            if combined_cov > 1.0 or (qlen - combined_bases) > max_uncovered:
                                continue
                            if max(a["cov"], b["cov"]) < 0.5:
                                continue
                            pairs.append((abs(combined_cov - 1.0), i, j))

                    pairs.sort(key=lambda x: x[0])
                    for _, i, j in pairs:
                        if not matched[i] and not matched[j]:
                            matched[i] = matched[j] = True
                            count += 1
                results.append({
                    "junction_name": os.path.basename(os.path.dirname(dirpath)),
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
            for idx, path in enumerate(entries):
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
        iso_nodes = path_dict[iso].nodes if iso in path_dict else []
        for idx, path in enumerate(entries):
            parent_clade_id = path.nodes[0].parent_clade_id if path.nodes else None

            # find position: end of the node just before the first deleted block
            position = None
            if path.nodes:
                first = path.nodes[0]
                for i, n in enumerate(iso_nodes):
                    if n.id == first.id and n.context == first.context:
                        for j in range(i - 1, -1, -1):
                            pred = iso_nodes[j]
                            if pred.type != "deletion" and pred.nid is not None:
                                position = pangraph.nodes[pred.nid].end
                                break
                        break

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
                    if n.nid is not None
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
                    "strand": "+" if path.nodes[0].strand else "-",
                    "position": position,
                    "parent_clade_id": parent_clade_id,
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
        for idx, inv_path in enumerate(inv_paths):
            nodes = inv_path.nodes
            if not nodes:
                continue

            parent_clade_id = nodes[0].parent_clade_id

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
                "parent_clade_id": parent_clade_id,
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


def annotate_events_with_mges(
    summaries,
    mge_dir="../results/junction_mges",
    min_length=200,
    fill_threshold=0.8,
):
    """
    Annotate structural events (insertions, deletions, translocations, inversions)
    with overlapping MGE annotations from per-junction GFF3 files.

    Annotations are matched per isolate: the GFF3 seqid is compared to the
    genome_name of each event row so that only annotations from the same isolate
    are considered.

    Coverage rules
    --------------
    Non-deletion events (insertions, ambiguous_insertions, translocations,
    inversions) have start_pos/end_pos and use two coverage fractions:
      - ann_cov  = overlap / annotation_length
      - ev_cov   = overlap / event_length

    Feature-specific thresholds (all use fill_threshold, default 0.8):
      Prophage      : ann_cov >= threshold AND ev_cov >= threshold → n_prophage
                      ann_cov >= threshold only                    → n_prophage_associated
      IS            : ann_cov >= threshold AND ev_cov >= threshold → n_is
                      ann_cov >= threshold only                    → n_is_associated
      Integron/CALIN: ann_cov >= threshold                        → n_integron
      Defense system: ann_cov >= threshold                        → n_defense_system

    Deletions only have a single position; an annotation is counted if that
    position falls within or on the border of the annotation (no coverage
    threshold applied).

    Columns added per event row
    ---------------------------
        n_prophage            : prophage annotations (full overlap)
        n_prophage_associated : prophage annotations (partial overlap)
        n_integron            : integron/CALIN annotations
        n_defense_system      : defense_system annotations
        n_is                  : IS annotations (full overlap)
        n_is_associated       : IS annotations (partial overlap)
        is_family             : IS subtype of the best-overlapping IS element (or None)
        is_families_associated: list of IS subtypes from associated IS hits
        n_mge                 : sum of all counts above
        mge_label             : priority label assigned as
                                Prophage > IS > Integron > Defense system >
                                Prophage (associated) > IS (associated) > None

    Parameters
    -------
    summaries : dict
        Output of combine_all_junctions_summaries(), keyed by event type.
    mge_dir : str
        Directory containing per-junction GFF3 files named {junction_name}.gff3.
    min_length : int
        Minimum length (bp) for both events and annotations. Default: 200.
    fill_threshold : float
        Fraction (0–1) of annotation coverage required for a hit. Default: 0.8 (80%).

    Returns
    -------
    annotated : dict
        Same structure as summaries but with MGE annotation columns added.
    mge_annotations : pd.DataFrame
        All MGE annotation rows (from all cached GFF3 files) with an added
        'junction_name' column and an 'overlaps_event' boolean column indicating
        whether the annotation is covered by at least one insertion, translocation,
        or inversion event (ann_cov >= fill_threshold) in any isolate.
    """
    # deletions only have 'position' — treat as a point interval
    point_types = {"deletions"}

    # cache GFF3 per junction to avoid re-reading
    _gff_cache = {}

    def _get_gff(junction_name):
        if junction_name not in _gff_cache:
            gff_path = os.path.join(mge_dir, f"{junction_name}.gff3")
            if os.path.isfile(gff_path):
                gff = read_gff3_annotations(gff_path)
                if not gff.empty and "length" in gff.columns:
                    gff = gff[gff["length"] >= min_length]
                _gff_cache[junction_name] = gff
            else:
                _gff_cache[junction_name] = pd.DataFrame()
        return _gff_cache[junction_name]

    def _overlapping(gff, ev_start, ev_end):
        """Return subset of gff rows that overlap [ev_start, ev_end] by any amount."""
        if gff.empty or "start" not in gff.columns or "end" not in gff.columns:
            return pd.DataFrame()
        overlap = gff[["start", "end"]].clip(lower=ev_start, upper=ev_end)
        overlap_len = (overlap["end"] - overlap["start"]).clip(lower=0)
        return gff[overlap_len > 0]

    def _within_or_border(gff, pos):
        """Return subset of gff rows where pos falls within or on the border of the annotation."""
        if gff.empty or "start" not in gff.columns or "end" not in gff.columns:
            return pd.DataFrame()
        return gff[(gff["start"] <= pos) & (gff["end"] >= pos)]

    def _annotate_df(df, is_point=False):
        if df is None or df.empty:
            return df
        if "length" in df.columns:
            df = df[df["length"] >= min_length].copy()
        else:
            print("No length filtering was applied because no length column was found in the dataframe.")

        results = []

        for _, row in df.iterrows():
            gff = _get_gff(row["junction_name"])

            # restrict to annotations on the same isolate sequence
            isolate = row.get("genome_name")
            if isolate is None or (isinstance(isolate, float) and np.isnan(isolate)):
                raise ValueError(
                    f"Event row for junction '{row.get('junction_name')}' has no genome_name. "
                    "All event rows must have isolate names to match against GFF annotations."
                )
            if gff.empty:
                gff_iso = gff
            elif "seqid" not in gff.columns:
                raise ValueError(
                    f"GFF for junction '{row.get('junction_name')}' has no 'seqid' column. "
                    "GFF files must contain isolate names in the seqid column."
                )
            else:
                gff_iso = gff[gff["seqid"] == isolate]

            empty_result = dict(
                n_prophage=0, n_prophage_associated=0,
                n_integron=0, n_defense_system=0,
                n_is=0, n_is_associated=0,
                is_family=None,
                is_families_associated=[],
            )

            if is_point:
                pos = row.get("position")
                if pos is None or (isinstance(pos, float) and np.isnan(pos)):
                    results.append(empty_result)
                    print(f"No position for deletion in junction {row.get("junction_name")} and genome {row.get("genome_name")}.")
                    continue
                hits = _within_or_border(gff_iso, int(pos))
                if hits.empty or "feature" not in hits.columns:
                    if not hits.empty and "feature" not in hits.columns:
                        print(f"Warning: 'feature' column missing in GFF for junction '{row.get('junction_name')}', isolate '{isolate}'.")
                    results.append(empty_result)
                    continue
                # for deletions: just count, no coverage thresholds
                res = empty_result.copy()
                res["n_prophage"]       = int((hits["feature"] == "prophage").sum())
                res["n_integron"]       = int(hits["feature"].isin(["integron", "CALIN"]).sum())
                res["n_defense_system"] = int((hits["feature"] == "defense_system").sum())
                is_hits = hits[hits["feature"] == "IS"]
                res["n_is"] = len(is_hits)
                if not is_hits.empty:
                    res["is_family"] = is_hits.iloc[0]["is_subtype"]
                results.append(res)
                continue

            # --- non-deletion events: apply coverage thresholds ---
            ev_start  = int(row["start_pos"])
            ev_end    = int(row["end_pos"])
            ev_length = max(ev_end - ev_start, 1)

            hits = _overlapping(gff_iso, ev_start, ev_end)
            if hits.empty or "feature" not in hits.columns:
                if not hits.empty and "feature" not in hits.columns:
                    print(f"Warning: 'feature' column missing in GFF for junction '{row.get('junction_name')}', isolate '{isolate}'.")
                results.append(empty_result)
                continue

            hits = hits.copy()
            ann_len = (hits["end"].astype(int) - hits["start"].astype(int)).clip(lower=1).values
            overlap = (hits["end"].astype(int).clip(upper=ev_end) - hits["start"].astype(int).clip(lower=ev_start)).clip(lower=0).values
            ann_cov = overlap / ann_len
            ev_cov  = overlap / ev_length
            feature = hits["feature"].values

            thr = fill_threshold
            res = empty_result.copy()

            ph_mask = (feature == "prophage")
            res["n_prophage"]            = int(((ann_cov >= thr) & (ev_cov >= thr) & ph_mask).sum())
            res["n_prophage_associated"] = int(((ann_cov >= thr) & (ev_cov < thr)  & ph_mask).sum())

            res["n_integron"]            = int(((ann_cov >= thr) & np.isin(feature, ["integron", "CALIN"])).sum())
            res["n_defense_system"]      = int(((ann_cov >= thr) & (feature == "defense_system")).sum())

            is_mask   = (feature == "IS")
            is_full   = (ann_cov >= thr) & (ev_cov >= thr) & is_mask
            is_assoc  = (ann_cov >= thr) & (ev_cov < thr)  & is_mask
            res["n_is"]           = int(is_full.sum())
            res["n_is_associated"] = int(is_assoc.sum())
            if is_full.any():
                res["is_family"] = hits["is_subtype"].values[is_full][overlap[is_full].argmax()]
            res["is_families_associated"] = [f for f in hits["is_subtype"].values[is_assoc] if f is not None]

            results.append(res)

        df = df.copy()
        for col in ("n_prophage", "n_prophage_associated", "n_integron",
                    "n_defense_system", "n_is", "n_is_associated", "is_family",
                    "is_families_associated"):
            df[col] = [r[col] for r in results]

        df["n_mge"] = (df["n_prophage"] + df["n_integron"] +
                       df["n_defense_system"] + df["n_is"] + df["n_is_associated"] + df["n_prophage_associated"])

        def _mge_label(row):
            if row["n_prophage"] > 0:             return "Prophage"
            if row["n_is"] > 0:                   return "IS"
            if row["n_integron"] > 0:             return "Integron"
            if row["n_defense_system"] > 0:       return "Defense system"
            if row["n_prophage_associated"] > 0:  return "Prophage (associated)"
            if row["n_is_associated"] > 0:        return "IS (associated)"
            return "None"

        df["mge_label"] = df.apply(_mge_label, axis=1)
        return df

    # build event_intervals upfront so _annotate_df can populate the GFF cache
    # and we don't need a second pass over summaries afterwards
    event_intervals = {}  # (junction_name, genome_name) -> np.array of shape (N, 2)
    _ev_lists = {}
    for dtype in ("insertions", "translocations", "inversions"):
        df = summaries.get(dtype)
        if df is None or df.empty:
            continue
        if "length" in df.columns:
            df = df[df["length"] >= min_length]
        for _, row in df.iterrows():
            key = (row["junction_name"], row["genome_name"])
            _ev_lists.setdefault(key, []).append((int(row["start_pos"]), int(row["end_pos"])))
    event_intervals = {k: np.array(v) for k, v in _ev_lists.items()} # values are transformed from lists to np.arrays

    annotated = {dtype: _annotate_df(df, is_point=(dtype in point_types))
                 for dtype, df in summaries.items()}

    # build mge_annotations: all GFF rows with overlaps_event flag, vectorized per (junction, isolate)
    annot_parts = []
    for junction_name, gff in _gff_cache.items():
        if gff.empty or "start" not in gff.columns:
            continue
        gff = gff.copy()
        gff["junction_name"] = junction_name
        gff["overlaps_event"] = False

        groups = gff.groupby("seqid") if "seqid" in gff.columns else [(None, gff)]
        for isolate, gff_iso in groups:
            iv = event_intervals.get((junction_name, isolate))
            if iv is None or len(iv) == 0:
                continue
            a_start  = gff_iso["start"].values[:, None]          # (M, 1)
            a_end    = gff_iso["end"].values[:, None]             # (M, 1)
            ann_len  = np.maximum(a_end - a_start, 1)
            overlap  = np.maximum(np.minimum(a_end, iv[:, 1]) -
                                  np.maximum(a_start, iv[:, 0]), 0)  # (M, N)
            ann_cov  = overlap / ann_len                          # (M, N)
            overlaps = (ann_cov >= fill_threshold).any(axis=1)   # (M,)
            gff.loc[gff_iso.index, "overlaps_event"] = overlaps

        annot_parts.append(gff)

    mge_annotations = pd.concat(annot_parts, ignore_index=True) if annot_parts else pd.DataFrame()

    return annotated, mge_annotations


def deduplicate_events(summaries, save_path=None):
    """
    Deduplicate all event summary DataFrames and combine into one DataFrame.

    For each event type:
      - Deduplicate by (junction_name, consensus, path, parent_clade_id) so
        each unique event within a cluster is represented exactly once
        regardless of which isolate carried it. Raises ValueError if any of
        these columns is missing.
      - Drop the isolate column (genome_name) since events are no longer
        tied to a specific isolate after deduplication.
      - Add an 'event_type' column with the event type string.

    Annotation columns (n_prophage, n_integron, n_defense_system, n_is,
    n_is_associated, is_family, is_families_associated, n_mge, mge_label, etc.)
    are resolved by majority vote across all isolates sharing the same event.
    If there is a single mode it is used directly; if multiple modes tie, numeric
    columns take the rounded mean of the modes, and string columns take the first
    mode. For list-valued columns (is_families_associated), a family is kept if
    it appears in more than half of the isolates (NaN/empty-list rows are
    included in the total count, so they count against the threshold). All other (non-annotation,
    non-isolate) columns take the value from the first occurrence.

    Origin/hits columns (n_hits_*, n_genomes_*, hits_in_genome,
    hits_in_plasmid) are summarised as the median across isolates (rounded to
    int). majority_organism is resolved by majority vote on the first two words
    of the organism name.

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
        path, parent_clade_id, length, ..., isolates (genome_name dropped,
        isolates is a list of all genome_names sharing the event).
    """
    # Maps summaries dict keys (plural, match directory names) to singular event_type labels
    event_type_map = {
        "insertions":           "insertion",
        "ambiguous_insertions": "ambiguous_insertion",
        "deletions":            "deletion",
        "translocations":       "translocation",
        "inversions":           "inversion",
    }

    # Column sets that require special aggregation
    annotation_cols = {
        "n_prophage", "n_prophage_associated",
        "n_integron", "n_defense_system",
        "n_is", "n_is_associated",
        "is_family", "is_families_associated", "n_mge", "mge_label",
    }
    origin_median_cols = {"hits_in_genome", "hits_in_plasmid"}  # plus n_hits_* and n_genomes_*

    # Aggregation helpers (defined once, used for every event type)
    def _majority(series):
        """Majority vote: single mode, or mean/first of tied modes."""
        m = series.dropna().mode()
        if m.empty:
            return series.iloc[0] if not series.empty else None
        if len(m) == 1:
            return m.iloc[0]
        return int(round(m.mean())) if pd.api.types.is_numeric_dtype(m) else m.iloc[0]

    def _majority_list(series):
        """Keep families that appear in more than half of all isolates (including NaN rows)."""
        n = len(series)
        counts = Counter(fam for lst in series for fam in (lst if isinstance(lst, list) else []))
        return [fam for fam, cnt in counts.items() if cnt > n / 2]

    def _median_int(series):
        """Median across isolates, rounded to int; 0 if all NaN."""
        return int(round(series.median(skipna=True))) if series.notna().any() else 0

    def _majority_organism(series):
        """Majority vote on the first two words of the organism name."""
        normalized = series.dropna().map(lambda x: " ".join(x.split()[:2]) if isinstance(x, str) else x)
        m = normalized.mode()
        return m.iloc[0] if not m.empty else None

    parts = []

    for dtype, etype_label in event_type_map.items():
        df = summaries.get(dtype)
        if df is None or df.empty:
            continue

        df = df.copy()

        # Deduplicate by (junction_name, consensus, path, parent_clade_id)
        dedup_cols = ["junction_name", "consensus", "path", "parent_clade_id"]
        missing = [c for c in dedup_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Cannot deduplicate {dtype}: missing columns {missing}")


        # Columns to skip (used as group keys or dropped)
        skip_cols = set(dedup_cols) | {"genome_name"}

        # Collect isolate lists before dropping genome_name
        if "genome_name" in df.columns:
            isolates_df = df.groupby(dedup_cols, as_index=False)["genome_name"].agg(list).rename(columns={"genome_name": "isolates"})
        else:
            isolates_df = None

        # Build agg_dict for every column in one pass
        agg_dict = {}
        for c in df.columns:
            if c in skip_cols:
                continue
            if c == "is_families_associated":
                agg_dict[c] = _majority_list
            elif c in annotation_cols:
                agg_dict[c] = _majority
            elif c == "majority_organism":
                agg_dict[c] = _majority_organism
            elif c.startswith("n_hits_") or c.startswith("n_genomes_") or c in origin_median_cols:
                agg_dict[c] = _median_int
            else:
                agg_dict[c] = "first"

        df = df.groupby(dedup_cols, as_index=False).agg(agg_dict)
        if isolates_df is not None:
            df = df.merge(isolates_df, on=dedup_cols, how="left")
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


def collect_hits_info_counts_with_self(
    search_root: str,
    sam_df: pd.DataFrame,
    min_pident: float = 100.0,
) -> pd.DataFrame:
    """
    Walk search_root for *.lexicmap.tsv files and count hits per insertion segment,
    bucketed by the relationship of each hit's source genome (sgenome biosample) to
    the query isolate.

    Input files
    -----------
    *.lexicmap.tsv : tab-separated, one hit per row. Required columns: sgenome, pident,
        qcovGnm.
    *.hits_info.tsv : optional companion file (same stem). Required columns: organism,
        pident, qcovGnm, sgenome, strain. Used to derive majority_organism, per-species
        hit counts, and n_hits_st131. Self-hits (own chromosome / plasmid) are excluded
        from all three.

    Hit bucketing (via sgenome biosample)
    --------------------------------------
    own_chromosome   : sgenome is a biosample of the query isolate's chromosome
    own_plasmid      : sgenome is a biosample of the query isolate's plasmid(s)
    other_chromosome : sgenome belongs to another isolate in sam_df (chromosome)
    other_plasmid    : sgenome belongs to another isolate in sam_df (plasmid)
    external         : sgenome is not present in sam_df at all (truly external to the study)

    The query isolate is identified by matching genome_name (from the filename) against
    sam_df.accession. Only hits with pident >= min_pident and qcovGnm >= 90% are counted.

    Majority organism
    -----------------
    Derived from the companion .hits_info.tsv (if present). Organism names are
    normalised to the first two words (genus + species) before the majority vote.
    Self-hits are excluded. Per-species counts are added as n_hits_{Genus_species} columns.

    Parameters
    ----------
    search_root : str
        Root directory to search recursively for *.lexicmap.tsv files. The path
        structure is expected to contain .../insertions/{junction_name}/{consensus}/...
    sam_df : pd.DataFrame
        Columns: isolate_id, accession, type ("chromosome"/"plasmid"), biosample.
    min_pident : float
        Minimum percent identity (inclusive) to count a hit. Default 100.0.

    Returns
    -------
    pd.DataFrame
        One row per *.lexicmap.tsv file with columns:
            junction_name, consensus, genome_name, segment,
            n_hits, n_genomes,
            n_hits_{cat}, n_genomes_{cat}  for each of the five categories above,
            majority_organism,
            n_hits_st131,                           # hits whose strain column contains "ST131"
            n_hits_{Genus_species}  per species found in hits_info (NaN filled to 0),
            file_path
    """
    # Build lookup structures from sam_df
    acc_to_isolate  = dict(zip(sam_df["accession"], sam_df["isolate_id"]))
    iso_chr_bs      = sam_df[sam_df["type"] == "chromosome"].groupby("isolate_id")["biosample"].apply(set).to_dict()
    iso_pla_bs      = sam_df[sam_df["type"] == "plasmid"].groupby("isolate_id")["biosample"].apply(set).to_dict()
    all_own_bs      = set(sam_df["biosample"])
    bs_to_type      = dict(zip(sam_df["biosample"], sam_df["type"]))

    cats = ("own_chromosome", "own_plasmid", "other_chromosome", "other_plasmid", "external")
    rows = []

    for p in sorted(Path(search_root).rglob("*.lexicmap.tsv")):
        # Extract junction_name and consensus from path
        parts = p.parts
        ins_idx = [i for i, x in enumerate(parts) if x == "insertions"][-1]
        junction_name = parts[ins_idx + 1]
        consensus     = parts[ins_idx + 2]

        # Extract genome_name and segment from filename
        stem      = p.name.replace(".lexicmap.tsv", "")
        seg_match = re.search(r"_segment_(\d+)$", stem)
        if seg_match:
            segment    = f"segment_{seg_match.group(1)}"
            genome_name = stem[: seg_match.start()]
        else:
            segment    = None
            genome_name = stem

        # Own-isolate biosample sets for bucketing
        isolate_id  = acc_to_isolate.get(genome_name)
        own_chr_bs  = iso_chr_bs.get(isolate_id, set())
        own_pla_bs  = iso_pla_bs.get(isolate_id, set())

        # Parse lexicmap hits
        hit_counts  = {c: 0 for c in cats}
        genome_sets = {c: set() for c in cats}
        n_hits      = 0

        with open(p) as f:
            cols = {name: idx for idx, name in enumerate(f.readline().strip().split("\t"))}
            sg_col      = cols["sgenome"]
            pident_col  = cols["pident"]
            qcovgnm_col = cols["qcovGnm"]

            for line in f:
                fields = line.strip().split("\t")
                if float(fields[pident_col]) < min_pident or float(fields[qcovgnm_col]) < 90.0:
                    continue
                n_hits += 1
                sg = fields[sg_col]
                if sg in own_chr_bs:
                    cat = "own_chromosome"
                elif sg in own_pla_bs:
                    cat = "own_plasmid"
                elif sg in all_own_bs:
                    cat = "other_chromosome" if bs_to_type[sg] == "chromosome" else "other_plasmid"
                else:
                    cat = "external"
                hit_counts[cat] += 1
                genome_sets[cat].add(sg)

        # Majority organism + per-species counts from companion hits_info.tsv
        majority_organism = None
        normalized_counts: dict = {}
        n_hits_st131 = 0
        info_path = Path(str(p).replace(".lexicmap.tsv", ".hits_info.tsv"))
        if info_path.exists():
            with open(info_path) as f:
                info_cols      = {name: idx for idx, name in enumerate(f.readline().strip().split("\t"))}
                org_col        = info_cols["organism"]
                ipident_col    = info_cols["pident"]
                iqcovgnm_col   = info_cols["qcovGnm"]
                isg_col        = info_cols["sgenome"]
                strain_col     = info_cols["strain"]
                organism_counts: dict = {}
                for line in f:
                    ifields = line.strip().split("\t")
                    if float(ifields[ipident_col]) < min_pident or float(ifields[iqcovgnm_col]) < 90.0:
                        continue
                    if ifields[isg_col] in all_own_bs:
                        continue
                    if len(ifields) <= org_col:
                        continue
                    if strain_col < len(ifields) and "ST131" in ifields[strain_col]:
                        n_hits_st131 += 1
                    org = ifields[org_col].strip()
                    if org:
                        organism_counts[org] = organism_counts.get(org, 0) + 1
            if organism_counts:
                for org, cnt in organism_counts.items():
                    species = " ".join(org.split()[:2])
                    normalized_counts[species] = normalized_counts.get(species, 0) + cnt
                majority_organism = max(normalized_counts, key=normalized_counts.get)

        row = dict(
            junction_name=junction_name,
            consensus=consensus,
            genome_name=genome_name,
            segment=segment,
            n_hits=n_hits,
            n_genomes=len(set.union(*genome_sets.values()) if any(genome_sets.values()) else set()),
            majority_organism=majority_organism,
            n_hits_st131=n_hits_st131,
            file_path=str(p),
        )
        for cat in cats:
            row[f"n_hits_{cat}"]    = hit_counts[cat]
            row[f"n_genomes_{cat}"] = len(genome_sets[cat])
        for species, cnt in normalized_counts.items():
            row[f"n_hits_{'_'.join(species.split())}"] = cnt

        rows.append(row)

    df = pd.DataFrame(rows)
    known_hit_cols = {f"n_hits_{cat}" for cat in cats} | {"n_hits"}
    species_cols = [c for c in df.columns if c.startswith("n_hits_") and c not in known_hit_cols]
    df[species_cols] = df[species_cols].fillna(0)
    return df
