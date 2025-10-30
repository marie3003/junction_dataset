import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import pypangraph as pp

from junction_analyis.helpers import get_tree_order
import junction_analyis.pangraph_utils as pu


def plot_heatmap_hover(sequence_comparison_df, diff = None, shared = None, show_tick_labels=False, similarity_measure = "shared_proportion"):
    """Plot heatmap showing isolate information with hover"""

    if similarity_measure == "shared_proportion":
   
        # build customdata for richer hover
        customdata = np.dstack([diff.values, shared.values])

        hovertemplate = (
            "path_i: %{y}<br>"
            "path_j: %{x}<br>"
            "similarity: %{z:.4f}<br>"
            "diff: %{customdata[0]}<br>"
            "shared: %{customdata[1]}<extra></extra>"
        )
        similarity_measure_full_name = "shared sequence similarity (1 - diff/shared)"
        similarity_measure_short_name = "Similarity"

    elif similarity_measure == "divergence_points":

        customdata = None
        hovertemplate = (
            "path_i: %{y}<br>"
            "path_j: %{x}<br>"
            "divergence points: %{z}<extra></extra>"
        )

        similarity_measure_full_name = "number of divergence points"
        similarity_measure_short_name = "Divergence points"

    elif similarity_measure == "shared_blocks":

        customdata = np.dstack([diff.values, shared.values])
        hovertemplate = (
            "path_i: %{y}<br>"
            "path_j: %{x}<br>"
            "Block Jaccard index: %{z}<extra></extra><br>"
            "# diff. blocks: %{customdata[0]}<br>"
            "# shared blocks: %{customdata[1]}"
        )

        similarity_measure_full_name = "jaccard index of shared blocks"
        similarity_measure_short_name = "Block Similarity"

    elif similarity_measure == "shared_edges":

        customdata = np.dstack([diff.values, shared.values])
        hovertemplate = (
            "path_i: %{y}<br>"
            "path_j: %{x}<br>"
            "Edge Jaccard index: %{z}<extra></extra><br>"
            "# diff. edges: %{customdata[0]}<br>"
            "# shared edges: %{customdata[1]}"
        )

        similarity_measure_full_name = "jaccard index of shared edges"
        similarity_measure_short_name = "Edge Similarity"

    fig = go.Figure(
        data=go.Heatmap(
            z=sequence_comparison_df.values,
            x=sequence_comparison_df.columns.astype(str),
            y=sequence_comparison_df.index.astype(str),
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorbar=dict(title=f"{similarity_measure_short_name}:"),
            zmin=np.nanmin(sequence_comparison_df.values),  # keeps scale stable if you filter later
            zmax=np.nanmax(sequence_comparison_df.values),
        )
    )

    if not show_tick_labels:
    # hide tick labels to prevent clutter; rely on hover for names
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        title=f"Pairwise Similarity of isolates based on {similarity_measure_full_name}",
        xaxis_title="path_j",
        yaxis_title="path_i",
        width=800,
        height=720,
    )

    fig.show()



def plot_junction_pangraph_combined(
    pan: pp.Pangraph,
    show_consensus: bool = False,
    consensus_paths: list = None,        # list[pu.Path] of Nodes with .id, .strand
    assignments: pd.DataFrame = None,    # index: isolate names, col 'best_consensus'
    order: str = "tree",
):
    """
    Plot junction graph blocks for all isolates in `pan`, optionally including consensus paths.

    Args
    ----
    pan : pp.Pangraph
        Pangraph object providing paths and node coordinates.
    show_consensus : bool
        Whether to include consensus tracks.
    consensus_paths : list
        consensus_paths[i] is a consensus "path"-like iterable of nodes,
        where each node has attributes .id (block id) and .strand (bool).
        Required if show_consensus=True.
    assignments : pd.DataFrame
        Must have index of isolate names and a column 'best_consensus'
        whose values are "consensus_1", "consensus_2", ...
        Required if show_consensus=True.
    order : str
        "tree" or anything else. If "tree", isolates are ordered by get_tree_order()
        whenever reasonable.
    """


    bdf = pan.to_blockstats_df()
    n_core = int(bdf["core"].sum())
    n_acc = int(len(bdf) - n_core)

    # distinct color generators for core / accessory
    cgen_acc = iter(sns.color_palette("rainbow", n_acc))
    cgen_core = iter(sns.color_palette("pastel", n_core))
    block_colors: dict = {}

    def get_block_color(block_id):
        """Return (and cache) a consistent color per block id."""
        if block_id not in block_colors:
            color = next(cgen_core) if bool(bdf.loc[block_id, "core"]) else next(cgen_acc)
            block_colors[block_id] = color
        return block_colors[block_id]

    # --- isolate ordering ---
    tree_order = get_tree_order() if order == "tree" else None
    if tree_order:
        isolates_ordered = tree_order
    else:
        isolates_ordered = list(pan.paths.keys())

    # --- helpers to actually draw bars ---
    def draw_isolate_track(isolate_name: str, y_val: int) -> int:
        """Plot one isolate using true genomic coordinates from pan."""
        if isolate_name not in pan.paths:
            return y_val
        p = pan.paths[isolate_name]
        for node_id in p.nodes:
            block, strand, start, end = pan.nodes[node_id][["block_id", "strand", "start", "end"]]
            ax.barh(
                y_val,
                width=end - start,
                left=start,
                color=get_block_color(block),
                edgecolor=("black" if strand else "red"),
            )
        y_labels.append(isolate_name)
        return y_val + 1

    def draw_consensus_track(cons_path, label: str, y_val: int) -> int:
        """
        Plot one consensus path. No absolute coords in pan,
        so we just accumulate block lengths along x.
        cons_path is iterable of nodes with .id and .strand
        """
        x_left = 0
        for node in cons_path:
            bid = node.id
            strand = node.strand
            block_len = int(bdf.loc[bid, "len"])
            ax.barh(
                y_val,
                width=block_len,
                left=x_left,
                color=get_block_color(bid),
                edgecolor=("black" if strand else "red"),
            )
            x_left += block_len
        y_labels.append(label)
        return y_val + 1

    est_rows = len(isolates_ordered)
    if show_consensus:
        est_rows += 2 * len(consensus_paths)

    fig, ax = plt.subplots(figsize=(12, max(4, est_rows * 0.22)))
    y = 0
    y_labels = []

    # === CASE 1: isolates-only mode ===
    if not show_consensus:
        for iso in isolates_ordered:
            y = draw_isolate_track(iso, y)

    # === CASE 2: show_consensus mode ===
    else:

        grouped = (
            assignments.reset_index()
            .groupby("best_consensus")["index"]
            .apply(list)
            .to_dict()
        )

        for i, cons_path in enumerate(consensus_paths):
            cons_label = f"consensus_{i+1}"

            isolates_for_this = grouped.get(cons_label, [])

            # respect tree ordering if we have it
            if tree_order:
                isolates_for_this = [iso for iso in tree_order if iso in isolates_for_this]

            # plot isolates for this consensus
            for iso in isolates_for_this:
                y = draw_isolate_track(iso, y)

            # plot consensus again
            y = draw_consensus_track(cons_path, cons_label, y)

        for i, cons_path in enumerate(consensus_paths):
            cons_label = f"consensus_{i+1}"
            y = draw_consensus_track(cons_path, cons_label, y)


    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # make consensus labels bold so they stand out
    for idx, tick in enumerate(ax.get_yticklabels()):
        if y_labels[idx].startswith("consensus_"):
            tick.set_fontweight("bold")

    ax.set_xlabel("genomic position (bp)")
    ax.grid(axis="x", alpha=0.4)

    ax.set_ylim(-1, len(y_labels))

    sns.despine()
    plt.tight_layout()
    return fig, ax


def plot_junction_pangraph_grouped(
    pan: pp.Pangraph,
    consensus_paths: list,              # list[pu.Path] of DeduplicatedNode (or Node)
    assignments: pd.DataFrame,          # index: isolate names, col 'best_consensus'
    order: str = "tree"
):
    # --- prepare inputs and colors ---
    path_dict = pan.to_path_dictionary()
    bdf = pan.to_blockstats_df()  # indexed by block id
    n_core = bdf["core"].sum()
    n_acc = len(bdf) - n_core
    cgen_acc = iter(sns.color_palette("rainbow", n_acc))
    cgen_core = iter(sns.color_palette("pastel", n_core))
    block_colors: dict = {}

    def get_block_color(bid):
        if bid not in block_colors:
            color = next(cgen_core) if bool(bdf.loc[bid, "core"]) else next(cgen_acc)
            block_colors[bid] = color
        return block_colors[bid]

    fig, ax = plt.subplots(figsize=(12, max(4, len(path_dict) * 0.25)))
    y = 0
    y_labels = []

    # --- helper: plot an isolate path from pan (uses actual coordinates) ---
    def plot_isolate_by_name(isolate: str, y: int) -> int:
        if isolate not in pan.paths:
            return y
        p = pan.paths[isolate]
        for node_id in p.nodes:
            block, strand, start, end = pan.nodes[node_id][["block_id", "strand", "start", "end"]]
            color = get_block_color(block)
            ax.barh(
                y,
                width=end - start,
                left=start,
                color=color,
                edgecolor=("black" if strand else "red"),
            )
        y_labels.append(isolate)
        return y + 1

    # --- helper: plot a consensus path (no coordinates; accumulate lengths) ---
    def plot_consensus(cons_path: pu.Path, label: str, y: int) -> int:
        left = 0
        # cons_path is a Path of Nodes/DeduplicatedNodes; use block.id and block.strand
        for node in cons_path:
            bid = node.id
            strand = node.strand
            block_len = int(bdf.loc[bid, "len"])
            color = get_block_color(bid)
            ax.barh(
                y,
                width=block_len,
                left=left,
                color=color,
                edgecolor=("black" if strand else "red"),
            )
            left += block_len
        y_labels.append(label)
        return y + 1

    # group isolates under each consensus label
    grouped = (
        assignments.reset_index()
        .groupby("best_consensus")["index"]
        .apply(list)
        .to_dict()
    )

    # optional ordering of isolates within each group
    tree_order = get_tree_order() if order == "tree" else None

    for i, cons_path in enumerate(consensus_paths):
        cons_label = f"consensus_{i+1}"

        # then plot assigned isolates under it
        isolates = grouped.get(cons_label, [])
        if tree_order:
            isolates = [iso for iso in tree_order if iso in isolates]
        for iso in isolates:
            y = plot_isolate_by_name(iso, y)
        
        # plot consensus last
        y = plot_consensus(cons_path, cons_label, y)

    # axes cosmetics
    # axes cosmetics
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)  # plain strings, once

    # make consensus labels bold
    for i, txt in enumerate(ax.get_yticklabels()):
        if y_labels[i].startswith("consensus_"):
            txt.set_fontweight("bold")   # or txt.set_weight("bold")
            # optional: txt.set_color("black")  # ensure visibility if you changed colors

    ax.set_yticklabels(y_labels)
    ax.set_xlabel("genomic position (bp)")
    ax.grid(axis="x", alpha=0.4)
    ax.set_ylim(-1, len(y_labels))
    sns.despine()
    plt.tight_layout()


def plot_junction_pangraph(pan: pp.Pangraph, add_consensus: bool = False, consensus_paths: list = None, order="tree"):

    if order == "tree":
        leaf_order = get_tree_order()

    path_dict = pan.to_path_dictionary()
    bdf = pan.to_blockstats_df()
    n_core = bdf["core"].sum()
    n_acc = len(bdf) - n_core
    cgen_acc = iter(sns.color_palette("rainbow", n_acc))
    cgen_core = iter(sns.color_palette("pastel", n_core))
    block_colors = {}

    fig, ax = plt.subplots(figsize=(12, len(path_dict) * 0.2))
    y = 0
    y_labels = []

    for name in leaf_order:
        if name not in pan.paths:
            continue
        path = pan.paths[name]
        for node_id in path.nodes:
            block, strand, start, end = pan.nodes[node_id][
                ["block_id", "strand", "start", "end"]
            ]
            if block not in block_colors:
                if bdf.loc[block, "core"]:
                    color = next(cgen_core)
                else:
                    color = next(cgen_acc)
                block_colors[block] = color
            else:
                color = block_colors[block]
            block_len = bdf.loc[block, "len"]
            edgecolor = "black" if strand else "red"
            ax.barh(
                y,
                width=end - start,
                left=start,
                color=color,
                edgecolor=edgecolor,
            )
        y_labels.append(name)
        y += 1

    if add_consensus:
        for i, cons_path in enumerate(consensus_paths):
            start = 0
            for block in cons_path:
                block_len = bdf.loc[block.id, "len"]
                ax.barh(
                    y,
                    width=block_len,
                    left=start,
                    color=block_colors[block.id],
                    edgecolor="black" if block.strand else "red",
                )
                start += block_len
            y_labels.append(f"consensus_{i+1}")
            y += 1
            

    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_xlabel("genomic position (bp)")
    #ax.set_title(f"Junction graph for edge {selected_edge}")
    ax.grid(axis="x", alpha=0.4)
    ax.set_ylim(-1, len(y_labels))
    sns.despine()
    plt.tight_layout()
