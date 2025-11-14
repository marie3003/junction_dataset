import numpy as np
import random

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import pypangraph as pp
from scipy.cluster.hierarchy import dendrogram

from junction_analysis.helpers import get_tree_order
import junction_analysis.pangraph_utils as pu


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

# interactive plotly version
def plot_junction_pangraph_interactive(
    pan: pp.Pangraph,
    show_consensus: bool = False,
    consensus_paths: list = None,
    assignments: pd.DataFrame = None,
    order: str = "tree",
    cluster_map: dict = None,
    title: str = "", 
):
    bdf = pan.to_blockstats_df()
    n_core = int(bdf["core"].sum())
    n_acc = int(len(bdf) - n_core)
    cgen_acc = iter(sns.color_palette("rainbow", n_acc))
    cgen_core = iter(sns.color_palette("pastel", n_core))
    block_colors: dict = {}

    def get_block_color(block_id):
        if block_id not in block_colors:
            color = next(cgen_core) if bool(bdf.loc[block_id, "core"]) else next(cgen_acc)
            if isinstance(color, tuple) and len(color) == 3:
                color = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
            block_colors[block_id] = color
        return block_colors[block_id]

    tree_order = get_tree_order() if order == "tree" else None
    isolates_ordered = tree_order if tree_order else list(pan.paths.keys())
    fig = go.Figure()
    y_labels = []
    y_seen = set()
    max_x = 0  # track max end to pad space for stars

    def _add_bar(label: str, left: int, width: int, color: str, strand: bool, block_id, block_pos: int):
        nonlocal max_x
        max_x = max(max_x, int(left) + int(width))
        fig.add_bar(
            x=[width],
            y=[label],
            base=[left],
            orientation="h",
            marker=dict(color=color, line=dict(color=("black" if strand else "red"), width=1)),
            customdata=[[left, width, left + width, block_id, strand, block_pos]],
            hovertemplate=(
                "label: %{y}"
                "<br>start: %{customdata[0]}"
                "<br>len: %{customdata[1]}"
                "<br>end: %{customdata[2]}"
                "<br>block: %{customdata[3]}"
                "<br>strand: %{customdata[4]:+, -}"
                "<br>block position: %{customdata[5]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )

    def draw_isolate_track(isolate_name: str):
        if isolate_name not in pan.paths:
            return
        p = pan.paths[isolate_name]
        for block_idx, node_id in enumerate(p.nodes):
            block, strand, start, end = pan.nodes[node_id][["block_id", "strand", "start", "end"]]
            _add_bar(
                label=isolate_name,
                left=int(start),
                width=int(end - start),
                color=get_block_color(block),
                strand=bool(strand),
                block_id=block,
                block_pos=block_idx,
            )
        if isolate_name not in y_seen:
            y_labels.append(isolate_name)
            y_seen.add(isolate_name)

    def draw_consensus_track(cons_path, label: str):
        x_left = 0
        for block_idx, node in enumerate(cons_path):
            bid = node.id
            strand = node.strand
            block_len = int(bdf.loc[bid, "len"])
            _add_bar(
                label=label,
                left=int(x_left),
                width=block_len,
                color=get_block_color(bid),
                strand=bool(strand),
                block_id=bid,
                block_pos=block_idx,
            )
            x_left += block_len
        if label not in y_seen:
            y_labels.append(label)
            y_seen.add(label)

    if not show_consensus:
        for iso in isolates_ordered:
            draw_isolate_track(iso)
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
            if tree_order:
                isolates_for_this = [iso for iso in tree_order if iso in isolates_for_this]
            for iso in isolates_for_this:
                draw_isolate_track(iso)
            draw_consensus_track(cons_path, cons_label)
        for i, cons_path in enumerate(consensus_paths):
            cons_label = f"consensus_{i+1}\u200b"
            draw_consensus_track(cons_path, cons_label)

    tickvals = y_labels
    ticktext = [f"<b>{y}</b>" if y.startswith("consensus_") else y for y in y_labels]

    # add a star per isolate based on cluster_map ---
    if cluster_map:
        # assign random colors per cluster id
        clusters = sorted(set(cluster_map.values()))
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
        random.shuffle(palette)
        cluster_color = {cid: palette[i % len(palette)] for i, cid in enumerate(clusters)}

        # position stars slightly left of the first base (add small negative pad)
        star_x = - 800

        xs, ys, cs = [], [], []
        for iso in y_labels:
            if iso.startswith("consensus_"):
                continue
            if iso in cluster_map:
                xs.append(star_x)
                ys.append(iso)
                cs.append(cluster_color[cluster_map[iso]])


        if xs:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(symbol="star", size=14, color=cs),
                hoverinfo="skip",
                showlegend=False,
            ))

        # --- simple legend stars for clusters ---
        for cid, color in cluster_color.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],  # invisible points just for legend
                mode="markers",
                marker=dict(symbol="star", size=14, color=color),
                name=f"Cluster {cid}",
                hoverinfo="skip",
            ))

        fig.update_layout(legend_title_text="Clusters")


    fig.update_layout(
        title=dict(
            text=title,
            x=0.05,               # align to the left edge (0 = left, 1 = right)
            y = 0.99,
            xanchor="left",    # anchor the title text to its left
            yanchor="top",
            yref="container",  # relative to the full container
            font=dict(size=18, family="Arial", color="black"),
            pad=dict(l=10, t=10),  # small left/top padding
        ),
        barmode="stack",
        bargap=0.08,
        xaxis=dict(
            title="genomic position (bp)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.2)",
            range=[-max(1, int(0.05 * max_x)), max_x],  # extend left to show stars
            zeroline=True,
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=y_labels,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        margin=dict(l=140, r=20, t=20, b=40),
        height=max(300, int(len(y_labels) * 22)),
        template="plotly_white",
    )
    return fig


def plot_junction_pangraph_combined(
    pan: pp.Pangraph,
    show_consensus: bool = False,
    consensus_paths: list = None,        # list[pu.Path] of Nodes with .id, .strand
    assignments: pd.DataFrame = None,    # index: isolate names, col 'best_consensus'
    order: str = "tree",
    cluster_map: dict = None,            # <--- NEW
):
    """
    Plot junction graph blocks for all isolates in `pan`, optionally including consensus paths.
    Also supports drawing cluster stars (one star per isolate row) if `cluster_map` is provided.
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

    # isolate ordering
    tree_order = get_tree_order() if order == "tree" else None
    isolates_ordered = tree_order if tree_order else list(pan.paths.keys())

    # helpers to draw bars
    max_x = 0
    min_x = 0

    def draw_isolate_track(isolate_name: str, y_val: int) -> int:
        nonlocal max_x, min_x
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
            max_x = max(max_x, float(end))
            min_x = min(min_x, float(start))
        y_labels.append(isolate_name)
        return y_val + 1

    def draw_consensus_track(cons_path, label: str, y_val: int) -> int:
        nonlocal max_x
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
        max_x = max(max_x, float(x_left))
        y_labels.append(label)
        return y_val + 1

    est_rows = len(isolates_ordered)
    if show_consensus:
        est_rows += 2 * len(consensus_paths)

    fig, ax = plt.subplots(figsize=(12, max(4, est_rows * 0.22)))
    y = 0
    y_labels = []

    # CASE 1: isolates-only
    if not show_consensus:
        for iso in isolates_ordered:
            y = draw_isolate_track(iso, y)

    # CASE 2: with consensus
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
            if tree_order:
                isolates_for_this = [iso for iso in tree_order if iso in isolates_for_this]

            for iso in isolates_for_this:
                y = draw_isolate_track(iso, y)

            y = draw_consensus_track(cons_path, cons_label, y)

        for i, cons_path in enumerate(consensus_paths):
            cons_label = f"consensus_{i+1}"
            y = draw_consensus_track(cons_path, cons_label, y)

    # y ticks and bold consensus labels
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    for idx, tick in enumerate(ax.get_yticklabels()):
        if y_labels[idx].startswith("consensus_"):
            tick.set_fontweight("bold")

    # axes labels & grid
    ax.set_xlabel("genomic position (bp)")
    ax.grid(axis="x", alpha=0.4)
    ax.set_ylim(-1, len(y_labels))

    # --- NEW: cluster stars ---
    if cluster_map:
        # assign random colors per cluster id (use a qualitative palette and shuffle)
        clusters = sorted(set(cluster_map.values()))
        palette = sns.color_palette("tab20", n_colors=max(20, len(clusters)))
        palette = [tuple(c) for c in palette]
        random.shuffle(palette)
        cluster_color = {cid: palette[i % len(palette)] for i, cid in enumerate(clusters)}

        # choose a star x-position left of content; similar to Plotly version
        span = max(1.0, max_x - min_x)
        star_x = min_x - 0.05 * span  # 5% to the left of the leftmost data
        # draw one star per isolate row (skip consensus labels)
        for row, iso in enumerate(y_labels):
            if iso.startswith("consensus_"):
                continue
            cid = cluster_map.get(iso)
            if cid is None:
                continue
            ax.scatter(
                star_x, row,
                marker="*",
                s=80,
                c=[cluster_color[cid]],
                linewidths=0.8,
                zorder=3,
            )

        # ensure stars are visible
        right = ax.get_xlim()[1]
        ax.set_xlim(star_x - 0.02 * span, right)

        # legend: one star per cluster
        handles = [
            Line2D(
                [0], [0],
                marker="*",
                linestyle="",
                markersize=15,
                markerfacecolor=cluster_color[cid],
                markeredgecolor="white",
                label=f"Cluster {cid}"
            )
            for cid in sorted(clusters)
        ]
        ax.legend(handles=handles, title="Clusters", loc="upper left", bbox_to_anchor=(1.01, 1.0))

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


def plot_dendrogram(Z, names):
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.title("Hierarchical clustering (p-distance)")
    plt.ylabel("p-distance")
    plt.tight_layout()
    plt.show()
