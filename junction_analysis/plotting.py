import numpy as np
import math
import random
import os

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorsys

import pandas as pd
import pypangraph as pp
from scipy.cluster.hierarchy import dendrogram

from junction_analysis.helpers import get_tree_order, read_gff3_annotations, read_gff3_cds_products
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
def _rgb_str(r, g, b):
    """
    Convert RGB values to an RGB color string.
    """
    return f"rgb({int(r)},{int(g)},{int(b)})"

def _shades_from_base_rgb(base_rgb, n: int):
    """
    Creates n shades from a base RGB color by varying lightness.
    """
    r, g, b = [x / 255.0 for x in base_rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n <= 1:
        return [_rgb_str(base_rgb[0], base_rgb[1], base_rgb[2])]
    ls = [0.85 - i * (0.7 / (n - 1)) for i in range(n)]
    out = []
    for li in ls:
        rr, gg, bb = colorsys.hls_to_rgb(h, max(0.0, min(1.0, li)), s)
        out.append(_rgb_str(rr * 255, gg * 255, bb * 255))
    return out

def _rgba(rgb_str: str, alpha: float) -> str:
    """
    Convert an RGB color string to RGBA with specified alpha value for transparency.
    """
    # rgb(1,2,3) -> rgba(1,2,3,0.4)
    nums = rgb_str.strip().removeprefix("rgb(").removesuffix(")")
    return f"rgba({nums},{alpha})"

def plot_junction_pangraph_interactive(
    pan: pp.Pangraph,
    show_consensus: bool = False,
    consensus_paths: list = None,
    assignments: pd.DataFrame = None,
    order: str = "tree",
    cluster_map: dict = None,
    add_cluster_annotation: bool = True,
    title: str = "",
    show_mges_annotations: bool = False,
    show_int_rec_annotations: bool = False,
    mges_gff_path: str = None,
    show_cds_annotations: bool = False,
    annotations_gff_path: str = None,
    annotation_alpha: float = 0.70,   # transparency for annotations
    cds_annotation_alpha: float = 0.30,  # transparency for CDS annotations
    show_indels: bool = False,
    indels_base_path: str = None,  # base path (e.g., results/atb_lookup) containing insertions/ and deletions/ subfolders
    junction_name: str = None,  # junction name for path construction (e.g., CIRMBUYJFK_f__CWCCKOQCWZ_r)
):
    """
    Plots the block structure of a junction pangraph using Plotly.
    The function can be used in four possible ways:
        1) Plot all isolates in the pangraph (show_consensus=False, consensus_paths=None, assignments=None)
        2) Plot isolates grouped by consensus paths (show_consensus=True, consensus_paths provided, assignments provided)
        3) Plot isolates grouped by consensus paths with cluster annotations that don't necessarily have to match the consensus assignments (add_cluster_annotations = True, cluster_map provided)
        4) Plot all isolates with prophage, defense_system, IS annotations. This works on top of adding consensus assignments and clustering or without (show_annotations=True, annotations_gff_path provided)
        5) Plot insertions and deletions on top of the block structure (show_indels=True, indels_path provided). Only works when consensus paths are defined.

    :param pan: pp.Pangraph, Pangraph object to plot
    :param show_consensus: bool, whether to show consensus paths
    :param consensus_paths: list of consensus paths to plot
    :param assignments: pd.DataFrame, assignments of isolates to consensus paths
    :param order: str, order of consensus path, if "tree" use order of core genome tree
    :param cluster_map: dict, mapping of isolate names to cluster IDs for annotation
    :param add_cluster_annotation: bool, whether to add cluster annotations
    :param title: str, Title of the plot
    :param show_annotations: bool, whether to show prophage, defense_system, IS annotations
    :param annotations_gff_path: str, path to GFF file with annotations
    :param annotation_alpha: float, transparency for annotations
    :param show_indels: bool, whether to show insertions and deletions
    :param indels_base_path: str, base path (e.g., results/atb_lookup) containing insertions/ and deletions/ subfolders
    :param junction_name: str, junction name for path construction (e.g., CIRMBUYJFK_f__CWCCKOQCWZ_r)
    """
    bdf = pan.to_blockstats_df()

    GREY_CORE = "rgb(220,220,220)"
    GREY_ACC  = "rgb(190,190,190)"

    n_core = int(bdf["core"].sum())
    n_acc = int(len(bdf) - n_core)
    cgen_acc = iter(sns.color_palette("rainbow", n_acc))
    cgen_core = iter(sns.color_palette("pastel", n_core))
    block_colors: dict = {}

    def get_block_color(block_id):
        # Only turn blocks grey for annotation overlays (not for indels alone)
        if show_mges_annotations or show_cds_annotations or show_int_rec_annotations:
            return GREY_CORE if bool(bdf.loc[block_id, "core"]) else GREY_ACC

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
    max_x = 0
    inversion_rects = []  # collect (label, left, width) for inverted blocks

    def _add_bar(label: str, left: int, width: int, color: str, strand: bool, block_id, block_pos: int):
        nonlocal max_x
        max_x = max(max_x, int(left) + int(width))
        if not strand:
            inversion_rects.append((label, int(left), int(width)))
        fig.add_bar(
            x=[width],
            y=[label],
            base=[left],
            orientation="h",
            marker=dict(color=color, line=dict(color=("black" if strand else "red"), width=1)),
            customdata=[[left, width, left + width, str(block_id), strand, block_pos]],
            hovertemplate=(
                "Label = %{y}"
                "<br>Start = %{customdata[0]}"
                "<br>Length = %{customdata[1]}"
                "<br>End = %{customdata[2]}"
                "<br>Block = %{customdata[3]}"
                "<br>Strand = %{customdata[4]:+, -}"
                "<br>Block position = %{customdata[5]}"
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

    # add a star per isolate based on cluster_map
    if cluster_map and add_cluster_annotation:
        clusters = sorted(set(cluster_map.values()))
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
        random.shuffle(palette)
        cluster_color = {cid: palette[i % len(palette)] for i, cid in enumerate(clusters)}

        star_x = -2000
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
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="star", size=14, color=cs),
                hoverinfo="skip", showlegend=False,
            ))

        for cid, color in cluster_color.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="star", size=14, color=color),
                name=f"Cluster {cid}",
                hoverinfo="skip",
            ))

        fig.update_layout(legend_title_text="Clusters")

    # overlay Integrase / Recombinase annotations (derived from CDS "product")
    if show_int_rec_annotations and annotations_gff_path:
        gdf = read_gff3_cds_products(annotations_gff_path)

        # filter CDS products containing integrase or recombinase (case-insensitive)
        prod = gdf["product"].fillna("").str.lower()
        ir = gdf[prod.str.contains("integrase") | prod.str.contains("recombinase")].copy()

        if not ir.empty:
            legend_added = False
            for label in y_labels:
                sub_ir = ir[ir["seqid"] == label]
                if sub_ir.empty:
                    continue

                showleg = not legend_added
                legend_added = True

                fig.add_trace(go.Bar(
                    x=(sub_ir["end"] - sub_ir["start"]).tolist(),
                    y=[label] * len(sub_ir),
                    base=(sub_ir["start"]).tolist(),
                    orientation="h",
                    marker=dict(color=_rgba("rgb(166,216,84)", annotation_alpha), line=dict(width=0)),
                    name="Integrase / Recombinase",
                    showlegend=showleg,
                    customdata=list(zip(sub_ir["end"].tolist(), sub_ir["product"].tolist())),
                    hovertemplate=(
                        "<b>Integrase / Recombinase</b>"
                        "<br>%{customdata[1]}"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{x:d}"
                        "<extra></extra>"
                    ),
                ))

    # overlay defense system, prophage, IS annotations
    if show_mges_annotations and mges_gff_path:
        ann = read_gff3_annotations(mges_gff_path)

        DEF_COLOR = "rgb(152,78,163)"  # purple, far from inversion red
        PROPH_COLOR = "rgb(27,158,119)"
        IS_BASE = (55, 126, 184)

        # stable IS subtype -> color mapping (computed once from whole file)
        is_types = sorted(ann.loc[ann["feature"] == "IS", "is_subtype"].dropna().unique())
        is_shades = _shades_from_base_rgb(IS_BASE, max(1, len(is_types)))
        is_color = {t: is_shades[i] for i, t in enumerate(is_types)}

        legend_seen = set()

        def _add_anno_bar(x, y, base, color_rgb, name, end):
            showleg = name not in legend_seen
            if showleg:
                legend_seen.add(name)

            fig.add_trace(go.Bar(
                x=x, # width
                y=y,
                base=base, # left start
                orientation="h",
                marker=dict(color=_rgba(color_rgb, annotation_alpha), line=dict(width=0)),
                name=name,
                showlegend=showleg,
                hovertemplate=f"<b>{name}</b><br>Start = %{{base:d}}<br>End = %{{customdata:d}}<br>Length = %{{x:d}}<extra></extra>",
                customdata=end,
            ))

        # add annotation bars for every row label (isolates + consensus), currently not done but one could add annotations to the consensus paths in the dataframe or gff3 file to also color the consensus tracks 
        for label in y_labels:
            sub = ann[ann["seqid"] == label]
            if sub.empty:
                continue

            # add prophage annotations
            ph = sub[sub["feature"] == "prophage"]
            if not ph.empty:
                _add_anno_bar(
                    x=(ph["end"] - ph["start"]).tolist(),
                    y=[label] * len(ph),
                    base=(ph["start"]).tolist(),
                    color_rgb=PROPH_COLOR,
                    name="Prophage",
                    end=ph["end"].tolist(),
                )

            # add defense system annotations
            ds = sub[sub["feature"] == "defense_system"]
            if not ds.empty:
                _add_anno_bar(
                    x=(ds["end"] - ds["start"]).tolist(),
                    y=[label] * len(ds),
                    base=(ds["start"]).tolist(),
                    color_rgb=DEF_COLOR,
                    name="Defense system",
                    end=ds["end"].tolist(),
                )

            # add IS annotations
            isdf = sub[sub["feature"] == "IS"].copy()
            if not isdf.empty:
                for type, istype_df in isdf.groupby("is_subtype", dropna=False):
                    type = type if pd.notna(type) else "IS"
                    name = f"IS:{type}"
                    col = is_color.get(type, _rgb_str(*IS_BASE))
                    _add_anno_bar(
                        x=(istype_df["end"] - istype_df["start"]).tolist(),
                        y=[label] * len(istype_df),
                        base=(istype_df["start"]).tolist(),
                        color_rgb=col,
                        name=name,
                        end=istype_df["end"].tolist(),
                    )

    # overlay gene CDS annotations (product labels)
    if show_cds_annotations and annotations_gff_path:
        gdf = read_gff3_cds_products(annotations_gff_path)

        CDS_COLOR = "rgb(240,228,66)"  # orange, distinct from inversion red / IS blue / prophage green / defense purple

        # keep a separate legend guard so "Genes (CDS)" appears once
        gene_legend_added = False

        for label in y_labels:
            subg = gdf[gdf["seqid"] == label]
            if subg.empty:
                continue

            # show legend only once globally
            showleg = not gene_legend_added
            gene_legend_added = True

            fig.add_trace(go.Bar(
                x=(subg["end"] - subg["start"]).tolist(),
                y=[label] * len(subg),
                base=(subg["start"]).tolist(),
                orientation="h",
                marker=dict(color=_rgba(CDS_COLOR, cds_annotation_alpha), line=dict(width=0)),
                name="Coding Sequence (CDS)",
                showlegend=showleg,
                customdata=list(zip(subg["end"].tolist(), subg["product"].tolist())),
                hovertemplate=(
                    "<b>CDS:</b> %{customdata[1]}"
                    "<br>Start = %{base:d}"
                    "<br>End = %{customdata[0]:d}"
                    "<br>Length = %{x:d}"
                    "<extra></extra>"
                ),
            ))

    # overlay insertions and deletions (only when consensus paths are defined)
    if show_indels and indels_base_path and junction_name and show_consensus and consensus_paths:

        DELETION_COLOR = "rgb(139,0,0)"  # dark red for deletions

        insertion_legend_added = False
        deletion_legend_added = False

        for i, cons_path in enumerate(consensus_paths):
            cons_label = f"consensus_{i+1}"

            # Load insertions for this consensus
            # Path: <indels_base_path>/insertions/<junction_name>/consensus<N>/insertions_summary.csv
            insertions_file = os.path.join(indels_base_path, "insertions", junction_name, f"consensus{i+1}", "insertions_summary.csv")
            if os.path.exists(insertions_file):
                ins_df = pd.read_csv(insertions_file)

                for label in y_labels:
                    # Match genome_name to isolate label (for isolates) or consensus label
                    sub_ins = ins_df[ins_df["genome_name"] == label]
                    if sub_ins.empty:
                        continue

                    showleg = not insertion_legend_added
                    insertion_legend_added = True

                    # Extract segment numbers from insertion names (e.g., "segment_0" -> "0")
                    segment_nums = [s.split("_")[-1] if "_" in s else s.replace("segment", "") for s in sub_ins["insertion"].tolist()]
                    block_counts = [str(p).count("[") for p in sub_ins["path"].tolist()]

                    fig.add_trace(go.Bar(
                        x=(sub_ins["end_pos"] - sub_ins["start_pos"]).tolist(),
                        y=[label] * len(sub_ins),
                        base=(sub_ins["start_pos"]).tolist(),
                        orientation="h",
                        marker=dict(
                            color="rgba(0,0,0,0)",  # transparent background
                            pattern=dict(
                                shape="/",  # diagonal lines
                                fgcolor="black",
                                size=6,
                                solidity=0.3,
                            ),
                            line=dict(width=1, color="black"),
                        ),
                        name="Insertion",
                        showlegend=showleg,
                        customdata=list(zip(
                            sub_ins["end_pos"].tolist(),
                            sub_ins["length"].tolist(),
                            segment_nums,
                            sub_ins["strand"].tolist(),
                            block_counts,
                        )),
                        hovertemplate=(
                            "<b>Insertion (#%{customdata[2]})</b>"
                            "<br>Start = %{base:d}"
                            "<br>End = %{customdata[0]:d}"
                            "<br>Length = %{customdata[1]:d}"
                            "<br>Strand = %{customdata[3]}"
                            "<br>Blocks = %{customdata[4]}"
                            "<extra></extra>"
                        ),
                    ))

        # Load deletions (all deletions are in a single file per junction)
        # Path: <indels_base_path>/deletions/<junction_name>/all_deletions_summary.csv
        deletions_file = os.path.join(indels_base_path, "deletions", junction_name, "all_deletions_summary.csv")
        if os.path.exists(deletions_file):
            del_df = pd.read_csv(deletions_file)

            # Collect all deletion markers, grouping by (label, position)
            del_grouped = {}  # (label, position) -> list of row dicts

            for label in y_labels:
                sub_del = del_df[del_df["genome_name"] == label]
                if sub_del.empty:
                    continue

                for _, row in sub_del.iterrows():
                    key = (label, row["position"])
                    del_name = row["deletion"]
                    del_num = del_name.replace("deletion", "") if "deletion" in del_name else del_name
                    n_blocks = str(row.get("path", "")).count("[")
                    del_grouped.setdefault(key, []).append({
                        "num": del_num, "length": row["length"], "strand": row.get("strand", ""), "blocks": n_blocks,
                    })

            if del_grouped:
                del_xs = []
                del_ys = []
                del_hovertexts = []

                for (label, pos), entries in del_grouped.items():
                    del_xs.append(pos)
                    del_ys.append(label)
                    lines = [f"<b>Deletion (#{e['num']})</b> Length={e['length']:g} Strand={e['strand']} Blocks={e['blocks']}" for e in entries]
                    hover = f"Position = {int(pos)}<br>" + "<br>".join(lines)
                    del_hovertexts.append(hover)

                fig.add_trace(go.Scatter(
                    x=del_xs,
                    y=del_ys,
                    mode="markers",
                    marker=dict(
                        symbol="line-ns",
                        size=20,
                        line=dict(color=_rgba(DELETION_COLOR, annotation_alpha), width=4),
                        color=_rgba(DELETION_COLOR, annotation_alpha),
                    ),
                    name="Deletion",
                    showlegend=True,
                    hovertemplate="%{text}<extra></extra>",
                    text=del_hovertexts,
                ))

    # Redraw inversion borders on top of overlays only when indels are shown
    if inversion_rects and show_indels and indels_base_path:
        fig.add_bar(
            x=[w for _, _, w in inversion_rects],
            y=[lbl for lbl, _, _ in inversion_rects],
            base=[l for _, l, _ in inversion_rects],
            orientation="h",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="red", width=1)),
            hoverinfo="skip",
            showlegend=False,
        )

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            symbol="square",
            size=12,
            color="rgba(0,0,0,0)",   # transparent fill
            line=dict(color="red", width=2),
        ),
        name="Inversion",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.05,
            y=0.99,
            xanchor="left",
            yanchor="top",
            yref="container",
            font=dict(size=18, family="Arial", color="black"),
            pad=dict(l=10, t=10),
        ),
        barmode=("overlay" if ((show_mges_annotations and mges_gff_path) or (show_cds_annotations and annotations_gff_path) or (show_int_rec_annotations and annotations_gff_path) or (show_indels and indels_base_path)) else "stack"), # if annotations or indels are on we want overlay (not stack) to allow them to be semi-transparent on top of blocks
        bargap=0.08,
        xaxis=dict(
            title="genomic position (bp)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.2)",
            range=[-max(1, int(0.05 * max_x)), max_x],
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
        margin=dict(l=140, r=20, t=100, b=40),
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

def plot_pairwise_distance_hist(distances, bins=30, figsize=(6, 4),
                                vline=None, vline_kwargs=None,
                                title="Pairwise Distance Distribution"):
    """
    Plot a histogram of pairwise distances with an optional vertical line.

    Parameters
    ----------
    distances : array-like
        1D array of pairwise distances.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.
    vline : float or None
        If provided, draw a vertical line at this x-position.
    vline_kwargs : dict or None
        Custom style for the vertical line.
    title : str
        Plot title.
    """
    distances = np.asarray(distances)

    if vline_kwargs is None:
        vline_kwargs = dict(color="black", linestyle="--", linewidth=1.5)

    fig, ax = plt.subplots(figsize=figsize)

    # histogram
    ax.hist(distances, bins=bins, density=True, alpha=0.7)

    # vertical line
    if vline is not None:
        ax.axvline(vline, **vline_kwargs)

    # labels
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_block_distance_distribution(pair_dists, block_list, bins=30, cols=4,
                                 figsize=(14, 20), vline=None, vline_kwargs=None, same_xrange=False):
    """
    pair_dists: dict {block_id: 1D numpy array of pairwise distances}
    block_list: list or Series of block IDs to plot
    vline: float or None — if given, draw vertical line at this x-value
    vline_kwargs: dict passed to ax.axvline() (e.g. color, linestyle)
    """
    block_list = list(block_list)
    n = len(block_list)

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # default style for vertical line
    if vline_kwargs is None:
        vline_kwargs = dict(color="red", linestyle="--", linewidth=1.5)
    
    if same_xrange:
        all_dists = []
        for block_id in block_list:
            d = pair_dists.get(block_id)
            if d is not None and d.size > 0:
                all_dists.append(d)

        all_dists_concat = np.concatenate(all_dists)

        # global x-axis range
        xmin = float(all_dists_concat.min())
        xmax = float(all_dists_concat.max())

    for ax, block_id in zip(axes, block_list):
        d = pair_dists.get(block_id)

        if d is None or d.size == 0:
            ax.text(0.5, 0.5, f"No data\nBlock {block_id}",
                    ha='center', va='center', fontsize=10)
            ax.set_axis_off()
            continue

        # histogram
        ax.hist(d, bins=bins, density=True, alpha=0.7)

        # vertical threshold line
        if vline is not None:
            ax.axvline(vline, **vline_kwargs)

        # titles & labels
        ax.set_title(f"Block {block_id}")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")

        if same_xrange:
            ax.set_xlim(xmin, xmax)

    # Hide unused axes
    for ax in axes[len(block_list):]:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

def plot_snp_pos_distribution(snp_pos, cutoff, bins=50, title=None):
    """
    Plot histogram of SNP positions and draw a vertical line at `cutoff`.

    Args:
        snp_pos (list[int]): SNP column positions.
        cutoff (int): cutoff position between core blocks (same coordinate system as snp_pos).
        bins (int): histogram bins.
        title (str|None): optional plot title.
    """
    if not snp_pos:
        raise ValueError("snp_pos is empty")

    xs = snp_pos
    x_cut = cutoff

    # Plot
    plt.figure()
    plt.hist(xs, bins=bins)
    plt.axvline(x_cut, linewidth=1.5, color = 'black', linestyle='--', label='Core block boundary')

    plt.xlabel("SNP position")
    plt.ylabel("Count")
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def plot_pangraph_base_for_dash(
    pan: pp.Pangraph,
    show_consensus: bool = False,
    consensus_paths: list = None,
    assignments: pd.DataFrame = None,
    order: str = "tree",
    cluster_map: dict = None,
    add_cluster_annotation: bool = True,
    title: str = "",
    grey_mode: bool = False,
):
    """
    Plots the base block structure of a junction pangraph using Plotly, without annotations.
    Returns the figure, y_labels, and max_x for potential further processing.
    """
    bdf = pan.to_blockstats_df()

    GREY_CORE = "rgb(220,220,220)"
    GREY_ACC = "rgb(190,190,190)"

    n_core = int(bdf["core"].sum())
    n_acc = int(len(bdf) - n_core)
    cgen_acc = iter(sns.color_palette("rainbow", n_acc))
    cgen_core = iter(sns.color_palette("pastel", n_core))
    block_colors: dict = {}

    def get_block_color(block_id):
        if grey_mode:
            return GREY_CORE if bool(bdf.loc[block_id, "core"]) else GREY_ACC

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
    max_x = 0
    inversion_rects = []  # collect (label, left, width) for inverted blocks

    def _add_bar(label: str, left: int, width: int, color: str, strand: bool, block_id, block_pos: int):
        nonlocal max_x
        max_x = max(max_x, int(left) + int(width))
        if not strand:
            inversion_rects.append((label, int(left), int(width)))
        fig.add_bar(
            x=[width],
            y=[label],
            base=[left],
            orientation="h",
            marker=dict(color=color, line=dict(color=("black" if strand else "red"), width=1)),
            customdata=[[left, width, left + width, str(block_id), strand, block_pos]],
            hovertemplate=(
                "Label = %{y}"
                "<br>Start = %{customdata[0]}"
                "<br>Length = %{customdata[1]}"
                "<br>End = %{customdata[2]}"
                "<br>Block = %{customdata[3]}"
                "<br>Strand = %{customdata[4]:+, -}"
                "<br>Block position = %{customdata[5]}"
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

    # add a star per isolate based on cluster_map
    if cluster_map and add_cluster_annotation:
        clusters = sorted(set(cluster_map.values()))
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
        random.shuffle(palette)
        cluster_color = {cid: palette[i % len(palette)] for i, cid in enumerate(clusters)}

        star_x = -2000
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
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="star", size=14, color=cs),
                hoverinfo="skip", showlegend=False,
            ))

        for cid, color in cluster_color.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="star", size=14, color=color),
                name=f"Cluster {cid}",
                hoverinfo="skip",
            ))

        fig.update_layout(legend_title_text="Legend")

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            symbol="square",
            size=12,
            color="rgba(0,0,0,0)",   # transparent fill
            line=dict(color="red", width=2),
        ),
        name="Inversion",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.05,
            y=0.99,
            xanchor="left",
            yanchor="top",
            yref="container",
            font=dict(size=18, family="Arial", color="black"),
            pad=dict(l=10, t=10),
        ),
        barmode="stack",
        bargap=0.08,
        xaxis=dict(
            title="genomic position (bp)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.2)",
            range=[-max(1, int(0.05 * max_x)), max_x],
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
        margin=dict(l=140, r=20, t=100, b=40),
        height=max(300, int(len(y_labels) * 22)),
        template="plotly_white",
    )
    return fig, y_labels, max_x, inversion_rects


def add_annotations_for_dash(
    fig: go.Figure,
    y_labels: list,
    show_mges_annotations: bool = False,
    show_int_rec_annotations: bool = False,
    mges_gff_path: str = None,
    show_cds_annotations: bool = False,
    annotations_gff_path: str = None,
    annotation_alpha: float = 0.70,
    cds_annotation_alpha: float = 0.30,
    show_indels: bool = False,
    indels_base_path: str = None,
    junction_name: str = None,
    consensus_paths: list = None,
    inversion_rects: list = None,
):
    """
    Adds annotation layers to an existing Plotly figure of a pangraph.
    """
    # overlay Integrase / Recombinase annotations (derived from CDS "product")
    if show_int_rec_annotations and annotations_gff_path:
        gdf = read_gff3_cds_products(annotations_gff_path)

        # filter CDS products containing integrase or recombinase (case-insensitive)
        prod = gdf["product"].fillna("").str.lower()
        ir = gdf[prod.str.contains("integrase") | prod.str.contains("recombinase")].copy()

        if not ir.empty:
            legend_added = False
            for label in y_labels:
                sub_ir = ir[ir["seqid"] == label]
                if sub_ir.empty:
                    continue

                showleg = not legend_added
                legend_added = True

                fig.add_trace(go.Bar(
                    x=(sub_ir["end"] - sub_ir["start"]).tolist(),
                    y=[label] * len(sub_ir),
                    base=(sub_ir["start"]).tolist(),
                    orientation="h",
                    marker=dict(color=_rgba("rgb(166,216,84)", annotation_alpha), line=dict(width=0)),
                    name="Integrase / Recombinase",
                    showlegend=showleg,
                    customdata=list(zip(sub_ir["end"].tolist(), sub_ir["product"].tolist())),
                    hovertemplate=(
                        "<b>Integrase / Recombinase</b>"
                        "<br>%{customdata[1]}"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{x:d}"
                        "<extra></extra>"
                    ),
                ))

    # overlay defense system, prophage, IS annotations
    if show_mges_annotations and mges_gff_path:
        ann = read_gff3_annotations(mges_gff_path)

        DEF_COLOR = "rgb(152,78,163)"  # purple, far from inversion red
        PROPH_COLOR = "rgb(27,158,119)"
        IS_BASE = (55, 126, 184)

        # stable IS subtype -> color mapping (computed once from whole file)
        is_types = sorted(ann.loc[ann["feature"] == "IS", "is_subtype"].dropna().unique())
        is_shades = _shades_from_base_rgb(IS_BASE, max(1, len(is_types)))
        is_color = {t: is_shades[i] for i, t in enumerate(is_types)}

        legend_seen = set()

        def _add_anno_bar(x, y, base, color_rgb, name, end):
            showleg = name not in legend_seen
            if showleg:
                legend_seen.add(name)

            fig.add_trace(go.Bar(
                x=x, # width
                y=y,
                base=base, # left start
                orientation="h",
                marker=dict(color=_rgba(color_rgb, annotation_alpha), line=dict(width=0)),
                name=name,
                showlegend=showleg,
                hovertemplate=f"<b>{name}</b><br>Start = %{{base:d}}<br>End = %{{customdata:d}}<br>Length = %{{x:d}}<extra></extra>",
                customdata=end,
            ))

        # add annotation bars for every row label (isolates + consensus), currently not done but one could add annotations to the consensus paths in the dataframe or gff3 file to also color the consensus tracks 
        for label in y_labels:
            sub = ann[ann["seqid"] == label]
            if sub.empty:
                continue

            # add prophage annotations
            ph = sub[sub["feature"] == "prophage"]
            if not ph.empty:
                _add_anno_bar(
                    x=(ph["end"] - ph["start"]).tolist(),
                    y=[label] * len(ph),
                    base=(ph["start"]).tolist(),
                    color_rgb=PROPH_COLOR,
                    name="Prophage",
                    end=ph["end"].tolist(),
                )

            # add defense system annotations
            ds = sub[sub["feature"] == "defense_system"]
            if not ds.empty:
                _add_anno_bar(
                    x=(ds["end"] - ds["start"]).tolist(),
                    y=[label] * len(ds),
                    base=(ds["start"]).tolist(),
                    color_rgb=DEF_COLOR,
                    name="Defense system",
                    end=ds["end"].tolist(),
                )

            # add IS annotations
            isdf = sub[sub["feature"] == "IS"].copy()
            if not isdf.empty:
                for type, istype_df in isdf.groupby("is_subtype", dropna=False):
                    type = type if pd.notna(type) else "IS"
                    name = f"IS:{type}"
                    col = is_color.get(type, _rgb_str(*IS_BASE))
                    _add_anno_bar(
                        x=(istype_df["end"] - istype_df["start"]).tolist(),
                        y=[label] * len(istype_df),
                        base=(istype_df["start"]).tolist(),
                        color_rgb=col,
                        name=name,
                        end=istype_df["end"].tolist(),
                    )

    # overlay gene CDS annotations (product labels)
    if show_cds_annotations and annotations_gff_path:
        gdf = read_gff3_cds_products(annotations_gff_path)

        CDS_COLOR = "rgb(240,228,66)"  # orange, distinct from inversion red / IS blue / prophage green / defense purple

        # keep a separate legend guard so "Genes (CDS)" appears once
        gene_legend_added = False

        for label in y_labels:
            subg = gdf[gdf["seqid"] == label]
            if subg.empty:
                continue

            # show legend only once globally
            showleg = not gene_legend_added
            gene_legend_added = True

            fig.add_trace(go.Bar(
                x=(subg["end"] - subg["start"]).tolist(),
                y=[label] * len(subg),
                base=(subg["start"]).tolist(),
                orientation="h",
                marker=dict(color=_rgba(CDS_COLOR, cds_annotation_alpha), line=dict(width=0)),
                name="Coding Sequence (CDS)",
                showlegend=showleg,
                customdata=list(zip(subg["end"].tolist(), subg["product"].tolist())),
                hovertemplate=(
                    "<b>CDS:</b> %{customdata[1]}"
                    "<br>Start = %{base:d}"
                    "<br>End = %{customdata[0]:d}"
                    "<br>Length = %{x:d}"
                    "<extra></extra>"
                ),
            ))

    # overlay insertions and deletions (only when consensus paths are defined)
    if show_indels and indels_base_path and junction_name and consensus_paths:
        import os

        DELETION_COLOR = "rgb(139,0,0)"  # dark red for deletions

        insertion_legend_added = False

        for i, cons_path in enumerate(consensus_paths):
            # Load insertions for this consensus
            # Path: <indels_base_path>/insertions/<junction_name>/consensus<N>/insertions_summary.csv
            insertions_file = os.path.join(indels_base_path, "insertions", junction_name, f"consensus{i+1}", "insertions_summary.csv")
            if os.path.exists(insertions_file):
                ins_df = pd.read_csv(insertions_file)

                for label in y_labels:
                    sub_ins = ins_df[ins_df["genome_name"] == label]
                    if sub_ins.empty:
                        continue

                    showleg = not insertion_legend_added
                    insertion_legend_added = True

                    # Extract segment numbers from insertion names (e.g., "segment_0" -> "0")
                    segment_nums = [s.split("_")[-1] if "_" in s else s.replace("segment", "") for s in sub_ins["insertion"].tolist()]
                    block_counts = [str(p).count("[") for p in sub_ins["path"].tolist()]

                    fig.add_trace(go.Bar(
                        x=(sub_ins["end_pos"] - sub_ins["start_pos"]).tolist(),
                        y=[label] * len(sub_ins),
                        base=(sub_ins["start_pos"]).tolist(),
                        orientation="h",
                        marker=dict(
                            color="rgba(0,0,0,0)",  # transparent background
                            pattern=dict(
                                shape="/",  # diagonal lines
                                fgcolor="black",
                                size=6,
                                solidity=0.3,
                            ),
                            line=dict(width=1, color="black"),
                        ),
                        name="Insertion",
                        showlegend=showleg,
                        customdata=list(zip(
                            sub_ins["end_pos"].tolist(),
                            sub_ins["length"].tolist(),
                            segment_nums,
                            sub_ins["strand"].tolist(),
                            block_counts,
                        )),
                        hovertemplate=(
                            "<b>Insertion (#%{customdata[2]})</b>"
                            "<br>Start = %{base:d}"
                            "<br>End = %{customdata[0]:d}"
                            "<br>Length = %{customdata[1]:d}"
                            "<br>Strand = %{customdata[3]}"
                            "<br>Blocks = %{customdata[4]}"
                            "<extra></extra>"
                        ),
                    ))

        # Load deletions (all deletions are in a single file per junction)
        # Path: <indels_base_path>/deletions/<junction_name>/all_deletions_summary.csv
        deletions_file = os.path.join(indels_base_path, "deletions", junction_name, "all_deletions_summary.csv")
        if os.path.exists(deletions_file):
            del_df = pd.read_csv(deletions_file)

            # Collect all deletion markers, grouping by (label, position)
            del_grouped = {}  # (label, position) -> list of row dicts

            for label in y_labels:
                sub_del = del_df[del_df["genome_name"] == label]
                if sub_del.empty:
                    continue

                for _, row in sub_del.iterrows():
                    key = (label, row["position"])
                    del_name = row["deletion"]
                    del_num = del_name.replace("deletion", "") if "deletion" in str(del_name) else del_name
                    n_blocks = str(row.get("path", "")).count("[")
                    del_grouped.setdefault(key, []).append({
                        "num": del_num, "length": row["length"], "strand": row.get("strand", ""), "blocks": n_blocks,
                    })

            if del_grouped:
                del_xs = []
                del_ys = []
                del_hovertexts = []

                for (label, pos), entries in del_grouped.items():
                    del_xs.append(pos)
                    del_ys.append(label)
                    lines = [f"<b>Deletion (#{e['num']})</b> Length={e['length']:g} Strand={e['strand']} Blocks={e['blocks']}" for e in entries]
                    hover = f"Position = {int(pos)}<br>" + "<br>".join(lines)
                    del_hovertexts.append(hover)

                fig.add_trace(go.Scatter(
                    x=del_xs,
                    y=del_ys,
                    mode="markers",
                    marker=dict(
                        symbol="line-ns",
                        size=20,
                        line=dict(color=DELETION_COLOR, width=4),
                        color=DELETION_COLOR,
                    ),
                    name="Deletion",
                    showlegend=True,
                    hovertemplate="%{text}<extra></extra>",
                    text=del_hovertexts,
                ))

    # Redraw inversion borders on top of overlays only when indels are shown
    if inversion_rects and show_indels and indels_base_path:
        fig.add_bar(
            x=[w for _, _, w in inversion_rects],
            y=[lbl for lbl, _, _ in inversion_rects],
            base=[l for _, l, _ in inversion_rects],
            orientation="h",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="red", width=1)),
            hoverinfo="skip",
            showlegend=False,
        )

    return fig