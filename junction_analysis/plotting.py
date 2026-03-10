import numpy as np
import math
import random
import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
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
        if ann.empty or "feature" not in ann.columns:
            ann = pd.DataFrame(columns=["seqid", "feature", "start", "end", "attrs", "is_subtype"])

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
                    pos_str = "N/A" if (pos != pos) else str(int(pos))  # NaN check
                    hover = f"Position = {pos_str}<br>" + "<br>".join(lines)
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

        # Load inversions
        INVERSION_COLOR = "red"

        inversion_legend_added = False

        for i, cons_path in enumerate(consensus_paths):
            inversions_file = os.path.join(indels_base_path, "inversions", junction_name, f"consensus_{i+1}", "inversions_summary.csv")
            if not os.path.exists(inversions_file):
                continue
            inv_df = pd.read_csv(inversions_file)

            for label in y_labels:
                sub_inv = inv_df[inv_df["genome_name"] == label]
                if sub_inv.empty:
                    continue

                showleg = not inversion_legend_added
                inversion_legend_added = True

                inv_nums = [str(name).replace("inversion", "") for name in sub_inv["inversion"].tolist()]
                block_counts = [str(p).count("[") for p in sub_inv["path"].tolist()]

                fig.add_trace(go.Bar(
                    x=(sub_inv["end_pos"] - sub_inv["start_pos"]).tolist(),
                    y=[label] * len(sub_inv),
                    base=(sub_inv["start_pos"]).tolist(),
                    orientation="h",
                    marker=dict(
                        color="rgba(0,0,0,0)",
                        pattern=dict(
                            shape="\\",
                            fgcolor=INVERSION_COLOR,
                            size=6,
                            solidity=0.3,
                        ),
                        line=dict(width=1, color=INVERSION_COLOR),
                    ),
                    name="Inversion",
                    showlegend=showleg,
                    customdata=list(zip(
                        sub_inv["end_pos"].tolist(),
                        sub_inv["length"].tolist(),
                        inv_nums,
                        sub_inv["strand"].tolist(),
                        block_counts,
                    )),
                    hovertemplate=(
                        "<b>Inversion (#%{customdata[2]})</b>"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{customdata[1]:d}"
                        "<br>Strand = %{customdata[3]}"
                        "<br>Blocks = %{customdata[4]}"
                        "<extra></extra>"
                    ),
                ))

        # Load translocations — draw arrows from start to end of translocated region
        TRANSLOCATION_COLOR = "rgb(0,0,255)"  # blue
        MIN_ARROW_SPAN = max_x * 0.01  # 1 % of total x range keeps arrowhead always visible
        any_trans_drawn = False

        for i, cons_path in enumerate(consensus_paths):
            trans_file = os.path.join(indels_base_path, "translocations", junction_name, f"consensus_{i+1}", "translocations_summary.csv")
            if not os.path.exists(trans_file):
                continue
            trans_df = pd.read_csv(trans_file)

            hover_x, hover_y, hover_custom = [], [], []

            for _, row in trans_df.iterrows():
                label = row["genome_name"]
                if label not in y_labels:
                    continue

                any_trans_drawn = True
                start = row["start_pos"]
                end   = row["end_pos"]
                if abs(end - start) < MIN_ARROW_SPAN:
                    mid   = (start + end) / 2
                    start = mid - MIN_ARROW_SPAN / 2
                    end   = mid + MIN_ARROW_SPAN / 2

                fig.add_annotation(
                    x=end, y=label,
                    ax=start, ay=label,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=TRANSLOCATION_COLOR,
                    text="",
                )

                hover_x.append((row["start_pos"] + row["end_pos"]) / 2)
                hover_y.append(label)
                hover_custom.append([
                    row["translocation"],
                    row["start_pos"],
                    row["end_pos"],
                    row["length"],
                ])

            if hover_x:
                fig.add_trace(go.Scatter(
                    x=hover_x,
                    y=hover_y,
                    mode="markers",
                    marker=dict(size=10, color="rgba(0,0,0,0)"),
                    customdata=hover_custom,
                    hovertemplate=(
                        "<b>Translocation</b><br>"
                        "Name: %{customdata[0]}<br>"
                        "Start: %{customdata[1]:d}<br>"
                        "End: %{customdata[2]:d}<br>"
                        "Length: %{customdata[3]:d}"
                        "<extra></extra>"
                    ),
                    name="Translocation",
                    showlegend=False,
                ))

        if any_trans_drawn:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="arrow-right", size=16, color=TRANSLOCATION_COLOR, angle=0),
                name="Translocation",
                showlegend=True,
                hoverinfo="skip",
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
        name="Negative strand",
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

def plot_pairwise_distance_hist(distances, bins=30, figsize=(7, 3.5),
                                vline=None, vline_kwargs=None,
                                percentage=False,
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
    percentage : bool
        If True, show y-axis as percentage of total distances. Default: False.
    title : str
        Plot title.
    """
    distances = np.asarray(distances)

    if vline_kwargs is None:
        vline_kwargs = dict(color="black", linestyle="--", linewidth=1.5)

    fig, ax = plt.subplots(figsize=figsize)

    bar_color = "#7394c2"
    ax.yaxis.grid(True, color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if percentage:
        counts, edges = np.histogram(distances, bins=bins)
        pcts = counts / counts.sum() * 100
        ax.bar(edges[:-1], pcts, width=np.diff(edges), align="edge",
               color=bar_color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_ylabel("% of pairwise distances", fontsize=11)
    else:
        ax.hist(distances, bins=bins, density=False,
                color=bar_color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_ylabel("Count", fontsize=11)

    if vline is not None:
        ax.axvline(vline, **vline_kwargs)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Pairwise patristic distance (substitutions per site)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
        name="Negative strand",
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


def plot_cluster_count_distribution(cluster_df, figsize=(7, 4), save_path=None):
    """
    Plot the distribution of the number of clusters per junction as a bar chart.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        Output of cluster_map_to_dataframe(). Must contain columns
        'junction_name' and 'n_clusters'.
    figsize : tuple
    save_path : str or None
        If provided, save the figure instead of showing it.
    """
    per_junction = cluster_df.drop_duplicates("junction_name")
    counts = per_junction["n_clusters"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index, counts.values, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of clusters per junction")
    ax.set_ylabel("Number of junctions")
    ax.set_title("Distribution of cluster counts across junctions")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_cluster_count_vs_diversity(
    cluster_df,
    jdf,
    diversity_columns,
    figsize=None,
    save_path=None,
):
    """
    Scatter plots of n_clusters per junction against junction diversity measures.

    One subplot per column in `diversity_columns`, laid out in a row.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        Output of cluster_map_to_dataframe(). Must contain 'junction_name'
        and 'n_clusters'.
    jdf : pd.DataFrame
        Junction summary DataFrame. Must contain 'junction_name' and all
        columns listed in `diversity_columns`.
    diversity_columns : list of str
        Column names in `jdf` to plot on the x-axis.
    figsize : tuple or None
        Figure size. Defaults to (4 * n_cols, 4).
    save_path : str or None
        If provided, save the figure; otherwise call plt.show().
    """
    n_cols = len(diversity_columns)
    if n_cols == 0:
        return

    if figsize is None:
        figsize = (4 * n_cols, 4)

    # One row per junction, merge n_clusters with jdf
    # cluster_df uses 'junction_name'; jdf uses 'edge'
    per_junction = cluster_df.drop_duplicates("junction_name")[["junction_name", "n_clusters"]]
    merged = per_junction.merge(jdf, left_on="junction_name", right_on="edge", how="inner")

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, diversity_columns):
        if col not in merged.columns:
            ax.set_visible(False)
            continue
        valid = merged[["n_clusters", col]].dropna()
        ax.scatter(valid[col], valid["n_clusters"], alpha=0.6, s=20,
                   color="#4C72B0", linewidths=0)
        ax.set_xlabel(col)
        ax.set_ylabel("Number of clusters")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cluster count vs. junction diversity")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_all_junctions_pairwise_distances(
    distances_df,
    exclude=None,
    bins=100,
    figsize=(7, 3.5),
    vline=None,
    vline_kwargs=None,
    log_y=False,
    percentage=False,
    title="default",
    save_path=None,
):
    """
    Plot a combined histogram of pairwise distances pooled across all junctions.

    Parameters
    ----------
    distances_df : pd.DataFrame
        Output of collect_all_pairwise_distances(), with columns
        'junction_name' and 'distance'.
    exclude : list of str or None
        Junction names to exclude from the plot. Default: None.
    bins : int
        Number of histogram bins. Default: 100.
    figsize : tuple
        Figure size.
    vline : float or None
        If provided, draw a vertical line at this x-position.
    vline_kwargs : dict or None
        Custom style for the vertical line.
    log_y : bool
        If True, use log scale on the y-axis. Default: False.
    percentage : bool
        If True, show y-axis as percentage of total distances. Default: False.
    save_path : str or None
        If provided, save the figure; otherwise call plt.show().
    """
    df = distances_df.copy()
    if exclude is not None:
        df = df[~df["junction_name"].isin(exclude)]

    distances = df["distance"].values
    if len(distances) == 0:
        print("No distances to plot.")
        return

    n_junctions = df["junction_name"].nunique()

    if vline_kwargs is None:
        vline_kwargs = dict(color="black", linestyle="--", linewidth=1.5)

    fig, ax = plt.subplots(figsize=figsize)

    bar_color = "#7394c2"
    ax.yaxis.grid(True, color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if percentage:
        counts, edges = np.histogram(distances, bins=bins)
        pcts = counts / counts.sum() * 100
        widths = np.diff(edges)
        ax.bar(edges[:-1], pcts, width=widths, align="edge",
               color=bar_color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_ylabel("% of pairwise distances", fontsize=11)
    else:
        ax.hist(distances, bins=bins, density=False,
                color=bar_color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_ylabel("Count", fontsize=11)

    if vline is not None:
        ax.axvline(vline, label=f"Pairwise distance cutoff for\nhomologous recombination ({vline})",
                   **vline_kwargs)
        ax.legend(frameon=False, fontsize=10)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Pairwise patristic distance (substitutions per site)", fontsize=11)
    ax.tick_params(labelsize=10)
    if title == "default":
        title = f"Core genome pairwise distances ({n_junctions} junctions)"
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_event_counts_distribution(counts_df, figsize=(10, 5), log_scale=True,
                                    log_x=None, log_y=None,
                                    subplots=False, share_y=True,
                                    save_path=None):
    """
    Plot the distribution of event counts per junction, coloured by event type.

    Each event type is shown as a normalised histogram (fraction of all junctions).
    For linear x, the zero bin is included as a normal bar at x=0.
    For log x, zero bins are shown in a separate narrow panel on the left
    with a broken-axis marker, because 0 cannot appear on a log x-axis.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Output of count_events_per_junction().
    figsize : tuple
        Matplotlib figure size.
    log_scale : bool
        Legacy parameter. If True (default), sets log_x=True and log_y=True
        unless those are specified explicitly.
    log_x : bool or None
        If True, use log scale on the x-axis (with broken-axis zero panel).
        Defaults to the value of log_scale.
    log_y : bool or None
        If True, use log scale on the y-axis.
        Defaults to the value of log_scale.
    subplots : bool
        If True, draw one subplot per event type stacked vertically, all sharing
        the same x-axis. If False (default), all types are overlaid on one axis.
    share_y : bool
        Only used when subplots=True. If True (default), all subplots share the
        same y-axis scale. If False, each subplot has an independent y-axis.
    save_path : str or None
        If provided, save the figure to this path instead of calling plt.show().
    """
    if log_x is None:
        log_x = log_scale
    if log_y is None:
        log_y = log_scale
    event_cols = {
        "insertion":     ("n_insertion",     "#4C72B0"),
        "deletion":      ("n_deletion",      "#DD8452"),
        "translocation": ("n_translocation", "#55A868"),
        "inversion":     ("n_inversion",     "#C44E52"),
    }
    n_types = len(event_cols)
    n_junctions = len(counts_df)
    gap_frac = 0.15

    max_count = max(
        counts_df[col].max()
        for col, _ in event_cols.values()
        if col in counts_df.columns
    )

    # ------------------------------------------------------------------ #
    # Subplots mode: one panel per event type, shared x-axis             #
    # ------------------------------------------------------------------ #
    if subplots:
        if not log_x:
            bin_edges = np.arange(0, max_count + 2) - 0.5
            fig, axes = plt.subplots(n_types, 1, figsize=figsize, sharex=True,
                                     sharey=share_y)
            fig.subplots_adjust(hspace=0.08)

            for ax, (label, (col, color)) in zip(axes, event_cols.items()):
                if col not in counts_df.columns:
                    ax.set_visible(False)
                    continue
                data = counts_df[col]
                counts_per_bin, _ = np.histogram(data, bins=bin_edges)
                bin_widths = np.diff(bin_edges)
                bin_centres = bin_edges[:-1] + bin_widths / 2
                ax.bar(bin_centres, counts_per_bin, width=bin_widths * (1 - gap_frac),
                       color=color, alpha=0.85, edgecolor="none")
                ax.set_ylabel(label.capitalize(), rotation=0, labelpad=60, va="center")
                if log_y:
                    ax.set_yscale("log")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            axes[-1].set_xlabel("Number of unique events per junction")
            fig.suptitle("Distribution of events per junction")
            plt.tight_layout()

        else:
            # Log scale: each row is a broken-axis pair (zero | log histogram)
            bin_edges = np.logspace(0, np.log10(max_count + 1), 30)

            fig, axes_pairs = plt.subplots(
                n_types, 2,
                figsize=figsize,
                gridspec_kw={"width_ratios": [1, 8], "wspace": 0.05},
            )
            fig.subplots_adjust(hspace=0.15)

            # Share y-axis across ALL panels (both columns, all rows)
            if share_y:
                ref_ax = axes_pairs[0, 0]
                for row in range(n_types):
                    for col_idx in range(2):
                        if not (row == 0 and col_idx == 0):
                            axes_pairs[row, col_idx].sharey(ref_ax)

            # Resolve axes sizes for bar-width matching
            fig.canvas.draw()

            for row, (label, (col, color)) in enumerate(event_cols.items()):
                ax0, ax1 = axes_pairs[row]

                if col not in counts_df.columns:
                    ax0.set_visible(False)
                    ax1.set_visible(False)
                    continue

                data = counts_df[col]

                # Match bar width in display pts across panels
                ax1_width_pts = ax1.get_window_extent().width
                log_range = np.log10(bin_edges[-1]) - np.log10(bin_edges[0])
                first_bin_log_frac = (np.log10(bin_edges[1]) - np.log10(bin_edges[0])) / log_range
                bar_pts = ax1_width_pts * first_bin_log_frac * (1 - gap_frac)
                ax0_width_pts = ax0.get_window_extent().width
                bar_width_ax0 = (bar_pts / ax0_width_pts) * 2.0  # ax0 spans [-1,1]

                # Zero bar
                n_zero = (data == 0).sum()
                ax0.bar(0, n_zero, width=bar_width_ax0,
                        color=color, alpha=0.85, edgecolor="none")
                ax0.set_xticks([0])
                ax0.set_xticklabels(["0"] if row == n_types - 1 else [""])
                ax0.set_xlim(-1.0, 1.0)
                if log_y:
                    ax0.set_yscale("log")
                ax0.spines["right"].set_visible(False)
                ax0.spines["top"].set_visible(False)
                ax0.tick_params(right=False, which="both")

                # Non-zero bins
                nonzero = data[data > 0]
                counts_per_bin, _ = np.histogram(nonzero, bins=bin_edges)
                bin_widths = np.diff(bin_edges)
                bw = bin_widths * (1 - gap_frac)
                bin_centres = bin_edges[:-1] + bin_widths / 2
                ax1.bar(bin_centres, counts_per_bin, width=bw,
                        color=color, alpha=0.85, edgecolor="none")
                ax1.set_xscale("log")
                ax1.spines["left"].set_visible(False)
                ax1.spines["top"].set_visible(False)
                ax1.tick_params(left=False, which="both")
                # Hide y tick labels on ax1 since it shares the scale with ax0
                ax1.tick_params(labelleft=False, which="both")
                if row < n_types - 1:
                    ax1.tick_params(labelbottom=False)
                    ax0.tick_params(labelbottom=False)

            axes_pairs[-1, 1].set_xlabel(
                "Number of unique events per junction (log scale)" if log_x
                else "Number of unique events per junction"
            )
            fig.suptitle("Distribution of events per junction")
            plt.tight_layout()

            # After layout is resolved, place per-row event-type labels and
            # the shared "Number of junctions" label to their left.
            fig.canvas.draw()
            x0 = axes_pairs[0, 0].get_position().x0
            for row, (label, (col, color)) in enumerate(event_cols.items()):
                ax0 = axes_pairs[row, 0]
                pos = ax0.get_position()
                y_center = (pos.y0 + pos.y1) / 2
                # event-type label horizontal, per row
                fig.text(x0 - 0.11, y_center, label.capitalize(),
                         va="center", ha="center", rotation=0, fontsize=9)
            # shared "Number of junctions" label centered across all rows
            fig.text(x0 - 0.045, 0.5, "Number of junctions",
                     va="center", ha="center", rotation="vertical", fontsize=9)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        return

    # ------------------------------------------------------------------ #
    # Non-log: single axis, 0 is a normal bin                            #
    # ------------------------------------------------------------------ #
    if not log_x:
        bin_edges = np.arange(0, max_count + 2) - 0.5  # integer-centred bins
        fig, ax = plt.subplots(figsize=figsize)

        for i, (label, (col, color)) in enumerate(event_cols.items()):
            if col not in counts_df.columns:
                continue
            data = counts_df[col]
            counts_per_bin, _ = np.histogram(data, bins=bin_edges)
            bin_widths = np.diff(bin_edges)
            usable = bin_widths * (1 - gap_frac)
            bar_width = usable / n_types
            bin_left = bin_edges[:-1] + bin_widths * (gap_frac / 2)
            x_centres = bin_left + i * bar_width + bar_width / 2
            ax.bar(x_centres, counts_per_bin, width=bar_width, color=color,
                   alpha=0.85, label=label.capitalize(), edgecolor="none")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Number of unique events per junction")
        ax.set_ylabel("Count (log scale)" if log_y else "Count")
        ax.set_title("Distribution of events per junction")
        ax.legend(title="Event type", frameon=True)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        return

    # ------------------------------------------------------------------ #
    # Log scale: broken axis — narrow left panel for zero, wide right     #
    # for log-spaced non-zero bins. Bar widths are matched in display pts #
    # so both panels look proportionate.                                  #
    # ------------------------------------------------------------------ #
    bin_edges = np.logspace(0, np.log10(max_count + 1), 30)

    fig, (ax0, ax1) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 8], "wspace": 0.05},
        sharey=True,
    )

    # Compute a representative bar width in display points from the right panel,
    # then use the same width (in ax0 data coords) for the zero bars.
    # We do this after the figure is created so the axes have a size.
    fig.canvas.draw()  # needed to resolve axes positions

    # Right panel: each histogram bar width in display points (use first bin as reference)
    ax1_width_pts = ax1.get_window_extent().width
    log_range = np.log10(bin_edges[-1]) - np.log10(bin_edges[0])
    first_bin_log_frac = (np.log10(bin_edges[1]) - np.log10(bin_edges[0])) / log_range
    one_type_frac = first_bin_log_frac * (1 - gap_frac) / n_types
    bar_pts = ax1_width_pts * one_type_frac  # display pts per bar in right panel

    # Convert that to data coords in ax0 (xlim will be set to [-1, 1])
    ax0_width_pts = ax0.get_window_extent().width
    ax0_xlim = 2.0  # ax0 spans [-1, 1]
    bar_width_ax0 = (bar_pts / ax0_width_pts) * ax0_xlim

    group_width_ax0 = bar_width_ax0 * n_types

    for i, (label, (col, color)) in enumerate(event_cols.items()):
        if col not in counts_df.columns:
            continue
        data = counts_df[col]

        # --- zero bar ---
        n_zero = (data == 0).sum()
        x_zero = -group_width_ax0 / 2 + i * bar_width_ax0 + bar_width_ax0 / 2
        ax0.bar(x_zero, n_zero, width=bar_width_ax0,
                color=color, alpha=0.85, edgecolor="none")

        # --- non-zero bins ---
        nonzero = data[data > 0]
        counts_per_bin, _ = np.histogram(nonzero, bins=bin_edges)
        bin_widths = np.diff(bin_edges)
        usable = bin_widths * (1 - gap_frac)
        bw = usable / n_types
        bin_left = bin_edges[:-1] + bin_widths * (gap_frac / 2)
        x_centres = bin_left + i * bw + bw / 2
        ax1.bar(x_centres, counts_per_bin, width=bw, color=color,
                alpha=0.85, label=label.capitalize(), edgecolor="none")


    # Left panel
    ax0.set_xticks([0])
    ax0.set_xticklabels(["0"])
    ax0.set_xlim(-1.0, 1.0)
    if log_y:
        ax0.set_yscale("log")
    ax0.set_ylabel("Count (log scale)" if log_y else "Count")
    ax0.spines["right"].set_visible(False)
    ax0.tick_params(right=False, which="both")

    # Right panel
    ax1.set_xscale("log")
    if log_y:
        ax1.set_yscale("log")
    ax1.set_xlabel("Number of unique events per junction (log scale)")
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(left=False, which="both")

    plt.tight_layout()

    fig.suptitle("Distribution of events per junction")
    ax1.legend(title="Event type", frameon=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_events_per_junction(counts_df, figsize=(12, 5), save_path=None):
    """
    Scatter plot: junctions on the x-axis (sorted by total event count),
    number of unique events on the y-axis, coloured by event type.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Output of count_events_per_junction(), with columns
        junction_name, n_insertions, n_deletions, n_translocations, n_inversions.
    figsize : tuple
        Matplotlib figure size.
    save_path : str or None
        If provided, save the figure to this path instead of calling plt.show().
    """
    event_cols = {
        "Insertion":     ("n_insertion",     "#4C72B0"),
        "Deletion":      ("n_deletion",      "#DD8452"),
        "Translocation": ("n_translocation", "#55A868"),
        "Inversion":     ("n_inversion",     "#C44E52"),
    }

    # Sort junctions by total event count descending
    total = sum(
        counts_df[col] for col, _ in event_cols.values() if col in counts_df.columns
    )
    order = total.argsort()[::-1].values
    sorted_df = counts_df.iloc[order].reset_index(drop=True)
    x = np.arange(len(sorted_df))

    fig, ax = plt.subplots(figsize=figsize)

    for label, (col, color) in event_cols.items():
        if col not in sorted_df.columns:
            continue
        y = sorted_df[col].values
        mask = y > 0
        ax.scatter(x[mask], y[mask], color=color, label=label, s=10, alpha=0.7, linewidths=0)

    ax.set_yscale("log")
    ax.set_xlabel("Junctions (sorted by total event count)")
    ax.set_ylabel("Number of unique events (log scale)")
    ax.set_title("Unique events per junction")
    ax.legend(title="Event type", frameon=True)
    ax.set_xlim(-1, len(sorted_df))

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_event_length_distribution(
    deduped_df,
    min_length_threshold=200,
    bins=50,
    log_x=False,
    log_y=True,
    y_max=None,
    x_max=None,
    filter_below_threshold=False,
    figsize=(10, 8),
    save_path=None,
):
    """
    Plot the length distribution of each event type as 4 histograms stacked
    vertically, using the same colour scheme as plot_event_counts_distribution.

    All subplots share the same x- and y-scales. A vertical red dashed line
    is drawn at `min_length_threshold` to mark the filtering cutoff.

    Parameters
    ----------
    deduped_df : pd.DataFrame
        Output of deduplicate_events(). Must have columns 'event_type' and 'length'.
    min_length_threshold : int or None
        Position of the red dashed cutoff line. Pass None to omit. Default: 200.
    bins : int
        Number of histogram bins. Default: 50.
    log_x : bool
        If True, use a log scale on the x-axis with log-spaced bin edges.
        Default: False.
    log_y : bool
        If True, use a log scale on the y-axis. Default: True.
    y_max : float or None
        If provided, cap the y-axis at this value. Default: None.
    x_max : float or None
        If provided, cap the x-axis at this value. Default: None.
    filter_below_threshold : bool
        If True, events with length < min_length_threshold are excluded from
        the plot. Has no effect if min_length_threshold is None. Default: False.
    figsize : tuple
        Matplotlib figure size.
    save_path : str or None
        If provided, save the figure; otherwise call plt.show().
    """
    event_cfg = [
        ("insertion",     "#4C72B0"),
        ("deletion",      "#DD8452"),
        ("translocation", "#55A868"),
        ("inversion",     "#C44E52"),
    ]

    plot_df = deduped_df.copy()
    if filter_below_threshold and min_length_threshold is not None:
        plot_df = plot_df[plot_df["length"] >= min_length_threshold]

    all_lengths = plot_df["length"].dropna().values
    if len(all_lengths) == 0:
        return

    x_min = max(all_lengths.min(), 1) if log_x else all_lengths.min()
    x_max_data = x_max if x_max is not None else all_lengths.max()

    if log_x:
        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max_data), bins + 1)
    else:
        bin_edges = np.linspace(x_min, x_max_data, bins + 1)

    fig, axes_grid = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.1)
    axes = axes_grid.flatten()

    for ax, (etype, color) in zip(axes, event_cfg):
        sub = plot_df[plot_df["event_type"] == etype]
        if not sub.empty and "length" in sub.columns:
            lengths = sub["length"].dropna().values
            ax.hist(lengths, bins=bin_edges, color=color, alpha=0.85, edgecolor="none")
        if min_length_threshold is not None and not filter_below_threshold:
            ax.axvline(min_length_threshold, color="red", linestyle="--", linewidth=1.2,
                       label=f"cutoff = {min_length_threshold} bp")
            ax.legend(frameon=False, fontsize=10)
        ax.set_title(etype.capitalize())
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=True)

    if y_max is not None:
        axes[0].set_ylim(top=y_max)
    if x_max is not None:
        axes[0].set_xlim(right=x_max)

    for ax in axes_grid[1, :]:
        ax.set_xlabel("Event length (bp)")
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("Count")

    fig.suptitle("Length distribution of events by type")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


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
    max_x: float = None,
):
    """
    Adds annotation layers to an existing Plotly figure of a pangraph.
    - Inversions: plotted with diagonal red lines (might be interrupted by insertions, in this case the insertion is also plotted within the inversion with red lines but not part of the length of the inversion)
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
        if ann.empty or "feature" not in ann.columns:
            ann = pd.DataFrame(columns=["seqid", "feature", "start", "end", "attrs", "is_subtype"])

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
                    pos_str = "N/A" if (pos != pos) else str(int(pos))  # NaN check
                    hover = f"Position = {pos_str}<br>" + "<br>".join(lines)
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

        # Load inversions
        INVERSION_COLOR = "red"

        inversion_legend_added = False

        for i, cons_path in enumerate(consensus_paths):
            inversions_file = os.path.join(indels_base_path, "inversions", junction_name, f"consensus_{i+1}", "inversions_summary.csv")
            if not os.path.exists(inversions_file):
                continue
            inv_df = pd.read_csv(inversions_file)

            for label in y_labels:
                sub_inv = inv_df[inv_df["genome_name"] == label]
                if sub_inv.empty:
                    continue

                showleg = not inversion_legend_added
                inversion_legend_added = True

                inv_nums = [str(name).replace("inversion", "") for name in sub_inv["inversion"].tolist()]
                block_counts = [str(p).count("[") for p in sub_inv["path"].tolist()]

                fig.add_trace(go.Bar(
                    x=(sub_inv["end_pos"] - sub_inv["start_pos"]).tolist(),
                    y=[label] * len(sub_inv),
                    base=(sub_inv["start_pos"]).tolist(),
                    orientation="h",
                    marker=dict(
                        color="rgba(0,0,0,0)",
                        pattern=dict(
                            shape="\\",
                            fgcolor=INVERSION_COLOR,
                            size=6,
                            solidity=0.3,
                        ),
                        line=dict(width=1, color=INVERSION_COLOR),
                    ),
                    name="Inversion",
                    showlegend=showleg,
                    customdata=list(zip(
                        sub_inv["end_pos"].tolist(),
                        sub_inv["length"].tolist(),
                        inv_nums,
                        sub_inv["strand"].tolist(),
                        block_counts,
                    )),
                    hovertemplate=(
                        "<b>Inversion (#%{customdata[2]})</b>"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{customdata[1]:d}"
                        "<br>Strand = %{customdata[3]}"
                        "<br>Blocks = %{customdata[4]}"
                        "<extra></extra>"
                    ),
                ))

        # Load translocations — draw arrows from start to end of translocated region
        TRANSLOCATION_COLOR = "rgb(0,0,255)"
        MIN_ARROW_SPAN = (max_x or 1) * 0.01
        any_trans_drawn = False

        for i, cons_path in enumerate(consensus_paths):
            trans_file = os.path.join(indels_base_path, "translocations", junction_name, f"consensus_{i+1}", "translocations_summary.csv")
            if not os.path.exists(trans_file):
                continue
            trans_df = pd.read_csv(trans_file)

            hover_x, hover_y, hover_custom = [], [], []

            for _, row in trans_df.iterrows():
                label = row["genome_name"]
                if label not in y_labels:
                    continue

                any_trans_drawn = True
                start = row["start_pos"]
                end   = row["end_pos"]
                if abs(end - start) < MIN_ARROW_SPAN:
                    mid   = (start + end) / 2
                    start = mid - MIN_ARROW_SPAN / 2
                    end   = mid + MIN_ARROW_SPAN / 2

                fig.add_annotation(
                    x=end, y=label,
                    ax=start, ay=label,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=TRANSLOCATION_COLOR,
                    text="",
                )

                hover_x.append((row["start_pos"] + row["end_pos"]) / 2)
                hover_y.append(label)
                hover_custom.append([
                    row["translocation"],
                    row["start_pos"],
                    row["end_pos"],
                    row["length"],
                ])

            if hover_x:
                fig.add_trace(go.Scatter(
                    x=hover_x,
                    y=hover_y,
                    mode="markers",
                    marker=dict(size=10, color="rgba(0,0,0,0)"),
                    customdata=hover_custom,
                    hovertemplate=(
                        "<b>Translocation</b><br>"
                        "Name: %{customdata[0]}<br>"
                        "Start: %{customdata[1]:d}<br>"
                        "End: %{customdata[2]:d}<br>"
                        "Length: %{customdata[3]:d}"
                        "<extra></extra>"
                    ),
                    name="Translocation",
                    showlegend=False,
                ))

        if any_trans_drawn:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="arrow-right", size=16, color=TRANSLOCATION_COLOR, angle=0),
                name="Translocation",
                showlegend=True,
                hoverinfo="skip",
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

def plot_marginal_scatter(
    df: pd.DataFrame,
    x_col: str = "tot_acc_len",
    y_col: str = "n_categories",
    figsize=(8, 6),
    y_jitter_low: float = 0.9,
    y_jitter_high: float = 1.1,
    fill_alpha: float = 0.6,
    edge_alpha: float = 1.0,
    scatter_size: float = 28,
    point_linewidth: float = 0.8,
    filled: bool = True,
    random_seed: int = 42,
    filter_y_gt_1: bool = True,
    color_by_n_clusters: bool = False,
    color_by_event_count=None,
    log_color_scale: bool = True,
    show_histograms: bool = True,
    subplot_arrangement: tuple = None,
    shared_colorbar: bool = True,
):
    """
    Scatter plot with marginal histograms, similar to panel (b),
    using multiplicative random jitter on the y-axis.

    This reproduces the old plotting style:
        y_jittered = y * uniform(y_jitter_low, y_jitter_high)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x_col : str
        Column for x-axis, e.g. "tot_acc_len".
    y_col : str
        Column for y-axis, e.g. "n_categories".
    figsize : tuple
        Figure size.
    y_jitter_low : float
        Lower bound of multiplicative y-jitter.
    y_jitter_high : float
        Upper bound of multiplicative y-jitter.
    fill_alpha : float
        Transparency of the dot fill. Default: 0.6.
    edge_alpha : float
        Transparency of the dot border. Default: 1.0.
    scatter_size : float
        Scatter marker size.
    point_linewidth : float
        Marker edge linewidth.
    filled : bool
        If True (default), dots are filled. If False, only the ring outline is shown.
    random_seed : int
        Seed for reproducible jitter.
    filter_y_gt_1 : bool
        If True, only plot rows with y_col > 1, like in your Altair snippet.
    color_by_n_clusters : bool
        If True, color scatter points and histogram bars by the 'n_clusters'
        column (discrete values 1–7). Default: False.
    color_by_event_count : str or None
        Column name (e.g. 'n_events', 'n_insertion') to color dots by
        continuously. Zero values are shown in gray; non-zero values use a
        continuous blue-to-red colormap. A colorbar is added. Default: None.
    log_color_scale : bool
        If True (default), use log scale for the event count colormap.
        If False, use linear scale.
    show_histograms : bool
        If True (default), show marginal histograms. If False, only the
        scatter plot is shown.
    subplot_arrangement : tuple or None
        If given (e.g. ``(2, 2)``), ``color_by_event_count`` must be a list
        of column names.  Creates a grid of nrows × ncols scatter panels,
        one per column, with a shared colormap/colorbar.  Histograms are
        suppressed in this mode.
    """

    # ------------------------------------------------------------------ #
    # Multi-panel mode                                                     #
    # ------------------------------------------------------------------ #
    if subplot_arrangement is not None:
        from matplotlib.colors import (
            to_rgba, LinearSegmentedColormap, LogNorm, Normalize,
        )
        from matplotlib.cm import ScalarMappable

        if not isinstance(color_by_event_count, list):
            raise ValueError(
                "When subplot_arrangement is given, color_by_event_count must be a list."
            )
        nrows, ncols = subplot_arrangement
        event_cols = color_by_event_count

        _event_cmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
        event_cmap = LinearSegmentedColormap.from_list("event_cmap", _event_cmap_colors)

        _axis_labels = {
            "tot_acc_len": "local pangenome length (bp)",
            "n_categories": "n. distinct paths",
        }
        _cbar_labels = {
            "n_events":        "n. events",
            "n_insertion":     "n. insertions",
            "n_deletion":      "n. deletions",
            "n_translocation": "n. translocations",
            "n_inversion":     "n. inversions",
        }

        # prepare data once
        all_cols = [x_col, y_col] + event_cols
        data = df[all_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
        data = data[(data[x_col] > 0) & (data[y_col] > 0)].copy()
        if filter_y_gt_1:
            data = data[data[y_col] > 1].copy()
        if data.empty:
            raise ValueError("No valid positive values left after filtering.")

        rng = np.random.default_rng(random_seed)
        y_jitter = rng.uniform(y_jitter_low, y_jitter_high, size=len(data))
        data["_y_jittered"] = data[y_col] * y_jitter

        # shared norm across all event columns
        vmax = max((data[c].max() for c in event_cols), default=1)
        if vmax <= 0:
            vmax = 1
        norm = LogNorm(vmin=1, vmax=vmax) if log_color_scale else Normalize(vmin=0, vmax=vmax)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            squeeze=False,
        )
        right_margin = 0.88 if shared_colorbar else 0.97
        fig.subplots_adjust(right=right_margin, hspace=0.45, wspace=0.35)

        for idx, col in enumerate(event_cols):
            ax = axes[idx // ncols][idx % ncols]
            event_vals = data[col].values
            nonzero_mask = event_vals > 0

            # per-panel norm when not shared
            if shared_colorbar:
                panel_norm = norm
            else:
                pmax = data[col].max() if data[col].max() > 0 else 1
                use_log = (col in log_color_scale) if isinstance(log_color_scale, list) else log_color_scale
                panel_norm = (LogNorm(vmin=1, vmax=pmax) if use_log
                              else Normalize(vmin=0, vmax=pmax))

            # gray for zero
            if (~nonzero_mask).any():
                gray_fc = (*to_rgba("#999999")[:3], fill_alpha) if filled else (0, 0, 0, 0)
                gray_ec = (*to_rgba("#999999")[:3], edge_alpha)
                ax.scatter(
                    data.loc[~nonzero_mask, x_col],
                    data.loc[~nonzero_mask, "_y_jittered"],
                    s=scatter_size, facecolors=gray_fc, edgecolors=gray_ec,
                    linewidths=point_linewidth,
                )

            # colored for non-zero
            if nonzero_mask.any():
                nd = data.loc[nonzero_mask].copy().sort_values(col, ascending=True)
                norm_vals = panel_norm(nd[col].values)
                face_colors = [(*event_cmap(v)[:3], fill_alpha) if filled else (0, 0, 0, 0)
                               for v in norm_vals]
                edge_colors = [(*event_cmap(v)[:3], edge_alpha) for v in norm_vals]
                ax.scatter(
                    nd[x_col], nd["_y_jittered"],
                    s=scatter_size, facecolors=face_colors, edgecolors=edge_colors,
                    linewidths=point_linewidth,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(_cbar_labels.get(col, col), fontsize=12)
            ax.set_xlabel(_axis_labels.get(x_col, x_col))
            ax.set_ylabel(_axis_labels.get(y_col, y_col))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if not shared_colorbar:
                sm_panel = ScalarMappable(cmap=event_cmap, norm=panel_norm)
                sm_panel.set_array([])
                fig.colorbar(sm_panel, ax=ax, fraction=0.046, pad=0.04)

        # hide unused axes
        for idx in range(len(event_cols), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # single shared colorbar
        if shared_colorbar:
            sm = ScalarMappable(cmap=event_cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("n. events", fontsize=9)

        return fig, axes

    # ------------------------------------------------------------------ #
    # Single-panel mode (original code below)                             #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(random_seed)

    cols = [x_col, y_col]
    if color_by_n_clusters:
        cols.append("n_clusters")
    if color_by_event_count is not None:
        cols.append(color_by_event_count)
    data = df[cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data[x_col] > 0) & (data[y_col] > 0)].copy()

    if filter_y_gt_1:
        data = data[data[y_col] > 1].copy()

    if color_by_n_clusters:
        cluster_values = sorted(data["n_clusters"].unique())
        _n_clusters_palette = {
            1: "#999999",  # gray
            2: "#b5d2f2",  # light blue
            3: "#7394c2",  # mid blue
            4: "#d991b4",  # pink
            5: "#80557e",  # purple
            6: "#a6444f",  # reddish
            7: "#a6444f",  # reddish (same, filtered out anyway)
        }
        cluster_colors = {v: _n_clusters_palette.get(v, "#999999") for v in cluster_values}

    if data.empty:
        raise ValueError("No valid positive values left after filtering.")

    # multiplicative y-jitter, exactly in the style of the old plot
    y_jitter = rng.uniform(y_jitter_low, y_jitter_high, size=len(data))
    data["_y_jittered"] = data[y_col] * y_jitter

    # layout
    fig = plt.figure(figsize=figsize)
    if show_histograms:
        gs = GridSpec(
            2, 2,
            width_ratios=[4, 1.2],
            height_ratios=[1.2, 4],
            hspace=0.08,
            wspace=0.08,
        )
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    else:
        ax_scatter = fig.add_subplot(1, 1, 1)
        ax_histx = None
        ax_histy = None

    # scatter
    if color_by_n_clusters:
        for v in cluster_values:
            mask = data["n_clusters"] == v
            from matplotlib.colors import to_rgba
            fc = (*to_rgba(cluster_colors[v])[:3], fill_alpha) if filled else (0, 0, 0, 0)
            ec = (*to_rgba(cluster_colors[v])[:3], edge_alpha)
            ax_scatter.scatter(
                data.loc[mask, x_col],
                data.loc[mask, "_y_jittered"],
                s=scatter_size,
                facecolors=fc,
                edgecolors=ec,
                linewidths=point_linewidth,
                label=str(int(v) - 1),
            )
        # legend added after layout below
        pass
    elif color_by_event_count is not None:
        from matplotlib.colors import to_rgba, LinearSegmentedColormap
        _event_cmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
        event_cmap = LinearSegmentedColormap.from_list("event_cmap", _event_cmap_colors)

        from matplotlib.colors import LogNorm, Normalize
        event_vals = data[color_by_event_count].values
        nonzero_mask = event_vals > 0
        vmax = event_vals.max() if event_vals.max() > 0 else 1
        log_norm = LogNorm(vmin=1, vmax=vmax) if log_color_scale else Normalize(vmin=0, vmax=vmax)

        # gray dots for zero
        if (~nonzero_mask).any():
            gray_fc = (*to_rgba("#999999")[:3], fill_alpha) if filled else (0, 0, 0, 0)
            gray_ec = (*to_rgba("#999999")[:3], edge_alpha)
            ax_scatter.scatter(
                data.loc[~nonzero_mask, x_col],
                data.loc[~nonzero_mask, "_y_jittered"],
                s=scatter_size, facecolors=gray_fc, edgecolors=gray_ec,
                linewidths=point_linewidth,
            )

        # colored dots for non-zero, log-normalized, sorted ascending so high counts plot on top
        if nonzero_mask.any():
            nonzero_data = data.loc[nonzero_mask].copy()
            nonzero_data = nonzero_data.sort_values(color_by_event_count, ascending=True)
            norm_vals = log_norm(nonzero_data[color_by_event_count].values)
            face_colors = [(*event_cmap(v)[:3], fill_alpha) if filled else (0, 0, 0, 0)
                           for v in norm_vals]
            edge_colors = [(*event_cmap(v)[:3], edge_alpha) for v in norm_vals]
            ax_scatter.scatter(
                nonzero_data[x_col],
                nonzero_data["_y_jittered"],
                s=scatter_size, facecolors=face_colors, edgecolors=edge_colors,
                linewidths=point_linewidth,
            )
    else:
        from matplotlib.colors import to_rgba
        fc = (*to_rgba("0.5")[:3], fill_alpha) if filled else (0, 0, 0, 0)
        ec = (*to_rgba("0.5")[:3], edge_alpha)
        ax_scatter.scatter(
            data[x_col],
            data["_y_jittered"],
            s=scatter_size,
            facecolors=fc,
            edgecolors=ec,
            linewidths=point_linewidth,
        )

    _axis_labels = {
        "tot_acc_len": "local pangenome length (bp)",
        "n_categories": "n. distinct paths",
    }
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(_axis_labels.get(x_col, x_col))
    ax_scatter.set_ylabel(_axis_labels.get(y_col, y_col))

    if show_histograms:
        # histograms from original, non-jittered data
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()

        x_bins = np.logspace(np.log10(x_min), np.log10(x_max), 25)
        y_bins = np.logspace(np.log10(y_min), np.log10(y_max), 25)

        if color_by_n_clusters:
            ax_histx.hist(
                [data.loc[data["n_clusters"] == v, x_col] for v in cluster_values],
                bins=x_bins,
                color=[cluster_colors[v] for v in cluster_values],
                stacked=True, edgecolor="none",
            )
            ax_histy.hist(
                [data.loc[data["n_clusters"] == v, y_col] for v in cluster_values],
                bins=y_bins, orientation="horizontal",
                color=[cluster_colors[v] for v in cluster_values],
                stacked=True, edgecolor="none",
            )
        elif color_by_event_count is not None:
            from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
            _hcmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
            hcmap = LinearSegmentedColormap.from_list("event_cmap", _hcmap_colors)
            hnorm = LogNorm(vmin=1, vmax=vmax) if log_color_scale else Normalize(vmin=0, vmax=vmax)

            def _colored_hist(ax, col, bins, orientation="vertical"):
                for i in range(len(bins) - 1):
                    mask = (data[col] >= bins[i]) & (data[col] < bins[i + 1])
                    if not mask.any():
                        continue
                    mean_val = data.loc[mask, color_by_event_count].mean()
                    color = "#999999" if mean_val <= 0 else hcmap(hnorm(mean_val))
                    count = mask.sum()
                    w = bins[i + 1] - bins[i]
                    if orientation == "vertical":
                        ax.bar(bins[i], count, width=w, align="edge", color=color, edgecolor="none")
                    else:
                        ax.barh(bins[i], count, height=w, align="edge", color=color, edgecolor="none")

            _colored_hist(ax_histx, x_col, x_bins, orientation="vertical")
            _colored_hist(ax_histy, y_col, y_bins, orientation="horizontal")
        else:
            ax_histx.hist(data[x_col], bins=x_bins, color="0.7", edgecolor="0.55")
            ax_histy.hist(data[y_col], bins=y_bins, orientation="horizontal",
                          color="0.7", edgecolor="0.55")

        ax_histx.set_xscale("log")
        ax_histx.set_ylabel("n. junctions")
        ax_histy.set_yscale("log")
        ax_histy.set_xlabel("n. junctions")
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

    for ax in [ax_scatter] + ([ax_histx, ax_histy] if show_histograms else []):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if color_by_n_clusters:
        fig.subplots_adjust(right=0.78)
        handles, labels = ax_scatter.get_legend_handles_labels()
        fig.legend(handles, labels,
                   title="n. homologous\nrecombination events",
                   frameon=False, fontsize=8,
                   loc="center left",
                   bbox_to_anchor=(0.80, 0.45))
    elif color_by_event_count is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
        _event_cmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
        event_cmap = LinearSegmentedColormap.from_list("event_cmap", _event_cmap_colors)
        _cb_norm = LogNorm(vmin=1, vmax=vmax) if log_color_scale else Normalize(vmin=0, vmax=vmax)
        sm = ScalarMappable(cmap=event_cmap, norm=_cb_norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.78)
        cbar_height = 0.55
        cbar_bottom = 0.12 if show_histograms else (1 - cbar_height) / 2
        cbar_ax = fig.add_axes([0.81, cbar_bottom, 0.03, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        _cbar_labels = {
            "n_events":       "n. events",
            "n_insertion":    "n. insertions",
            "n_deletion":     "n. deletions",
            "n_translocation":"n. translocations",
            "n_inversion":    "n. inversions",
        }
        cbar.set_label(_cbar_labels.get(color_by_event_count, color_by_event_count), fontsize=9)

    if show_histograms:
        return fig, (ax_scatter, ax_histx, ax_histy)
    return fig, (ax_scatter,)