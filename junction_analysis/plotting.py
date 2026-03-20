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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba, LinearSegmentedColormap, LogNorm, Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
import colorsys

import pandas as pd

COLORS = {
    "reddish":    "#a6444f",
    "teal":       "#57a8b8",
    "purple":     "#80557e",
    "light_blue": "#b5d2f2",
    "pink":       "#d991b4",
    "dark_blue":  "#397398",
    "mid_blue":   "#7394c2",
    "gray":       "#7a7a7a",
    "wine":       "#7a2030",
    "rosa":     "#c8a2c8",

}
import pypangraph as pp
from scipy.cluster.hierarchy import dendrogram

from junction_analysis.helpers import get_tree_order, read_gff3_annotations, read_gff3_cds_products, read_gff3_trna, add_silhouette_scores
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
    show_trna_annotations: bool = False,
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
        for iso in reversed(isolates_ordered):
            draw_isolate_track(iso)
    else:
        grouped = (
            assignments.reset_index()
            .groupby("best_consensus")["index"]
            .apply(list)
            .to_dict()
        )
        # Draw groups in reverse order so consensus_1 ends up nearest the top (overview)
        for i, cons_path in reversed(list(enumerate(consensus_paths))):
            cons_label = f"consensus_{i+1}"
            isolates_for_this = grouped.get(cons_label, [])
            if tree_order:
                isolates_for_this = [iso for iso in tree_order if iso in isolates_for_this]
            # Reversed so tree-top isolate appears at visual top of the group
            for iso in reversed(isolates_for_this):
                draw_isolate_track(iso)
            draw_consensus_track(cons_path, cons_label)
        # Overview tracks drawn last → appear at top of chart; reversed so consensus_1 is topmost
        for i, cons_path in reversed(list(enumerate(consensus_paths))):
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

        # place stars inside the left margin reserved by the axis range (-5% of max_x)
        star_x = -max(1, int(0.035 * max_x))
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

        # filter CDS products containing integrase, recombinase, or transposase (case-insensitive)
        prod = gdf["product"].fillna("").str.lower()
        ir = gdf[prod.str.contains("integrase") | prod.str.contains("recombinase") | prod.str.contains("transposase")].copy()

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
                    name="Integrase / Recombinase / Transposase",
                    showlegend=showleg,
                    customdata=list(zip(sub_ir["end"].tolist(), sub_ir["product"].tolist())),
                    hovertemplate=(
                        "<b>Integrase / Recombinase / Transposase</b>"
                        "<br>%{customdata[1]}"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{x:d}"
                        "<extra></extra>"
                    ),
                ))

    # overlay tRNA / tmRNA annotations
    TRNA_COLOR = "rgb(127,201,127)"  # mid sage-green between integrase (166,216,84) and prophage (27,158,119)
    if show_trna_annotations and annotations_gff_path:
        tdf = read_gff3_trna(annotations_gff_path)
        if not tdf.empty:
            trna_legend_added = False
            for label in y_labels:
                sub_t = tdf[tdf["seqid"] == label]
                if sub_t.empty:
                    continue
                showleg = not trna_legend_added
                trna_legend_added = True
                fig.add_trace(go.Bar(
                    x=(sub_t["end"] - sub_t["start"]).tolist(),
                    y=[label] * len(sub_t),
                    base=sub_t["start"].tolist(),
                    orientation="h",
                    marker=dict(color=_rgba(TRNA_COLOR, annotation_alpha), line=dict(width=0)),
                    name="tRNA / tmRNA",
                    showlegend=showleg,
                    customdata=list(zip(
                        sub_t["end"].tolist(),
                        sub_t["product"].fillna("").tolist(),
                        sub_t["feature"].tolist(),
                        (sub_t["end"] - sub_t["start"] + 1).tolist(),
                    )),
                    hovertemplate=(
                        "<b>%{customdata[2]}: %{customdata[1]}</b>"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{customdata[3]:d}"
                        "<extra></extra>"
                    ),
                ))

    # overlay defense system, prophage, IS annotations
    if show_mges_annotations and mges_gff_path:
        ann = read_gff3_annotations(mges_gff_path)
        if ann.empty or "feature" not in ann.columns:
            ann = pd.DataFrame(columns=["seqid", "feature", "start", "end", "attrs", "is_subtype"])

        DEF_COLOR = "rgb(152,78,163)"    # medium purple
        INT_COLOR = "rgb(196,150,210)"   # lighter purple for integrons
        PROPH_COLOR = "rgb(27,158,119)"
        IS_BASE = (55, 126, 184)

        # stable IS subtype -> color mapping (computed once from whole file)
        is_types = sorted(ann.loc[ann["feature"] == "IS", "is_subtype"].dropna().unique())
        is_shades = _shades_from_base_rgb(IS_BASE, max(1, len(is_types)))
        is_color = {t: is_shades[i] for i, t in enumerate(is_types)}

        legend_seen = set()

        def _add_anno_bar(x, y, base, color_rgb, name, end, length):
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
                hovertemplate=f"<b>{name}</b><br>Start = %{{base:d}}<br>End = %{{customdata[0]:d}}<br>Length = %{{customdata[1]:d}}<extra></extra>",
                customdata=list(zip(end, length)),
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
                    length=ph["length"].tolist(),
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
                    length=ds["length"].tolist(),
                )

            # add integron annotations (CALIN feature type)
            it = sub[sub["feature"].isin(["integron", "CALIN"])]
            if not it.empty:
                _add_anno_bar(
                    x=(it["end"] - it["start"]).tolist(),
                    y=[label] * len(it),
                    base=(it["start"]).tolist(),
                    color_rgb=INT_COLOR,
                    name="Integron",
                    end=it["end"].tolist(),
                    length=it["length"].tolist(),
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
                        length=istype_df["length"].tolist(),
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

            del_xs, del_ys, del_customdata = [], [], []
            for label in y_labels:
                sub_del = del_df[del_df["genome_name"] == label]
                if sub_del.empty:
                    continue
                for _, row in sub_del.iterrows():
                    del_name = row["deletion"]
                    del_num = del_name.replace("deletion", "") if "deletion" in str(del_name) else del_name
                    n_blocks = str(row.get("path", "")).count("[")
                    del_xs.append(row["position"])
                    del_ys.append(label)
                    del_customdata.append([del_num, row["length"], row.get("strand", ""), n_blocks])

            if del_xs:
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
                    customdata=del_customdata,
                    hovertemplate=(
                        "<b>Deletion (#%{customdata[0]})</b>"
                        "<br>Position = %{x:d}"
                        "<br>Length = %{customdata[1]:d}"
                        "<br>Strand = %{customdata[2]}"
                        "<br>Blocks = %{customdata[3]}"
                        "<extra></extra>"
                    ),
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
            text=(
                f"{title}<br>"
                f"<span style='font-size:12px; color:#666; font-weight:normal;'>"
                f"Isolates are ordered by their position in the core-genome phylogeny, grouped by consensus path assignment."
                f"</span>"
            ),
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
        margin=dict(l=140, r=20, t=140, b=40),
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
                                 figsize=None, vline=None, vline_kwargs=None, same_xrange=False):
    """
    pair_dists: dict {block_id: 1D numpy array of pairwise distances}
    block_list: list or Series of block IDs to plot
    vline: float or None — if given, draw vertical line at this x-value
    vline_kwargs: dict passed to ax.axvline() (e.g. color, linestyle)
    """
    block_list = list(block_list)
    n = len(block_list)

    rows = math.ceil(n / cols)
    if figsize is None:
        figsize = (cols * 3, rows * 2.5)
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
            lbl = str(block_id[0])[:8] if isinstance(block_id, tuple) else str(block_id)[:12]
            ax.text(0.5, 0.5, f"No data\n{lbl}", ha='center', va='center', fontsize=7)
            ax.set_axis_off()
            continue

        # histogram
        ax.hist(d, bins=bins, density=True, alpha=0.7)

        # vertical threshold line
        if vline is not None:
            ax.axvline(vline, **vline_kwargs)

        # titles & labels
        if isinstance(block_id, tuple):
            label = f"{str(block_id[0])[:8]}\n{block_id[1]}"
        else:
            label = str(block_id)[:12]
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Distance", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)

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
        for iso in reversed(isolates_ordered):
            draw_isolate_track(iso)
    else:
        grouped = (
            assignments.reset_index()
            .groupby("best_consensus")["index"]
            .apply(list)
            .to_dict()
        )
        # Draw groups in reverse order so consensus_1 ends up nearest the top (overview)
        for i, cons_path in reversed(list(enumerate(consensus_paths))):
            cons_label = f"consensus_{i+1}"
            isolates_for_this = grouped.get(cons_label, [])
            if tree_order:
                isolates_for_this = [iso for iso in tree_order if iso in isolates_for_this]
            # Reversed so tree-top isolate appears at visual top of the group
            for iso in reversed(isolates_for_this):
                draw_isolate_track(iso)
            draw_consensus_track(cons_path, cons_label)
        # Overview tracks drawn last → appear at top of chart; reversed so consensus_1 is topmost
        for i, cons_path in reversed(list(enumerate(consensus_paths))):
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

        # place stars inside the left margin reserved by the axis range (-5% of max_x)
        star_x = -max(1, int(0.035 * max_x))
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
            text=(
                f"{title}<br>"
                f"<span style='font-size:17px; color:#666; font-weight:normal;'>"
                f"Isolates are ordered by their position in the core-genome phylogeny, grouped by consensus path assignment."
                f"</span>"
            ),
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
        margin=dict(l=140, r=20, t=140, b=40),
        height=max(300, int(len(y_labels) * 22)),
        template="plotly_white",
    )
    return fig, y_labels, max_x, inversion_rects


def plot_cluster_count_distribution(df, figsize=(6, 4), save_path=None):
    """
    Plot the distribution of the number of clusters per junction as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'n_clusters'. One row per junction.
    figsize : tuple
    save_path : str or None
        If provided, save the figure instead of showing it.
    """
    _n_clusters_palette = {
        0: COLORS["gray"],
        1: COLORS["light_blue"],
        2: COLORS["mid_blue"],
        3: COLORS["teal"],
        4: COLORS["dark_blue"],
        5: COLORS["purple"],
        6: COLORS["pink"],
        7: COLORS["reddish"],
        8: COLORS["wine"],
    }

    counts = (df["n_clusters"] - 1).value_counts().sort_index()
    bar_colors = [_n_clusters_palette.get(v, COLORS["wine"]) for v in counts.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index, counts.values, color=bar_colors,
           edgecolor="white", linewidth=0.6, width=0.85)

    # annotate each bar with its count
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + counts.values.max() * 0.01, str(y),
                ha="center", va="bottom", fontsize=12, color="black")

    ax.set_xlabel("n. additional clusters per junction", fontsize=16)
    ax.set_ylabel("n. junctions", fontsize=16)
    ax.tick_params(labelsize=14)
    x_max_tick = max(counts.index.max(), 10)
    all_ticks = list(range(counts.index.min(), x_max_tick + 1))
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([str(v) for v in all_ticks])
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.grid(True, color="0.92", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(counts.index.min() - 0.6, x_max_tick + 0.6)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_cluster_count_vs_diversity(
    df,
    diversity_columns,
    figsize=None,
    save_path=None,
):
    """
    Scatter plots of n_clusters per junction against junction diversity measures.

    One subplot per column in `diversity_columns`, laid out in a row.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'n_clusters' and all columns listed in `diversity_columns`.
        One row per junction.
    diversity_columns : list of str
        Column names to plot on the x-axis.
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

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, diversity_columns):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        valid = df[["n_clusters", col]].dropna()
        ax.scatter(valid[col], valid["n_clusters"], alpha=0.6, s=20,
                   color=COLORS["mid_blue"], linewidths=0)
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


def plot_junction_heatmap(
    df,
    x_col,
    y_col,
    heatmap_col,
    x_label=None,
    y_label=None,
    legend_label=None,
    title=None,
    figsize=(6, 4),
    save_path=None,
):
    """
    Heatmap of counts for two categorical columns, with cell annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `x_col`, `y_col`, and `heatmap_col`. One row per junction.
    x_col : str
        Column for x-axis (columns of heatmap).
    y_col : str
        Column for y-axis (rows of heatmap).
    heatmap_col : str
        Unused directly — the cell values are counts of rows per (y_col, x_col) bin.
        Both `x_col` and `y_col` should already be categorical or discrete.
    x_label : str or None
        X-axis label. Defaults to `x_col`.
    y_label : str or None
        Y-axis label. Defaults to `y_col`.
    legend_label : str or None
        Colorbar label. Defaults to "Number of junctions".
    title : str or None
    figsize : tuple
    save_path : str or None
    """
    data = df[[y_col, x_col]].dropna()

    ct = (
        data.groupby([y_col, x_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    x_cats = ct.columns.tolist()
    y_cats = ct.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues")

    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels(x_cats)
    ax.set_yticks(range(len(y_cats)))
    ax.set_yticklabels(y_cats)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)

    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            if val > 0:
                text_color = "white" if val > ct.values.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax, label=legend_label or "Number of junctions")
    if title:
        ax.set_title(title)
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
    cumulative=False,
    log_x=False,
    distance_col="distance",
    title="default",
    xlabel=None,
    ylabel=None,
    legend_title=None,
    cluster_df=None,
    xlim=None,
    save_path=None,
):
    """
    Plot a combined histogram (or declining CDF) of pairwise distances pooled across all junctions.

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
    cumulative : bool
        If True, plot a declining cumulative distribution (fraction/% of distances
        >= x) instead of a histogram. Default: False.
    log_x : bool
        If True and cumulative=True, use log scale on the x-axis. Default: False.
    save_path : str or None
        If provided, save the figure; otherwise call plt.show().
    """
    df = distances_df.copy()
    if exclude is not None:
        df = df[~df["junction_name"].isin(exclude)]

    if len(df) == 0:
        print("No distances to plot.")
        return

    n_junctions = df["junction_name"].nunique()

    if vline_kwargs is None:
        vline_kwargs = dict(color="black", linestyle="--", linewidth=1.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.grid(True, color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # build groups: split by within/between cluster if cluster_df provided
    if cluster_df is not None:
        meta_cols = {"junction_name", "n_clusters", "n_isolates"}
        isolate_cols = [c for c in cluster_df.columns if c not in meta_cols]
        # build (junction_name, isolate) -> cluster_id lookup
        records = []
        for _, row in cluster_df.iterrows():
            jname = row["junction_name"]
            for iso in isolate_cols:
                if pd.notna(row[iso]):
                    records.append((jname, iso, int(row[iso])))
        cl_lookup = pd.DataFrame(records, columns=["junction_name", "isolate", "cluster"])
        df = df.merge(
            cl_lookup.rename(columns={"isolate": "isolate_1", "cluster": "cl_1"}),
            on=["junction_name", "isolate_1"], how="left"
        ).merge(
            cl_lookup.rename(columns={"isolate": "isolate_2", "cluster": "cl_2"}),
            on=["junction_name", "isolate_2"], how="left"
        )
        groups = [
            (df[df["cl_1"] == df["cl_2"]][distance_col].dropna().values,  COLORS["dark_blue"], "within cluster"),
            (df[df["cl_1"] != df["cl_2"]][distance_col].dropna().values,  COLORS["reddish"],   "between cluster"),
        ]
    else:
        groups = [(df[distance_col].dropna().values, COLORS["mid_blue"], None)]

    def _plot_group(distances, color, label):
        if len(distances) == 0:
            return
        if cumulative:
            sorted_d = np.sort(distances)
            y = np.arange(len(sorted_d), 0, -1) / len(sorted_d)
            if percentage:
                y = y * 100
            ax.plot(sorted_d, y, color=color, linewidth=1.5, zorder=2, label=label)
        elif percentage:
            counts, edges = np.histogram(distances, bins=bins)
            pcts = counts / counts.sum() * 100
            widths = np.diff(edges)
            ax.bar(edges[:-1], pcts, width=widths, align="edge",
                   color=color, edgecolor="white", linewidth=0.5, zorder=2, label=label, alpha=0.7)
        else:
            ax.hist(distances, bins=bins, color=color, edgecolor="white",
                    linewidth=0.5, zorder=2, label=label, alpha=0.7)

    for distances, color, label in groups:
        _plot_group(distances, color, label)

    if cluster_df is not None:
        ax.legend(frameon=False, fontsize=10)

    if cumulative:
        ax.set_ylabel(ylabel if ylabel is not None else ("% of pairwise distances ≥ x" if percentage else "Fraction of pairwise distances ≥ x"), fontsize=11)
    elif percentage:
        ax.set_ylabel(ylabel if ylabel is not None else "% of pairwise distances", fontsize=11)
    else:
        ax.set_ylabel(ylabel if ylabel is not None else "Count", fontsize=11)

    if vline is not None:
        default_label = f"Pairwise distance cutoff for\nhomologous recombination ({vline})"
        ax.axvline(vline, label=legend_title if legend_title is not None else default_label,
                   **vline_kwargs)
        ax.legend(frameon=False, fontsize=10)
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel if xlabel is not None else "Pairwise patristic distance (substitutions per site)", fontsize=11)
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
    color_by_mge=False,
    insertions_only=False,
    insertions_by_is_family=False,
    deletions_only=False,
    ccdf_event=None,
    color_by_majority_organism=False,
    color_by_species_counts=False,
    n_species=5,
    legend_title=None,
    figsize=None,
    save_path=None,
):
    """
    Plot the length distribution of each event type as histograms.

    All subplots share the same x- and y-scales. A vertical dashed line
    is drawn at `min_length_threshold` to mark the filtering cutoff.

    Parameters
    ----------
    deduped_df : pd.DataFrame
        Output of deduplicate_events(). Must have columns 'event_type' and 'length'.
    min_length_threshold : int or None
        Position of the cutoff line. Pass None to omit. Default: 200.
    bins : int
        Number of histogram bins. Default: 50.
    log_x : bool
        If True, use a log scale on the x-axis with log-spaced bin edges.
    log_y : bool
        If True, use a log scale on the y-axis. Default: True.
    y_max : float or None
        If provided, cap the y-axis at this value.
    x_max : float or None
        If provided, cap the x-axis at this value.
    filter_below_threshold : bool
        If True, exclude events shorter than min_length_threshold.
    color_by_mge : bool
        If True, stack bars by MGE association using the same colors as
        plot_event_mge_stacked_bar. Default: False.
    insertions_only : bool
        If True, show only a single panel for insertions. Default: False.
    figsize : tuple or None
        Figure size. Defaults to (6, 4) for insertions_only, (10, 8) otherwise.
    save_path : str or None
        If provided, save the figure; otherwise call plt.show().
    """
    # MGE colors + alphas matching plot_event_mge_stacked_bar
    _MGE_STYLE = {
        "Prophage":              ("#a6444f", 1.0),
        "Prophage (associated)": ("#c98087", 1.0),
        "IS":                    ("#7394c2", 1.0),
        "IS (associated)":       ("#b5d2f2", 1.0),
        "Defense system":        ("#80557e", 1.0),
        "Integron":              ("#d991b4", 1.0),
        "None":                  ("#7a7a7a", 1.0),
    }
    _MGE_DISPLAY_NAME = {
        "Defense system": "Defense system (associated)",
        "Integron":       "Integron (associated)",
    }
    mge_order = ["None", "IS (associated)", "IS",
                 "Defense system", "Integron",
                 "Prophage (associated)", "Prophage"]

    event_cfg = [
        ("insertion",     "#7394c2"),
        ("deletion",      "#DD8452"),
        ("translocation", "#55A868"),
        ("inversion",     "#C44E52"),
    ]

    plot_df = deduped_df[
        deduped_df["event_type"] != "ambiguous_insertion"
    ].copy()
    if filter_below_threshold and min_length_threshold is not None:
        plot_df = plot_df[plot_df["length"] >= min_length_threshold]

    # --- single-event CCDF shortcut ---------------------------------------------
    _ccdf_event = None
    if deletions_only:
        _ccdf_event = "deletion"
    elif ccdf_event is not None:
        _ccdf_event = ccdf_event

    if _ccdf_event is not None:
        _etype_label = {
            "deletion":      "Deletion",
            "translocation": "Translocation",
            "inversion":     "Inversion",
            "insertion":     "Insertion",
        }.get(_ccdf_event, _ccdf_event.capitalize())

        lengths = plot_df[plot_df["event_type"] == _ccdf_event]["length"].dropna().sort_values().values
        if len(lengths) == 0:
            return
        if figsize is None:
            figsize = (6, 4)
        fig, ax = plt.subplots(figsize=figsize)
        ccdf = 1 - np.arange(0, len(lengths)) / len(lengths)
        ax.plot(lengths, ccdf, color="#7394c2", linewidth=2.5)
        ax.fill_between(lengths, ccdf, alpha=0.12, color="#7394c2")
        if x_max is not None:
            ax.set_xlim(right=x_max)
        ax.set_xlim(left=lengths.min())
        ax.set_ylim(0, 1.02)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(f"{_etype_label} length (bp)", fontsize=15)
        ax.set_ylabel(f"Fraction of {_etype_label.lower()}s ≥ length", fontsize=15)
        ax.tick_params(axis="both", labelsize=13, length=4)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        return fig, ax

    # --- insertions by IS family shortcut ---------------------------------------
    if insertions_by_is_family:
        sub = plot_df[plot_df["event_type"] == "insertion"].copy()
        if sub.empty:
            return
        if figsize is None:
            figsize = (6, 4)

        IS_BASE = (55, 126, 184)

        # collect IS families from is_family + is_families_associated columns
        full_families = sorted(sub["is_family"].dropna().unique()) if "is_family" in sub.columns else []
        assoc_families = sorted({
            f for cell in sub.get("is_families_associated", pd.Series(dtype=object)).dropna()
            for f in (cell if isinstance(cell, list) else [])
        })
        all_families = sorted(set(full_families) | set(assoc_families))

        # lightest shade reserved for "IS (associated)" (no specific family)
        n_shades = max(len(all_families), 1)
        shades = _shades_from_base_rgb(IS_BASE, n_shades)

        # family → shade (darkest shades for named families, lightest for associated)
        def _hex_from_rgb_str(s):
            vals = [int(v) for v in s.replace("rgb(", "").replace(")", "").split(",")]
            return "#{:02x}{:02x}{:02x}".format(*vals)

        family_color = {fam: _hex_from_rgb_str(shades[i]) for i, fam in enumerate(all_families)}
        assoc_color  = COLORS["rosa"]   # rosa, clearly distinct from blue IS family shades
        none_color   = COLORS["gray"]

        def _row_label(row):
            if "is_family" in row and pd.notna(row["is_family"]):
                return row["is_family"]
            assoc = row.get("is_families_associated", None)
            if isinstance(assoc, list) and assoc:
                return "__associated__"
            return "__none__"

        sub["_is_label"] = sub.apply(_row_label, axis=1)

        # stack order: none → associated → named families (darkest on top)
        stack_order = ["__none__", "__associated__"] + all_families
        color_map = {"__none__": none_color, "__associated__": assoc_color, **family_color}
        display_map = {"__none__": "Not associated", "__associated__": "IS (associated)"}

        all_ins_lengths = sub["length"].dropna().values
        x_min_i = max(all_ins_lengths.min(), 1) if log_x else all_ins_lengths.min()
        x_max_i = x_max if x_max is not None else all_ins_lengths.max()
        if log_x:
            bin_edges_i = np.logspace(np.log10(x_min_i), np.log10(x_max_i), bins + 1)
        else:
            bin_edges_i = np.linspace(x_min_i, x_max_i, bins + 1)

        fig, ax = plt.subplots(figsize=figsize)
        bottoms = np.zeros(len(bin_edges_i) - 1)
        handles = []
        for lbl in stack_order:
            lengths = sub.loc[sub["_is_label"] == lbl, "length"].dropna().values
            if len(lengths) == 0:
                continue
            vals, _ = np.histogram(lengths, bins=bin_edges_i)
            ax.bar(
                bin_edges_i[:-1], vals, width=np.diff(bin_edges_i),
                bottom=bottoms, align="edge",
                color=color_map[lbl], edgecolor="white", linewidth=0.4,
            )
            handles.append((plt.Rectangle((0, 0), 1, 1, color=color_map[lbl], linewidth=0),
                             display_map.get(lbl, lbl)))
            bottoms = bottoms + vals

        if min_length_threshold is not None and not filter_below_threshold:
            ax.axvline(min_length_threshold, color="#7a7a7a", linestyle="--", linewidth=1.2)

        h, l = zip(*reversed(handles))
        ax.legend(h, l, frameon=False, fontsize=11,
                  title=legend_title if legend_title is not None else "IS family", title_fontsize=12,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


        ax.set_xlabel("Event length (bp)", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        if y_max is not None:
            ax.set_ylim(top=y_max)
        if x_max is not None:
            ax.set_xlim(right=x_max)
        ax.tick_params(axis="both", labelsize=12, length=3)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(right=0.72)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        return fig, ax

    # --- shared helper for organism coloring modes ------------------------------
    def _organism_stacked_bar(sub, bin_edges_s, stack_order, color_map, legend_title_str):
        fig, ax = plt.subplots(figsize=figsize)
        bottoms = np.zeros(len(bin_edges_s) - 1)
        handles = []
        for lbl in stack_order:
            lengths = sub.loc[sub["_species_label"] == lbl, "length"].dropna().values
            if len(lengths) == 0:
                continue
            vals, _ = np.histogram(lengths, bins=bin_edges_s)
            ax.bar(
                bin_edges_s[:-1], vals, width=np.diff(bin_edges_s),
                bottom=bottoms, align="edge",
                color=color_map[lbl], edgecolor="white", linewidth=0.4,
            )
            handles.append((plt.Rectangle((0, 0), 1, 1, color=color_map[lbl], linewidth=0), lbl))
            bottoms = bottoms + vals

        if min_length_threshold is not None and not filter_below_threshold:
            ax.axvline(min_length_threshold, color="#7a7a7a", linestyle="--", linewidth=1.2)

        h, l = zip(*reversed(handles))
        ax.legend(h, l, frameon=False, fontsize=11,
                  title=legend_title_str, title_fontsize=12,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_xlabel("Insertion length (bp)", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        if y_max is not None:
            ax.set_ylim(top=y_max)
        if x_max is not None:
            ax.set_xlim(right=x_max)
        ax.tick_params(axis="both", labelsize=12, length=3)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(right=0.72)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        return fig, ax

    _species_palette = [COLORS["light_blue"], COLORS["mid_blue"], COLORS["dark_blue"],
                        COLORS["purple"], COLORS["rosa"], COLORS["reddish"], COLORS["wine"]]

    # --- mode 1: color each insertion by its majority_organism column -----------
    if color_by_majority_organism:
        sub = plot_df[plot_df["event_type"] == "insertion"].copy()
        if sub.empty:
            return
        if figsize is None:
            figsize = (6, 4)

        if "majority_organism" in sub.columns:
            sub["_species"] = sub["majority_organism"].apply(
                lambda x: " ".join(str(x).split()[:2]) if pd.notna(x) else None
            )
        else:
            sub["_species"] = None

        species_counts = sub["_species"].value_counts()
        all_species = list(species_counts.index)
        sub["_species_label"] = sub["_species"].apply(
            lambda x: x if pd.notna(x) else "unknown"
        )

        stack_order = all_species + (["unknown"] if sub["_species_label"].eq("unknown").any() else [])
        color_map = {s: _species_palette[i % len(_species_palette)] for i, s in enumerate(all_species)}
        color_map["unknown"] = COLORS["gray"]

        all_lengths = sub["length"].dropna().values
        x_min_s = max(all_lengths.min(), 1) if log_x else all_lengths.min()
        x_max_s = x_max if x_max is not None else all_lengths.max()
        bin_edges_s = (
            np.logspace(np.log10(x_min_s), np.log10(x_max_s), bins + 1)
            if log_x else np.linspace(x_min_s, x_max_s, bins + 1)
        )
        return _organism_stacked_bar(sub, bin_edges_s, stack_order, color_map, "Majority organism")

    # --- mode 2: color each bar by relative species hit counts ------------------
    if color_by_species_counts:
        sub = plot_df[plot_df["event_type"] == "insertion"].copy()
        if sub.empty:
            return
        if figsize is None:
            figsize = (6, 4)

        known_cats = {"n_hits_own_chromosome", "n_hits_own_plasmid", "n_hits_other_chromosome",
                      "n_hits_other_plasmid", "n_hits_external"}
        _exclude = {"bacterium", "Candidatus bacterium"}
        species_cols = [c for c in sub.columns if c.startswith("n_hits_") and c not in known_cats
                        and c.replace("n_hits_", "").replace("_", " ") not in _exclude]

        species_totals = sub[species_cols].sum().sort_values(ascending=False)
        top_species_cols = list(species_totals.index[:n_species])
        other_species_cols = [c for c in species_cols if c not in top_species_cols]
        top_species_labels = [c.replace("n_hits_", "").replace("_", " ") for c in top_species_cols]

        color_map = {c: _species_palette[i % len(_species_palette)] for i, c in enumerate(top_species_cols)}
        color_map["_other"] = COLORS["gray"]

        all_lengths = sub["length"].dropna().values
        x_min_s = max(all_lengths.min(), 1) if log_x else all_lengths.min()
        x_max_s = x_max if x_max is not None else all_lengths.max()
        bin_edges_s = (
            np.logspace(np.log10(x_min_s), np.log10(x_max_s), bins + 1)
            if log_x else np.linspace(x_min_s, x_max_s, bins + 1)
        )

        bin_idx = np.digitize(all_lengths, bin_edges_s) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_edges_s) - 2)
        sub = sub.loc[sub["length"].dropna().index].copy()
        sub["_bin"] = bin_idx

        bar_counts, _ = np.histogram(all_lengths, bins=bin_edges_s)
        total_per_bin = np.array([sub[sub["_bin"] == b][species_cols].sum(axis=1).sum()
                                  for b in range(len(bin_edges_s) - 1)])

        fig, ax = plt.subplots(figsize=figsize)
        bottoms = np.zeros(len(bin_edges_s) - 1)
        handles = []

        for sc in top_species_cols + ["_other"]:
            bin_totals = np.array([
                sub[sub["_bin"] == b][other_species_cols].sum(axis=1).sum()
                if sc == "_other"
                else sub[sub["_bin"] == b][sc].sum()
                for b in range(len(bin_edges_s) - 1)
            ])
            rel = np.where(total_per_bin > 0, bin_totals / total_per_bin, 0)
            heights = rel * bar_counts
            label = "Other" if sc == "_other" else sc.replace("n_hits_", "").replace("_", " ")
            ax.bar(bin_edges_s[:-1], heights, width=np.diff(bin_edges_s),
                   bottom=bottoms, align="edge",
                   color=color_map[sc], edgecolor="white", linewidth=0.4)
            handles.append((plt.Rectangle((0, 0), 1, 1, color=color_map[sc], linewidth=0), label))
            bottoms = bottoms + heights

        if min_length_threshold is not None and not filter_below_threshold:
            ax.axvline(min_length_threshold, color="#7a7a7a", linestyle="--", linewidth=1.2)

        h, l = zip(*reversed(handles))
        ax.legend(h, l, frameon=False, fontsize=11,
                  title=legend_title if legend_title is not None else "Species (hit counts)",
                  title_fontsize=12, loc="upper right")
        ax.set_xlabel("Insertion length (bp)", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        if y_max is not None:
            ax.set_ylim(top=y_max)
        if x_max is not None:
            ax.set_xlim(right=x_max)
        ax.tick_params(axis="both", labelsize=12, length=3)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        return fig, ax

    if insertions_only:
        active_cfg = [c for c in event_cfg if c[0] == "insertion"]
    else:
        active_cfg = event_cfg

    all_lengths = plot_df[
        plot_df["event_type"].isin([e for e, _ in active_cfg])
    ]["length"].dropna().values
    if len(all_lengths) == 0:
        return

    x_min = max(all_lengths.min(), 1) if log_x else all_lengths.min()
    x_max_data = x_max if x_max is not None else all_lengths.max()

    if log_x:
        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max_data), bins + 1)
    else:
        bin_edges = np.linspace(x_min, x_max_data, bins + 1)

    if insertions_only:
        if figsize is None:
            figsize = (6, 4)
        fig, ax_single = plt.subplots(figsize=figsize)
        axes_list = [ax_single]
        axes_grid = None
    else:
        if figsize is None:
            figsize = (10, 8)
        fig, axes_grid = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.3, wspace=0.12)
        axes_list = axes_grid.flatten()

    for ax, (etype, color) in zip(axes_list, active_cfg):
        sub = plot_df[plot_df["event_type"] == etype]

        if not sub.empty and "length" in sub.columns:
            if color_by_mge:
                lengths_by_label = {
                    lbl: sub.loc[sub["mge_label"] == lbl, "length"].dropna().values
                    for lbl in mge_order
                }
                active_labels = [lbl for lbl in mge_order if len(lengths_by_label[lbl]) > 0]
                colors = [_MGE_STYLE[lbl][0] for lbl in active_labels]
                alphas = [_MGE_STYLE[lbl][1] for lbl in active_labels]

                # matplotlib stacked hist doesn't support per-bar alpha;
                # draw layers manually from bottom up
                bottoms = np.zeros(len(bin_edges) - 1)
                handles = []
                for lbl, c, a in zip(active_labels, colors, alphas):
                    vals, _ = np.histogram(lengths_by_label[lbl], bins=bin_edges)
                    ax.bar(
                        bin_edges[:-1], vals, width=np.diff(bin_edges),
                        bottom=bottoms, align="edge",
                        color=c, alpha=a, edgecolor="white", linewidth=0.4,
                    )
                    handles.append(plt.Rectangle((0, 0), 1, 1, color=c, alpha=a, linewidth=0))
                    bottoms = bottoms + vals

                display_labels = [_MGE_DISPLAY_NAME.get(l, l) for l in active_labels]
                ax.legend(
                    handles[::-1], display_labels[::-1],
                    frameon=False, fontsize=11,
                    title=legend_title if legend_title is not None else "Mobile genetic elements", title_fontsize=12,
                )
            else:
                ax.hist(
                    sub["length"].dropna().values, bins=bin_edges,
                    color=color, edgecolor="white", linewidth=0.4,
                )

        if min_length_threshold is not None and not filter_below_threshold:
            ax.axvline(
                min_length_threshold, color="#7a7a7a",
                linestyle="--", linewidth=1.2,
                label=f"cutoff = {min_length_threshold} bp",
            )
            if not color_by_mge:
                ax.legend(frameon=False, fontsize=10)

        if not insertions_only:
            ax.set_title(etype.capitalize(), fontsize=14, fontweight="normal", pad=6)
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12, length=3)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    if y_max is not None:
        axes_list[0].set_ylim(top=y_max)
    if x_max is not None:
        axes_list[0].set_xlim(right=x_max)

    if insertions_only:
        axes_list[0].set_xlabel("Event length (bp)", fontsize=14)
        axes_list[0].set_ylabel("Count", fontsize=14)
    else:
        for ax in axes_grid[1, :]:
            ax.set_xlabel("Event length (bp)", fontsize=14)
        for ax in axes_grid[:, 0]:
            ax.set_ylabel("Count", fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def add_annotations_for_dash(
    fig: go.Figure,
    y_labels: list,
    show_mges_annotations: bool = False,
    show_int_rec_annotations: bool = False,
    show_trna_annotations: bool = False,
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

        # filter CDS products containing integrase, recombinase, or transposase (case-insensitive)
        prod = gdf["product"].fillna("").str.lower()
        ir = gdf[prod.str.contains("integrase") | prod.str.contains("recombinase") | prod.str.contains("transposase")].copy()

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
                    name="Integrase / Recombinase / Transposase",
                    showlegend=showleg,
                    customdata=list(zip(sub_ir["end"].tolist(), sub_ir["product"].tolist(), sub_ir["length"].tolist())),
                    hovertemplate=(
                        "<b>Integrase / Recombinase / Transposase</b>"
                        "<br>%{customdata[1]}"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{customdata[2]:d}"
                        "<extra></extra>"
                    ),
                ))

    # overlay tRNA / tmRNA annotations
    TRNA_COLOR = "rgb(127,201,127)"  # mid sage-green between integrase (166,216,84) and prophage (27,158,119)
    if show_trna_annotations and annotations_gff_path:
        tdf = read_gff3_trna(annotations_gff_path)
        if not tdf.empty:
            trna_legend_added = False
            for label in y_labels:
                sub_t = tdf[tdf["seqid"] == label]
                if sub_t.empty:
                    continue
                showleg = not trna_legend_added
                trna_legend_added = True
                fig.add_trace(go.Bar(
                    x=(sub_t["end"] - sub_t["start"]).tolist(),
                    y=[label] * len(sub_t),
                    base=sub_t["start"].tolist(),
                    orientation="h",
                    marker=dict(color=_rgba(TRNA_COLOR, annotation_alpha), line=dict(width=0)),
                    name="tRNA / tmRNA",
                    showlegend=showleg,
                    customdata=list(zip(
                        sub_t["end"].tolist(),
                        sub_t["product"].fillna("").tolist(),
                        sub_t["feature"].tolist(),
                        sub_t["length"].tolist(),
                    )),
                    hovertemplate=(
                        "<b>%{customdata[2]}: %{customdata[1]}</b>"
                        "<br>Start = %{base:d}"
                        "<br>End = %{customdata[0]:d}"
                        "<br>Length = %{customdata[3]:d}"
                        "<extra></extra>"
                    ),
                ))

    # overlay defense system, prophage, IS annotations
    if show_mges_annotations and mges_gff_path:
        ann = read_gff3_annotations(mges_gff_path)
        if ann.empty or "feature" not in ann.columns:
            ann = pd.DataFrame(columns=["seqid", "feature", "start", "end", "attrs", "is_subtype"])

        DEF_COLOR = "rgb(152,78,163)"    # medium purple
        INT_COLOR = "rgb(196,150,210)"   # lighter purple for integrons
        PROPH_COLOR = "rgb(27,158,119)"
        IS_BASE = (55, 126, 184)

        # stable IS subtype -> color mapping (computed once from whole file)
        is_types = sorted(ann.loc[ann["feature"] == "IS", "is_subtype"].dropna().unique())
        is_shades = _shades_from_base_rgb(IS_BASE, max(1, len(is_types)))
        is_color = {t: is_shades[i] for i, t in enumerate(is_types)}

        legend_seen = set()

        def _add_anno_bar(x, y, base, color_rgb, name, end, length):
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
                hovertemplate=f"<b>{name}</b><br>Start = %{{base:d}}<br>End = %{{customdata[0]:d}}<br>Length = %{{customdata[1]:d}}<extra></extra>",
                customdata=list(zip(end, length)),
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
                    length = ph["length"].tolist(),
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
                    length=ds["length"].tolist()
                )

            # add integron annotations (CALIN feature type)
            it = sub[sub["feature"].isin(["integron", "CALIN"])]
            if not it.empty:
                _add_anno_bar(
                    x=(it["end"] - it["start"]).tolist(),
                    y=[label] * len(it),
                    base=(it["start"]).tolist(),
                    color_rgb=INT_COLOR,
                    name="Integron",
                    end=it["end"].tolist(),
                    length=it["length"].tolist()
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
                        length=istype_df["length"].tolist()
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
                customdata=list(zip(subg["end"].tolist(), subg["product"].tolist(), subg["length"].tolist())),
                hovertemplate=(
                    "<b>CDS:</b> %{customdata[1]}"
                    "<br>Start = %{base:d}"
                    "<br>End = %{customdata[0]:d}"
                    "<br>Length = %{customdata[2]:d}"
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

            del_xs, del_ys, del_customdata = [], [], []
            for label in y_labels:
                sub_del = del_df[del_df["genome_name"] == label]
                if sub_del.empty:
                    continue
                for _, row in sub_del.iterrows():
                    del_name = row["deletion"]
                    del_num = del_name.replace("deletion", "") if "deletion" in str(del_name) else del_name
                    n_blocks = str(row.get("path", "")).count("[")
                    del_xs.append(row["position"])
                    del_ys.append(label)
                    del_customdata.append([del_num, row["length"], row.get("strand", ""), n_blocks])

            if del_xs:
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
                    customdata=del_customdata,
                    hovertemplate=(
                        "<b>Deletion (#%{customdata[0]})</b>"
                        "<br>Position = %{x:d}"
                        "<br>Length = %{customdata[1]:d}"
                        "<br>Strand = %{customdata[2]}"
                        "<br>Blocks = %{customdata[3]}"
                        "<extra></extra>"
                    ),
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
    power_norm_color_scale: bool = False,
    log_x: bool = True,
    log_y: bool = True,
    show_histograms: bool = True,
    subplot_arrangement: tuple = None,
    shared_colorbar: bool = True,
    colorbar_label: str = None,
    save_path: str = None,
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
        fig.subplots_adjust(hspace=0.5, wspace=0.38)

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
                gray_fc = (*to_rgba("#cccccc")[:3], fill_alpha) if filled else (0, 0, 0, 0)
                gray_ec = (*to_rgba("#aaaaaa")[:3], edge_alpha)
                ax.scatter(
                    data.loc[~nonzero_mask, x_col],
                    data.loc[~nonzero_mask, "_y_jittered"],
                    s=scatter_size, facecolors=gray_fc, edgecolors=gray_ec,
                    linewidths=point_linewidth, zorder=1,
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
                    linewidths=point_linewidth, zorder=2,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(_cbar_labels.get(col, col), fontsize=14, pad=6)
            ax.set_xlabel(_axis_labels.get(x_col, x_col), fontsize=13)
            ax.set_ylabel(_axis_labels.get(y_col, y_col), fontsize=13)
            ax.tick_params(axis="both", labelsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.4)

            if not shared_colorbar:
                sm_panel = ScalarMappable(cmap=event_cmap, norm=panel_norm)
                sm_panel.set_array([])
                cbar = fig.colorbar(sm_panel, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(_cbar_labels.get(col, col), fontsize=11)
                cbar.ax.tick_params(labelsize=10)

        # hide unused axes
        for idx in range(len(event_cols), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # single shared colorbar
        if shared_colorbar:
            sm = ScalarMappable(cmap=event_cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("n. events", fontsize=12)
            cbar.ax.tick_params(labelsize=11)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
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
            1:  COLORS["gray"],        # 0 additional clusters
            2:  COLORS["light_blue"],  # 1 additional cluster
            3:  COLORS["mid_blue"],
            4:  COLORS["teal"],
            5:  COLORS["dark_blue"],
            6:  COLORS["purple"],
            7:  COLORS["pink"],
            8:  COLORS["reddish"],
            9:  COLORS["wine"],
            10: COLORS["wine"],
            11: COLORS["wine"],
            12: COLORS["wine"],
        }
        cluster_colors = {v: _n_clusters_palette.get(v, COLORS["wine"]) for v in cluster_values}

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
        if power_norm_color_scale:
            _event_cmap_colors = ["#b5d2f2", "#7394c2", "#a6444f"]
        else:
            _event_cmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
        event_cmap = LinearSegmentedColormap.from_list("event_cmap", _event_cmap_colors)

        event_vals = data[color_by_event_count].values
        nonzero_mask = event_vals > 0
        vmax = event_vals.max() if event_vals.max() > 0 else 1
        if power_norm_color_scale:
            log_norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
        elif log_color_scale:
            log_norm = LogNorm(vmin=1, vmax=vmax)
        else:
            log_norm = Normalize(vmin=0, vmax=vmax)

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
    if log_x:
        ax_scatter.set_xscale("log")
    if log_y:
        ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(_axis_labels.get(x_col, x_col), fontsize=15)
    ax_scatter.set_ylabel(_axis_labels.get(y_col, y_col), fontsize=15)
    ax_scatter.tick_params(axis="both", labelsize=13)
    ax_scatter.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)

    if show_histograms:
        # histograms from original, non-jittered data
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()

        x_bins = np.logspace(np.log10(max(x_min, 1e-10)), np.log10(x_max), 25) if log_x else np.linspace(x_min, x_max, 25)
        y_bins = np.logspace(np.log10(max(y_min, 1e-10)), np.log10(y_max), 25) if log_y else np.linspace(y_min, y_max, 25)

        if color_by_n_clusters:
            ax_histx.hist(
                [data.loc[data["n_clusters"] == v, x_col] for v in cluster_values],
                bins=x_bins,
                color=[cluster_colors[v] for v in cluster_values],
                stacked=True, edgecolor="white", linewidth=0.4,
            )
            ax_histy.hist(
                [data.loc[data["n_clusters"] == v, y_col] for v in cluster_values],
                bins=y_bins, orientation="horizontal",
                color=[cluster_colors[v] for v in cluster_values],
                stacked=True, edgecolor="white", linewidth=0.4,
            )
        elif color_by_event_count is not None:
            _hcmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
            hcmap = LinearSegmentedColormap.from_list("event_cmap", _hcmap_colors)
            hnorm = log_norm

            def _colored_hist(ax, col, bins, orientation="vertical"):
                for i in range(len(bins) - 1):
                    mask = (data[col] >= bins[i]) & (data[col] < bins[i + 1])
                    if not mask.any():
                        continue
                    median_val = data.loc[mask, color_by_event_count].median()
                    color = "#999999" if median_val <= 0 else hcmap(hnorm(median_val))
                    count = mask.sum()
                    w = bins[i + 1] - bins[i]
                    if orientation == "vertical":
                        ax.bar(bins[i], count, width=w, align="edge", color=color,
                               edgecolor="white", linewidth=0.4)
                    else:
                        ax.barh(bins[i], count, height=w, align="edge", color=color,
                                edgecolor="white", linewidth=0.4)

            _colored_hist(ax_histx, x_col, x_bins, orientation="vertical")
            _colored_hist(ax_histy, y_col, y_bins, orientation="horizontal")
        else:
            ax_histx.hist(data[x_col], bins=x_bins, color=COLORS["mid_blue"],
                          edgecolor="white", linewidth=0.4)
            ax_histy.hist(data[y_col], bins=y_bins, orientation="horizontal",
                          color=COLORS["mid_blue"], edgecolor="white", linewidth=0.4)

        if log_x:
            ax_histx.set_xscale("log")
        if log_y:
            ax_histy.set_yscale("log")
        ax_histx.set_ylabel("n. junctions", fontsize=13)
        ax_histy.set_xlabel("n. junctions", fontsize=13)
        ax_histx.tick_params(axis="y", labelsize=12)
        ax_histy.tick_params(axis="x", labelsize=12)
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

    for ax in [ax_scatter] + ([ax_histx, ax_histy] if show_histograms else []):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if color_by_n_clusters:
        fig.subplots_adjust(right=0.76)
        handles, labels = ax_scatter.get_legend_handles_labels()
        fig.legend(handles, labels,
                   title="n. additional\nclusters",
                   title_fontsize=13,
                   frameon=False, fontsize=13,
                   loc="center left",
                   bbox_to_anchor=(0.78, 0.38))
    elif color_by_event_count is not None:
        if power_norm_color_scale:
            _event_cmap_colors = ["#b5d2f2", "#7394c2", "#a6444f"]
        else:
            _event_cmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
        event_cmap = LinearSegmentedColormap.from_list("event_cmap", _event_cmap_colors)
        _cb_norm = log_norm
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
        _label = colorbar_label if colorbar_label is not None else _cbar_labels.get(color_by_event_count, color_by_event_count)
        cbar.set_label(_label, fontsize=14)
        cbar.ax.tick_params(labelsize=13)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show_histograms:
        return fig, (ax_scatter, ax_histx, ax_histy)
    return fig, (ax_scatter,)

def plot_score_threshold_violin(
    dfs,
    thresholds,
    score_col="silhouette_score",
    figsize=(8, 5),
    ylim=None,
    xlabel=None,
    ylabel=None,
    legend_labels=None,
    legend_title=None,
    save_path=None,
):
    """
    Plot per-threshold violin plots for one or more score columns.

    For multiple score columns, violins are placed side by side within each
    threshold group, each column in a distinct color.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        One DataFrame per threshold, already filtered and with score columns present.
    thresholds : list
        Threshold labels for the x-axis, matching `dfs`.
    score_col : str or list of str
        Column(s) to plot. If a list, one violin per column per threshold.
    figsize : tuple
    ylim : tuple or None
        (min, max) for the y-axis.
    xlabel : str or None
        Override the x-axis label.
    ylabel : str or None
        Override the y-axis label.
    legend_labels : list of str or None
        Override legend labels for each score column.
    legend_title : str or None
        Title for the legend.
    save_path : str or None
    """
    _COL_COLORS = [COLORS["reddish"], COLORS["mid_blue"], COLORS["purple"], COLORS["light_blue"]]

    if isinstance(score_col, str):
        score_col = [score_col]

    n_thresh = len(thresholds)
    n_cols = len(score_col)
    width = 0.8 / n_cols if n_cols > 1 else 0.8
    offsets = np.linspace(-(n_cols - 1) / 2, (n_cols - 1) / 2, n_cols) * width

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # subtle horizontal gridlines behind everything
    ax.yaxis.grid(True, color="0.88", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = []

    all_scores = []
    for col_idx, col in enumerate(score_col):
        color = _COL_COLORS[col_idx % len(_COL_COLORS)]
        all_positions = [i + offsets[col_idx] for i in range(n_thresh)]
        all_data = [df[col].dropna().values for df in dfs]
        all_scores.extend([s for s in all_data if len(s) > 0])

        # violinplot requires >= 2 points; filter out insufficient datasets
        valid = [(pos, d) for pos, d in zip(all_positions, all_data) if len(d) >= 2]
        if not valid:
            continue
        positions, data = zip(*valid)

        parts = ax.violinplot(data, positions=list(positions), widths=width * 0.75,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

        # strip plot with aggressive jitter
        for pos, d in zip(positions, data):
            jitter = np.random.uniform(-width * 0.32, width * 0.32, size=len(d))
            ax.scatter(pos + jitter, d, color=color, s=11, alpha=0.4, zorder=1,
                       linewidths=0.4, edgecolors="black")

        label = legend_labels[col_idx] if legend_labels and col_idx < len(legend_labels) else col
        legend_handles.append(plt.matplotlib.patches.Patch(facecolor=color, alpha=0.7, label=label))

    ax.axhline(0, color=COLORS["gray"], linewidth=1.0, linestyle="--", zorder=1)

    ax.set_xticks(range(n_thresh))
    if all(isinstance(t, (int, float)) for t in thresholds):
        tick_labels = [f"{t * 1000:g}" for t in thresholds]
        x_unit = " (×10⁻³)"
    else:
        tick_labels = [str(t) for t in thresholds]
        x_unit = ""
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel(f"{xlabel if xlabel is not None else 'Branch-length threshold'}{x_unit}", fontsize=14)
    ax.set_ylabel(ylabel if ylabel is not None else "Score", fontsize=14)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif all_scores:
        combined = np.concatenate(all_scores)
        margin = (combined.max() - combined.min()) * 0.05
        ax.set_ylim(combined.min() - margin, combined.max() + margin)

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_cols > 1:
        leg = ax.legend(handles=legend_handles, frameon=False, title=legend_title,
                        bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0,
                        fontsize=13, title_fontsize=14)
        leg._legend_box.align = "left"
        if legend_title is not None:
            leg.get_title().set_ha("left")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_core_alignment_lengths(
    junction_df,
    block_df=None,
    col="aln_length_nogap",
    bins=50,
    cumulative=False,
    log_x=False,
    log_y=False,
    figsize=(8, 4),
    save_path=None,
):
    """
    Plot the distribution of core alignment lengths for junction-level and/or
    per-block alignments.

    Parameters
    ----------
    junction_df : pd.DataFrame
        Output of get_core_alignment_lengths() junction_df — one row per junction.
    block_df : pd.DataFrame or None
        Output of get_core_alignment_lengths() block_df — one row per core block.
        If provided, both series are shown together.
    col : str
        Column to plot. Default: 'aln_length_nogap'.
    bins : int
        Number of histogram bins.
    cumulative : bool
        If True, plot declining CDFs instead of histograms.
    log_x : bool
    log_y : bool
    figsize : tuple
    save_path : str or None
    """
    junction_values = junction_df[col].dropna().values
    block_values = block_df[col].dropna().values if block_df is not None else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.grid(True, color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if cumulative:
        def _plot_cdf(vals, color, label):
            sv = np.sort(vals)
            y = np.arange(len(sv), 0, -1) / len(sv) * 100
            ax.plot(sv, y, color=color, linewidth=1.5, label=label)

        if block_values is not None:
            _plot_cdf(block_values, COLORS["mid_blue"], "Individual core blocks")
        _plot_cdf(junction_values, COLORS["reddish"], "Combined core blocks (per junction)")
        ax.set_ylabel("Fraction of alignments ≥ x (%)", fontsize=13)
        ax.legend(fontsize=11)
    else:
        if block_values is not None:
            ax.hist(block_values, bins=bins, color=COLORS["mid_blue"], edgecolor="white",
                    linewidth=0.5, alpha=0.7, label="Individual core blocks",
                    weights=np.ones(len(block_values)) / len(block_values) * 100)
        ax.hist(junction_values, bins=bins, color=COLORS["reddish"], edgecolor="white",
                linewidth=0.5, alpha=0.7, label="Combined core blocks (per junction)",
                weights=np.ones(len(junction_values)) / len(junction_values) * 100)
        ax.set_ylabel("Relative frequency (%)", fontsize=13)
        ax.legend(fontsize=11)

    ax.set_xlabel("Alignment length (bp)", fontsize=13)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_block_avg_pairwise_dist(
    df,
    bins=100,
    cumulative=False,
    log_x=False,
    log_y=False,
    vline=None,
    vline_kwargs=None,
    figsize=(8, 4),
    save_path=None,
):
    """
    Plot the distribution of avg_pairwise_dist across all blocks and junctions.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_all_block_alignment_stats(), must contain avg_pairwise_dist column.
    bins : int
    cumulative : bool
        If True, plot a declining CDF (fraction of blocks with avg_pairwise_dist >= x).
    log_x : bool
    log_y : bool
    vline : float or None
        Draw a vertical line at this x-value (e.g. the gain/loss threshold).
    vline_kwargs : dict or None
    figsize : tuple
    save_path : str or None
    """
    values = df["avg_pairwise_dist"].dropna().values

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.grid(True, color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if cumulative:
        sorted_v = np.sort(values)
        y = np.arange(len(sorted_v), 0, -1) / len(sorted_v) * 100
        ax.plot(sorted_v, y, color=COLORS["mid_blue"], linewidth=1.5)
        ax.set_ylabel("% of blocks with avg pairwise dist ≥ x", fontsize=11)
    else:
        ax.hist(values, bins=bins, color=COLORS["mid_blue"], edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Number of blocks", fontsize=11)

    ax.set_xlabel("Average pairwise distance", fontsize=11)

    if vline is not None:
        kw = dict(color=COLORS["reddish"], linestyle="--", linewidth=1.5)
        if vline_kwargs:
            kw.update(vline_kwargs)
        ax.axvline(vline, **kw)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_ambiguous_block_distances(
    ambiguity_df,
    bins=100,
    cumulative=False,
    log_x=False,
    log_y=False,
    vline=None,
    vline_kwargs=None,
    figsize=(8, 4),
    save_path=None,
):
    """
    Plot the distribution of avg_pairwise_dist values from decide_ambiguities,
    unpacking the list stored in each row of ambiguity_df.avg_pairwise_dists.

    Parameters
    ----------
    ambiguity_df : pd.DataFrame
        Output of find_consensus_paths_core(), must contain avg_pairwise_dists column
        where each cell is a list of float values.
    bins : int
    cumulative : bool
        If True, plot a declining CDF (fraction of blocks with avg_pairwise_dist >= x).
    log_x : bool
    log_y : bool
    vline : float or None
        Draw a vertical line at this x-value (e.g. the gain/loss threshold).
    vline_kwargs : dict or None
    figsize : tuple
    save_path : str or None
    """
    import ast, re

    raw = ambiguity_df["avg_pairwise_dists"].dropna()
    values = []
    for entry in raw:
        if isinstance(entry, str):
            # parse floats directly from string, handles np.float64(...) and nan
            entry = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?=\s*[,\)])", entry)]
        values.extend([float(v) for v in entry if not np.isnan(float(v))])
    values = np.array(values)

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.grid(True, color="0.88", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if cumulative:
        sv = np.sort(values)
        y = np.arange(len(sv), 0, -1) / len(sv) * 100
        ax.plot(sv, y, color=COLORS["mid_blue"], linewidth=1.5)
        ax.set_ylabel("Fraction of blocks ≥ x (%)", fontsize=13)
    else:
        ax.hist(values, bins=bins, color=COLORS["mid_blue"], edgecolor="white", linewidth=0.5,
                weights=np.ones(len(values)) / len(values) * 100)
        ax.set_ylabel("Relative frequency (%)", fontsize=13)

    ax.set_xlabel("Average pairwise distance", fontsize=13)

    if vline is not None:
        kw = dict(color=COLORS["reddish"], linestyle="--", linewidth=1.5)
        if vline_kwargs:
            kw.update(vline_kwargs)
        ax.axvline(vline, **kw)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_scatter(
    df,
    x_col,
    y_col,
    xlabel=None,
    ylabel=None,
    log_x=False,
    log_y=False,
    alpha=0.6,
    size=20,
    figsize=None,
    save_path=None,
):
    """
    Simple scatter plot with optional subplots for multiple x columns.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str or list of str
        One or more columns for the x-axis. If a list, one subplot per column.
    y_col : str
        Column for the y-axis.
    xlabel : str or list of str or None
        X-axis label(s). If None, uses column name(s).
    ylabel : str or None
        Y-axis label. If None, uses column name.
    log_x : bool
    log_y : bool
    alpha : float
    size : float
        Marker size.
    figsize : tuple or None
        Auto-sized if None.
    save_path : str or None
    """
    if isinstance(x_col, str):
        x_col = [x_col]
    if isinstance(xlabel, str) or xlabel is None:
        xlabel = [xlabel] * len(x_col)

    n = len(x_col)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, xcol, xlbl in zip(axes, x_col, xlabel):
        plot_data = df[[xcol, y_col]].dropna()
        ax.scatter(plot_data[xcol], plot_data[y_col],
                   color=COLORS["mid_blue"], alpha=alpha, s=size,
                   linewidths=0.3, edgecolors="black")

        ax.set_xlabel(xlbl if xlbl is not None else xcol, fontsize=12)
        ax.set_ylabel(ylabel if ylabel is not None else y_col, fontsize=12)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_snp_gap_histogram(df, bins=100, log_y=False, log_x=False, cumulative=False, figsize=(8, 4), save_path=None):
    """
    Plot a histogram or declining CDF of inter-SNP gap lengths pooled across all junctions and blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Output of core_block_snp_gaps(). Must contain column 'gap_lengths'.
        Each entry can be a list or a string representation of a list.
    bins : int
    log_y : bool
    log_x : bool
    cumulative : bool
        If True, plot a declining cumulative distribution (fraction of gaps >= x).
    figsize : tuple
    save_path : str or None
    """
    import ast

    all_gaps = []
    for val in df["gap_lengths"]:
        if isinstance(val, str):
            val = ast.literal_eval(val)
        all_gaps.extend(val)

    all_gaps = np.array(all_gaps)

    fig, ax = plt.subplots(figsize=figsize)

    if cumulative:
        sorted_gaps = np.sort(all_gaps)
        y = np.arange(len(sorted_gaps), 0, -1) / len(sorted_gaps)
        ax.plot(sorted_gaps, y, color=COLORS["mid_blue"], linewidth=1.5)
        ax.set_ylabel("Fraction of gaps ≥ x", fontsize=11)
    else:
        ax.hist(all_gaps, bins=bins, color=COLORS["mid_blue"], edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Count", fontsize=11)

    ax.set_xlabel("Gap length between SNPs (bp)", fontsize=11)
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_snp_position_cdfs(df, figsize=(8, 4), alpha=0.3, highlight_junctions=None, save_path=None):
    """
    For each row in `df`, normalize SNP positions to [0, 1] using aln_length
    and plot a cumulative distribution curve. All curves are shown in the same axes.

    Parameters
    ----------
    df : pd.DataFrame
        Output of core_block_snp_gaps(). Must contain columns
        'snp_positions', 'aln_length', and 'junction_name'. Each entry in
        'snp_positions' can be a list or a string representation of a list.
    figsize : tuple
    alpha : float
        Transparency of individual curves. Default: 0.3.
    highlight_junctions : list of str or None
        Junction names to highlight in red. All others are plotted in mid blue.
    save_path : str or None
    """
    import ast

    highlight_set = set(highlight_junctions) if highlight_junctions else set()

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        snp_pos = row["snp_positions"]
        if isinstance(snp_pos, str):
            snp_pos = ast.literal_eval(snp_pos)
        if len(snp_pos) == 0:
            continue

        aln_length = row["aln_length"]
        normalized = np.sort(np.array(snp_pos) / aln_length)
        y = np.arange(1, len(normalized) + 1) / len(normalized)

        jname = row.get("junction_name", "")
        color = COLORS["reddish"] if jname in highlight_set else COLORS["mid_blue"]  # reddish or mid blue
        ax.plot(normalized, y, color=color, alpha=alpha, linewidth=0.8)

    ax.set_xlabel("Normalized SNP position (0 = start, 1 = end)", fontsize=11)
    ax.set_ylabel("Cumulative fraction of SNPs", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

def plot_all_junctions_pairwise_distances_zoom(
    distances_df,
    exclude=None,
    bins=100,
    figsize=(8, 4.2),
    vline=None,
    vline_kwargs=None,
    log_y=False,
    percentage=False,
    cumulative=False,
    log_x=False,
    distance_col="distance",
    title="default",
    xlabel=None,
    ylabel=None,
    legend_title=None,
    save_path=None,
    zoom_xlim=None,
    zoom_ylim=None,
    inset=True,
    inset_loc="upper right",
    inset_width="38%",
    inset_height="48%",
):
    """
    Plot pooled pairwise distances across all junctions.

    If zoom_xlim is given and inset=True, add a zoomed inset panel.
    """
    df = distances_df.copy()
    if exclude is not None:
        df = df[~df["junction_name"].isin(exclude)]

    distances = df[distance_col].dropna().values
    if len(distances) == 0:
        print("No distances to plot.")
        return

    n_junctions = df["junction_name"].nunique()

    if vline_kwargs is None:
        vline_kwargs = dict(color="black", linestyle="--", linewidth=2.5)

    if title == "default":
        title = f"Core genome pairwise distances ({n_junctions} junctions)"

    x_label = xlabel if xlabel is not None else "Pairwise patristic distance (substitutions per site)"
    if ylabel is not None:
        y_label = ylabel
    else:
        if cumulative:
            y_label = "% of pairwise distances ≥ x" if percentage else "Fraction of pairwise distances ≥ x"
        elif percentage:
            y_label = "% of pairwise distances"
        else:
            y_label = "Count"

    bar_color = COLORS["mid_blue"]

    def _draw(ax, show_legend=True):
        ax.yaxis.grid(True, color="0.9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        if cumulative:
            sorted_d = np.sort(distances)
            y = np.arange(len(sorted_d), 0, -1) / len(sorted_d)
            if percentage:
                y *= 100
            ax.plot(sorted_d, y, color=bar_color, linewidth=2.5, zorder=2)
        elif percentage:
            counts, edges = np.histogram(distances, bins=bins)
            pcts = counts / counts.sum() * 100
            widths = np.diff(edges)
            ax.bar(
                edges[:-1],
                pcts,
                width=widths,
                align="edge",
                color=bar_color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
        else:
            ax.hist(
                distances,
                bins=bins,
                density=False,
                color=bar_color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

        if vline is not None:
            default_label = f"Branch length cutoff for\nhomologous recombination ({vline})"
            ax.axvline(
                vline,
                label=legend_title if legend_title is not None else default_label,
                **vline_kwargs,
            )
            if show_legend:
                ax.legend(frameon=False, fontsize=18, loc="upper right", bbox_to_anchor=(0.4, 0.2))

        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=16)

    fig, ax = plt.subplots(figsize=figsize)
    _draw(ax, show_legend=True)

    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    if title is not None:
        ax.set_title(title, fontsize=22)

    if zoom_xlim is not None and inset:
        axins = inset_axes(
            ax,
            width="38%",
            height="48%",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        _draw(axins, show_legend=False)

        axins.set_xlim(*zoom_xlim)
        if zoom_ylim is not None:
            axins.set_ylim(*zoom_ylim)

        if log_x:
            axins.set_xscale("log")
        if log_y:
            axins.set_yscale("log")

        axins.tick_params(labelsize=16)
        axins.set_title("Zoom", fontsize=18, pad=4)

        # slightly softer inset styling
        for spine in axins.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.4")

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_clustering_threshold_sweep(df, figsize=(7,4), save_path=None, normalize=False):

    fig, ax = plt.subplots(figsize=figsize)

    metrics = [
        "n_junctions_with_clusters",
        "n_additional_clusters",
        "n_events",
        "n_insertions",
        "n_deletions",
        "n_translocations",
        "n_inversions",
        "n_inconsistant_cluster",
        "n_inconsistant_junctions"
    ]

    plot_df = df.copy()

    # convert thresholds to numeric so x-axis reflects real positions
    plot_df["threshold"] = plot_df["threshold"].astype(float)
    plot_df = plot_df.sort_values("threshold")

    if normalize:
        for m in metrics:
            max_v = plot_df[m].max()
            if max_v != 0:
                plot_df[m] = plot_df[m] / max_v

    for metric in metrics:
        ax.plot(
            plot_df["threshold"],
            plot_df[metric],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=metric.replace("_", " ")
        )

    ax.set_xlabel("branch-length threshold", fontsize=11)

    if normalize:
        ax.set_ylabel("normalized value (value / max)", fontsize=11)
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylabel("count", fontsize=11)

    # show ticks exactly at the thresholds
    ax.set_xticks(plot_df["threshold"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_fitch_tree(tree, iso_to_cluster, state_sets, junction_name, ax=None, save_path=None):
    """
    Plot a phylogenetic tree with branches colored by Fitch parsimony cluster assignment.

    A branch (parent -> child) is drawn in a cluster's color if state_sets[child]
    is a singleton {c}, meaning all tips below that child share the same cluster c.
    Mixed branches (state_sets has >1 element) are drawn in gray.

    Parameters
    ----------
    tree : Bio.Phylo tree
        Pruned tree containing only the isolates present at this junction.
    iso_to_cluster : dict
        Mapping isolate_name -> cluster_id (int).
    state_sets : dict
        Mapping clade -> set of cluster ids, from the Fitch bottom-up pass.
    junction_name : str
        Used as the plot title.
    ax : matplotlib Axes or None
        If None, a new figure is created.
    save_path : str or None
        If provided, save figure to this path instead of showing.
    """
    import matplotlib.patches as mpatches

    # remap cluster ids to consecutive 0, 1, 2, ...
    observed_clusters = sorted(set(iso_to_cluster.values()))
    cl_remap = {old: new for new, old in enumerate(observed_clusters)}
    iso_to_cluster = {iso: cl_remap[cl] for iso, cl in iso_to_cluster.items()}
    state_sets = {clade: {cl_remap[c] for c in ss} for clade, ss in state_sets.items()}

    n_tips = len(iso_to_cluster)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, n_tips * 0.18)))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    # --- cluster color palette ---
    cluster_ids = sorted(set(iso_to_cluster.values()))
    palette = [
        COLORS["teal"], COLORS["reddish"], COLORS["dark_blue"],
        COLORS["purple"], COLORS["pink"], COLORS["mid_blue"],
        COLORS["wine"], COLORS["light_blue"], COLORS["rosa"],
    ]
    cl_color = {cl: palette[i % len(palette)] for i, cl in enumerate(cluster_ids)}
    mixed_color = COLORS["gray"]

    # --- build parent map ---
    parent_map = {}
    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            parent_map[child] = clade

    # --- y positions: tips in pre-order traversal order ---
    tips_in_order = [c for c in tree.find_clades(order="preorder") if c.is_terminal()]
    tip_y = {tip: i for i, tip in enumerate(tips_in_order)}

    # --- x positions: cumulative branch lengths from root ---
    node_x = {tree.root: 0.0}
    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            bl = child.branch_length if child.branch_length is not None else 0.0
            node_x[child] = node_x[clade] + bl

    # --- y positions for internal nodes: midpoint of children's y range ---
    node_y = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            node_y[clade] = tip_y[clade]
        else:
            child_ys = [node_y[c] for c in clade.clades]
            node_y[clade] = (min(child_ys) + max(child_ys)) / 2.0

    def _clade_color(clade):
        ss = state_sets.get(clade, set())
        return cl_color[next(iter(ss))] if len(ss) == 1 else mixed_color

    # --- draw branches and vertical connectors ---
    lw = 1.8
    for clade in tree.find_clades(order="preorder"):
        # vertical connector at internal nodes
        if not clade.is_terminal():
            # sort children by y position
            children_sorted = sorted(clade.clades, key=lambda c: node_y[c])
            # draw segments between consecutive children, split at midpoint
            for i in range(len(children_sorted) - 1):
                lower = children_sorted[i]
                upper = children_sorted[i + 1]
                y_lo = node_y[lower]
                y_hi = node_y[upper]
                y_mid = (y_lo + y_hi) / 2.0
                c_lo = _clade_color(lower)
                c_hi = _clade_color(upper)
                ax.plot([node_x[clade], node_x[clade]], [y_lo, y_mid],
                        color=c_lo, linewidth=lw, solid_capstyle="butt")
                ax.plot([node_x[clade], node_x[clade]], [y_mid, y_hi],
                        color=c_hi, linewidth=lw, solid_capstyle="butt")

        # horizontal branch from parent to this node
        if clade is tree.root:
            continue
        parent = parent_map[clade]
        color = _clade_color(clade)
        ax.plot(
            [node_x[parent], node_x[clade]],
            [node_y[clade], node_y[clade]],
            color=color, linewidth=lw, solid_capstyle="butt",
        )

    # --- tip labels colored by cluster ---
    max_x = max(node_x.values())
    for tip in tips_in_order:
        cl = iso_to_cluster.get(tip.name)
        color = cl_color[cl] if cl is not None else mixed_color
        ax.text(
            max_x * 1.01, node_y[tip], tip.name,
            va="center", ha="left", fontsize=7, color=color,
        )

    # --- legend (top left) ---
    handles = [
        mpatches.Patch(color=cl_color[cl], label=f"cluster {cl}")
        for cl in cluster_ids
    ]
    ax.legend(handles=handles, fontsize=11, frameon=False,
              loc="upper left", bbox_to_anchor=(0, 1))

    ax.set_title(junction_name, fontsize=14, pad=4)
    ax.set_xlabel("branch length", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(left=0, right=max_x * 1.35)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Fitch tree plot to {save_path}")
        if created_fig:
            plt.close(fig)
    elif created_fig:
        plt.show()


def plot_path_comparison_lengths(
    path_comparison_df,
    cols=("shared_length", "exclusive_length_1", "exclusive_length_2", "exclusive_length"),
    labels=("shared", "exclusive (cluster 1)", "exclusive (cluster 2)", "exclusive (combined)"),
    bins=40,
    histogram=True,
    cumulative=True,
    log_x=False,
    log_y=False,
    figsize=None,
    save_path=None,
):
    """
    Plot histogram and/or reverse CDF of path comparison lengths.

    Parameters
    ----------
    path_comparison_df : pd.DataFrame
        Output of compare_consensus_paths loop, one row per junction.
    cols : sequence of str
        Columns to plot.
    labels : sequence of str
        Legend labels corresponding to each column.
    bins : int
    histogram : bool
        If True, plot the histogram panel.
    cumulative : bool
        If True, plot the reverse CDF panel.
    log_x, log_y : bool
    figsize : tuple or None
        Defaults to (5, 4) per panel.
    save_path : str or None
    """
    n_panels = int(histogram) + int(cumulative)
    if n_panels == 0:
        return

    if figsize is None:
        figsize = (5 * n_panels, 4)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_iter = iter(axes)
    ax_hist = next(panel_iter) if histogram else None
    ax_cdf  = next(panel_iter) if cumulative else None

    color_list = [
        COLORS["mid_blue"], COLORS["reddish"], COLORS["reddish"],
        COLORS["reddish"], COLORS["reddish"], COLORS["reddish"],
    ]

    for ax in axes:
        ax.yaxis.grid(True, color="0.92", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for col, label, color in zip(cols, labels, color_list):
        if col not in path_comparison_df.columns:
            continue
        vals = path_comparison_df[col].dropna().values

        if ax_hist is not None:
            weights = np.ones(len(vals)) / len(vals) * 100
            ax_hist.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.5,
                         alpha=0.75, label=label, weights=weights)

        if ax_cdf is not None:
            sv = np.sort(vals)
            y = np.arange(len(sv), 0, -1) / len(sv) * 100
            # prepend a point at x=0 so the fill covers the full width of the line
            sv_fill = np.concatenate([[0], sv])
            y_fill = np.concatenate([[100], y])
            ax_cdf.plot(sv, y, color=color, linewidth=2.0, label=label)
            ax_cdf.fill_between(sv_fill, y_fill, alpha=0.12, color=color)

    if ax_hist is not None:
        ax_hist.set_xlabel("Length (bp)", fontsize=16)
        ax_hist.set_ylabel("Relative frequency (%)", fontsize=16)
        ax_hist.tick_params(labelsize=14)
        if len(cols) > 1:
            ax_hist.legend(fontsize=13, frameon=False)

    if ax_cdf is not None:
        ax_cdf.set_xlabel("Length (bp)", fontsize=16)
        ax_cdf.set_ylabel("Fraction of junctions ≥ x (%)", fontsize=16)
        ax_cdf.tick_params(labelsize=14)
        ax_cdf.set_ylim(0, 102)
        if len(cols) > 1:
            ax_cdf.legend(fontsize=13, frameon=False)

    for ax in axes:
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_event_mge_heatmap(
    deduplicated_events_df,
    figsize=(7, 5),
    save_path=None,
):
    """
    Heatmap of event types vs. MGE association.

    For each event type (rows) and MGE category (columns), shows the count of
    events associated with that MGE type. Events with n_mge == 0 are shown in
    a 'None' column. Annotated with counts. Color scale matches plot_marginal_scatter.

    Parameters
    ----------
    deduplicated_events_df : pd.DataFrame
        Output of deduplicate_events(), must contain 'event_type', 'n_prophage',
        'n_integron', 'n_defense_system', 'n_is', 'n_mge'.
    figsize : tuple
    save_path : str or None
    """
    df = deduplicated_events_df[
        deduplicated_events_df["event_type"] != "ambiguous_insertion"
    ].copy()

    event_order = ["insertion", "deletion", "translocation", "inversion"]
    event_order = [e for e in event_order if e in df["event_type"].unique()]

    mge_cols = {
        "Prophage":              "n_prophage",
        "Integron":              "n_integron",
        "Defense system":        "n_defense_system",
        "IS":                    "n_is",
        "Prophage (associated)": "n_prophage_associated",
        "IS (associated)":       "n_is_associated",
    }

    rows = []
    for etype in event_order:
        sub = df[df["event_type"] == etype]
        row = {}
        for label, col in mge_cols.items():
            row[label] = int((sub[col] > 0).sum()) if col in sub.columns else 0
        row["None"] = int((sub["n_mge"] == 0).sum()) if "n_mge" in sub.columns else len(sub)
        rows.append(row)

    heatmap_df = pd.DataFrame(rows, index=event_order)
    col_order = ["Integron", "Prophage", "IS", "Defense system",
                 "Prophage (associated)", "IS (associated)", "None"]
    heatmap_df = heatmap_df[col_order]

    _cmap_colors = ["#b5d2f2", "#7394c2", "#397398", "#80557e", "#d991b4", "#a6444f"]
    cmap = LinearSegmentedColormap.from_list("event_mge_cmap", _cmap_colors)
    cmap.set_bad(color="#aaaaaa")  # grey for zero / masked cells

    values = heatmap_df.values.astype(float)
    vmin = max(values[values > 0].min(), 1) if (values > 0).any() else 1
    vmax = values.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # mask zeros so they render as grey
    masked_values = np.ma.masked_where(values == 0, values)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, norm=norm)

    # draw grey background for zero cells explicitly
    ax.imshow(np.where(values == 0, 1, np.nan), aspect="auto",
              cmap=LinearSegmentedColormap.from_list("grey_only", ["#aaaaaa", "#aaaaaa"]),
              vmin=0, vmax=1, zorder=0)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=13, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(event_order)))
    ax.set_yticklabels([e.capitalize() for e in event_order], fontsize=13)

    # annotate cells with counts; use log-normalized brightness to pick text color
    for i in range(len(event_order)):
        for j in range(len(col_order)):
            val = heatmap_df.iloc[i, j]
            if val == 0:
                text_color = "white"
            else:
                normed = norm(val)
                text_color = "white" if normed > 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("n. events", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("MGE category", fontsize=14, labelpad=10)
    ax.set_ylabel("Event type", fontsize=14, labelpad=8)
    ax.xaxis.set_label_position("bottom")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_event_mge_stacked_bar(
    deduplicated_events_df,
    figsize=(7, 5),
    save_path=None,
):
    """
    Stacked bar chart of event types vs. MGE association.

    Each bar shows the count of events per event type, stacked by MGE category.
    Full associations (Prophage, IS) are shown in solid colors; associated/partial
    and always-associated categories (Defense system, Integron) are shown
    semi-transparent. None is grey.
    """
    # --- color / style config ---------------------------------------------------
    _CAT_STYLE = {
        # label          : (hex_color,  alpha)
        "Prophage":              ("#a6444f", 1.0),   # reddish, solid
        "Prophage (associated)": ("#c98087", 1.0),   # lighter reddish, solid
        "IS":                    ("#7394c2", 1.0),   # mid_blue, solid
        "IS (associated)":       ("#b5d2f2", 1.0),   # light_blue, solid
        "Defense system":        ("#80557e", 1.0),   # purple, solid
        "Integron":              ("#d991b4", 1.0),   # pink, solid
        "None":                  ("#7a7a7a", 1.0),   # grey
    }
    _DISPLAY_NAME = {
        "Defense system":        "Defense system (associated)",
        "Integron":              "Integron (associated)",
    }
    # stack order bottom → top (legend shows top → bottom = Prophage first)
    _STACK_ORDER = [
        "None", "IS (associated)", "IS",
        "Defense system", "Integron",
        "Prophage (associated)", "Prophage",
    ]

    mge_cols = {
        "Prophage":              "n_prophage",
        "Prophage (associated)": "n_prophage_associated",
        "IS":                    "n_is",
        "IS (associated)":       "n_is_associated",
        "Defense system":        "n_defense_system",
        "Integron":              "n_integron",
    }

    df = deduplicated_events_df[
        deduplicated_events_df["event_type"] != "ambiguous_insertion"
    ].copy()

    event_order = ["insertion", "deletion", "translocation", "inversion"]
    event_order = [e for e in event_order if e in df["event_type"].unique()]

    # build counts dataframe
    rows = {}
    for etype in event_order:
        sub = df[df["event_type"] == etype]
        row = {}
        for label, col in mge_cols.items():
            row[label] = int((sub[col] > 0).sum()) if col in sub.columns else 0
        row["None"] = int((sub["n_mge"] == 0).sum()) if "n_mge" in sub.columns else len(sub)
        rows[etype] = row

    counts = pd.DataFrame(rows, index=_STACK_ORDER).T  # shape: event_order x categories

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(event_order))
    bottoms = np.zeros(len(event_order))

    bar_handles = []
    for cat in _STACK_ORDER:
        color, alpha = _CAT_STYLE[cat]
        vals = counts[cat].values.astype(float)
        bars = ax.bar(
            x, vals, bottom=bottoms,
            color=color, alpha=alpha,
            width=0.6, edgecolor="white", linewidth=0.5,
        )
        # legend handle with correct visual appearance
        display = _DISPLAY_NAME.get(cat, cat)
        handle = plt.Rectangle((0, 0), 1, 1, color=color,
                                alpha=alpha, linewidth=0)
        bar_handles.append((handle, display))
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in event_order], fontsize=13)
    ax.set_ylabel("n. events", fontsize=14)
    ax.set_xlabel("Event type", fontsize=14)
    ax.tick_params(axis="both", labelsize=12, length=0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    handles, labels = zip(*reversed(bar_handles))
    ax.legend(handles, labels, fontsize=11, frameon=False,
              loc="upper right", title="MGE category",
              title_fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig, ax

def plot_hits_distribution(
    df,
    column,
    bins=50,
    bin_width=None,
    cumulative=False,
    log_x=False,
    log_y=False,
    title=None,
    color_by=None,
    color_by_species=False,
    n_species=5,
    legend_labels=None,
    legend_title="",
    xlabel=None,
    figsize=(8, 4),
    save_path=None,
    two_colors=False,
    stacked=False,
):
    """
    Plot the distribution of one or more hits/genomes columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column(s).
    column : str or list of str
        Column(s) to plot, e.g. 'n_hits_external' or
        ['n_hits_external', 'n_genomes_external'].
    bins : int
        Number of histogram bins (ignored in cumulative mode).
    cumulative : bool
        If True, plot a reverse ECDF (fraction >= x) instead of a histogram.
    log_x : bool
        Use log scale on the x-axis. Zero values are dropped automatically.
    log_y : bool
        Use log scale on the y-axis.
    title : str or None
        Plot title; defaults to the column name (or first column if multiple).
    color_by : str or None
        Column name to color by (e.g. 'majority_organism'). When set, a
        stacked histogram is drawn with one color per category. Only one
        column may be plotted in this mode.
    figsize : tuple
    save_path : str or None
    """
    _palette = [COLORS["mid_blue"], COLORS["reddish"]] if two_colors else [
        COLORS["light_blue"], COLORS["mid_blue"], COLORS["purple"], COLORS["dark_blue"],
        COLORS["reddish"], COLORS["rosa"], COLORS["teal"], COLORS["wine"], COLORS["pink"]]
    _species_default_palette = [
        COLORS["light_blue"], COLORS["mid_blue"], COLORS["dark_blue"],
        COLORS["purple"], COLORS["rosa"], COLORS["reddish"]]

    columns = [column] if isinstance(column, str) else column

    fig, ax = plt.subplots(figsize=figsize)

    # Compute shared bin edges across all columns
    all_values = pd.concat([df[col].dropna() for col in columns])
    if log_x:
        all_values = all_values[all_values > 0]
    if log_x:
        shared_bins = np.logspace(np.log10(all_values.min()), np.log10(all_values.max()), bins + 1)
    elif bin_width is not None:
        shared_bins = np.arange(all_values.min(), all_values.max() + bin_width, bin_width)
    else:
        shared_bins = np.linspace(all_values.min(), all_values.max(), bins + 1)

    if color_by is not None:
        col = columns[0]
        categories = df[color_by].dropna().unique()
        categories = sorted(categories, key=lambda x: -(df[color_by] == x).sum())
        cat_colors = {cat: _palette[i % len(_palette)] for i, cat in enumerate(categories)}
        # also include rows where color_by is NaN
        if df[color_by].isna().any():
            categories = list(categories) + [None]
            cat_colors[None] = COLORS["gray"]

        bottoms = np.zeros(len(shared_bins) - 1)
        for cat in categories:
            mask = df[color_by].isna() if cat is None else (df[color_by] == cat)
            vals = df.loc[mask, col].dropna()
            if log_x:
                vals = vals[vals > 0]
            counts, _ = np.histogram(vals, bins=shared_bins)
            label = "unknown" if cat is None else cat
            ax.bar(
                shared_bins[:-1], counts, width=np.diff(shared_bins),
                bottom=bottoms, align="edge",
                color=cat_colors[cat], label=label, edgecolor="white", linewidth=0.3,
            )
            bottoms = bottoms + counts

        # apply custom legend labels in order of categories
        if legend_labels is not None:
            handles, _ = ax.get_legend_handles_labels()
            labels = list(legend_labels) + ["unknown"] * max(0, len(handles) - len(legend_labels))
        else:
            handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=10, frameon=False, title=legend_title, title_fontsize=11, loc="upper right")
        ax.set_ylabel("Count", fontsize=13)
        ax.set_xlabel(xlabel if xlabel is not None else col, fontsize=13)

    elif color_by_species:
        col = columns[0]
        # identify species columns (n_hits_<Genus>_<species>)
        known_cats = {"n_hits_own_chromosome", "n_hits_own_plasmid", "n_hits_other_chromosome",
                      "n_hits_other_plasmid", "n_hits_external"}
        species_cols = [c for c in df.columns if c.startswith("n_hits_") and c not in known_cats
                        and not c.startswith("n_hits_n_")]

        # exclude generic/uninformative species labels
        _exclude = {"bacterium", "Candidatus bacterium"}
        species_cols = [c for c in species_cols if c.replace("n_hits_", "").replace("_", " ") not in _exclude]

        # pick n most abundant species by total sum
        species_totals = df[species_cols].sum().sort_values(ascending=False)
        top_species = list(species_totals.index[:n_species])
        other_species = [c for c in species_cols if c not in top_species]

        species_labels = [c.replace("n_hits_", "").replace("_", " ") for c in top_species]
        species_colors = {c: _species_default_palette[i % len(_species_default_palette)] for i, c in enumerate(top_species)}
        species_colors["_other"] = COLORS["gray"]

        values = df[col].dropna()
        if log_x:
            values = values[values > 0]

        # assign each row to a bin
        bin_idx = np.digitize(values, shared_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(shared_bins) - 2)
        df_plot = df.loc[values.index].copy()
        df_plot["_bin"] = bin_idx

        bar_widths = np.diff(shared_bins)
        bottoms = np.zeros(len(shared_bins) - 1)

        for i, sc in enumerate(top_species + ["_other"]):
            bin_totals = np.zeros(len(shared_bins) - 1)
            for b in range(len(shared_bins) - 1):
                in_bin = df_plot[df_plot["_bin"] == b]
                if sc == "_other":
                    bin_totals[b] = in_bin[other_species].sum(axis=1).sum() if other_species else 0
                else:
                    bin_totals[b] = in_bin[sc].sum() if sc in in_bin.columns else 0

            # normalize to relative within each bar
            bar_counts, _ = np.histogram(values, bins=shared_bins)
            total_species_per_bin = np.zeros(len(shared_bins) - 1)
            for b in range(len(shared_bins) - 1):
                in_bin = df_plot[df_plot["_bin"] == b]
                total_species_per_bin[b] = in_bin[species_cols].sum(axis=1).sum()

            rel = np.where(total_species_per_bin > 0, bin_totals / total_species_per_bin, 0)
            heights = rel * bar_counts

            label = "Other" if sc == "_other" else sc.replace("n_hits_", "").replace("_", " ")
            if legend_labels is not None and i < len(legend_labels):
                label = legend_labels[i]

            color = species_colors[sc]
            ax.bar(shared_bins[:-1], heights, width=bar_widths, bottom=bottoms,
                   align="edge", color=color, label=label, edgecolor="white", linewidth=0.3)
            bottoms = bottoms + heights

        ax.legend(fontsize=10, frameon=False, title=legend_title, title_fontsize=11, loc="upper right")
        ax.set_ylabel("Count", fontsize=13)
        ax.set_xlabel(xlabel if xlabel is not None else col, fontsize=13)

    else:
        all_counts = []
        for i, col in enumerate(columns):
            color = _palette[i % len(_palette)]
            values = df[col].dropna()
            if log_x:
                values = values[values > 0]
            lbl = legend_labels[i] if legend_labels is not None and i < len(legend_labels) else col

            if cumulative:
                sorted_vals = np.sort(values)
                ecdf = 1 - np.arange(0, len(sorted_vals)) / len(sorted_vals)
                ax.plot(sorted_vals, ecdf, color=color, linewidth=1.5, label=lbl)
            elif stacked:
                counts, _ = np.histogram(values, bins=shared_bins)
                all_counts.append((counts, color, lbl))
            else:
                ax.hist(values, bins=shared_bins, color=color, edgecolor="white",
                        linewidth=0.4, alpha=0.7, label=lbl)

        if stacked and not cumulative:
            bottoms = np.zeros(len(shared_bins) - 1)
            bar_widths = np.diff(shared_bins)
            for counts, color, lbl in all_counts:
                ax.bar(shared_bins[:-1], counts, width=bar_widths, bottom=bottoms,
                       align="edge", color=color, label=lbl, edgecolor="white", linewidth=0.4)
                bottoms = bottoms + counts

        if cumulative:
            ax.set_ylabel("Fraction ≥ x", fontsize=13)
        else:
            ax.set_ylabel("Count", fontsize=13)

        if len(columns) > 1:
            ax.legend(fontsize=11, frameon=False, title=legend_title, title_fontsize=11, loc="upper right")

        ax.set_xlabel(xlabel if xlabel is not None else (columns[0] if len(columns) == 1 else ""), fontsize=13)

    ax.set_title(title or "", fontsize=14)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.tick_params(axis="both", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig, ax
