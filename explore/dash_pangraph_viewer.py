import copy
import random

from dash import Dash, dcc, html, Input, Output, Patch  # CHANGED: added Patch

from pathlib import Path
import sys

# Path to this script file
HERE = Path(__file__).resolve()

# Repo root = parent of junction_analysis/results/data
# If script is in explore/, parents[1] is repo root.
# If script is in junction_analysis/, parents[1] is also repo root.
REPO_ROOT = HERE.parents[1]

# Make imports work (repo root must be on sys.path so `import junction_analysis` works)
sys.path.insert(0, str(REPO_ROOT))

from junction_analysis.plotting import plot_pangraph_base_for_dash, add_annotations_for_dash, _rgb_str, _shades_from_base_rgb, _rgba
from junction_analysis.consensus import find_consensus_paths_core

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pypangraph as pp


def _trace_groups_from_fig(fig):
    """
    Find trace indices for:
      - blocks: bar traces with customdata containing block_id at [0][3] AND hovertemplate contains '<br>block:'
      - mges: bar traces named Prophage / Defense system / IS:*
      - intrec: bar traces named 'Integrase / Recombinase'
      - cds: bar traces named 'Coding Sequence (CDS)'
      - insertions: bar traces named 'Insertion'
      - deletions: scatter traces named 'Deletion'
    """
    groups = {"blocks": [], "mges": [], "intrec": [], "cds": [], "insertions": [], "deletions": [], "inversions": [], "translocations": []}

    for i, tr in enumerate(fig.data):
        ttype = getattr(tr, "type", None)

        name = getattr(tr, "name", "") or ""
        hover = getattr(tr, "hovertemplate", "") or ""
        if isinstance(hover, (list, tuple)):
            hover = hover[0] if hover else ""
        cd = getattr(tr, "customdata", None)

        # block traces: your _add_bar sets hovertemplate with "<br>Block = " and customdata includes block_id at index 3
        if ttype == "bar" and ("<br>Block = " in hover) and (cd is not None) and len(cd) > 0 and len(cd[0]) >= 4:
            groups["blocks"].append(i)
            continue

        # annotation traces: your function sets these names explicitly
        if name == "Integrase / Recombinase":
            groups["intrec"].append(i)
        elif name == "Coding Sequence (CDS)":
            groups["cds"].append(i)
        elif name in ("Prophage", "Defense system") or name.startswith("IS:"):
            groups["mges"].append(i)
        elif name == "Insertion":
            groups["insertions"].append(i)
        elif name == "Deletion":
            groups["deletions"].append(i)
        elif name == "Inversion":
            groups["inversions"].append(i)
        elif name == "Translocation":
            groups["translocations"].append(i)

    return groups


def _compute_block_colors_for_figure(fig, pan):
    """
    Precompute per-block-trace colors for:
      - colored mode (based on *existing* marker colors in the base figure)
      - grey mode (based on core/accessory from pan.to_blockstats_df())

    This avoids replicating seaborn palette logic, and guarantees "colored" matches your original output.
    """
    bdf = pan.to_blockstats_df()
    core_by_bid = {str(bid): bool(bdf.loc[bid, "core"]) for bid in bdf.index}

    GREY_CORE = "rgb(220,220,220)"
    GREY_ACC = "rgb(190,190,190)"

    groups = _trace_groups_from_fig(fig)
    block_idxs = groups["blocks"]

    colored = []
    grey = []
    for tidx in block_idxs:
        tr = fig.data[tidx]
        bid_str = str(tr.customdata[0][3])

        # colored: whatever your original function assigned
        colored.append(tr.marker.color)

        # grey: derived from core/accessory
        is_core = core_by_bid.get(bid_str, False)
        grey.append(GREY_CORE if is_core else GREY_ACC)

    return groups, colored, grey


def make_junction_dash_app(
    *,
    pan,
    consensus_paths_plotting,
    assignment_df_plotting,
    cluster_map_core,
    mges_gff_path: str,
    annotations_gff_path: str,
    order: str = "tree",
    title: str = "Junction viewer",
    initial_selection=("mges",),  # e.g. ("mges","intrec","cds","indels") or ()
    show_indels: bool = False,
    indels_base_path: str = None,
    junction_name: str = None,
):
    """
    Creates a Dash app showing the pangraph with annotation toggles.
    """
    # 1. Build a base figure with colored blocks (no annotations yet)
    base_fig, y_labels, max_x, inversion_rects = plot_pangraph_base_for_dash(
        pan=pan,
        show_consensus=True,
        consensus_paths=consensus_paths_plotting,
        assignments=assignment_df_plotting,
        order=order,
        cluster_map=cluster_map_core,
        add_cluster_annotation=True,
        title=title,
        grey_mode=False,  # Start with colored blocks
    )

    # 2. Add all annotation traces to the figure (they will be hidden by default)
    base_fig = add_annotations_for_dash(
        fig=base_fig,
        y_labels=y_labels,
        show_mges_annotations=True,
        show_int_rec_annotations=True,
        show_cds_annotations=True,
        mges_gff_path=mges_gff_path,
        annotations_gff_path=annotations_gff_path,
        annotation_alpha=0.70,
        cds_annotation_alpha=0.30,
        show_indels=show_indels,
        indels_base_path=indels_base_path,
        junction_name=junction_name,
        consensus_paths=consensus_paths_plotting,
        inversion_rects=inversion_rects,
        max_x=max_x,
    )

    base_fig.update_layout(
        uirevision="junction-viewer",
        autosize=False,
        height=5000,
        width=1800,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    groups, block_colors_colored, block_colors_grey = _compute_block_colors_for_figure(base_fig, pan)

    # Store original hover templates for blocks to be able to restore them
    original_hovertemplates = [base_fig.data[tidx].hovertemplate for tidx in groups["blocks"]]

    # Store original hover templates for annotations
    anno_indices = groups["mges"] + groups["intrec"] + groups["cds"]
    original_anno_hovertemplates = {tidx: base_fig.data[tidx].hovertemplate for tidx in anno_indices}

    # Store original hover templates for indels
    indel_indices = groups["insertions"] + groups["deletions"] + groups["inversions"] + groups["translocations"]
    original_indel_hovertemplates = {tidx: base_fig.data[tidx].hovertemplate for tidx in indel_indices}

    # Build lookup: block_id -> set of trace indices for highlighting
    block_id_to_traces = {}
    for tidx in groups["blocks"]:
        tr = base_fig.data[tidx]
        for row in tr.customdata:
            bid_str = str(row[3])
            block_id_to_traces.setdefault(bid_str, set()).add(tidx)

    # Store original line colors/widths for block traces to restore when search is cleared
    original_block_lines = {}
    for tidx in groups["blocks"]:
        tr = base_fig.data[tidx]
        original_block_lines[tidx] = {
            "color": tr.marker.line.color,
            "width": tr.marker.line.width,
        }

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "Arial", "margin": "12px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "18px", "marginBottom": "10px"},
                children=[
                    html.Div("Annotations:", style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id="anno-toggle",
                        options=[
                            {"label": "MGEs", "value": "mges"},
                            {"label": "Integrase/Recombinase", "value": "intrec"},
                            {"label": "CDS", "value": "cds"},
                            {"label": "Insertions/Deletions", "value": "indels", "disabled": not show_indels},
                        ],
                        value=list(initial_selection),
                        inline=True,
                        persistence=True,
                        persistence_type="memory",
                    ),
                    html.Div("Hover options:", style={"fontWeight": "bold", "marginLeft": "20px"}),
                    dcc.Checklist(
                        id="hover-toggle",
                        options=[
                            {"label": "Disable block hover", "value": "disable_block_hover"},
                            {"label": "Disable annotation hover", "value": "disable_anno_hover"},
                            {"label": "Disable indel hover", "value": "disable_indel_hover"},
                        ],
                        value=[],
                        inline=True,
                        persistence=True,
                        persistence_type="memory",
                    ),
                    html.Div("Search block ID:", style={"fontWeight": "bold", "marginLeft": "20px"}),
                    dcc.Input(
                        id="block-search",
                        type="text",
                        placeholder="Enter block ID...",
                        debounce=True,
                        style={"width": "220px"},
                        persistence=True,
                        persistence_type="memory",
                    ),
                    html.Span(id="search-count", style={"marginLeft": "8px", "color": "grey"}),
                ],
            ),
            dcc.Graph(id="graph", figure=base_fig),
        ],
    )

    @app.callback(
        Output("graph", "figure"),
        Output("search-count", "children"),
        Input("anno-toggle", "value"),
        Input("hover-toggle", "value"),
        Input("block-search", "value"),
    )
    def update_figure(selected_annos, selected_options, search_query):
        selected_annos = set(selected_annos or [])
        selected_options = set(selected_options or [])
        any_on = len(selected_annos) > 0
        # Only turn blocks grey for non-indel annotations
        non_indel_annos_on = bool(selected_annos - {"indels"})

        # Determine which traces to highlight based on block ID search
        search_query = (search_query or "").strip()
        highlighted_traces = set()
        if search_query:
            for bid, tidxs in block_id_to_traces.items():
                if search_query in bid:
                    highlighted_traces.update(tidxs)

        patch = Patch()

        # 1) Toggle annotation trace visibility
        def _set_visible(idxs, on):
            for i in idxs:
                patch["data"][i]["visible"] = bool(on)

        _set_visible(groups["mges"], "mges" in selected_annos)
        _set_visible(groups["intrec"], "intrec" in selected_annos)
        _set_visible(groups["cds"], "cds" in selected_annos)
        _set_visible(groups["insertions"], "indels" in selected_annos)
        _set_visible(groups["deletions"], "indels" in selected_annos)
        _set_visible(groups["inversions"], "indels" in selected_annos)
        _set_visible(groups["translocations"], "indels" in selected_annos)

        # Toggle translocation arrow annotations
        n_annotations = len(base_fig.layout.annotations or [])
        for ai in range(n_annotations):
            patch["layout"]["annotations"][ai]["visible"] = ("indels" in selected_annos)

        # 2) Enforce block colors (grey only if non-indel annotations are on)
        colors = block_colors_grey if non_indel_annos_on else block_colors_colored
        for j, tidx in enumerate(groups["blocks"]):
            patch["data"][tidx]["marker"]["color"] = colors[j]

        # 2b) Highlight blocks matching search query
        for tidx in groups["blocks"]:
            if tidx in highlighted_traces:
                patch["data"][tidx]["marker"]["line"]["color"] = "magenta"
                patch["data"][tidx]["marker"]["line"]["width"] = 3
            else:
                patch["data"][tidx]["marker"]["line"]["color"] = original_block_lines[tidx]["color"]
                patch["data"][tidx]["marker"]["line"]["width"] = original_block_lines[tidx]["width"]

        # 3) Toggle block hover information
        disable_block_hover = "disable_block_hover" in selected_options
        if disable_block_hover:
            for tidx in groups["blocks"]:
                patch["data"][tidx]["hoverinfo"] = "skip"
                patch["data"][tidx]["hovertemplate"] = None
        else:
            for i, tidx in enumerate(groups["blocks"]):
                patch["data"][tidx]["hoverinfo"] = "all"
                patch["data"][tidx]["hovertemplate"] = original_hovertemplates[i]

        # 4) Toggle annotation hover information
        disable_anno_hover = "disable_anno_hover" in selected_options
        for tidx in anno_indices:
            if disable_anno_hover:
                patch["data"][tidx]["hoverinfo"] = "skip"
                patch["data"][tidx]["hovertemplate"] = None
            else:
                patch["data"][tidx]["hoverinfo"] = "all"
                patch["data"][tidx]["hovertemplate"] = original_anno_hovertemplates[tidx]

        # 5) Toggle indel hover information
        disable_indel_hover = "disable_indel_hover" in selected_options
        for tidx in indel_indices:
            if disable_indel_hover:
                patch["data"][tidx]["hoverinfo"] = "skip"
                patch["data"][tidx]["hovertemplate"] = None
            else:
                patch["data"][tidx]["hoverinfo"] = "all"
                patch["data"][tidx]["hovertemplate"] = original_indel_hovertemplates[tidx]

        # 6) barmode: overlay when annotations are on; stack otherwise
        patch["layout"]["barmode"] = "overlay" if any_on else "stack"

        # Search result count
        search_msg = ""
        if search_query:
            search_msg = f"{len(highlighted_traces)} traces matched"

        return patch, search_msg

    return app


if __name__ == "__main__":
    
    #junction_name = "XXVMWZCEKI_r__YUOECYBHUS_r" # consensus definition very conservative here, investigate
    #junction_name = "PLTCZQCVRD_f__RYYAQMEJGY_f"
    #junction_name = "NOAJDCSIVA_f__NZXBIFMPMA_r"
    junction_name = "RYYAQMEJGY_r__ZTHKZYHPIX_f"
    #junction_name = "CIRMBUYJFK_f__CWCCKOQCWZ_r"
    #junction_name = "ATPWUNKKID_f__KKPYPKGMXA_f"

    pangraph_path = REPO_ROOT / "results" / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    mges_gff_path = REPO_ROOT / "results" / "junction_mges" / f"{junction_name}.gff3"
    annotations_gff_path = REPO_ROOT / "results" / "junction_annotations" / f"{junction_name}.gff"
    tree_path = REPO_ROOT / "config" / "polished_tree.nwk"
    in_del_path = REPO_ROOT / "results" / "atb_lookup" 

    cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, all_root_states, all_root_states_unqiue = find_consensus_paths_core(
        junction_name,
        plot_consensus=False,
        plot_annotations=False,
        plot_pair_dist=False,
        plot_snp_dist=False,
        plot_ambiguities=False,
        clustering_bl_thresh=0.005,
        consensus_criterium="core_genome_tree",
        tree_path=str(tree_path),
    )

    app = make_junction_dash_app(
        pan=pangraph,
        consensus_paths_plotting=consensus_paths_plotting,
        assignment_df_plotting=assignment_df_plotting,
        cluster_map_core=cluster_map_core,
        mges_gff_path=str(mges_gff_path),
        annotations_gff_path=str(annotations_gff_path),
        order="tree",
        title=f"Junction Block Structure ({junction_name})",
        initial_selection=("mges", "indels"),  # change to () to start with no annotations
        indels_base_path=str(in_del_path),
        show_indels=True,
        junction_name=junction_name,
    )

    app.run(debug=False)  # CHANGED: turn off debug for normal use
