from dash import Dash, dcc, html, Input, Output, Patch, State

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

from junction_analysis.plotting import plot_pangraph_base_for_dash, add_annotations_for_dash
from junction_analysis.consensus import find_consensus_paths_core, find_consensus_paths_edge, find_consensus_paths_refined, save_consensus_cache, load_consensus_cache
from junction_analysis.annotate_insertions import (
    # get_insertions_deletions_from_consensus,  # replaced by new pipeline
    detect_event_blocks,
    group_events_by_clade,
    find_combined_events,
    write_insertions_fasta,
    # summarize_ambiguous_insertions,  # no longer used
    summarize_deletions_consensus,
    summarize_inversions_consensus,
    load_all_insertions_summaries,
    # load_all_ambiguous_insertions_summaries,  # no longer used
    load_all_deletions_summaries,
    load_all_inversions_summaries,
)

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pypangraph as pp

# Species color palette for pie charts
_PIE_COLORS = [
    "#a6444f", "#57a8b8", "#80557e", "#b5d2f2", "#d991b4",
    "#397398", "#7394c2", "#7a7a7a", "#7a2030", "#5b8c5a",
    "#c9a84c", "#e07b39", "#6b4c9a", "#4a7c8e", "#c75b7a",
]



_LEXICMAP_SUMMARY_PATH  = REPO_ROOT / "results" / "atb_lookup" / "insertions" / "lexicmap_hits_summary_100.csv"
_SELF_HITS_PATH         = REPO_ROOT / "results" / "atb_lookup" / "insertions" / "self_hits_combined.csv"


_CAT_COLS = ["n_hits_own_chromosome", "n_hits_own_plasmid",
             "n_hits_other_chromosome", "n_hits_other_plasmid", "n_hits_external"]
_NON_SPECIES = set(_CAT_COLS) | {"n_hits_st131"}


def _build_insertion_pie_lookup(junction_name: str, indels_base_path: str) -> dict:
    """
    Build lookup: (genome_name, segment_key) -> {"species": {...}, "cats": {...}}.

    Both species distribution and category counts come from the per-genome
    lexicmap_hits_summary_100.csv, so every insertion reflects its specific
    genome's actual hits (no deduplication).
    """
    if not _LEXICMAP_SUMMARY_PATH.exists():
        return {}

    summ = pd.read_csv(_LEXICMAP_SUMMARY_PATH)
    summ_jn = summ[summ["junction_name"] == junction_name]
    if summ_jn.empty:
        return {}

    species_cols  = [c for c in summ_jn.columns
                     if c.startswith("n_hits_") and c not in _NON_SPECIES]
    cat_cols_present = [c for c in _CAT_COLS if c in summ_jn.columns]

    lookup = {}
    for _, row in summ_jn.iterrows():
        key = (row["genome_name"], row["segment"])
        if key not in lookup:
            lookup[key] = {
                "species": {col[len("n_hits_"):]: float(row[col])
                            for col in species_cols if row[col] > 0},
                "cats":    {col[len("n_hits_"):]: int(row[col])
                            for col in cat_cols_present},
            }
    return lookup


def _make_pie_figure(species_counts: dict) -> go.Figure:
    """Build a small Plotly pie figure from a species counts dict."""
    if not species_counts:
        fig = go.Figure()
        fig.update_layout(
            height=220, width=340,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=20, b=5),
            annotations=[dict(text="No hits", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=13, color="#888"), xref="paper", yref="paper")],
        )
        return fig
    items = sorted(species_counts.items(), key=lambda x: -x[1])
    if len(items) > 8:
        other_val = sum(v for _, v in items[8:])
        items = items[:8] + [("Other", other_val)]
    values = [v for _, v in items]
    labels = [f"{k.replace('_', ' ')} ({int(v):,})" for k, v in items]

    # Assign colors: fixed for special labels, cycle palette for the rest
    palette = iter(_PIE_COLORS)
    colors = []
    for lbl, _ in zip(labels, values):
        if lbl.startswith("Other ") or lbl == "Other":
            colors.append("#7a7a7a")
        elif lbl.startswith("Escherichia coli"):
            colors.append("#b5d2f2")
        else:
            colors.append(next(palette, "#7a7a7a"))
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color="white", width=0.5)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value:.0f} hits (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        height=220,
        width=340,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9), orientation="v", x=1.0),
        showlegend=True,
    )
    return fig


def _make_category_pie_figure(cats: dict) -> go.Figure:
    """Pie chart showing own chromosome / own plasmid / other chromosome / other plasmid / external."""
    _CAT_LABELS = {
        "own_chromosome":   "Own chromosome",
        "own_plasmid":      "Own plasmid",
        "other_chromosome": "Other chromosome",
        "other_plasmid":    "Other plasmid",
        "external":         "External",
    }
    _CAT_COLORS = {
        "own_chromosome":   "#397398",
        "own_plasmid":      "#7394c2",
        "other_chromosome": "#57a8b8",
        "other_plasmid":    "#b5d2f2",
        "external":         "#7a7a7a",
    }
    items = [(k, int(cats.get(k, 0))) for k in _CAT_LABELS]
    items = [(k, v) for k, v in items if v > 0]
    if not items:
        return go.Figure()
    values = [v for _, v in items]
    labels = [f"{_CAT_LABELS[k]} ({v:,})" for k, v in items]
    colors = [_CAT_COLORS[k] for k, _ in items]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color="white", width=0.5)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value:,d} hits (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        height=220,
        width=300,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9), orientation="v", x=1.0),
        showlegend=True,
    )
    return fig


def _trace_groups_from_fig(fig):
    """
    Find trace indices for:
      - blocks: bar traces with customdata containing block_id at [0][3] AND hovertemplate contains '<br>block:'
      - mges: bar traces named Prophage / Defense system / IS:*
      - intrec: bar traces named 'Integrase / Recombinase / Transposase'
      - cds: bar traces named 'Coding Sequence (CDS)'
      - insertions: bar traces named 'Insertion'
      - deletions: scatter traces named 'Deletion'
    """
    groups = {"blocks": [], "mges": [], "intrec": [], "cds": [], "trna": [], "insertions": [], "deletions": [], "inversions": [], "translocations": [], "clustering": []}

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
        if name == "Integrase / Recombinase / Transposase":
            groups["intrec"].append(i)
        elif name == "Coding Sequence (CDS)":
            groups["cds"].append(i)
        elif name == "tRNA / tmRNA":
            groups["trna"].append(i)
        elif name in ("Prophage", "Defense system", "Integron") or name.startswith("IS:"):
            groups["mges"].append(i)
        elif name == "Insertion":
            groups["insertions"].append(i)
        elif name == "Deletion":
            groups["deletions"].append(i)
        elif name == "Inversion":
            groups["inversions"].append(i)
        elif name == "Translocation":
            groups["translocations"].append(i)
        elif ttype == "scatter" and (name.startswith("Cluster ") or (not name and getattr(tr, "hoverinfo", None) == "skip" and getattr(tr, "showlegend", True) is False)):
            groups["clustering"].append(i)

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
    show_trna: bool = False,
    indels_base_path: str = None,
    junction_name: str = None,
    path_dict: dict = None,
):
    """
    Creates a Dash app showing the pangraph with annotation toggles.
    """
    # build nid -> suffixed block id lookup from path_dict if provided
    nid_to_block_id = {}
    if path_dict is not None:
        for iso, path in path_dict.items():
            for node in path.nodes:
                if node.nid is not None:
                    nid_to_block_id[str(node.nid)] = node.id

    def _build_fig(show_consensus):
        fig, yl, mx, inv_rects = plot_pangraph_base_for_dash(
            pan=pan,
            show_consensus=show_consensus,
            consensus_paths=consensus_paths_plotting,
            assignments=assignment_df_plotting,
            order=order,
            cluster_map=cluster_map_core,
            add_cluster_annotation=True,
            title=title,
            grey_mode=False,
            nid_to_block_id=nid_to_block_id if nid_to_block_id else None,
        )
        fig = add_annotations_for_dash(
            fig=fig,
            y_labels=yl,
            show_mges_annotations=mges_gff_path is not None,
            show_int_rec_annotations=annotations_gff_path is not None,
            show_trna_annotations=show_trna and annotations_gff_path is not None,
            show_cds_annotations=annotations_gff_path is not None,
            mges_gff_path=mges_gff_path,
            annotations_gff_path=annotations_gff_path,
            annotation_alpha=0.70,
            cds_annotation_alpha=0.30,
            show_indels=show_indels,
            indels_base_path=indels_base_path,
            junction_name=junction_name,
            consensus_paths=consensus_paths_plotting,
            inversion_rects=inv_rects,
            max_x=mx,
        )
        fig.update_layout(
            uirevision="junction-viewer",
            autosize=False,
            height=5000,
            width=1800,
            margin=dict(l=10, r=10, t=140, b=10),
        )
        return fig

    # Build pie chart lookup: (genome_name, segment) -> species counts via ctx join
    _pie_lookup = (
        _build_insertion_pie_lookup(junction_name, indels_base_path)
        if (junction_name and indels_base_path)
        else {}
    )

    # Self-hits lookup: (genome_name, insertion) -> {hits_in_genome, hits_in_plasmid}
    _self_lookup = {}
    if junction_name and _SELF_HITS_PATH.exists():
        _sh = pd.read_csv(_SELF_HITS_PATH)
        _sh = _sh[_sh["junction_name"] == junction_name]
        for _, _row in _sh.iterrows():
            _key = (_row["genome_name"], _row["insertion"])
            if _key not in _self_lookup:
                _self_lookup[_key] = {
                    "hits_in_genome":  int(_row.get("hits_in_genome",  0) or 0),
                    "hits_in_plasmid": int(_row.get("hits_in_plasmid", 0) or 0),
                }

    # Pre-build both figures
    base_fig_consensus = _build_fig(show_consensus=True)
    base_fig_tree = _build_fig(show_consensus=False)

    base_fig = base_fig_consensus  # default

    # Initialise translocation arrow visibility to match initial_selection
    _indels_initially_on = "indels" in initial_selection
    for _fig in (base_fig_consensus, base_fig_tree):
        for _ann in (_fig.layout.annotations or []):
            _ann.visible = _indels_initially_on

    def _precompute(fig):
        grps, colored, grey = _compute_block_colors_for_figure(fig, pan)
        orig_hover = [fig.data[tidx].hovertemplate for tidx in grps["blocks"]]
        anno_idx = grps["mges"] + grps["intrec"] + grps["cds"] + grps["trna"]
        orig_anno_hover = {tidx: fig.data[tidx].hovertemplate for tidx in anno_idx}
        indel_idx = grps["insertions"] + grps["deletions"] + grps["inversions"] + grps["translocations"]
        orig_indel_hover = {tidx: fig.data[tidx].hovertemplate for tidx in indel_idx}
        bid_to_traces = {}
        for tidx in grps["blocks"]:
            for row in fig.data[tidx].customdata:
                bid_str = str(row[3])
                bid_to_traces.setdefault(bid_str, set()).add(tidx)
        orig_lines = {
            tidx: {"color": fig.data[tidx].marker.line.color, "width": fig.data[tidx].marker.line.width}
            for tidx in grps["blocks"]
        }
        return grps, colored, grey, orig_hover, anno_idx, orig_anno_hover, indel_idx, orig_indel_hover, bid_to_traces, orig_lines

    precomputed_consensus = _precompute(base_fig_consensus)
    precomputed_tree = _precompute(base_fig_tree)

    # Clustering traces start visible (toggle is on by default)
    for _grps, _fig in ((precomputed_consensus[0], base_fig_consensus), (precomputed_tree[0], base_fig_tree)):
        for _tidx in _grps["clustering"]:
            _fig.data[_tidx].visible = True

    # unpack defaults (consensus view)
    groups, block_colors_colored, block_colors_grey, original_hovertemplates, \
        anno_indices, original_anno_hovertemplates, indel_indices, \
        original_indel_hovertemplates, block_id_to_traces, original_block_lines = precomputed_consensus

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
                            {"label": "Integrase/Recombinase/Transposase", "value": "intrec"},
                            {"label": "CDS", "value": "cds"},
                            {"label": "Insertions/Deletions", "value": "indels", "disabled": not show_indels},
                            {"label": "tRNA/tmRNA", "value": "trna", "disabled": not show_trna},
                        ],
                        value=list(initial_selection),
                        inline=True,
                        persistence=True,
                        persistence_type="memory",
                    ),
                    html.Div("Options:", style={"fontWeight": "bold", "marginLeft": "20px", "whiteSpace": "nowrap"}),
                    dcc.Checklist(
                        id="hover-toggle",
                        options=[
                            {"label": "Disable block hover", "value": "disable_block_hover"},
                            {"label": "Disable annotation hover", "value": "disable_anno_hover"},
                            {"label": "Disable indel hover", "value": "disable_indel_hover"},
                            {"label": "Show consensus", "value": "show_consensus"},
                            {"label": "Show clustering", "value": "show_clustering"},
                        ],
                        value=["show_consensus", "show_clustering"],
                        inline=True,
                        persistence=True,
                        persistence_type="memory",
                    ),
                    html.Div("Search block ID:", style={"fontWeight": "bold", "marginLeft": "20px", "whiteSpace": "nowrap"}),
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
            html.Div(
                id="insertion-pie-panel",
                style={
                    "position": "fixed",
                    "bottom": "20px",
                    "right": "20px",
                    "width": "360px",
                    "background": "rgba(255,255,255,0.96)",
                    "border": "1px solid #ccc",
                    "borderRadius": "6px",
                    "padding": "8px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
                    "zIndex": 1000,
                    "display": "none",
                },
                children=[
                    html.Div(
                        id="insertion-self-hits",
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        id="insertion-pie-title",
                        style={"fontWeight": "bold", "fontSize": "11px",
                               "marginBottom": "4px", "color": "#444"},
                    ),
                    html.Div(
                        id="insertion-charts-row",
                        style={"display": "none", "gap": "0px"},
                        children=[
                            html.Div(
                                id="insertion-cats-col",
                                style={"display": "none"},
                                children=[
                                    html.Div("Hit categories", style={"fontSize": "10px", "color": "#666", "textAlign": "center"}),
                                    dcc.Graph(
                                        id="insertion-pie-cats",
                                        figure=go.Figure(layout=dict(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=220, width=300)),
                                        config={"displayModeBar": False},
                                        style={"height": "220px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                id="insertion-species-col",
                                children=[
                                    html.Div("External species", style={"fontSize": "10px", "color": "#666", "textAlign": "center"}),
                                    dcc.Graph(
                                        id="insertion-pie",
                                        figure=go.Figure(layout=dict(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=220, width=340)),
                                        config={"displayModeBar": False},
                                        style={"height": "220px"},
                                    ),
                                    html.Div(
                                        id="insertion-no-hits",
                                        style={"display": "none", "fontSize": "12px", "color": "#888",
                                               "padding": "80px 40px", "textAlign": "center"},
                                        children="No external hits",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ] if _pie_lookup else [],
            ),
        ],
    )

    app.layout.children.insert(0, dcc.Store(id="consensus-store", data=True))

    @app.callback(
        Output("graph", "figure"),
        Output("search-count", "children"),
        Output("consensus-store", "data"),
        Input("anno-toggle", "value"),
        Input("hover-toggle", "value"),
        Input("block-search", "value"),
        State("consensus-store", "data"),
    )
    def update_figure(selected_annos, selected_options, search_query, prev_consensus):
        selected_annos = set(selected_annos or [])
        selected_options = set(selected_options or [])
        any_on = len(selected_annos) > 0
        non_indel_annos_on = bool(selected_annos - {"indels"})

        use_consensus = "show_consensus" in selected_options
        consensus_changed = use_consensus != prev_consensus

        if use_consensus:
            (groups, block_colors_colored, block_colors_grey, original_hovertemplates,
             anno_indices, original_anno_hovertemplates, indel_indices,
             original_indel_hovertemplates, block_id_to_traces, original_block_lines) = precomputed_consensus
        else:
            (groups, block_colors_colored, block_colors_grey, original_hovertemplates,
             anno_indices, original_anno_hovertemplates, indel_indices,
             original_indel_hovertemplates, block_id_to_traces, original_block_lines) = precomputed_tree

        search_query = (search_query or "").strip()
        highlighted_traces = set()
        if search_query:
            for bid, tidxs in block_id_to_traces.items():
                if search_query in bid:
                    highlighted_traces.update(tidxs)

        # When consensus mode switches, build a full figure copy with all states applied
        if consensus_changed:
            import copy as _copy
            active_fig = _copy.deepcopy(base_fig_consensus if use_consensus else base_fig_tree)
            colors = block_colors_grey if non_indel_annos_on else block_colors_colored
            for j, tidx in enumerate(groups["blocks"]):
                active_fig.data[tidx].marker.color = colors[j]
            for tidx in groups["mges"]: active_fig.data[tidx].visible = "mges" in selected_annos
            for tidx in groups["intrec"]: active_fig.data[tidx].visible = "intrec" in selected_annos
            for tidx in groups["cds"]: active_fig.data[tidx].visible = "cds" in selected_annos
            for tidx in groups["trna"]: active_fig.data[tidx].visible = "trna" in selected_annos
            for tidx in groups["insertions"] + groups["deletions"] + groups["inversions"] + groups["translocations"]:
                active_fig.data[tidx].visible = "indels" in selected_annos
            for ann in (active_fig.layout.annotations or []):
                ann.visible = "indels" in selected_annos
            for tidx in groups["clustering"]:
                active_fig.data[tidx].visible = "show_clustering" in selected_options
            active_fig.layout.barmode = "overlay" if any_on else "stack"
            search_msg = f"{len(highlighted_traces)} traces matched" if search_query else ""
            return active_fig, search_msg, use_consensus

        patch = Patch()

        # 1) Toggle annotation trace visibility
        def _set_visible(idxs, on):
            for i in idxs:
                patch["data"][i]["visible"] = bool(on)

        _set_visible(groups["mges"], "mges" in selected_annos)
        _set_visible(groups["intrec"], "intrec" in selected_annos)
        _set_visible(groups["cds"], "cds" in selected_annos)
        _set_visible(groups["trna"], "trna" in selected_annos)
        _set_visible(groups["insertions"], "indels" in selected_annos)
        _set_visible(groups["deletions"], "indels" in selected_annos)
        _set_visible(groups["inversions"], "indels" in selected_annos)
        _set_visible(groups["translocations"], "indels" in selected_annos)
        _set_visible(groups["clustering"], "show_clustering" in selected_options)

        # Toggle translocation arrow annotations
        active_fig_ref = base_fig_consensus if use_consensus else base_fig_tree
        for ai in range(len(active_fig_ref.layout.annotations or [])):
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

        return patch, search_msg, use_consensus

    # ── Insertion pie chart callback ──────────────────────────────────────────
    if _pie_lookup:
        _panel_base = {
            "position": "fixed", "bottom": "20px", "right": "20px",
            "width": "660px", "background": "rgba(255,255,255,0.96)",
            "border": "1px solid #ccc", "borderRadius": "6px",
            "padding": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
            "zIndex": 1000,
        }
        _panel_hidden  = {**_panel_base, "display": "none"}
        _panel_visible = {**_panel_base, "display": "block"}

        @app.callback(
            Output("insertion-self-hits", "children"),
            Output("insertion-pie-cats", "figure"),
            Output("insertion-pie", "figure"),
            Output("insertion-pie-title", "children"),
            Output("insertion-charts-row", "style"),
            Output("insertion-cats-col", "style"),
            Output("insertion-pie", "style"),
            Output("insertion-no-hits", "style"),
            Output("insertion-pie-panel", "style"),
            Input("graph", "hoverData"),
        )
        def update_insertion_pie(hover_data):
            empty = go.Figure(layout=dict(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=220, width=300,
            ))
            _pie_shown    = {"height": "220px"}
            _pie_hidden   = {"display": "none"}
            _no_hits_shown  = {"display": "block", "fontSize": "12px", "color": "#888",
                               "padding": "80px 40px", "textAlign": "center"}
            _no_hits_hidden = {"display": "none"}

            def _hidden():
                return (None, empty, empty, "",
                        {"display": "none"}, {"display": "none"},
                        _pie_hidden, _no_hits_hidden, _panel_hidden)

            if not hover_data:
                return _hidden()

            point = hover_data["points"][0]
            curve = point.get("curveNumber")
            if curve is None or curve >= len(base_fig_consensus.data):
                return _hidden()

            tr = base_fig_consensus.data[curve]
            if getattr(tr, "name", "") != "Insertion":
                return _hidden()

            pt_idx = point.get("pointIndex", 0)
            cd = getattr(tr, "customdata", None)
            if cd is None or pt_idx >= len(cd):
                return _hidden()

            segment_num = cd[pt_idx][2]
            genome_name = tr.y[pt_idx] if hasattr(tr.y, "__len__") else tr.y
            segment_key = f"segment_{segment_num}"

            # Self-hits section
            self_info = _self_lookup.get((genome_name, segment_key))
            if self_info:
                self_block = html.Div([
                    html.Span(
                        f"Self hits — insertion #{segment_num} ({genome_name})",
                        style={"fontWeight": "bold", "fontSize": "11px", "color": "#444"},
                    ),
                    html.Br(),
                    html.Span(
                        f"Hits in own chromosome: {self_info['hits_in_genome']}",
                        style={"fontSize": "11px"},
                    ),
                    html.Span("   ", style={"whiteSpace": "pre"}),
                    html.Span(
                        f"Hits in own plasmid: {self_info['hits_in_plasmid']}",
                        style={"fontSize": "11px"},
                    ),
                    html.Hr(style={"margin": "6px 0"}),
                ])
            else:
                self_block = None

            entry = _pie_lookup.get((genome_name, segment_key))
            has_cats    = bool(entry and any(entry.get("cats", {}).values()))
            has_species = bool(entry and entry.get("species"))

            title   = f"ATB hits — insertion #{segment_num} ({genome_name})"
            cat_fig = _make_category_pie_figure(entry.get("cats", {})) if has_cats else empty
            sp_fig  = _make_pie_figure(entry["species"]) if has_species else empty

            show_row = has_cats or has_species
            return (
                self_block,
                cat_fig,
                sp_fig,
                title,
                {"display": "flex", "gap": "0px"} if show_row else {"display": "none"},
                {"display": "block"} if has_cats else {"display": "none"},
                _pie_shown if has_species else _pie_hidden,
                _no_hits_hidden if has_species else _no_hits_shown,
                _panel_visible,
            )

    return app


if __name__ == "__main__":
    import argparse

    # Detect CLI vs. interactive (VSCode) run
    _cli = len(sys.argv) > 1

    if _cli:
        parser = argparse.ArgumentParser(
            description="Launch the Dash pangraph viewer for a given junction."
        )
        parser.add_argument(
            "junction_name",
            help="Junction name to visualise, e.g. CIRMBUYJFK_f__CWCCKOQCWZ_r",
        )
        parser.add_argument(
            "--recompute",
            action="store_true",
            help="Force recomputation of clustering, consensus paths and indel summaries "
                 "even if cached results already exist.",
        )
        parser.add_argument(
            "--mges-gff",
            default=None,
            metavar="PATH",
            help="Path to the MGE GFF3 file. If omitted, MGE annotations are not shown.",
        )
        parser.add_argument(
            "--annotations-gff",
            default=None,
            metavar="PATH",
            help="Path to the gene annotations GFF file. "
                 "If omitted, CDS/integrase/tRNA annotations are not shown.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8050,
            help="Port to run the Dash server on (default: 8050).",
        )
        args = parser.parse_args()
        junction_name        = args.junction_name
        recompute            = args.recompute
        port                 = args.port

        _default_mges = REPO_ROOT / "results" / "junction_mges" / f"{junction_name}.gff3"
        _default_ann  = REPO_ROOT / "results" / "junction_annotations" / f"{junction_name}.gff"

        if args.mges_gff:
            mges_gff_path = Path(args.mges_gff)
        elif _default_mges.exists():
            mges_gff_path = _default_mges
        else:
            mges_gff_path = None

        if args.annotations_gff:
            annotations_gff_path = Path(args.annotations_gff)
        elif _default_ann.exists():
            annotations_gff_path = _default_ann
        else:
            annotations_gff_path = None
    else:
        # ── VSCode interactive run ── edit junction_name here ──────────────────
        #junction_name = "XXVMWZCEKI_r__YUOECYBHUS_r" # consensus definition very conservative here, investigate
        #junction_name = "ITEMFVYTUE_r__JVDYVQZUBR_r"
        #junction_name = "XXVMWZCEKI_r__YUOECYBHUS_r"
        #junction_name = "PLTCZQCVRD_f__RYYAQMEJGY_f"
        #junction_name = "NOAJDCSIVA_f__NZXBIFMPMA_r"
        #junction_name = "EJPOGALASQ_f__KUIFCLFQSI_r"
        junction_name = "RYYAQMEJGY_r__ZTHKZYHPIX_f"
        #junction_name = "IHKFSQQUKE_r__KPBYGJHRZJ_f" # --> look up again, its the prophage one that would get two clusters
        #junction_name = "AWYUJYFNGP_f__GCNKXNFARN_r"
        #junction_name = "JVNRLCFAVD_f__PLTCZQCVRD_r" # example for a junction that isn't split enough
        #junction_name = "XXVMWZCEKI_r__YUOECYBHUS_r" # --> wild junctions with 12 clusters
        #junction_name = "OBEJYXNUDN_r__ZTHKZYHPIX_r"
        #junction_name = "AGVTAJTYER_r__OZLYYMOKWU_f" # interesting has one insertion in its own cluster and then the same one again in a different cluster but inverted
        #junction_name = "CIRMBUYJFK_f__CWCCKOQCWZ_r"
        #junction_name = "ATPWUNKKID_f__KKPYPKGMXA_f"
        #junction_name = 'KGWWUZQEKD_r__UXLELLOQVR_r'
        recompute            = True
        mges_gff_path        = REPO_ROOT / "results" / "junction_mges" / f"{junction_name}.gff3"
        annotations_gff_path = REPO_ROOT / "results" / "junction_annotations" / f"{junction_name}.gff"
        port                 = 8050

    pangraph_path = REPO_ROOT / "results" / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    tree_path    = REPO_ROOT / "config" / "polished_tree.nwk"
    in_del_path  = REPO_ROOT / "results" / "atb_lookup_new"

    cache_path = REPO_ROOT / "results" / "consensus_analysis" / junction_name / "dash_cache.json"
    path_dict = None
    if not recompute and cache_path.exists():
        print(f"Loading consensus cache from {cache_path}")
        cluster_map_core, consensus_paths_plotting, assignment_df_plotting = load_consensus_cache(cache_path)
    else:
        print("Cache not found, running find_consensus_paths_refined ...")
        cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, *_ = find_consensus_paths_refined(
            junction_name,
            clustering_bl_thresh=0.001,
            tree_path=str(tree_path),
            rare_context_thresh=0.2, z_score_threshold=3.5, seq_dedup_length_threshold=0.05
        )
        # cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting, *_ = find_consensus_paths_core(
        #     junction_name,
        #     plot_consensus=False,
        #     plot_annotations=False,
        #     plot_pair_dist=False,
        #     plot_snp_dist=False,
        #     plot_ambiguities=False,
        #     clustering_bl_thresh=0.005,
        #     consensus_criterium="core_genome_tree",
        #     tree_path=str(tree_path),
        # )
        # cluster_map_core, consensus_paths_core, path_dict, consensus_paths_plotting, assignment_df_plotting = find_consensus_paths_edge(
        #     junction_name,
        #     clustering_bl_thresh=0.001,
        #     tree_path=str(tree_path),
        # )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_consensus_cache(cache_path, cluster_map_core, consensus_paths_plotting, assignment_df_plotting)
        print(f"Cache saved to {cache_path}")

    # compute indels on the fly only if the consensus1 folder doesn't exist yet (or --recompute)
    _consensus1_dir = in_del_path / "insertions" / junction_name / "consensus1"
    _all_ins_summary = in_del_path / "insertions" / junction_name / "all_insertions_summary.csv"
    if recompute or not _consensus1_dir.exists():
        print("Indel summaries not found, computing on the fly ...")
        if path_dict is None:
            print("path_dict not available from cache — re-running find_consensus_paths_refined ...")
            _, _, path_dict, *_ = find_consensus_paths_refined(
                junction_name,
                clustering_bl_thresh=0.001,
                tree_path=str(tree_path),
                rare_context_thresh=0.2, z_score_threshold=3.5, seq_dedup_length_threshold=0.05,
            )
            # _, _, path_dict, *_ = find_consensus_paths_core(
            #     junction_name,
            #     plot_consensus=False,
            #     plot_annotations=False,
            #     plot_pair_dist=False,
            #     plot_snp_dist=False,
            #     plot_ambiguities=False,
            #     clustering_bl_thresh=0.005,
            #     consensus_criterium="core_genome_tree",
            #     tree_path=str(tree_path),
            # )
            # _, _, path_dict, *_ = find_consensus_paths_edge(
            #     junction_name,
            #     clustering_bl_thresh=0.001,
            #     tree_path=str(tree_path),
            # )
        for i in range(1, len(consensus_paths_plotting) + 1):
            isolates_i = assignment_df_plotting[
                assignment_df_plotting["best_consensus"] == f"consensus_{i}"
            ].index.tolist()
            deduplicated_paths_i = {iso: path_dict[iso] for iso in isolates_i if iso in path_dict}

            df, path_dict_annotated, consensus_annotated = detect_event_blocks(
                deduplicated_paths_i, consensus_paths_plotting[i - 1]
            )
            _, path_dict_annotated = group_events_by_clade(
                df, path_dict_annotated, isolates_i,
                tree_path=str(tree_path), method="fitch", polytomy_threshold = 0.7
            )
            insertions, deletions, inversions = find_combined_events(
                path_dict_annotated, consensus_annotated,
                tree_path=str(tree_path), isolate_list=isolates_i
            )

            write_insertions_fasta(junction_name, pangraph, insertions, consensus=i,
                                   save_df=True, parent_dir=str(in_del_path / "insertions"))
            summarize_inversions_consensus(inversions, junction_name, pangraph, consensus_id=i,
                                           parent_dir=str(in_del_path / "inversions"), save_df=True)
            summarize_deletions_consensus(deletions, junction_name, pangraph, path_dict_annotated,
                                          assignment_df_plotting, consensus_id=i,
                                          parent_dir=str(in_del_path / "deletions"),
                                          rerun_alignment=False, save_df=True)

        load_all_insertions_summaries(str(in_del_path / "insertions"), junction_name, save_df=True)
        load_all_deletions_summaries(str(in_del_path / "deletions"), junction_name, save_df=True)
        load_all_inversions_summaries(str(in_del_path / "inversions"), junction_name, save_df=True)
        print("Indel summaries computed and saved.")
    elif not _all_ins_summary.exists():
        print(
            f"No insertion summary file found for {junction_name} "
            f"(checked: {_all_ins_summary}). "
            "This is either because no insertions were detected for this junction, "
            "or the precomputed summary file is missing."
        )

    app = make_junction_dash_app(
        pan=pangraph,
        consensus_paths_plotting=consensus_paths_plotting,
        assignment_df_plotting=assignment_df_plotting,
        cluster_map_core=cluster_map_core,
        mges_gff_path=str(mges_gff_path) if mges_gff_path is not None else None,
        annotations_gff_path=str(annotations_gff_path) if annotations_gff_path is not None else None,
        order="tree",
        title=f"Junction Block Structure ({junction_name})",
        initial_selection=(*(["mges"] if mges_gff_path is not None else []), "indels"),
        indels_base_path=str(in_del_path),
        show_indels=True,
        path_dict=path_dict,
        show_trna=True,
        junction_name=junction_name,
    )

    app.run(debug=False, port=port)
