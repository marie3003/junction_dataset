import copy
import random

from dash import Dash, dcc, html, Input, Output, Patch, State, ctx

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
from junction_analysis.consensus import find_consensus_paths_core, save_consensus_cache, load_consensus_cache

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
      - intrec: bar traces named 'Integrase / Recombinase / Transposase'
      - cds: bar traces named 'Coding Sequence (CDS)'
      - insertions: bar traces named 'Insertion'
      - deletions: scatter traces named 'Deletion'
    """
    groups = {"blocks": [], "mges": [], "intrec": [], "cds": [], "trna": [], "insertions": [], "deletions": [], "inversions": [], "translocations": []}

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
    show_trna: bool = False,
    indels_base_path: str = None,
    junction_name: str = None,
):
    """
    Creates a Dash app showing the pangraph with annotation toggles.
    """
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
        )
        fig = add_annotations_for_dash(
            fig=fig,
            y_labels=yl,
            show_mges_annotations=True,
            show_int_rec_annotations=True,
            show_trna_annotations=show_trna,
            show_cds_annotations=True,
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

    # Pre-build both figures
    base_fig_consensus = _build_fig(show_consensus=True)
    base_fig_tree = _build_fig(show_consensus=False)
    base_fig = base_fig_consensus  # default

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
                        ],
                        value=["show_consensus"],
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

        return patch, search_msg, use_consensus

    return app


if __name__ == "__main__":
    
    #junction_name = "XXVMWZCEKI_r__YUOECYBHUS_r" # consensus definition very conservative here, investigate
    #junction_name = "ITEMFVYTUE_r__JVDYVQZUBR_r"
    #junction_name = "MUWGUWCDTU_r__UTYAQKFQDH_r"
    #junction_name = "PLTCZQCVRD_f__RYYAQMEJGY_f"
    #junction_name = "NOAJDCSIVA_f__NZXBIFMPMA_r"
    #junction_name = "EJPOGALASQ_f__KUIFCLFQSI_r"
    #junction_name = "GPKQYOCEJI_r__NKVSUZGURN_f"
    #junction_name = "IHKFSQQUKE_r__KPBYGJHRZJ_f" # --> look up again, its the prophage one that would get two clusters
    junction_name = "AFFODHUCNW_r__VRDEBAMMSO_r"
    #junction_name = "RYYAQMEJGY_r__ZTHKZYHPIX_f"
    #junction_name = "XXVMWZCEKI_r__YUOECYBHUS_r" # --> wild junctions with 12 clusters
    #junction_name = "JPYVXRYZLU_f__UUBXUCAQCF_f"
    #junction_name = "NPQDSPAYII_f__VJEJDHVKTM_f"
    #junction_name = "CIRMBUYJFK_f__CWCCKOQCWZ_r"
    #junction_name = "ATPWUNKKID_f__KKPYPKGMXA_f"
    #junction_name = 'KGWWUZQEKD_r__UXLELLOQVR_r'

    pangraph_path = REPO_ROOT / "results" / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    mges_gff_path = REPO_ROOT / "results" / "junction_mges" / f"{junction_name}.gff3"
    annotations_gff_path = REPO_ROOT / "results" / "junction_annotations" / f"{junction_name}.gff"
    tree_path = REPO_ROOT / "config" / "polished_tree.nwk"
    in_del_path = REPO_ROOT / "results" / "atb_lookup" 

    cache_path = REPO_ROOT / "results" / "consensus_analysis" / junction_name / "dash_cache.json"
    if cache_path.exists():
        print(f"Loading consensus cache from {cache_path}")
        cluster_map_core, consensus_paths_plotting, assignment_df_plotting = load_consensus_cache(cache_path)
    else:
        print("Cache not found, running find_consensus_paths_core ...")
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
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_consensus_cache(cache_path, cluster_map_core, consensus_paths_plotting, assignment_df_plotting)
        print(f"Cache saved to {cache_path}")

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
        show_trna=True,
        junction_name=junction_name,
    )

    app.run(debug=False)  # CHANGED: turn off debug for normal use
