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

from junction_analysis.plotting import plot_junction_pangraph_interactive, _rgb_str, _shades_from_base_rgb, _rgba
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
    """
    groups = {"blocks": [], "mges": [], "intrec": [], "cds": []}

    for i, tr in enumerate(fig.data):
        ttype = getattr(tr, "type", None)

        name = getattr(tr, "name", "") or ""
        hover = getattr(tr, "hovertemplate", "") or ""
        cd = getattr(tr, "customdata", None)

        # block traces: your _add_bar sets hovertemplate with "<br>block:" and customdata includes block_id at index 3
        if ttype == "bar" and ("<br>block:" in hover) and (cd is not None) and len(cd) > 0 and len(cd[0]) >= 4:
            groups["blocks"].append(i)
            continue

        # annotation traces: your function sets these names explicitly
        if name == "Integrase / Recombinase":
            groups["intrec"].append(i)
        elif name == "Coding Sequence (CDS)":
            groups["cds"].append(i)
        elif name in ("Prophage", "Defense system") or name.startswith("IS:"):
            groups["mges"].append(i)

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
    junction_name: str,
    mges_gff_path: str,
    annotations_gff_path: str,
    order: str = "tree",
    title: str = "Junction viewer",
    initial_selection=("mges",),  # e.g. ("mges","intrec","cds") or ()
):
    """
    Creates a Dash app using your original plot_junction_pangraph_interactive().

    NOTE: We build the base fig with all annotations present so the UI can toggle them.
    """
    # Build a base figure with ALL annotation traces created
    base_fig = plot_junction_pangraph_interactive(
        pan,
        show_consensus=True,
        consensus_paths=consensus_paths_plotting,
        assignments=assignment_df_plotting,
        order=order,
        cluster_map=cluster_map_core,
        add_cluster_annotation=True,
        title=title,
        show_mges_annotations=True,
        show_int_rec_annotations=True,
        show_cds_annotations=True,
        mges_gff_path=mges_gff_path,
        annotations_gff_path=annotations_gff_path,
        annotation_alpha=0.70,
        cds_annotation_alpha=0.30,
    )
    base_fig.update_layout(
        uirevision="junction-viewer",
        autosize=False,
        height=5000,
        width=1800,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    groups, block_colors_colored, block_colors_grey = _compute_block_colors_for_figure(base_fig, pan)

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
                        ],
                        value=list(initial_selection),
                        inline=True,
                        persistence=True,              # CHANGED: added persistence
                        persistence_type="memory",     # CHANGED: added persistence_type
                    ),
                    html.Div("(any ON → grey blocks)", style={"color": "#555"}),
                ],
            ),
            dcc.Graph(id="graph", figure=base_fig),
        ],
    )

    @app.callback(
        Output("graph", "figure"),
        Input("anno-toggle", "value"),
    )
    def update_figure(selected):
        # CHANGED: use Patch instead of deepcopy(base_fig)
        selected = set(selected or [])
        any_on = len(selected) > 0

        patch = Patch()

        # 1) Toggle annotation trace visibility
        def _set_visible(idxs, on):
            for i in idxs:
                patch["data"][i]["visible"] = bool(on)

        _set_visible(groups["mges"], "mges" in selected)
        _set_visible(groups["intrec"], "intrec" in selected)
        _set_visible(groups["cds"], "cds" in selected)

        # 2) Enforce block colors (grey if any annotation on)
        colors = block_colors_grey if any_on else block_colors_colored
        for j, tidx in enumerate(groups["blocks"]):
            patch["data"][tidx]["marker"]["color"] = colors[j]

        # 3) barmode: overlay when annotations are on; stack otherwise
        patch["layout"]["barmode"] = "overlay" if any_on else "stack"

        return patch

    return app


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # You should already have these variables in scope from your analysis code:
    # pangraph, consensus_paths_plotting, assignment_df_plotting, cluster_map_core, junction_name
    # and paths:
    junction_name = "RYYAQMEJGY_r__ZTHKZYHPIX_f"

    pangraph_path = REPO_ROOT / "results" / "junction_pangraphs" / f"{junction_name}.json"
    pangraph = pp.Pangraph.from_json(str(pangraph_path))

    mges_gff_path = REPO_ROOT / "results" / "junction_mges" / f"{junction_name}.gff3"
    annotations_gff_path = REPO_ROOT / "results" / "junction_annotations" / f"{junction_name}.gff"
    tree_path = REPO_ROOT / "config" / "polished_tree.nwk"
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
        junction_name=junction_name,
        mges_gff_path=str(mges_gff_path),
        annotations_gff_path=str(annotations_gff_path),
        order="tree",
        title="Junction Block Structure (Dash)",
        initial_selection=("mges",),  # change to () to start with no annotations
    )

    app.run(debug=False)  # CHANGED: turn off debug for normal use
