import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import altair as alt
    import numpy as np
    import pypangraph as pp
    from Bio import Phylo
    from collections import defaultdict

    return Phylo, alt, mo, np, pd, plt, pp, sns


@app.cell
def _(pd):
    _fname = f"config/junction_stats.csv"
    jdf = pd.read_csv(_fname)
    jdf
    return (jdf,)


@app.cell
def _(alt, jdf, mo, np):
    # _xvar = "nonempty_acc_len"
    _xvar = "pangenome_len"
    _yvar = "n_categories"
    _cvar = "nonempty_freq"
    _yvar_jiggle = _yvar + "_jiggle"
    jdf[_yvar_jiggle] = jdf[_yvar] * np.random.uniform(0.9, 1.1, size=len(jdf))
    mask = jdf["n_categories"] > 1
    chart = mo.ui.altair_chart(
        alt.Chart(jdf[mask])
        .mark_point(opacity=0.5)
        .encode(
            x=alt.X(_xvar).scale(
                type="symlog",
                constant=100,
                bins=[
                    0,
                    100,
                    1000,
                    2000,
                    5000,
                    10000,
                    20000,
                    50000,
                    100000,
                    200000,
                    500000,
                    1000000,
                ],
            ),
            y=alt.Y(_yvar_jiggle).scale(type="log"),
            color=alt.Color(_cvar, scale=alt.Scale(scheme="blueorange")),
        )
        .properties(width=700, height=500)
    )
    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(chart, pp):
    selected_edge = chart.value["edge"].iloc[0]
    _fname = f"results/junction_pangraphs/{selected_edge}.json"
    pan = pp.Pangraph.from_json(_fname)
    return pan, selected_edge


@app.cell
def _(jdf, mo, pan):
    # print information:
    N_genomes = len(pan.paths)
    N_genomes_tot = jdf["n_iso"].max()
    entry = jdf.query("edge == @selected_edge").iloc[0]
    N_path_categories = entry["n_categories"]
    acc_gen_len = entry["pangenome_len"]
    mo.md(f"""
    ## summary stats

    - N. genomes: {N_genomes} / {N_genomes_tot}
    - N. path categories: {N_path_categories}
    - total accessory pangenome: {acc_gen_len} bp
    """)
    return


@app.cell
def _(Phylo):
    _fname = f"config/polished_tree.nwk"
    _tree = Phylo.read(_fname, "newick")
    _tree.root_at_midpoint()
    _tree.ladderize()
    # extract the order of isolates from the tree
    leaf_order = [leaf.name for leaf in _tree.get_terminals()]
    return (leaf_order,)


@app.cell
def _(leaf_order, mo, pan, plt, selected_edge, sns):
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
    with mo.status.progress_bar(title="plotting...", total=len(pan.paths)) as pbar:
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
            pbar.update()
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_xlabel("genomic position (bp)")
    ax.set_title(f"Junction graph for edge {selected_edge}")
    ax.grid(axis="x", alpha=0.4)
    ax.set_ylim(-1, len(y_labels))
    sns.despine()
    plt.tight_layout()
    ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
