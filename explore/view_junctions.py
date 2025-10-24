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
    from Bio import Phylo, SeqIO
    from collections import defaultdict

    import subprocess
    import os
    return Phylo, SeqIO, alt, mo, np, pd, plt, pp, sns, subprocess


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
    #selected_edge = "CIRMBUYJFK_f__CWCCKOQCWZ_r"
    #selected_edge = "ATPWUNKKID_f__KKPYPKGMXA_f"
    #selected_edge = jdf.iloc[0].edge
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
def _(jdf):
    jdf_filtered = jdf[(jdf.singleton == True) & (jdf.nonempty_freq < 0.01) & (jdf.n_iso > 5)]
    jdf_filtered
    # 280 out of 548 junctions are singletons (= either one gain or one loss), # 245 out of these 280 junctions are gains, of which one only has 2 isolates
    # continue with 244 junctions
    return (jdf_filtered,)


@app.cell
def _(SeqIO, pd, pp, subprocess):
    def analyze_insertion_sequences(jdf_filtered):

        results = []

        for _, row in jdf_filtered.iterrows():
            junction = row.edge

            # Load pangraph
            j_pan = pp.Pangraph.from_json(f"results/junction_pangraphs/{junction}.json")

            # Find gain block
            j_stats_df = j_pan.to_blockstats_df()
            # all single junctions have exactly one insertion (one block with one sequence)
            gain_block_id = j_stats_df[j_stats_df["count"] == 1].index[0] 
            j_gain_sequence = j_pan.blocks[gain_block_id].consensus()

            # Find isolate name
            j_gain_node_id = j_pan.blocks[gain_block_id].to_biopython_alignment()[0].id
            j_gain_path_id = j_pan.nodes[j_gain_node_id].path_id
            genome_name = j_pan.paths.idx_to_name[j_gain_path_id]

            # Convert genome from gbk to fasta
            input_gbk = f"data/gbk/{genome_name}.gbk"
            genome_fasta = f"results/genome_fastas/{genome_name}.fasta"
            with open(genome_fasta, "w") as out_f:
                for record in SeqIO.parse(input_gbk, "genbank"):
                    SeqIO.write(record, out_f, "fasta")

            # Write insertion sequence as fasta
            insertion_seq_path = f"results/insertion_sequences/{genome_name}_{j_gain_node_id}.fasta"
            with open(insertion_seq_path, "w") as f:
                f.write(f">{j_gain_node_id}\n{j_gain_sequence}\n")

            # Get junction position in this genome
            junction_fasta_path = f"results/junction_sequences/{junction}.fa"
            with open(junction_fasta_path) as junction_fasta:
                for line in junction_fasta:
                    if line.startswith(">") and line.startswith(">" + genome_name):
                        parts = line.strip().split()
                        junction_start, junction_end = parts[1].split("-")
                        junction_strand = parts[2].strip("()")

            # Get ALL insertion types + defense systems + prophages from GFF3
            gff3_path = f"results/junction_mges/{junction}.gff3"

            insertion_types_list = []
            defense_systems_list = []
            prophages_list = []

            with open(gff3_path) as gff:
                for line in gff:
                    if line.startswith("#"):
                        continue
                    # insertion sequence (ISxxxx) — e.g., ID=NZ_...|IS1380|...
                    if "\tIS\t" in line:
                        attrs = line.strip().split("ID=")[1].split(";")[0]
                        parts = attrs.split("|")
                        if len(parts) >= 2:
                            is_type = parts[1]
                            if is_type not in insertion_types_list:
                                insertion_types_list.append(is_type)

                    # defense system — e.g., ID=NZ_..._MazEF_8
                    if "\tdefense_system\t" in line:
                        attrs = line.strip().split("ID=")[1].split(";")[0]
                        defense_system = attrs.split("_", 1)[1] if "_" in attrs else attrs
                        if defense_system not in defense_systems_list:
                            defense_systems_list.append(defense_system)

                    # prophage — e.g., ID=NZ_...|provirus_3783810_3795098
                    if "\tprophage\t" in line:
                        attrs = line.strip().split("ID=")[1].split(";")[0]
                        prophage = attrs.split("|", 1)[1] if "|" in attrs else attrs
                        if prophage not in prophages_list:
                            prophages_list.append(prophage)

            # Store as semicolon-joined strings (or keep lists if you prefer)
            insertion_type = ";".join(insertion_types_list) if insertion_types_list else None
            defense_system = ";".join(defense_systems_list) if defense_systems_list else None
            prophage = ";".join(prophages_list) if prophages_list else None

            # Run minimap2
            paf_path = f"results/insertion_sequences_hits/{genome_name}_{j_gain_node_id}"
            subprocess.run([
                "minimap2", "-x", "asm5", "-N", "50", "-p", "0.9", "-k", "19", "--eqx",
                genome_fasta,
                insertion_seq_path
            ], stdout=open(paf_path, "w"))

            # Count hits with <1% divergence
            count = 0
            with open(paf_path) as paf_file:
                for hit in paf_file:
                    if "dv:f:" in hit:
                        dv = float(hit.split("dv:f:")[1].split("\t")[0])
                        fields = hit.split("\t")
                        query_len = int(fields[1])
                        query_start = int(fields[2])
                        query_end = int(fields[3])
                        coverage = (query_end - query_start) / query_len if query_len > 0 else 0
                        if dv < 0.01 and coverage >= 0.9:
                            count += 1

            # Append results
            results.append({
                "junction": junction,
                "isolate": genome_name,
                "junction_start": junction_start,
                "junction_end": junction_end,
                "junction_strand": junction_strand,
                "node_id": j_gain_node_id,
                "insertion_type": insertion_type,
                "defense_system": defense_system,
                "prophage": prophage,
                #"insertion_sequence": j_gain_sequence,
                "hits_in_genome": count
            })

        insertions_df = pd.DataFrame(results)
        return insertions_df
    return


@app.cell
def _():
    # took 2m 40s
    #insertions_df = analyze_insertion_sequences(jdf_filtered)
    #insertions_df.to_csv("results/insertion_sequences_hits/insertions_df.csv", index = False)
    #insertions_df
    return


@app.cell
def _(pd):
    insertions_df = pd.read_csv("results/insertion_sequences_hits/insertions_df.csv")
    insertions_df
    return (insertions_df,)


@app.cell
def _(insertions_df, plt):
    # Handle missing insertion_type entries (NaN or empty)
    insertions_df['insertion_type'] = insertions_df['insertion_type'].fillna('Unknown')
    insertions_df.loc[insertions_df['insertion_type'].str.strip() == '', 'insertion_type'] = 'Unknown'

    # Extract the first insertion type (before any ';')
    insertions_df['first_insertion_type'] = insertions_df['insertion_type'].str.split(';').str[0]

    # Collect all groups including "Unknown"
    labels = insertions_df['first_insertion_type'].unique()
    data_to_plot = [
        insertions_df.loc[insertions_df['first_insertion_type'] == label, 'hits_in_genome']
        for label in labels
    ]

    # Plot stacked histogram
    plt.figure(figsize=(8, 5))
    plt.hist(
        data_to_plot,
        bins=20,
        stacked=True,      # ✅ stack bars
        label=labels,
        alpha=0.9
    )

    plt.xlabel("Hits in genome")
    plt.ylabel("Count")
    plt.title("Stacked Histogram of Hits of Insertion Sequence in Genome by First Insertion Type")
    plt.legend(title="Insertion Type")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(insertions_df):
    insertions_df.hits_in_genome.hist(bins = 20)
    return


@app.cell
def _(jdf_filtered, pp):
    # extract insertion sequence from these junctions
    # start with example
    junction_1 = jdf_filtered.edge.iloc[1]
    j1_pan_name = f"results/junction_pangraphs/{junction_1}.json"
    j1_pan = pp.Pangraph.from_json(j1_pan_name)
    j1_pan
    return (j1_pan,)


@app.cell
def _(j1_pan):
    j1_stats_df = j1_pan.to_blockstats_df()
    print(j1_stats_df)
    return (j1_stats_df,)


@app.cell
def _(j1_pan, j1_stats_df):
    gain_block_id = j1_stats_df[j1_stats_df["count"] == 1].index[0]
    j1_gain_sequence = j1_pan.blocks[gain_block_id].consensus()
    print(len(j1_gain_sequence))
    j1_gain_sequence
    return (gain_block_id,)


@app.cell
def _(gain_block_id, j1_pan):
    j1_pan.blocks[gain_block_id].alignment.generate_alignment()
    return


@app.cell
def _(gain_block_id, j1_pan):
    # in which genome is this block / sequence

    # get node
    j1_gain_node_id = j1_pan.blocks[gain_block_id].to_biopython_alignment()[0].id
    # get path
    j1_gain_path_id = j1_pan.nodes[j1_gain_node_id].path_id
    genome_name = j1_pan.paths.idx_to_name[j1_gain_path_id]
    genome_name
    #j1_pan.paths[j1_gain_path_id]
    return


@app.cell
def _():
    # get junction annotation from junction_mges
    return


@app.cell
def _(Phylo):
    _fname = f"config/polished_tree.nwk"
    test_tree = Phylo.read(_fname, "newick")
    print(test_tree)
    Phylo.draw_ascii(test_tree)
    Phylo.draw(test_tree)
    return (test_tree,)


@app.cell
def _(test_tree):
    test_tree.get_nonterminals()
    test_tree.get_terminals()
    test_tree.root
    return


if __name__ == "__main__":
    app.run()
