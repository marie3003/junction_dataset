import pandas as pd
import json


configfile: "config/config.yaml"


# load the list of accession numbers
acc_nums = pd.read_csv(config["acc_nums_file"], header=None)[0].tolist()
print(f"N. isolates: {len(acc_nums)}")
# load the table of junction positions
with open(config["junction_positions_file"]) as f:
    junc_pos = json.load(f)
# exclude single-isolate junctions
junc_pos = {j: p for j, p in junc_pos.items() if len(p) > 1}
# list of junction IDs
junc_ids = list(junc_pos.keys())
print(f"N. junctions: {len(junc_ids)}")
# list of plasmids
with open(config["plasmids_file"]) as f:
    plasmids = json.load(f)


rule download_gbk:
    output:
        "data/gbk/{acc}.gbk",
    conda:
        "config/conda_envs/entrez_direct.yaml"
    shell:
        """
        efetch -db nucleotide -id {wildcards.acc} -format gbwithparts > {output}
        """


rule extract_junction_sequences:
    input:
        gbk=expand(rules.download_gbk.output, acc=acc_nums),
        j_pos=config["junction_positions_file"],
    output:
        fa="results/junction_sequences/{junc}.fa",
        gff="results/junction_annotations/{junc}.gff",
    conda:
        "config/conda_envs/bioinfo.yaml"
    log:
        "logs/extract_junctions/{junc}.log",
    shell:
        """
        python scripts/extract_junctions.py \
            --gbk-fld "data/gbk" \
            --junc-id {wildcards.junc} \
            --junc-pos-file {input.j_pos} \
            --out-fa {output.fa} \
            --out-ann {output.gff} \
            &> {log}
        """


rule build_junction_pangraph:
    input:
        fa=rules.extract_junction_sequences.output.fa,
    output:
        "results/junction_pangraphs/{junc}.json",
    params:
        opt="-s 20 -a 100 -b 5 -l 100",
    shell:
        """
        pangraph build {input.fa} {params.opt} -o {output}
        """


rule genome_lengths:
    input:
        expand(rules.download_gbk.output, acc=acc_nums),
    output:
        "results/genome_lengths.csv",
    conda:
        "config/conda_envs/bioinfo.yaml"
    shell:
        """
        python scripts/genome_lengths.py \
            --output {output} \
            --gbk_files {input}
        """


rule plasmids:
    input:
        rules.download_gbk.output,
    output:
        "results/plasmids/{acc_iso}/{acc}.gbk",
    shell:
        """
        cp {input} {output}
        """


def all_plasmid_outputs(wildcards):
    outs = []
    for iso, plasmid_list in plasmids.items():
        outs.extend(expand(rules.plasmids.output, acc_iso=iso, acc=plasmid_list))
    return outs


rule all:
    input:
        expand(rules.build_junction_pangraph.output, junc=junc_ids),
        rules.genome_lengths.output,
        all_plasmid_outputs,


localrules:
    download_gbk,
