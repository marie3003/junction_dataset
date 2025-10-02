import pandas as pd


configfile: "config/config.yaml"


# load the list of accession numbers
acc_nums = pd.read_csv(config["acc_nums_file"], header=None)[0].tolist()
print(f"N. isolates: {len(acc_nums)}")
# load the table of junction positions


rule download_gbk:
    output:
        "data/gbk/{acc}.gbk",
    conda:
        "config/conda_envs/entrez_direct.yaml"
    shell:
        """
        efetch -db nucleotide -id {wildcards.acc} -format gbwithparts > {output}
        """


rule all:
    input:
        expand(rules.download_gbk.output, acc=acc_nums),
