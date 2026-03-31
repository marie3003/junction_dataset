rule genomad_download_db:
    output:
        db=directory("data/genomad_db"),
    conda:
        "../config/conda_envs/genomad.yaml"
    shell:
        """
        genomad download-database data/
        """


rule genomad_run:
    input:
        db=rules.genomad_download_db.output.db,
        fa=rules.gbk_to_fa.output.fa,
    output:
        d=directory("data/genomad/{acc}"),
        s="data/genomad/{acc}/{acc}_summary/{acc}_virus_summary.tsv",
    log:
        "logs/genomad/{acc}.log",
    conda:
        "../config/conda_envs/genomad.yaml"
    shell:
        """
        genomad end-to-end {input.fa} {output.d} {input.db} \
            --cleanup \
            --threads 4 \
            &> {log}
        """


rule genomad_preformat:
    input:
        lambda w: expand(rules.genomad_run.output.s, acc=acc_nums),
    output:
        "results/mges/genomad.csv",
    conda:
        "../config/conda_envs/bioinfo.yaml"
    shell:
        """
        python3 scripts/genomad_df_preformat.py \
            --input_tsvs {input} \
            --output_df {output}
        """


rule defensefinder_models_download:
    output:
        directory("data/defensefinder_models"),
    conda:
        "../config/conda_envs/defensefinder.yaml"
    shell:
        """
        TMPDIR=$(mktemp -d -t defensefinder_model_download_XXXXXXXXX)
        echo "created temporary directory $TMPDIR"
        defense-finder update --models-dir {output}
        rm -r $TMPDIR
        """


rule defensefinder_find:
    input:
        fa=rules.gbk_to_fa.output.fa,
        mod=rules.defensefinder_models_download.output,
    output:
        a=directory("data/defense_finder/{acc}"),
        g="data/defense_finder/{acc}/{acc}_defense_finder_genes.tsv",
        s="data/defense_finder/{acc}/{acc}_defense_finder_systems.tsv",
        p="data/defense_finder/{acc}/{acc}.prt",
    conda:
        "../config/conda_envs/defensefinder.yaml"
    shell:
        """
        defense-finder run \
            -o {output.a} \
            --models-dir {input.mod} \
            {input.fa}
        """


rule defensefinder_gene_location:
    input:
        g=rules.defensefinder_find.output.g,
        p=rules.defensefinder_find.output.p,
    output:
        temp("data/defense_finder/{acc}/{acc}_genes_loc.tsv"),
    conda:
        "../config/conda_envs/bioinfo.yaml"
    shell:
        """
        python3 scripts/annotations/defensefinder_gene_location.py \
            --input_gene_df {input.g} \
            --proteins {input.p} \
            --output_gene_df {output}
        """


rule defensefinder_preformat:
    input:
        s=expand(rules.defensefinder_find.output.s, acc=acc_nums),
        g=expand(rules.defensefinder_gene_location.output, acc=acc_nums),
    output:
        "results/mges/defensefinder.csv",
    conda:
        "../config/conda_envs/bioinfo.yaml"
    shell:
        """
        python3 scripts/annotations/defensefinder_df_preformat.py \
            --input_genes {input.g} \
            --input_systems {input.s} \
            --output_df {output}
        """


rule mge_assign_positions:
    input:
        el="results/mges/{tool}.csv",
        j_pos=config["junction_positions_file"],
        iso_len=rules.genome_lengths.output,
    output:
        "results/mges_to_junctions/{tool}.csv",
    conda:
        "../config/conda_envs/bioinfo.yaml"
    params:
        zero_based="--zero_based",
        random="",
    shell:
        """
        python3 scripts/assign_junctions.py \
            --iso_len {input.iso_len} \
            --junction_pos_json {input.j_pos} \
            --element_pos_df {input.el} \
            --output_pos {output} \
            {params.zero_based} \
            {params.random}
        """


rule mges_all:
    input:
        expand(rules.mge_assign_positions.output, tool=["genomad", "defensefinder"]),
