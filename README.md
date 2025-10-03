# Junction dataset

This repository contains a simple pipeline to prepare and format the dataset of genomic junctions from the _E. coli_ ST131 collection described in [our paper](https://academic.oup.com/mbe/article/42/1/msae272/7942412).

## Setup

To run the pipeline you will need to have [Snakemake](https://snakemake.readthedocs.io/en/stable/) (tested on v9.11) and [Conda](https://docs.conda.io/en/latest/) installed.
Moreover you will need to have the [PanGraph](https://github.com/neherlab/pangraph) (v1.2.1) binary available in your PATH.

## Usage

Run the pipeline simply with:

```sh
snakemake --use-conda --cores <num_cores> all
```

## Viewing junctions

You can explore junctions visually with [marimo](https://marimo.io/) by running:

```sh
marimo run explore/view_junctions.py
```