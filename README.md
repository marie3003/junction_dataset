# Evolutionary dynamics shaping accessory genome hotspots in closely related *E. coli* ST131 genomes

Bacterial genomes show substantial structural diversity driven by horizontal gene transfer, homologous recombination, and mobile genetic elements (MGEs). This diversity is often concentrated in accessory genome “hotspots,” which play a key role in adaptation and the acquisition of clinically relevant traits such as antibiotic resistance.

In this thesis, we investigate the evolutionary mechanisms shaping these hotspots in 222 closely related *Escherichia coli* ST131 genomes using a pangenome graph framework.

Building on local pangenome graphs generated with PanGraph, we develop a computational pipeline to:

- cluster genomic junctions based on homologous recombination signals  
- reconstruct ancestral genome structures  
- detect structural events, including insertions, deletions, inversions, and translocations  

These events are further characterized using functional annotation and sequence similarity searches to infer their likely origins.

## Key findings

- Homologous recombination frequently contributes to sequence divergence but only rarely results in structural changes  
- MGEs are the primary drivers of accessory genome variability  
- Insertions occur much more frequently than deletions in secondary structural events and are often associated with insertion sequences and prophages  
- Hotspot diversity is not strongly correlated with recombination frequency, suggesting that repeated MGE activity—not recombination alone—drives structural complexity  

Finally, we integrate structural, functional, and evolutionary information into an interactive visualization framework, enabling systematic exploration of accessory genome hotspots.

---

## Pipeline: Preparing genomic junction data

This repository includes a pipeline to reproduce the genomic junction dataset described in the paper by Molari et al.:
https://academic.oup.com/mbe/article/42/1/msae272/7942412

The pipeline prepares and formats genomic junctions from the *E. coli* ST131 collection for downstream analysis.

### Setup

To run the pipeline you will need to have [Snakemake](https://snakemake.readthedocs.io/en/stable/) (tested on v9.11) and [Conda](https://docs.conda.io/en/latest/) installed.
Moreover you will need to have the [PanGraph](https://github.com/neherlab/pangraph) (v1.2.1) binary available in your PATH.

### Usage

Run the pipeline simply with:

```sh
snakemake --use-conda --cores <num_cores> all
```

### Viewing junctions

You can explore junctions visually with [marimo](https://marimo.io/) by running:

```sh
marimo run explore/view_junctions.py
```

---

## Detailed analysis of accessory genome junctions

All analysis code is located in the `junction_analysis` directory.

The `explore` folder contains notebooks used throughout the thesis. These notebooks document the full workflow, validation steps, and downstream analyses.

### Main workflows

- `explore/workflow_all_junctions.ipynb`  
  End-to-end workflow: junction clustering, consensus path inference, and detection of structural events  

- `explore/analyze_clustering_results.ipynb`  
  Evaluation of clustering results and validation of the chosen branch length cutoff  

- `explore/validate_consensus_creation_event_detection.ipynb`  
  Validation of consensus reconstruction and event detection heuristics  

- `explore/analyze_secondary_events.ipynb`  
  Analysis of structural events, including MGE annotations  

- `explore/analyze_insertion_origins.ipynb`  
  Investigation of the origins of inserted genomic segments  

### Ongoing analyses

- `explore/analyze_homologous_recombinations.ipynb`  
  Analysis of homologous recombination across the core (and potentially accessory) genome  

---

## Interactive visualization

An interactive dashboard for exploring junctions can be launched with:

```bash
python explore/dash_pangraph_viewer.py CIRMBUYJFK_f__CWCCKOQCWZ_r  \
--mges-gff results/junction_mges/CIRMBUYJFK_f__CWCCKOQCWZ_r.gff3 \
--annotations-gff results/junction_annotations/CIRMBUYJFK_f__CWCCKOQCWZ_r.gff \
--port 8051
```
Providing GFF files enables visualization of functional annotations, including:
- mobile genetic elements (MGEs)
- integrases, recombinases, and transposases
- coding sequences (CDS)
- tRNA and tmRNA genes
If no GFF paths are specified, these annotations will not be displayed.

You can optionally recompute clustering, consensus paths, and secondary structural events directly within the dashboard:
```bash
python explore/dash_pangraph_viewer.py CIRMBUYJFK_f__CWCCKOQCWZ_r --recompute \
--port 8052
```