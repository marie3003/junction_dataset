from pathlib import Path
import subprocess

import pandas as pd

import pypangraph as pp
from Bio import SeqIO

from junction_analyis.helpers import write_isolate_fasta


# create block fasta files
def create_block_msas(example_junction): # 2 min 14

    example_pangraph = pp.Pangraph.from_json(f"../results/junction_pangraphs/{example_junction}.json")

    parent_dir = Path(f"../results/block_fastas/{example_junction}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    aligned_dir = Path(f"../results/block_alignments/{example_junction}")
    aligned_dir.mkdir(parents=True, exist_ok=True)

    for block in example_pangraph.blocks:
        output_path = parent_dir / f"block_{block.id}_sequences.fa"
        write_isolate_fasta(example_pangraph, block, output_path)

    for fasta_file in parent_dir.glob("block_*_sequences.fa"):
        aln_file = aligned_dir / fasta_file.name.replace("_sequences.fa", "_aln.fa")
        
        # Run MAFFT quietly (you can remove '--quiet' if you want verbose output)
        subprocess.run(
            ["mafft", "--quiet", str(fasta_file)],
            stdout=open(aln_file, "w"),
            check=True
        )

def read_fasta_alignment(path):
    for rec in SeqIO.parse(path, "fasta"):
        yield rec.id, str(rec.seq)

def core_bounds(records):
    """Return (start, end) of intersection of non-gap spans across sequences.
    - start: position of first non-gap character
    - end : position of last non-gap character
    """
    seqs = [s for _, s in records]
    L = len(seqs[0])

    firsts = [s.find(s.replace('-', '')[0]) if '-' in s else 0 for s in seqs if set(s) != {'-'}] #s.find finds first occurrence of substring, s.replace delets all '-', [0] gives first character
    lasts  = [s.rfind(s.replace('-', '')[-1]) if '-' in s else L-1 for s in seqs if set(s) != {'-'}]

    start, end = max(firsts), min(lasts)
    return (start,end)


def analyze_alignment(path):
    recs = list(read_fasta_alignment(path))

    start, end = core_bounds(recs)

    seqs = [s for _, s in recs]
    length = len(seqs[0])

    left_overhang = start
    right_overhang = length - end - 1
    core_len = end - start + 1

    mismatch_cols = 0
    for j in range(start, end + 1):
        bases = {s[j] for s in seqs}
        if len(bases) >= 2:           # ≥2 distinct symbols
            mismatch_cols += 1

    return dict(file=path.name,
                block_id=path.stem.replace("_aln","").replace("block_",""),
                n_seqs=len(recs),
                alignment_len = length,
                core_len=core_len,
                left_overhang=left_overhang,
                right_overhang=right_overhang,
                mismatch_columns=mismatch_cols,
                mismatch_fraction=(mismatch_cols / core_len) if core_len > 0 else 0.0)

def summarize_block_msas(junction_name, save_df = True):
    aligned_dir = Path(f"../results/block_alignments/{junction_name}")
    results = [analyze_alignment(p) for p in sorted(aligned_dir.glob("block_*_aln.fa"))]
    summary_df = pd.DataFrame(results).sort_values("block_id")

    if save_df:
        out_csv = Path(f"../results/block_alignments/{junction_name}_alignment_stats.csv")
        summary_df.to_csv(out_csv, index=False)

    return summary_df