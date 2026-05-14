import pandas as pd
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results"
CONFIG    = REPO_ROOT / "config"

GFF_PATH            = RESULTS / "gubbins" / "pan_gubbins_ST131_ABC__asm20-100-5.recombination_predictions.gff"
COORDS_PATH         = RESULTS / "gubbins" / "full_corealignment_coords.csv"
JUNCTION_STATS_PATH = RESULTS / "junction_summary_df.csv"

COORDS_ANNOTATED_PATH    = RESULTS / "gubbins" / "full_corealignment_coords_annotated.csv"
LONG_OUTPUT_PATH         = RESULTS / "gubbins" / "recombination_block_long.csv"
JUNCTION_LONG_OUTPUT_PATH = RESULTS / "gubbins" / "recombination_junction_long.csv"


# =============================================================================
# Part 1: Annotate core alignment coords with junctions
# =============================================================================

coords_df = pd.read_csv(COORDS_PATH)
coords_df = coords_df.sort_values("start").reset_index(drop=True)

block_ids       = coords_df["block_id"].tolist()
block_id_to_idx = {bid: i for i, bid in enumerate(block_ids)}
n_blocks        = len(block_ids)

junc_df = pd.read_csv(JUNCTION_STATS_PATH)

# Build block_id -> list of junction edges that span it (shorter circular path)
block_to_junctions = {}
for _, jrow in junc_df.iterrows():
    edge  = jrow["junction_name"]
    parts = edge.split("__")
    if len(parts) != 2:
        continue
    bid1 = re.sub(r"_[fr]$", "", parts[0])
    bid2 = re.sub(r"_[fr]$", "", parts[1])
    if bid1 not in block_id_to_idx or bid2 not in block_id_to_idx:
        continue
    lo = min(block_id_to_idx[bid1], block_id_to_idx[bid2])
    hi = max(block_id_to_idx[bid1], block_id_to_idx[bid2])
    path_inner = block_ids[lo : hi + 1]
    path_outer = block_ids[hi:] + block_ids[: lo + 1]
    spanning   = path_inner if len(path_inner) <= len(path_outer) else path_outer
    for bid in spanning:
        block_to_junctions.setdefault(bid, []).append(edge)

coords_df["junctions"] = coords_df["block_id"].map(
    lambda bid: block_to_junctions.get(bid, [])
)

coords_df.to_csv(COORDS_ANNOTATED_PATH, index=False)
print(f"Saved annotated coords to {COORDS_ANNOTATED_PATH}")


# =============================================================================
# Part 2: Build recombination-block dataframes
# =============================================================================

# --- Parse GFF recombination predictions ---

def _parse_attributes(attr_str):
    attrs = {}
    for part in attr_str.strip().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip().strip('"')
    return attrs

records = []
with open(GFF_PATH) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 9:
            continue
        start     = int(parts[3])
        end       = int(parts[4])
        attrs     = _parse_attributes(parts[8])
        branch    = attrs.get("node", "")
        taxa_str  = attrs.get("taxa", "")
        taxa      = [t.strip() for t in taxa_str.split(",") if t.strip()]
        snp_count = int(attrs.get("snp_count", 0))
        records.append({
            "rec_id":    f"rec_{len(records)}",
            "start":     start,
            "end":       end,
            "length":    end - start + 1,
            "branch":    branch,
            "taxa":      taxa,
            "snp_count": snp_count,
        })

rec_df = pd.DataFrame(records)
print(f"Parsed {len(rec_df)} recombination events from GFF.")

# --- For each recombination event find overlapping core blocks ---

def _overlapping_blocks(rec_start, rec_end, coords_df):
    mask = (coords_df["start"] <= rec_end) & (coords_df["end"] >= rec_start)
    ov = coords_df[mask].copy()
    ov["overlap_length"]     = ov["end"].clip(upper=rec_end) - ov["start"].clip(lower=rec_start)
    ov["pct_block_affected"] = (ov["overlap_length"] / ov["len"] * 100).round(2)
    return ov

# --- Long format: one row per (recombination event, core block) ---

event_cols = ["rec_id", "start", "end", "length", "branch", "taxa", "snp_count"]

rows = []
for _, row in rec_df.iterrows():
    ov = _overlapping_blocks(row["start"], row["end"], coords_df)
    for _, block_row in ov.iterrows():
        rows.append({
            **{col: row[col] for col in event_cols},
            "block_id":           block_row["block_id"],
            "block_length":       block_row["len"],
            "overlap_length":     block_row["overlap_length"],
            "pct_block_affected": block_row["pct_block_affected"],
        })

long_df = pd.DataFrame(rows)
long_df.to_csv(LONG_OUTPUT_PATH, index=False)
print(f"Saved recombination-block long format to {LONG_OUTPUT_PATH} ({len(long_df)} rows)")


# =============================================================================
# Part 3: Junction x recombination event long format
# =============================================================================

# Load annotated coords to get junction -> blocks mapping
coords_ann = pd.read_csv(COORDS_ANNOTATED_PATH)
coords_ann["junctions"] = coords_ann["junctions"].apply(eval)

# Build junction -> list of block rows (as dicts with start, end, len)
junction_to_blocks = {}
for _, brow in coords_ann.iterrows():
    for junc in brow["junctions"]:
        junction_to_blocks.setdefault(junc, []).append({
            "block_id": brow["block_id"],
            "start":    brow["start"],
            "end":      brow["end"],
            "len":      brow["len"],
        })

event_cols = ["rec_id", "start", "end", "length", "branch", "taxa", "snp_count"]

junc_rows = []
for _, rec_row in rec_df.iterrows():
    rec_start = rec_row["start"]
    rec_end   = rec_row["end"]

    for junc, blocks in junction_to_blocks.items():
        total_overlap      = 0
        n_blocks_affected  = 0
        combined_block_len = sum(b["len"] for b in blocks)
        n_blocks_junction  = len(blocks)

        for b in blocks:
            overlap = max(0, min(b["end"], rec_end) - max(b["start"], rec_start))
            if overlap > 0:
                total_overlap     += overlap
                n_blocks_affected += 1

        if n_blocks_affected == 0:
            continue

        junc_rows.append({
            **{col: rec_row[col] for col in event_cols},
            "junction_name":          junc,
            "overlap_length":         total_overlap,
            "combined_core_length":   combined_block_len,
            "pct_core_length_covered": round(total_overlap / combined_block_len * 100, 2) if combined_block_len > 0 else 0,
            "n_blocks_affected":      n_blocks_affected,
            "n_blocks_junction":      n_blocks_junction,
            "pct_blocks_affected":    round(n_blocks_affected / n_blocks_junction * 100, 2),
        })

junc_long_df = pd.DataFrame(junc_rows)
junc_long_df.to_csv(JUNCTION_LONG_OUTPUT_PATH, index=False)
print(f"Saved junction long format to {JUNCTION_LONG_OUTPUT_PATH} ({len(junc_long_df)} rows)")
