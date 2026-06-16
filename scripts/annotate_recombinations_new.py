import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results" / "gubbins_new"

GFF_PATH    = RESULTS / "gubbins.recombination_predictions.gff"
COORDS_PATH = RESULTS / "ungapped_coords.csv"

OUTPUT_PATH = RESULTS / "recombination_block_long.csv"


def _parse_attributes(attr_str):
    attrs = {}
    for part in attr_str.strip().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip().strip('"')
    return attrs


# --- Parse GFF ---

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

# --- Load block coords ---

coords_df = pd.read_csv(COORDS_PATH)

# --- Build long-format: one row per (recombination event, overlapping block) ---

event_cols = ["rec_id", "start", "end", "length", "branch", "taxa", "snp_count"]

rows = []
for _, rec_row in rec_df.iterrows():
    rec_start = rec_row["start"]
    rec_end   = rec_row["end"]

    mask = (coords_df["start"] <= rec_end) & (coords_df["end"] >= rec_start)
    overlapping = coords_df[mask]

    for _, block in overlapping.iterrows():
        overlap_len = min(block["end"], rec_end) - max(block["start"], rec_start)
        rows.append({
            **{col: rec_row[col] for col in event_cols},
            "block_id":           block["block_id"],
            "block_length":       block["len"],
            "overlap_length":     overlap_len,
            "pct_block_affected": round(overlap_len / block["len"] * 100, 2),
        })

long_df = pd.DataFrame(rows, columns=[
    "rec_id", "start", "end", "length", "branch", "taxa", "snp_count",
    "block_id", "block_length", "overlap_length", "pct_block_affected",
])
long_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved recombination-block long format to {OUTPUT_PATH} ({len(long_df)} rows)")