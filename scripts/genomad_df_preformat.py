import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_tsvs", required=True, nargs="+", help="Input TSV files to concatenate"
    )
    parser.add_argument("--output_df", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Read and concatenate all input TSV files
    dfs = [pd.read_csv(tsv, sep="\t") for tsv in args.input_tsvs]
    df = pd.concat(dfs, ignore_index=True)

    columns_final = ["id", "iso", "beg", "end", "type"]

    df["iso"] = df["seq_name"].str.split("|").str[0]
    df["beg"] = df["coordinates"].str.split("-").str[0].astype(int)
    df["end"] = df["coordinates"].str.split("-").str[1].astype(int)
    df["type"] = "prophage"
    df.rename(columns={"seq_name": "id"}, inplace=True)
    df = df[columns_final]
    df.set_index("id", inplace=True, verify_integrity=True)
    df.to_csv(args.output_df)
