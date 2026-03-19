#!/usr/bin/env python3
"""
csv_blocks_to_txt.py
--------------------
Converts block CSV files produced by block_identification.R
(columns: Chrom, Start_Mb, End_Mb, Bloodline)
into the per-chromosome .txt format expected by the unsupervised autoencoder
(format: start_bp <TAB> end_bp <TAB> bloodline, no header).

Usage:
    python csv_blocks_to_txt.py \
        --input_dir  /path/to/R_output_dir \
        --output_dir /path/to/block_txt_dir
"""

import argparse
import csv
import os


def convert_one(csv_path, output_dir):
    """Convert a single *_Final_Blocks.csv to per-chromosome .txt files."""
    per_chrom = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            chrom     = row["Chrom"].strip()
            start_bp  = int(round(float(row["Start_Mb"]) * 1_000_000))
            end_bp    = int(round(float(row["End_Mb"])   * 1_000_000))
            bloodline = row["Bloodline"].strip()
            per_chrom.setdefault(chrom, []).append((start_bp, end_bp, bloodline))

    for chrom, blocks in per_chrom.items():
        out_path = os.path.join(output_dir, f"{chrom}.txt")
        # If the .txt already exists (e.g., from a previous run), we overwrite.
        with open(out_path, "w") as fh:
            for start, end, bl in sorted(blocks, key=lambda x: x[0]):
                fh.write(f"{start}\t{end}\t{bl}\n")
        print(f"  Written: {out_path}  ({len(blocks)} blocks)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert R block CSV files to autoencoder-compatible .txt files."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing *_Final_Blocks.csv files.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for per-chromosome .txt files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith("_Final_Blocks.csv")]
    if not csv_files:
        print("No *_Final_Blocks.csv files found in:", args.input_dir)
        return

    print(f"Found {len(csv_files)} CSV file(s). Converting…")
    for fname in sorted(csv_files):
        csv_path = os.path.join(args.input_dir, fname)
        print(f"Processing: {fname}")
        convert_one(csv_path, args.output_dir)

    print(f"\nDone. Block .txt files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
