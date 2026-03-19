#!/usr/bin/env python3
"""
mapping_counts_to_blocks.py
----------------------------
Converts k-mer mapping count TSV files (output of map_kmers_to_genome.py)
into per-chromosome block .txt files for the unsupervised autoencoder.

Logic:
  For each window, the dominant species is determined by highest k-mer count
  (must also exceed --dominance_threshold as a fraction of total counts).
  Adjacent windows with the same dominant species are merged into one block.
  Windows with no dominant species are labelled "LowInfo".

Input TSV format (from map_kmers_to_genome.py):
  #Start  End  Species1  Species2  [Species3 ...]
  0       100000  1341  822
  100000  200000  902   818

Output .txt format (for unsupervised --block_dir):
  Start   End     Bloodline
  0       500000  Species1
  500000  1200000 Species2
  ...

Usage:
  python mapping_counts_to_blocks.py \\
      --input_dir         /path/to/mapping_tsv_dir \\
      --output_dir        /path/to/block_txt_dir \\
      [--dominance_thr    0.55] \\
      [--min_counts       10]
"""
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir",      required=True,
                   help="Directory containing *_mapping.tsv files")
    p.add_argument("--output_dir",     required=True,
                   help="Output directory for per-chromosome .txt block files")
    p.add_argument("--dominance_thr",  type=float, default=0.55,
                   help="Min fraction for dominant-species call (default: 0.55)")
    p.add_argument("--min_counts",     type=int,   default=10,
                   help="Min total k-mer count in a window to attempt a call "
                        "(below this → LowInfo, default: 10)")
    return p.parse_args()


def call_dominant(counts_dict, dominance_thr, min_counts):
    """Return dominant species name, or 'LowInfo' / 'Mixed'."""
    total = sum(counts_dict.values())
    if total < min_counts:
        return "LowInfo"
    best_sp   = max(counts_dict, key=counts_dict.get)
    best_frac = counts_dict[best_sp] / total
    if best_frac >= dominance_thr:
        return best_sp
    return "Mixed"


def rle_merge(windows):
    """Run-length encode adjacent windows with the same bloodline."""
    if not windows:
        return []
    merged = []
    cur_start, cur_end, cur_bl = windows[0]
    for start, end, bl in windows[1:]:
        if bl == cur_bl:
            cur_end = end
        else:
            merged.append((cur_start, cur_end, cur_bl))
            cur_start, cur_end, cur_bl = start, end, bl
    merged.append((cur_start, cur_end, cur_bl))
    return merged


def convert_file(tsv_path, output_dir, dominance_thr, min_counts):
    species = []
    windows = []

    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Header: #Start  End  Sp1  Sp2 ...
                parts = line.lstrip("#").split("\t")
                species = [p.strip() for p in parts[2:]]
                continue
            parts = line.split("\t")
            if len(parts) < 2 + len(species):
                continue
            start = int(parts[0])
            end   = int(parts[1])
            counts = {sp: int(parts[2 + i]) for i, sp in enumerate(species)}
            bl = call_dominant(counts, dominance_thr, min_counts)
            windows.append((start, end, bl))

    if not windows:
        print(f"  [WARN] No windows parsed from {os.path.basename(tsv_path)}")
        return 0

    blocks  = rle_merge(windows)
    chrom   = os.path.basename(tsv_path).replace("_mapping.tsv", "")
    out_path = os.path.join(output_dir, f"{chrom}.txt")

    with open(out_path, "w") as fh:
        fh.write("Start\tEnd\tBloodline\n")
        for start, end, bl in blocks:
            fh.write(f"{start}\t{end}\t{bl}\n")

    print(f"  Written: {out_path}  ({len(blocks)} blocks from {len(windows)} windows)")
    return len(blocks)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tsv_files = [f for f in os.listdir(args.input_dir)
                 if f.endswith("_mapping.tsv")]
    if not tsv_files:
        print(f"[ERROR] No *_mapping.tsv files found in: {args.input_dir}")
        return

    print(f"Found {len(tsv_files)} mapping TSV file(s). Converting...")
    total_blocks = 0
    for fname in sorted(tsv_files):
        n = convert_file(os.path.join(args.input_dir, fname),
                         args.output_dir, args.dominance_thr, args.min_counts)
        total_blocks += n

    print(f"\nDone. {total_blocks} total blocks written to: {args.output_dir}")
    print("Feed to unsupervised module with:  --block_dir", args.output_dir)


if __name__ == "__main__":
    main()
