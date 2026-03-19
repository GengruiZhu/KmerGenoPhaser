#!/usr/bin/env python3
"""
check_and_fix_blocks.py
-----------------------
Validates block annotation files against chromosome lengths in a reference FASTA.
Rules applied per chromosome:
  1. No block file found → generate NoData blocks (NODATA_WINDOW-sized) for whole chromosome.
  2. Last block ends beyond chromosome → trim its end to chromosome length.
  3. Gap between block end and chromosome end → append NoData blocks (NODATA_WINDOW-sized).
  4. Gap inside the block list (e.g. 0…start of first block) → fill with NoData blocks.

Writes corrected files to --output_block_dir (one file per chromosome, e.g. Chr1A.txt).
Format: start_bp <TAB> end_bp <TAB> bloodline  (0-based half-open intervals in bp)
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# FASTA utilities
# ---------------------------------------------------------------------------

def get_chrom_lengths(fasta_path):
    """Return {chrom_name: length_bp} for every sequence in the FASTA."""
    lengths = {}
    current = None
    length = 0
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current is not None:
                    lengths[current] = length
                current = line[1:].split()[0]
                length = 0
            else:
                length += len(line)
    if current is not None:
        lengths[current] = length
    return lengths


# ---------------------------------------------------------------------------
# Block file I/O
# ---------------------------------------------------------------------------

def read_block_file(path):
    """
    Read a block txt file.
    Expected format (tab-separated): start_bp  end_bp  bloodline
    Returns list of (start, end, bloodline) tuples.
    """
    blocks = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
                bloodline = parts[2].strip() if len(parts) > 2 else "NoData"
                blocks.append((start, end, bloodline))
            except ValueError:
                continue
    return sorted(blocks, key=lambda x: x[0])


def write_block_file(path, blocks):
    """Write (start, end, bloodline) tuples to a tab-separated file."""
    with open(path, "w") as fh:
        for start, end, bl in blocks:
            fh.write(f"{start}\t{end}\t{bl}\n")


# ---------------------------------------------------------------------------
# Core fix logic
# ---------------------------------------------------------------------------

def fill_gap_with_nodata(pos, end, window):
    """Generate NoData blocks covering [pos, end) with given window size."""
    result = []
    cur = pos
    while cur < end:
        blk_end = min(cur + window, end)
        result.append((cur, blk_end, "NoData"))
        cur = blk_end
    return result


def fix_blocks_for_chrom(chrom, chrom_len, block_file, nodata_window):
    """
    Return the corrected block list for one chromosome.
    """
    # ── Case 1: No block file ──────────────────────────────────────────────
    if block_file is None or not os.path.exists(block_file):
        print(f"  {chrom} ({chrom_len:,} bp): no block file → full NoData coverage")
        return fill_gap_with_nodata(0, chrom_len, nodata_window)

    blocks = read_block_file(block_file)

    if not blocks:
        print(f"  {chrom} ({chrom_len:,} bp): empty block file → full NoData coverage")
        return fill_gap_with_nodata(0, chrom_len, nodata_window)

    # ── Trim blocks that overshoot chromosome end ──────────────────────────
    trimmed = []
    n_trimmed = 0
    for (start, end, bl) in blocks:
        if start >= chrom_len:
            n_trimmed += 1
            continue                       # drop entirely out-of-range block
        if end > chrom_len:
            end = chrom_len               # trim end
            n_trimmed += 1
        trimmed.append((start, end, bl))

    if n_trimmed:
        print(f"  {chrom}: {n_trimmed} block(s) trimmed/dropped (exceeded {chrom_len:,} bp)")

    # ── Fill internal gaps + leading gap + trailing gap ────────────────────
    filled = []
    pos = 0                                # cursor over chromosome

    for (start, end, bl) in trimmed:
        if start > pos:
            # Gap before this block
            filled.extend(fill_gap_with_nodata(pos, start, nodata_window))
        filled.append((start, end, bl))
        pos = end

    if pos < chrom_len:
        # Gap after last block → append NoData
        n_nodata_added = (chrom_len - pos + nodata_window - 1) // nodata_window
        print(f"  {chrom}: +{n_nodata_added} NoData block(s) appended "
              f"(chromosome extends {chrom_len - pos:,} bp beyond block coverage)")
        filled.extend(fill_gap_with_nodata(pos, chrom_len, nodata_window))

    return filled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check block annotation files against chromosome FASTA lengths "
                    "and fix mismatches (trim / add NoData blocks)."
    )
    parser.add_argument("--input_fasta",      required=True,
                        help="Reference FASTA file (used to read chromosome lengths).")
    parser.add_argument("--block_dir",        default=None,
                        help="Directory containing block .txt files (ChrXX.txt). "
                             "If omitted, all chromosomes get full NoData coverage.")
    parser.add_argument("--output_block_dir", required=True,
                        help="Output directory for fixed block files.")
    parser.add_argument("--nodata_window",    type=int, default=1_000_000,
                        help="Window size (bp) used when generating NoData padding blocks "
                             "(default: 1000000).")
    parser.add_argument("--target_chroms",    nargs="*", default=None,
                        help="Only process these chromosome names. "
                             "Default: all chromosomes in the FASTA.")
    args = parser.parse_args()

    os.makedirs(args.output_block_dir, exist_ok=True)

    print(f"Reading chromosome lengths from: {args.input_fasta}")
    chrom_lengths = get_chrom_lengths(args.input_fasta)
    print(f"  → {len(chrom_lengths)} chromosomes found\n")

    target = set(args.target_chroms) if args.target_chroms else set(chrom_lengths.keys())
    changed = 0

    for chrom in sorted(chrom_lengths.keys()):
        if chrom not in target:
            continue

        chrom_len = chrom_lengths[chrom]
        block_file = None
        if args.block_dir:
            candidate = os.path.join(args.block_dir, f"{chrom}.txt")
            if os.path.exists(candidate):
                block_file = candidate

        fixed_blocks = fix_blocks_for_chrom(chrom, chrom_len, block_file, args.nodata_window)

        out_path = os.path.join(args.output_block_dir, f"{chrom}.txt")
        write_block_file(out_path, fixed_blocks)
        changed += 1

    print(f"\nDone. {changed} chromosome block file(s) written to: {args.output_block_dir}")


if __name__ == "__main__":
    main()
