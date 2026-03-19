#!/usr/bin/env python3
"""
extract_block_features.py
按 block 区域提取 k-mer 频率特征，用于自编码器血缘分析。

三项改进：
  1. k-mer 范围从 min_kmer 到 max_kmer（默认 1~5）
     --block_dir 可选：不传则以整条染色体为单位提取（染色体级模式）
  2. --block_dir 可选：不传则以整条染色体为单位，bloodline 标记为 "NoData"
  3. 自动计算特征维度并写入 pickle 元信息，供训练脚本读取
"""

import numpy as np
from collections import Counter
import pickle
import argparse
from Bio import SeqIO
import pandas as pd
import os
import re
import sys


# =============================================================================
# Feature computation
# =============================================================================

def _all_kmers(k):
    """Return sorted list of all 4^k k-mers over ACGT."""
    bases = ['A', 'C', 'G', 'T']
    result = ['']
    for _ in range(k):
        result = [prev + b for prev in result for b in bases]
    return result


# Pre-build kmer index tables for speed
_KMER_TABLES = {}


def _get_kmer_table(k):
    if k not in _KMER_TABLES:
        kmers = _all_kmers(k)
        _KMER_TABLES[k] = {kmer: idx for idx, kmer in enumerate(kmers)}
    return _KMER_TABLES[k]


def compute_kmer_freq(seq, k):
    """
    Compute k-mer frequency vector of length 4^k.
    Ignores windows containing 'N'.
    """
    table = _get_kmer_table(k)
    n = len(table)
    counts = np.zeros(n, dtype=np.float64)
    valid = 0
    for i in range(len(seq) - k + 1):
        sub = seq[i:i + k]
        if 'N' not in sub:
            idx = table.get(sub)
            if idx is not None:
                counts[idx] += 1
                valid += 1
    if valid == 0:
        return counts          # all zeros
    return counts / valid


def extract_features(seq, min_kmer=1, max_kmer=5):
    """
    Concatenate k-mer frequency vectors for k in [min_kmer, max_kmer].
    Total dimensions = sum(4^k for k in range(min_kmer, max_kmer+1)).
    """
    seq = seq.upper()
    valid_bases = sum(1 for b in seq if b in 'ACGT')
    if valid_bases == 0:
        return None

    parts = []
    for k in range(min_kmer, max_kmer + 1):
        parts.append(compute_kmer_freq(seq, k))

    return np.concatenate(parts)


def feature_dim(min_kmer, max_kmer):
    return sum(4 ** k for k in range(min_kmer, max_kmer + 1))


# =============================================================================
# Block annotation loading
# =============================================================================

def _try_parse_block_df(filepath, chrom):
    """
    Try several formats:
      Format A (with header): Chrom, Start, End, Bloodline
      Format B (with header): Start, End, Bloodline  (or Start_bp, End_bp, Bloodline)
      Format C (no header):   start_bp<TAB>end_bp<TAB>bloodline
    Returns list of (start, end, bloodline).
    """
    try:
        df = pd.read_csv(filepath, sep='\t', header=0)
    except Exception:
        return []

    # Normalise column names
    col_map = {c.lower().strip(): c for c in df.columns}

    def _getcol(candidates):
        for cand in candidates:
            if cand in col_map:
                return df[col_map[cand]]
        return None

    start_col = _getcol(['start', 'start_bp', 'start_mb'])
    end_col   = _getcol(['end',   'end_bp',   'end_mb'])
    bl_col    = _getcol(['bloodline', 'blood_line', 'ancestry', 'label'])

    # If column names look like numbers (no-header file), fall back
    if start_col is None:
        try:
            df2 = pd.read_csv(filepath, sep='\t', header=None)
            if df2.shape[1] >= 3:
                start_col = df2.iloc[:, 0]
                end_col   = df2.iloc[:, 1]
                bl_col    = df2.iloc[:, 2]
            else:
                return []
        except Exception:
            return []

    if start_col is None or end_col is None or bl_col is None:
        return []

    # Detect Mb vs bp (if max start > 10000 assume bp, else assume Mb)
    try:
        starts = start_col.astype(float).tolist()
        ends   = end_col.astype(float).tolist()
    except Exception:
        return []

    if starts and max(starts) <= 10000:
        # looks like Mb
        starts = [int(s * 1_000_000) for s in starts]
        ends   = [int(e * 1_000_000) for e in ends]
    else:
        starts = [int(s) for s in starts]
        ends   = [int(e) for e in ends]

    bloodlines = bl_col.astype(str).str.strip().tolist()
    return sorted(zip(starts, ends, bloodlines), key=lambda x: x[0])


def load_block_annotations(block_dir, target_chroms=None):
    """
    Returns {chrom: [(start_bp, end_bp, bloodline), ...]}
    """
    blocks = {}
    for filename in sorted(os.listdir(block_dir)):
        if not filename.endswith('.txt'):
            continue
        chrom = filename[:-4]
        if target_chroms and chrom not in target_chroms:
            continue
        filepath = os.path.join(block_dir, filename)
        parsed = _try_parse_block_df(filepath, chrom)
        if parsed:
            blocks[chrom] = parsed
            print(f"  Loaded {len(parsed):>4d} blocks ← {filename}")
        else:
            print(f"  [WARN] Could not parse {filename}, skipping.")
    return blocks


def chromosome_level_blocks(sequences, target_chroms=None):
    """
    No block_dir: treat each whole chromosome as one 'NoData' block.
    """
    blocks = {}
    for chrom, seq in sequences.items():
        if target_chroms and chrom not in target_chroms:
            continue
        blocks[chrom] = [(0, len(seq), 'NoData')]
    return blocks


# =============================================================================
# Name cleaning
# =============================================================================

def clean_bloodline_name(raw):
    cleaned = re.sub(r'\d+$', '', str(raw).strip())
    return cleaned.capitalize() if cleaned else raw


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-k k-mer features from block/chromosome regions."
    )
    parser.add_argument('--input_fasta',    required=True,
                        help="Reference FASTA file.")
    parser.add_argument('--block_dir',      default=None,
                        help="Directory with block annotation .txt files. "
                             "If omitted, each chromosome is treated as one block.")
    parser.add_argument('--output_pickle',  required=True,
                        help="Output pickle path.")
    parser.add_argument('--min_block_size', type=int, default=10000,
                        help="Skip blocks shorter than this (bp). Default: 10000.")
    parser.add_argument('--min_kmer',       type=int, default=1,
                        help="Minimum k for k-mer frequency (default: 1). "
                             "k=1 → mononucleotide, k=2 → dinucleotide, etc.")
    parser.add_argument('--max_kmer',       type=int, default=5,
                        help="Maximum k for k-mer frequency (default: 5). "
                             "Feature dim = sum(4^k, k=min_kmer..max_kmer).")
    parser.add_argument('--target_chroms',  nargs='+', default=None,
                        help="Only process listed chromosomes.")

    args = parser.parse_args()

    # ── Sanity checks ────────────────────────────────────────────────────────
    if args.min_kmer < 1:
        print("[ERROR] --min_kmer must be >= 1"); sys.exit(1)
    if args.max_kmer < args.min_kmer:
        print("[ERROR] --max_kmer must be >= --min_kmer"); sys.exit(1)
    if args.max_kmer > 7:
        print("[WARN] max_kmer > 7 produces very large feature vectors; "
              "ensure you have enough RAM.")

    fdim = feature_dim(args.min_kmer, args.max_kmer)

    print("=" * 70)
    print("EXTRACT BLOCK FEATURES")
    print("=" * 70)
    print(f"  Input FASTA    : {args.input_fasta}")
    print(f"  Block dir      : {args.block_dir or '<none — chromosome-level mode>'}")
    print(f"  k-mer range    : {args.min_kmer} – {args.max_kmer}")
    print(f"  Feature dim    : {fdim}  "
          f"({' + '.join(str(4**k)+'(k='+str(k)+')' for k in range(args.min_kmer, args.max_kmer+1))})")
    print(f"  Min block size : {args.min_block_size:,} bp")
    if args.target_chroms:
        print(f"  Target chroms  : {args.target_chroms}")
    print("=" * 70)

    # ── Load sequences ────────────────────────────────────────────────────────
    print("\n[1/3] Loading sequences...")
    sequences = {}
    for record in SeqIO.parse(args.input_fasta, 'fasta'):
        cid = record.id
        if args.target_chroms and cid not in args.target_chroms:
            continue
        sequences[cid] = str(record.seq).upper()
        print(f"  {cid}: {len(sequences[cid]):,} bp")

    if not sequences:
        print("[ERROR] No sequences loaded. Check --input_fasta and --target_chroms.")
        sys.exit(1)

    # ── Load / build blocks ───────────────────────────────────────────────────
    print("\n[2/3] Loading block annotations...")
    if args.block_dir:
        blocks = load_block_annotations(args.block_dir, args.target_chroms)
        if not blocks:
            print("[ERROR] No blocks loaded from block_dir.")
            sys.exit(1)
    else:
        print("  No block_dir provided → chromosome-level mode (one block per chrom).")
        blocks = chromosome_level_blocks(sequences, args.target_chroms)

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n[3/3] Extracting features...")
    data = {}           # block_id → feature vector
    n_ok = 0
    n_skip_size = 0
    n_skip_seq  = 0

    for chrom in sorted(blocks.keys()):
        if chrom not in sequences:
            print(f"  [WARN] {chrom} not in FASTA, skipping.")
            continue

        seq = sequences[chrom]
        chrom_blocks = blocks[chrom]
        bl_counter = {}   # bloodline → count within this chrom

        print(f"\n  {chrom}: {len(chrom_blocks)} block(s)")

        for idx, (start, end, raw_bl) in enumerate(chrom_blocks):
            blen = end - start
            if blen < args.min_block_size:
                n_skip_size += 1
                continue

            bloodline = clean_bloodline_name(raw_bl)
            bl_counter[bloodline] = bl_counter.get(bloodline, 0) + 1
            bnum = bl_counter[bloodline]

            block_seq = seq[start:min(end, len(seq))]
            feat = extract_features(block_seq, args.min_kmer, args.max_kmer)

            if feat is None:
                n_skip_seq += 1
                continue

            block_id = f"{chrom}_{bloodline}_{bnum}"
            data[block_id] = feat
            n_ok += 1

            if (idx + 1) % 10 == 0:
                print(f"    ... {idx+1}/{len(chrom_blocks)}", end='\r')

        print(f"    done: {len(chrom_blocks)} blocks processed")

    print(f"\n{'='*70}")
    print(f"  Extracted  : {n_ok:,} blocks")
    print(f"  Skipped    : {n_skip_size:,} (too small) + {n_skip_seq:,} (no valid bases)")
    print(f"  Feature dim: {fdim}")

    if not data:
        print("[ERROR] No features extracted.")
        sys.exit(1)

    # Bloodline distribution
    bl_dist = {}
    for bid in data:
        parts = bid.split('_')
        # block_id format: ChrXX_Bloodline_N  (Chrom may have underscore too)
        # The bloodline is the part between last '_N' suffix and the chrom prefix
        # Safe: split on '_' and the second-to-last is the counter
        bl = '_'.join(parts[1:-1])
        bl_dist[bl] = bl_dist.get(bl, 0) + 1

    print("\n  Bloodline distribution:")
    for bl in sorted(bl_dist):
        print(f"    {bl}: {bl_dist[bl]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        'features': data,
        'meta': {
            'min_kmer': args.min_kmer,
            'max_kmer': args.max_kmer,
            'feature_dim': fdim,
            'n_blocks': n_ok,
        }
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pickle)), exist_ok=True)
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\n  ✓ Saved to: {args.output_pickle}")
    print(f"    Set INPUT_DIM={fdim} in your training script or config.")
    print("=" * 70)


if __name__ == '__main__':
    main()
