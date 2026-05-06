#!/usr/bin/env python3
"""
extract_block_features_fft.py  ─  KmerGenoPhaser v1.1
======================================================
Block-level feature extraction with selectable encoding strategy.

Encoding modes (--encoding):
  kmer    K-mer frequency only (k=MIN_KMER..MAX_KMER). Original behaviour.
          dim = sum(4^k, k=MIN..MAX),  e.g. k=1..5 → 1364
  fft     Complex-encoded FFT magnitude only.
          dim = FFT_SIZE  (default 1024)
  concat  K-mer frequency concatenated with complex FFT.  ← RECOMMENDED
          dim = kmer_dim + FFT_SIZE,  e.g. 1364 + 1024 = 2388

Complex encoding (Vari-code / biological-meaning preserving):
  A =  1+1j   (purine   + amino-group)
  G =  1-1j   (purine   + keto-group)
  C = -1+1j   (pyrimidine + amino-group)
  T = -1-1j   (pyrimidine + keto-group)
  N / other → 0+0j

Compared with the old ASCII encoding (np.int8, A=65 etc.) this design
keeps the purine/pyrimidine (real axis) and amino/keto (imaginary axis)
distinctions orthogonal in the complex plane, which yields more biologically
meaningful spectral peaks (see Session log 2026-03-31).

Output
------
A pickle file containing:
  { block_id: np.ndarray(shape=(dim,), dtype=float64), ... }

block_id format:  {Chrom}_{Bloodline}_{n}
  e.g.  Chr1G_Officinarum_15

Compatible with train_adaptive_unsupervised.py.

Usage
-----
python extract_block_features_fft.py \\
    --input_fasta   target.fasta \\
    --block_dir     blocks/ \\
    --output_pickle features.pkl \\
    --encoding      concat \\
    --min_kmer      1 \\
    --max_kmer      5 \\
    --fft_size      1024 \\
    --target_chroms Chr1A Chr1B Chr1C

Author : Gengrui Zhu
Version: 1.1 (2026-03-31)
"""

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict
from itertools import product

import numpy as np
from Bio import SeqIO
from scipy.fftpack import fft


# ──────────────────────────────────────────────────────────────────────────────
# Complex (vari-code) encoding
# ──────────────────────────────────────────────────────────────────────────────

_COMPLEX_MAP: dict = {
    'A':  1 + 1j,
    'C': -1 + 1j,
    'G':  1 - 1j,
    'T': -1 - 1j,
}


def encode_complex(seq: str) -> np.ndarray:
    """Map each ACGT base to its complex vari-code value; unknowns → 0+0j."""
    return np.array(
        [_COMPLEX_MAP.get(b, 0 + 0j) for b in seq.upper()],
        dtype=np.complex128,
    )


def compute_fft_features(seq: str, fft_size: int = 1024) -> np.ndarray:
    """
    Complex-encode a sequence, compute FFT, return magnitude spectrum.

    The sequence is truncated to `fft_size` or zero-padded if shorter.
    Output dimension is always exactly `fft_size`.
    """
    z = encode_complex(seq)
    if len(z) >= fft_size:
        z = z[:fft_size]
    else:
        z = np.pad(z, (0, fft_size - len(z)))
    return np.abs(fft(z)).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# K-mer frequency encoding
# ──────────────────────────────────────────────────────────────────────────────

def build_kmer_index(min_kmer: int, max_kmer: int) -> dict:
    """Build an ordered {kmer_str: position} dict for all k in [min_kmer, max_kmer]."""
    idx: dict = {}
    pos = 0
    for k in range(min_kmer, max_kmer + 1):
        for tup in product('ACGT', repeat=k):
            idx[''.join(tup)] = pos
            pos += 1
    return idx


def kmer_total_dim(min_kmer: int, max_kmer: int) -> int:
    return sum(4 ** k for k in range(min_kmer, max_kmer + 1))


def compute_kmer_features(
    seq: str,
    kmer_index: dict,
    min_kmer: int,
    max_kmer: int,
) -> np.ndarray:
    """
    Count all k-mers in [min_kmer, max_kmer] and return a normalised
    frequency vector.  Unknown / ambiguous bases are skipped.
    """
    vec = np.zeros(len(kmer_index), dtype=np.float64)
    seq_upper = seq.upper()
    total = 0
    for k in range(min_kmer, max_kmer + 1):
        for i in range(len(seq_upper) - k + 1):
            mer = seq_upper[i: i + k]
            if mer in kmer_index:
                vec[kmer_index[mer]] += 1
                total += 1
    if total > 0:
        vec /= total
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Feature dispatch
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(
    seq: str,
    encoding: str,
    min_kmer: int,
    max_kmer: int,
    kmer_index: dict,
    fft_size: int,
) -> np.ndarray:
    """Return a 1-D feature vector for the given sequence and encoding mode."""
    if encoding == 'kmer':
        return compute_kmer_features(seq, kmer_index, min_kmer, max_kmer)

    elif encoding == 'fft':
        return compute_fft_features(seq, fft_size)

    elif encoding == 'concat':
        kf = compute_kmer_features(seq, kmer_index, min_kmer, max_kmer)
        ff = compute_fft_features(seq, fft_size)
        return np.concatenate([kf, ff])

    else:
        raise ValueError(f"Unknown encoding mode: '{encoding}'. "
                         "Choose from: kmer, fft, concat")


def feature_dim(encoding: str, min_kmer: int, max_kmer: int, fft_size: int) -> int:
    kd = kmer_total_dim(min_kmer, max_kmer)
    return {'kmer': kd, 'fft': fft_size, 'concat': kd + fft_size}[encoding]


# ──────────────────────────────────────────────────────────────────────────────
# Block file reader
# ──────────────────────────────────────────────────────────────────────────────

def read_block_file(path: str):
    """
    Parse a block .txt file (whitespace-separated, with header).
    Auto-detects 3-column vs 4-column format from header.
    """
    with open(path) as fh:
        header = fh.readline()
        header_parts = header.strip().split()
        has_chrom_col = (len(header_parts) >= 4)

        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split()          # ← 用 split() 代替 split('\t')，兼容空格/tab 混排
            min_cols = 4 if has_chrom_col else 3
            if len(parts) < min_cols:
                print(f"  [WARN] {path}:{lineno}: expected {min_cols} columns, got {len(parts)}, skipping")
                continue
            try:
                if has_chrom_col:
                    start, end = int(parts[1]), int(parts[2])
                    bloodline = parts[3].strip()
                else:
                    start, end = int(parts[0]), int(parts[1])
                    bloodline = parts[2].strip()
            except ValueError:
                print(f"  [WARN] {path}:{lineno}: non-integer start/end, skipping")
                continue
            yield start, end, bloodline


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Block feature extraction with selectable encoding  (KmerGenoPhaser v1.1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Encoding modes:
  kmer    K-mer frequencies only          dim = sum(4^k, k=MIN..MAX)
  fft     Complex FFT magnitudes only     dim = FFT_SIZE
  concat  K-mer + FFT concatenated        dim = kmer_dim + FFT_SIZE  [default]

Complex encoding:  A=1+j  C=-1+j  G=1-j  T=-1-j
""",
    )
    # Required
    parser.add_argument('--input_fasta',   required=True,
                        help='Target genome FASTA file')
    parser.add_argument('--block_dir',     required=True,
                        help='Directory of block .txt files (one per chromosome)')
    parser.add_argument('--output_pickle', required=True,
                        help='Output pickle path')

    # Encoding options (v1.1)
    parser.add_argument('--encoding', choices=['kmer', 'fft', 'concat'],
                        default='concat',
                        help='Feature encoding strategy (default: concat)')
    parser.add_argument('--fft_size', type=int, default=1024,
                        help='FFT points / output spectrum dimension (default: 1024)')

    # K-mer options
    parser.add_argument('--min_kmer', type=int, default=1,
                        help='Minimum k-mer size (default: 1)')
    parser.add_argument('--max_kmer', type=int, default=5,
                        help='Maximum k-mer size (default: 5)')

    # Filter options (kept from original script)
    parser.add_argument('--min_block_size', type=int, default=0,
                        help='Skip blocks shorter than this (bp) (default: 0)')
    parser.add_argument('--target_chroms',  nargs='+', default=None,
                        help='Only process these chromosomes; default = all')

    args = parser.parse_args()

    # ── pre-compute index once ──────────────────────────────────────────────
    kmer_index = build_kmer_index(args.min_kmer, args.max_kmer)
    dim = feature_dim(args.encoding, args.min_kmer, args.max_kmer, args.fft_size)

    print("=" * 68)
    print("  extract_block_features_fft.py  (KmerGenoPhaser v1.1)")
    print("=" * 68)
    print(f"  Encoding   : {args.encoding}")
    if args.encoding in ('kmer', 'concat'):
        print(f"  K-mer range: {args.min_kmer} – {args.max_kmer}  "
              f"(dim={kmer_total_dim(args.min_kmer, args.max_kmer)})")
    if args.encoding in ('fft', 'concat'):
        print(f"  FFT size   : {args.fft_size}")
        print(f"  Complex map: A=1+j  C=-1+j  G=1-j  T=-1-j")
    print(f"  INPUT_DIM  : {dim}")
    print(f"  Block dir  : {args.block_dir}")
    print("=" * 68)

    # ── load genome ─────────────────────────────────────────────────────────
    print("[Loading] Reading genome FASTA …")
    genome: dict = {}
    for rec in SeqIO.parse(args.input_fasta, 'fasta'):
        genome[rec.id] = str(rec.seq)
    print(f"  Loaded {len(genome)} sequence(s)")

    # ── determine block files ────────────────────────────────────────────────
    block_files = sorted(glob.glob(os.path.join(args.block_dir, '*.txt')))
    if not block_files:
        print(f"[ERROR] No .txt block files found in: {args.block_dir}", file=sys.stderr)
        sys.exit(1)

    if args.target_chroms:
        target_set = set(args.target_chroms)
        block_files = [
            bf for bf in block_files
            if os.path.splitext(os.path.basename(bf))[0] in target_set
        ]
        print(f"  Filtering to {len(block_files)} target chromosome(s): "
              f"{', '.join(args.target_chroms)}")

    # ── process blocks ───────────────────────────────────────────────────────
    block_data: dict = {}
    skipped_chrom = 0
    skipped_size  = 0
    skipped_empty = 0

    print(f"\n[Step 1] Extracting features from {len(block_files)} block file(s) …\n")

    for bf in block_files:
        chrom_name = os.path.splitext(os.path.basename(bf))[0]

        if chrom_name not in genome:
            print(f"  [SKIP] {chrom_name}: not found in FASTA")
            skipped_chrom += 1
            continue

        chrom_seq   = genome[chrom_name]
        chrom_len   = len(chrom_seq)
        print(f"  Processing {chrom_name}  ({chrom_len:,} bp) …")

        # per-bloodline counter → reproducible block_ids
        bl_counter: dict = defaultdict(int)

        for start, end, bloodline in read_block_file(bf):
            # size filter
            if (end - start) < args.min_block_size:
                skipped_size += 1
                continue

            # bounds guard
            end = min(end, chrom_len)
            if end <= start:
                continue

            subseq = chrom_seq[start:end]

            # skip sequences with no ACGT content
            valid = sum(1 for b in subseq.upper() if b in 'ACGT')
            if valid == 0:
                skipped_empty += 1
                continue

            feat = extract_features(
                subseq, args.encoding,
                args.min_kmer, args.max_kmer,
                kmer_index, args.fft_size,
            )

            bl_counter[bloodline] += 1
            block_id = f"{chrom_name}_{bloodline}_{bl_counter[bloodline]}"
            block_data[block_id] = feat

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pickle)), exist_ok=True)
    with open(args.output_pickle, 'wb') as fh:
        pickle.dump(block_data, fh)

    print()
    print("=" * 68)
    print(f"  Total blocks saved : {len(block_data)}")
    print(f"  Skipped (no FASTA) : {skipped_chrom} chrom(s)")
    print(f"  Skipped (too small): {skipped_size} block(s)")
    print(f"  Skipped (no ACGT)  : {skipped_empty} block(s)")
    print(f"  Feature dim        : {dim}")
    print(f"  OUTPUT_PICKLE      : {args.output_pickle}")
    print()
    print(f">>> Set INPUT_DIM={dim} in your training script / conf <<<")
    print("=" * 68)


if __name__ == '__main__':
    main()
