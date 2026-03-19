#!/usr/bin/env python3
"""
diag_dosage_curve_ref_or_alt.py
--------------------------------
Compute per-window diagnostic allele dosage ratios for a target sample,
using diagnostic sites from make_diag_sites_ref_or_alt.py.

Output bedGraph columns:
  CHROM  win_start  win_end  ratio  n_diag_reads  n_total_reads

Usage:
  python diag_dosage_curve_ref_or_alt.py \
      --vcf           merged.vcf.gz \
      --diag_tsv      diag_GroupA_vs_GroupB.tsv \
      --target_sample MyHybrid.1 \
      --window        1000000 \
      --output        MyHybrid.1_Chr1_GroupAdiag_1Mb.bedGraph \
      [--min_site_dp  5]
"""
import argparse
import sys
from cyvcf2 import VCF


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf",           required=True, help="VCF path")
    p.add_argument("--diag_tsv",      required=True,
                   help="Diagnostic sites TSV from make_diag_sites_ref_or_alt.py")
    p.add_argument("--target_sample", required=True, help="Sample name in VCF")
    p.add_argument("--window",        type=int, default=1000000,
                   help="Window size in bp (default: 1000000)")
    p.add_argument("--output",        required=True, help="Output bedGraph path")
    p.add_argument("--min_site_dp",   type=int, default=5,
                   help="Min total depth per site for a target sample (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load diagnostic dict: chrom -> {pos: diag_is_alt}
    diag = {}
    with open(args.diag_tsv) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom, pos, is_alt = parts[0], int(parts[1]), int(parts[2])
            diag.setdefault(chrom, {})[pos] = is_alt

    vcf     = VCF(args.vcf)
    samples = vcf.samples
    if args.target_sample not in samples:
        sys.exit(f"[ERROR] Target sample '{args.target_sample}' not found in VCF.")
    ti = samples.index(args.target_sample)

    acc_num = {}  # sum of diagnostic allele reads per window
    acc_den = {}  # sum of total reads per window

    for rec in vcf:
        d = diag.get(rec.CHROM)
        if not d:
            continue
        is_alt = d.get(rec.POS)
        if is_alt is None:
            continue

        AD = rec.format("AD")
        if AD is None:
            continue
        ad = AD[ti]
        if ad is None or ad[0] < 0 or ad[1] < 0:
            continue
        tot = ad[0] + ad[1]
        if tot < args.min_site_dp:
            continue

        diag_reads = ad[1] if is_alt == 1 else ad[0]
        win_start  = ((rec.POS - 1) // args.window) * args.window
        key        = (rec.CHROM, win_start)
        acc_num[key] = acc_num.get(key, 0) + diag_reads
        acc_den[key] = acc_den.get(key, 0) + tot

    written = 0
    with open(args.output, "w") as out:
        for (chrom, win_start) in sorted(acc_den.keys()):
            den = acc_den[(chrom, win_start)]
            num = acc_num.get((chrom, win_start), 0)
            if den <= 0:
                continue
            r = num / den
            out.write(
                f"{chrom}\t{win_start}\t{win_start + args.window}"
                f"\t{r:.6f}\t{num}\t{den}\n"
            )
            written += 1

    print(f"[INFO] Windows written: {written}  →  {args.output}")


if __name__ == "__main__":
    main()
