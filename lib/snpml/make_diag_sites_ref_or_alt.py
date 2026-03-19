#!/usr/bin/env python3
"""
make_diag_sites_ref_or_alt.py
------------------------------
Find diagnostic SNP sites that are near-fixed in one ancestor group (group2)
vs. near-fixed for the other allele in the other group (group1).

Output TSV: CHROM  POS  diag_is_alt
  diag_is_alt=1  →  group2 fixed ALT,  group1 fixed REF
  diag_is_alt=0  →  group2 fixed REF,  group1 fixed ALT

Usage:
  python make_diag_sites_ref_or_alt.py \
      --vcf       merged.vcf.gz \
      --group1    /path/group1_samples.txt \
      --group2    /path/group2_samples.txt \
      --output    diag_GroupA_vs_GroupB.tsv \
      [--min_per_sample_dp 3] \
      [--min_called_samples 3] \
      [--min_group_totdp 40] \
      [--af_g2_min 0.95] \
      [--af_g1_max 0.05] \
      [--max_site_dp 200]
"""
import argparse
import sys
from cyvcf2 import VCF


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf",    required=True, help="VCF path (bgzipped + tabix indexed)")
    p.add_argument("--group1", required=True,
                   help="File with group1 sample names (one per line) — "
                        "this group should be near-fixed for REF at diagnostic sites")
    p.add_argument("--group2", required=True,
                   help="File with group2 sample names (one per line) — "
                        "this group should be near-fixed for ALT at diagnostic sites")
    p.add_argument("--output", required=True, help="Output TSV path")
    p.add_argument("--min_per_sample_dp", type=int,   default=3,    metavar="INT")
    p.add_argument("--min_called_samples", type=int,  default=3,    metavar="INT")
    p.add_argument("--min_group_totdp",   type=int,   default=40,   metavar="INT")
    p.add_argument("--af_g2_min",         type=float, default=0.95, metavar="FLOAT",
                   help="Min ALT freq in group2 for diag site (default 0.95)")
    p.add_argument("--af_g1_max",         type=float, default=0.05, metavar="FLOAT",
                   help="Max ALT freq in group1 for diag site (default 0.05)")
    p.add_argument("--max_site_dp",       type=int,   default=200,  metavar="INT",
                   help="Per-sample depth cap (filter copy-number artifacts)")
    return p.parse_args()


def load_sample_set(path):
    with open(path) as fh:
        return set(x.strip() for x in fh if x.strip())


def sum_ref_alt(rec, idxs, min_dp, max_dp):
    AD = rec.format("AD")
    DP = rec.format("DP")
    ref_sum = alt_sum = tot_sum = called = 0
    for i in idxs:
        if AD is None or DP is None:
            continue
        dp = DP[i]
        ad = AD[i]
        if dp is None or ad is None:
            continue
        if dp <= 0 or ad[0] < 0 or ad[1] < 0:
            continue
        if dp < min_dp or dp > max_dp:
            continue
        tot = ad[0] + ad[1]
        if tot <= 0:
            continue
        ref_sum += ad[0]
        alt_sum += ad[1]
        tot_sum += tot
        called  += 1
    return ref_sum, alt_sum, tot_sum, called


def main():
    args = parse_args()

    g1_set = load_sample_set(args.group1)
    g2_set = load_sample_set(args.group2)

    vcf     = VCF(args.vcf)
    samples = vcf.samples
    g1_idx  = [i for i, s in enumerate(samples) if s in g1_set]
    g2_idx  = [i for i, s in enumerate(samples) if s in g2_set]

    if not g1_idx:
        sys.exit(f"[ERROR] No group1 samples found in VCF. "
                 f"Check {args.group1} vs VCF header.")
    if not g2_idx:
        sys.exit(f"[ERROR] No group2 samples found in VCF. "
                 f"Check {args.group2} vs VCF header.")

    print(f"[INFO] Group1 samples matched: {len(g1_idx)}")
    print(f"[INFO] Group2 samples matched: {len(g2_idx)}")

    written = 0
    with open(args.output, "w") as out:
        out.write("CHROM\tPOS\tdiag_is_alt\n")
        for rec in vcf:
            g1_ref, g1_alt, g1_tot, g1_called = sum_ref_alt(
                rec, g1_idx, args.min_per_sample_dp, args.max_site_dp)
            g2_ref, g2_alt, g2_tot, g2_called = sum_ref_alt(
                rec, g2_idx, args.min_per_sample_dp, args.max_site_dp)

            if (g1_called < args.min_called_samples or
                g2_called < args.min_called_samples):
                continue
            if g1_tot < args.min_group_totdp or g2_tot < args.min_group_totdp:
                continue

            af_g1_alt = g1_alt / g1_tot
            af_g2_alt = g2_alt / g2_tot

            # group2 fixed ALT, group1 fixed REF
            if af_g2_alt >= args.af_g2_min and af_g1_alt <= args.af_g1_max:
                out.write(f"{rec.CHROM}\t{rec.POS}\t1\n")
                written += 1
            # group2 fixed REF, group1 fixed ALT
            elif ((1 - af_g2_alt) >= args.af_g2_min and
                  (1 - af_g1_alt) <= args.af_g1_max):
                out.write(f"{rec.CHROM}\t{rec.POS}\t0\n")
                written += 1

    print(f"[INFO] Diagnostic sites written: {written}  →  {args.output}")


if __name__ == "__main__":
    main()
