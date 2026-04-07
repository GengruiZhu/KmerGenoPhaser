import pandas as pd
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Score distribution analysis and bin-based equalization."
    )
    parser.add_argument("--input_file",  required=True,
                        help="The merged unique kmer file (TSV: Kmer / FinalScore / Species)")
    parser.add_argument("--output_file", required=True,
                        help="Final equalized kmer file")
    parser.add_argument("--min_score",   type=float, default=0.5,
                        help="Minimum specificity score threshold (default: 0.5)")
    parser.add_argument("--bin_size",    type=float, default=1.0,
                        help="Score bin size for distribution analysis (default: 1.0)")
    args = parser.parse_args()

    # ── 1. Read & basic filter ────────────────────────────────────────────────
    print(f"Reading merged file: {args.input_file}")
    df = pd.read_csv(args.input_file, sep="\t")

    required_cols = {"Kmer", "FinalScore", "Species"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"[ERROR] Input file missing columns: {missing}")

    df = df[df["FinalScore"] >= args.min_score].copy()

    if df.empty:
        print(
            f"[ERROR] No k-mers remain after applying --min_score {args.min_score}.\n"
            f"        Try lowering --min_score."
        )
        return

    # ── 2. Detect species count dynamically ───────────────────────────────────
    # FIX: was hardcoded as 3; now reads the actual number from the data so
    #      the script works for 2-ancestor species (e.g. Brassica carinata)
    #      and 3-ancestor species (e.g. wheat) alike.
    all_species  = sorted(df["Species"].unique())
    n_species    = len(all_species)

    print(f"\nDetected {n_species} species: {', '.join(all_species)}")

    if n_species < 2:
        print("[ERROR] At least 2 species are required for equalization.")
        return

    # ── 3. Bin assignment ─────────────────────────────────────────────────────
    df["Score_Bin"] = (df["FinalScore"] / args.bin_size).astype(int)

    # ── 4. Print score gradient distribution ──────────────────────────────────
    print("\nScore Gradient Distribution:")
    dist_stats = (
        df.groupby(["Score_Bin", "Species"])
          .size()
          .unstack(fill_value=0)
    )
    print(dist_stats)

    # ── 5. Bin-wise equalization ───────────────────────────────────────────────
    final_list = []
    print("\nBin-wise Equalization Progress:")
    print(f"{'Bin_Range':<12} | {'Min_Count':<10} | {'Status'}")
    print("-" * 45)

    for bin_val, group in df.groupby("Score_Bin"):
        counts      = group["Species"].value_counts()
        bin_label   = (
            f"{bin_val * args.bin_size:.1f}-"
            f"{(bin_val + 1) * args.bin_size:.1f}"
        )

        # A bin is valid only when EVERY species has at least one k-mer in it.
        # FIX: compare against n_species (dynamic), not the hardcoded literal 3.
        if len(counts) < n_species:
            min_c = 0
        else:
            min_c = int(counts.min())

        if min_c > 0:
            sampled = (
                group
                .groupby("Species", group_keys=False)
                .apply(lambda x: x.sample(n=min_c, random_state=42))
            )
            final_list.append(sampled)
            print(f"{bin_label:<12} | {min_c:<10} | Balanced")
        else:
            # Show which species are absent to ease debugging
            absent = [sp for sp in all_species if sp not in counts.index]
            reason = (
                f"Skipped (missing: {', '.join(absent)})"
                if absent
                else "Skipped (Species missing)"
            )
            print(f"{bin_label:<12} | {min_c:<10} | {reason}")

    print("-" * 45)

    # ── 6. Guard: nothing survived ────────────────────────────────────────────
    if not final_list:
        print(
            "[ERROR] No k-mers survived equalization.\n"
            "        Likely causes:\n"
            "          1) --min_score is too high — try lowering it.\n"
            "          2) One species has very few k-mers in every score bin;\n"
            "             inspect the Score Gradient Distribution above."
        )
        return

    # ── 7. Write output ───────────────────────────────────────────────────────
    final_df = pd.concat(final_list).reset_index(drop=True)
    final_df[["Kmer", "FinalScore", "Species"]].to_csv(
        args.output_file, sep="\t", index=False
    )

    total = len(final_df)
    print(f"Success! Total K-mers (Strictly Equalized): {total}")
    for sp in all_species:
        count = int((final_df["Species"] == sp).sum())
        print(f"  Verify {sp}: {count} K-mers")


if __name__ == "__main__":
    main()
