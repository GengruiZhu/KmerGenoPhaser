#!/usr/bin/env python3
"""
map_kmers_to_genome.py
-----------------------
Maps species-specific k-mers onto a target genome using a sliding window,
producing a counts-per-window TSV for each chromosome.

Output TSV format (one file per chromosome):
  #Start  End  Species1  Species2  [...]
  0       100000  1341  822
  100000  200000  902   818
"""
import os
import argparse
import time
import multiprocessing
from Bio import SeqIO


def get_canonical_kmer(kmer):
    rev = kmer.translate(str.maketrans("ATCG", "TAGC"))[::-1]
    return kmer if kmer < rev else rev


def init_worker(db_ref):
    global global_kmer_db
    global_kmer_db = db_ref


def process_chromosome(args):
    chrom_name, chrom_seq, k, window_size, output_dir, species_list = args
    seq_len     = len(chrom_seq)
    num_windows = (seq_len + window_size - 1) // window_size
    counts      = {sp: [0] * num_windows for sp in species_list}

    for i in range(seq_len - k + 1):
        kmer = get_canonical_kmer(chrom_seq[i:i + k].upper())
        sp   = global_kmer_db.get(kmer)
        if sp:
            counts[sp][i // window_size] += 1

    out_p = os.path.join(output_dir, f"{chrom_name}_mapping.tsv")
    with open(out_p, "w") as f:
        f.write("#Start\tEnd\t" + "\t".join(species_list) + "\n")
        for w in range(num_windows):
            start = w * window_size
            end   = min(start + window_size, seq_len)
            f.write(f"{start}\t{end}\t"
                    + "\t".join(str(counts[sp][w]) for sp in species_list)
                    + "\n")
    return chrom_name


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged_kmer_file", required=True,
                        help="Equalized k-mer file (Kmer, FinalScore, Species columns)")
    parser.add_argument("--genome_file",      required=True,
                        help="Target genome FASTA")
    parser.add_argument("--output_dir",       required=True,
                        help="Output directory for *_mapping.tsv files")
    parser.add_argument("--species_list",     required=True,
                        help="Comma-separated species names (must match Species column)")
    parser.add_argument("--k",                type=int, required=True,
                        help="K-mer size")
    parser.add_argument("--threads",          type=int, default=10,
                        help="CPU threads (default: 10)")
    parser.add_argument("--window_size",      type=int, default=100000,
                        help="Sliding window size in bp (default: 100000 = 100 kb)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load k-mer database
    print("Loading k-mer database...")
    local_db       = {}
    conflict_kmers = set()

    df = pd.read_csv(args.merged_kmer_file, sep="\t")
    for _, row in df.iterrows():
        kmer, sp = row["Kmer"], row["Species"]
        if kmer in local_db and local_db[kmer] != sp:
            conflict_kmers.add(kmer)
        else:
            local_db[kmer] = sp

    for c in conflict_kmers:
        del local_db[c]

    print(f"  DB loaded: {len(local_db):,} unique k-mers "
          f"({len(conflict_kmers):,} conflicts removed)")

    # 2. Parse genome and dispatch
    species_list = args.species_list.split(",")
    tasks = []
    for rec in SeqIO.parse(args.genome_file, "fasta"):
        tasks.append((rec.id, str(rec.seq),
                      args.k, args.window_size,
                      args.output_dir, species_list))

    print(f"Mapping {len(tasks)} chromosome(s) "
          f"| window={args.window_size:,} bp | threads={args.threads}")
    t0 = time.time()

    with multiprocessing.Pool(processes=args.threads,
                              initializer=init_worker,
                              initargs=(local_db,)) as pool:
        results = pool.map(process_chromosome, tasks)

    elapsed = time.time() - t0
    print(f"Done. Mapped {len(results)} chromosome(s) in {elapsed:.1f}s")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
