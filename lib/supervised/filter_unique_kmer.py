import pandas as pd
import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--kmer_db_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--species', required=True)
    parser.add_argument('--other_species', required=True)
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()

    # 加载候选集
    cand_file = os.path.join(args.input_dir, f"{args.species}_top_weighted_k{args.k}_complex.txt")
    df = pd.read_csv(cand_file, sep='\t')
    
    # 逐个减去背景全集
    for other in args.other_species.split(','):
        bg_file = os.path.join(args.kmer_db_dir, f"{other}_k{args.k}.fa")
        bg_set = set()
        with open(bg_file, 'r') as f:
            for line in f:
                bg_set.add(line.split('\t')[0].upper())
        df = df[~df['Kmer'].isin(bg_set)]
        print(f"After {other} filter, {len(df)} left.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, f"{args.species}_unique_k{args.k}_complex.txt"), sep='\t', index=False)

if __name__ == "__main__":
    main()
