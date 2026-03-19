import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Score distribution analysis and bin-based equalization.")
    parser.add_argument('--input_file', required=True, help="The merged unique kmer file")
    parser.add_argument('--output_file', required=True, help="Final equalized kmer file")
    parser.add_argument('--min_score', type=float, default=0.5, help="Minimum specificity score threshold")
    parser.add_argument('--bin_size', type=float, default=1.0, help="Score bin size for distribution analysis")
    args = parser.parse_args()

    print(f"Reading merged file: {args.input_file}")
    # 使用流式读取或分块处理以节省内存
    df = pd.read_csv(args.input_file, sep='\t')
    
    # 1. 基础过滤
    df = df[df['FinalScore'] >= args.min_score]
    
    # 2. 划分得分区间 (Bins)
    df['Score_Bin'] = (df['FinalScore'] / args.bin_size).astype(int)
    
    # 3. 统计并打印梯度分布
    print("\nScore Gradient Distribution:")
    dist_stats = df.groupby(['Score_Bin', 'Species']).size().unstack(fill_value=0)
    print(dist_stats)

    # 4. 执行平衡采样 (Equalization)
    # 在每一个 Bin 内，取三个物种中数量最少的那个作为基准
    final_list = []
    print("\nBin-wise Equalization Progress:")
    print(f"{'Bin_Range':<12} | {'Min_Count':<10} | {'Status'}")
    print("-" * 45)

    for bin_val, group in df.groupby('Score_Bin'):
        counts = group['Species'].value_counts()
        if len(counts) < 3:
            # 如果某个物种在这个区间没有 K-mer，为了严格平衡，这个区间通常弃用或报警
            min_c = 0 
        else:
            min_c = counts.min()
        
        bin_label = f"{bin_val*args.bin_size:.1f}-{(bin_val+1)*args.bin_size:.1f}"
        
        if min_c > 0:
            # 每个物种采样相同数量
            sampled_group = group.groupby('Species').apply(lambda x: x.sample(n=min_c, random_state=42))
            final_list.append(sampled_group)
            print(f"{bin_label:<12} | {min_c:<10} | Balanced")
        else:
            print(f"{bin_label:<12} | {min_c:<10} | Skipped (Species missing)")

    if not final_list:
        print("Error: No kmers survived equalization. Try lowering --min_score.")
        return

    # 5. 合并并保存
    final_df = pd.concat(final_list).reset_index(drop=True)
    final_df[['Kmer', 'FinalScore', 'Species']].to_csv(args.output_file, sep='\t', index=False)
    
    print("-" * 45)
    print(f"Success! Total K-mers (Strictly Equalized): {len(final_df)}")
    for sp in df['Species'].unique():
        count = len(final_df[final_df['Species'] == sp])
        print(f"Verify {sp}: {count} K-mers")

if __name__ == "__main__":
    main()
