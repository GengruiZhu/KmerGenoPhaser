#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
from Bio import SeqIO
from scipy.fftpack import fft

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', required=True)
    parser.add_argument('--output_pickle', required=True)
    parser.add_argument('--window_size', type=int, default=10000)
    args = parser.parse_args()

    # 以 window 为 key 保存特征
    window_data = {}
    print(f"--- Extracting Spectral Features for each {args.window_size}bp Window ---")
    
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        seq_len = len(record.seq)
        num_windows = seq_len // args.window_size
        print(f"Processing {record.id}...")
        
        for i in range(num_windows):
            start = i * args.window_size
            subseq = record.seq[start : start + args.window_size]
            
            # 使用 FFT 提取频谱特征 (沿用你之前的逻辑)
            # 转换为 int8 并取前 1024 点做 FFT
            feat = np.abs(fft(np.frombuffer(str(subseq).encode(), dtype=np.int8)[:1024]))
            
            window_id = f"{record.id}_{i+1:05d}"
            window_data[window_id] = feat

    with open(args.output_pickle, 'wb') as f:
        pickle.dump(window_data, f)
    print(f"Total Windows: {len(window_data)}")

if __name__ == "__main__":
    main()
