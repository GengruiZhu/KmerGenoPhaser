import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os, argparse, heapq, random

# 编码逻辑
def complex_encode(kmer):
    mapping = {'A': [1.0, 1.0], 'C': [-1.0, 1.0], 'G': [1.0, -1.0], 'T': [-1.0, -1.0]}
    encoded = []
    for base in kmer:
        encoded.extend(mapping.get(base, [0.0, 0.0]))
    return np.array(encoded, dtype=np.float32)

def is_low_complexity(kmer, k):
    if len(set(kmer)) == 1: return True # AAAAA
    # 简单的二核苷酸重复检测
    if kmer[:2] * (k // 2) == kmer[:(k // 2) * 2]: return True
    return False

# 核心修正：更安全的生成器，避免 GeneratorExit 错误
def parse_kmc_txt(fasta_file, k):
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2: continue
                kmer = parts[0].upper()
                if len(kmer) != k: continue
                # yield kmer, count
                yield kmer, int(parts[1])
    except GeneratorExit:
        # 显式允许生成器关闭
        return
    except Exception as e:
        print(f"Warning reading {fasta_file}: {e}")
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmer_db_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--species', required=True)
    parser.add_argument('--other_species', required=True)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--top_percent', type=float, default=0.2)
    # 兼容性参数（保留但忽略，防止报错）
    parser.add_argument('--input_dir', required=False) 
    parser.add_argument('--max_compare', required=False)
    parser.add_argument('--chunk_size', required=False)
    
    args = parser.parse_args()

    # 1. 采样背景质心
    other_centroids = []
    for spec in args.other_species.split(','):
        f = os.path.join(args.kmer_db_dir, f"{spec}_k{args.k}.fa")
        if not os.path.exists(f): 
            print(f"Warning: Background file {f} not found.")
            continue
            
        samples = []
        # 只读取前 50w 行做采样，避免全读
        count_read = 0
        for kmer, count in parse_kmc_txt(f, args.k):
            count_read += 1
            if count < 5: continue
            if len(samples) < 20000: 
                samples.append(complex_encode(kmer))
            elif random.random() < 0.05: # 蓄水池采样
                samples[random.randint(0, 19999)] = complex_encode(kmer)
            if count_read > 500000: break # 强制截断，加速背景采样
            
        if samples: other_centroids.append(np.mean(samples, axis=0))
    
    if not other_centroids:
        print("Error: No background centroids found.")
        return

    other_centroids = np.array(other_centroids)

    # 2. 计算目标物种得分
    target_f = os.path.join(args.kmer_db_dir, f"{args.species}_k{args.k}.fa")
    print(f"Scoring {args.species} from {target_f}...")
    
    # 预估文件行数 (快速扫描)
    # 如果文件太大，直接硬编码一个最大值，或者只取前N个高频
    top_n = 2000000 # 保留前200万个
    heap = []
    
    # 预拟合 StandardScaler (只读前 10000 行)
    scaler = StandardScaler()
    sample_fit = []
    for kmer, _ in parse_kmc_txt(target_f, args.k):
        sample_fit.append(complex_encode(kmer))
        if len(sample_fit) >= 10000: break
    
    if not sample_fit:
        print("Error: Target file empty.")
        return
        
    scaler.fit(sample_fit)

    # 正式流式处理
    processed_count = 0
    for kmer, count in parse_kmc_txt(target_f, args.k):
        processed_count += 1
        if processed_count % 1000000 == 0:
            print(f"Processed {processed_count} kmers...", end='\r')

        if is_low_complexity(kmer, args.k): continue
        
        encoded = complex_encode(kmer).reshape(1, -1)
        scaled = scaler.transform(encoded)
        others_scaled = scaler.transform(other_centroids)
        
        # 欧氏距离 * log(Count)
        dist = cdist(scaled, others_scaled, metric='euclidean').min()
        score = dist * np.log1p(count) 
        
        # 维护小顶堆
        if len(heap) < top_n:
            heapq.heappush(heap, (score, kmer, count))
        elif score > heap[0][0]:
            heapq.heappushpop(heap, (score, kmer, count))

    # 3. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_p = os.path.join(args.output_dir, f"{args.species}_top_weighted_k{args.k}_complex.txt")
    
    # 堆排序输出（从大到小）
    results = sorted(heap, key=lambda x: x[0], reverse=True)
    
    with open(out_p, 'w') as f:
        f.write("FinalScore\tKmer\tCount\n") # 只有三列
        for s, k, c in results:
            f.write(f"{s:.6f}\t{k}\t{c}\n")
            
    print(f"\nDone. Saved top {len(results)} kmers to {out_p}")

if __name__ == "__main__":
    main()
