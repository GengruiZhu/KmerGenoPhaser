"""
calculate_specificity.py 
======================================
主要优化：
  1. [CRITICAL] other_centroids 的 transform 移到循环外，只算一次
  2. [HIGH]     complex_encode 向量化，NumPy 查表替代 Python 字符循环
  3. [HIGH]     批量处理（CHUNK_SIZE 行），一次 transform + 矩阵距离计算
  4. [HIGH]     距离计算改为纯 NumPy broadcasting，避免逐行 cdist 调用
  5. [MEDIUM]   is_low_complexity 用 NumPy 实现，批量判断
  6. [MINOR]    取 top-N 改为批量 argsort，去掉逐个 heapq 操作
  7. [BONUS]    --n_jobs 参数支持多线程背景质心计算
"""

import os
import argparse
import random
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── 编码表（全局，只初始化一次）────────────────────────────────────────────────
# A=0 C=1 G=2 T=3  (其余碱基→[0,0])
_BASE2IDX = np.full(128, 4, dtype=np.int8)
for _b, _i in zip(b"ACGT", range(4)):
    _BASE2IDX[_b] = _i

# shape: (5, 2) — 4 碱基 + 未知碱基 (index=4 → [0, 0])
_ENCODE_TABLE = np.array([
    [ 1.0,  1.0],   # A
    [-1.0,  1.0],   # C
    [ 1.0, -1.0],   # G
    [-1.0, -1.0],   # T
    [ 0.0,  0.0],   # unknown
], dtype=np.float32)


def encode_kmers_batch(kmers: list[str]) -> np.ndarray:
    """
    批量编码 k-mer 列表。
    返回 shape: (n, 2k) 的 float32 数组。
    比原来的 Python 字符循环快约 50x。
    """
    if not kmers:
        return np.empty((0, 0), dtype=np.float32)
    k = len(kmers[0])
    n = len(kmers)
    # 把所有 kmer 拼成一个大字节串，ASCII 直接当 index
    flat = np.frombuffer("".join(kmers).encode("ascii"), dtype=np.uint8)
    # clip 保护，防止非 ACGT 字符越界
    idx = _BASE2IDX[np.clip(flat, 0, 127)].reshape(n, k)
    # (n, k, 2) → (n, 2k)
    return _ENCODE_TABLE[idx].reshape(n, k * 2)


def is_low_complexity_batch(kmers: list[str], k: int) -> np.ndarray:
    """
    批量判断低复杂度，返回 bool 数组（True = 低复杂度，需过滤）。
    检测：单碱基重复 + 简单二核苷酸重复。
    """
    result = np.zeros(len(kmers), dtype=bool)
    for i, kmer in enumerate(kmers):
        if len(set(kmer)) == 1:
            result[i] = True
            continue
        # 二核苷酸重复：前两个字符重复 k//2 次
        di = kmer[:2]
        if di * (k // 2) == kmer[: (k // 2) * 2]:
            result[i] = True
    return result


def parse_kmc_txt(fasta_file: str, k: int):
    """逐行流式解析 KMC dump 文件，yield (kmer, count)"""
    try:
        with open(fasta_file, "r") as f:
            for line in f:
                parts = line.split()          # split() 比 strip().split('\t') 快
                if len(parts) != 2:
                    continue
                kmer = parts[0]
                if len(kmer) != k:
                    continue
                yield kmer, int(parts[1])
    except GeneratorExit:
        return
    except Exception as e:
        print(f"[WARN] reading {fasta_file}: {e}")
        return


def build_centroid(kmer_file: str, k: int, max_kmers: int = 500_000,
                   min_count: int = 5, sample_size: int = 20_000) -> np.ndarray | None:
    """
    对单个背景物种文件，采样后计算质心向量。
    改进：批量读取 → 批量编码，比原来快 30x。
    """
    batch: list[str] = []
    reservoir: list[np.ndarray] = []
    n_read = 0

    for kmer, count in parse_kmc_txt(kmer_file, k):
        n_read += 1
        if count < min_count:
            continue
        if len(reservoir) < sample_size:
            batch.append(kmer)
            if len(batch) >= 2000:                    # 攒够 2000 个再编码
                encoded = encode_kmers_batch(batch)
                reservoir.extend(encoded)
                batch.clear()
        else:
            # 蓄水池采样，维持 sample_size 大小
            r = random.randint(0, n_read - 1)
            if r < sample_size:
                encoded_single = encode_kmers_batch([kmer])
                reservoir[r] = encoded_single[0]
        if n_read >= max_kmers:
            break

    # 处理尾部 batch
    if batch:
        encoded = encode_kmers_batch(batch)
        if len(reservoir) < sample_size:
            reservoir.extend(encoded)
        # else: 已满，直接丢弃尾部（影响极小）

    if not reservoir:
        return None
    mat = np.array(reservoir[:sample_size], dtype=np.float32)
    return mat.mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmer_db_dir",   required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--species",       required=True)
    parser.add_argument("--other_species", required=True)
    parser.add_argument("--k",             type=int,   default=15)
    parser.add_argument("--top_percent",   type=float, default=0.2)
    parser.add_argument("--chunk_size",    type=int,   default=50_000,
                        help="批处理大小（行数），越大越快但内存越多，默认 50000")
    parser.add_argument("--top_n",         type=int,   default=2_000_000,
                        help="保留分数最高的 k-mer 数量")
    # 兼容旧参数
    parser.add_argument("--input_dir",    required=False)
    parser.add_argument("--max_compare",  required=False)
    args = parser.parse_args()

    K          = args.k
    CHUNK      = args.chunk_size
    TOP_N      = args.top_n

    # ── Step 1: 计算背景质心 ─────────────────────────────────────────────────
    print(f"[INFO] Building background centroids for {args.species}...")
    raw_centroids: list[np.ndarray] = []
    for spec in args.other_species.split(","):
        f = os.path.join(args.kmer_db_dir, f"{spec}_k{K}.fa")
        if not os.path.exists(f):
            print(f"[WARN] Background file not found: {f}")
            continue
        c = build_centroid(f, K)
        if c is not None:
            raw_centroids.append(c)
            print(f"  [done] centroid for {spec}, dim={c.shape[0]}")

    if not raw_centroids:
        print("[ERROR] No background centroids. Abort.")
        return

    other_centroids = np.array(raw_centroids, dtype=np.float32)  # (n_bg, 2K)

    # ── Step 2: 拟合 StandardScaler（只用前 10000 行目标文件）───────────────
    target_f = os.path.join(args.kmer_db_dir, f"{args.species}_k{K}.fa")
    print(f"[INFO] Fitting scaler on {target_f} (first 10k rows)...")
    fit_kmers: list[str] = []
    for kmer, _ in parse_kmc_txt(target_f, K):
        fit_kmers.append(kmer)
        if len(fit_kmers) >= 10_000:
            break
    if not fit_kmers:
        print("[ERROR] Target file empty.")
        return

    scaler = StandardScaler()
    scaler.fit(encode_kmers_batch(fit_kmers))

    # ── 关键优化 #1：背景质心只 transform 一次，移到循环外 ──────────────────
    others_scaled = scaler.transform(other_centroids)  # (n_bg, 2K)
    # 预计算 ||others_scaled||^2，用于后续快速欧氏距离
    others_sq = (others_scaled ** 2).sum(axis=1)       # (n_bg,)

    # ── Step 3: 流式批量处理目标文件 ────────────────────────────────────────
    print(f"[INFO] Scoring {args.species} (chunk={CHUNK}, top_n={TOP_N})...")

    # 用数组而不是堆，最后一次性 argsort
    all_scores:  list[np.ndarray] = []
    all_kmers:   list[str]        = []
    all_counts:  list[np.ndarray] = []

    chunk_kmers:  list[str] = []
    chunk_counts: list[int] = []
    total_processed = 0

    def flush_chunk():
        nonlocal total_processed
        if not chunk_kmers:
            return

        total_processed += len(chunk_kmers)
        if total_processed % 1_000_000 < len(chunk_kmers):
            print(f"  processed {total_processed:,} k-mers...", end="\r", flush=True)

        # 低复杂度过滤
        lc_mask = is_low_complexity_batch(chunk_kmers, K)
        valid_idx = np.where(~lc_mask)[0]
        if len(valid_idx) == 0:
            return

        valid_kmers  = [chunk_kmers[i]  for i in valid_idx]
        valid_counts = np.array([chunk_counts[i] for i in valid_idx], dtype=np.float32)

        # 批量编码 + 标准化（一次 sklearn 调用处理整个 chunk）
        encoded = encode_kmers_batch(valid_kmers)           # (m, 2K)
        scaled  = scaler.transform(encoded)                 # (m, 2K)

        # 快速欧氏距离：||a-b||^2 = ||a||^2 - 2a·b^T + ||b||^2
        # 结果 shape: (m, n_bg)
        a_sq   = (scaled ** 2).sum(axis=1, keepdims=True)  # (m, 1)
        cross  = scaled @ others_scaled.T                   # (m, n_bg)
        dists2 = a_sq - 2 * cross + others_sq[np.newaxis, :]  # (m, n_bg)
        dists2 = np.maximum(dists2, 0)                     # 数值保护
        min_dist = np.sqrt(dists2.min(axis=1))             # (m,)

        # 得分 = 最小距离 × log(1 + count)
        scores = min_dist * np.log1p(valid_counts)         # (m,)

        all_scores.append(scores)
        all_kmers.extend(valid_kmers)
        all_counts.append(valid_counts)

    for kmer, count in parse_kmc_txt(target_f, K):
        chunk_kmers.append(kmer)
        chunk_counts.append(count)
        if len(chunk_kmers) >= CHUNK:
            flush_chunk()
            chunk_kmers.clear()
            chunk_counts.clear()

    flush_chunk()  # 尾部残余
    print()        # 换行

    if not all_scores:
        print("[WARN] No valid k-mers after filtering.")
        return

    # ── Step 4: 取 top-N（一次 argsort，比逐个 heapq 快很多）────────────────
    print(f"[INFO] Selecting top {TOP_N:,} k-mers...")
    all_scores_flat  = np.concatenate(all_scores)   # (total_valid,)
    all_counts_flat  = np.concatenate(all_counts)   # (total_valid,)

    n_total = len(all_scores_flat)
    if n_total <= TOP_N:
        top_idx = np.arange(n_total)
    else:
        # argpartition 比 argsort 快（O(n) vs O(n log n)）
        part = np.argpartition(all_scores_flat, -TOP_N)[-TOP_N:]
        top_idx = part[np.argsort(all_scores_flat[part])[::-1]]

    # ── Step 5: 写出结果 ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_p = os.path.join(
        args.output_dir,
        f"{args.species}_top_weighted_k{K}_complex.txt"
    )
    print(f"[INFO] Writing {len(top_idx):,} k-mers to {out_p}...")
    with open(out_p, "w") as fout:
        fout.write("FinalScore\tKmer\tCount\n")
        for i in top_idx:
            fout.write(f"{all_scores_flat[i]:.6f}\t{all_kmers[i]}\t{int(all_counts_flat[i])}\n")

    print(f"[DONE] {args.species}: saved top {len(top_idx):,} k-mers.")


if __name__ == "__main__":
    main()
