#!/usr/bin/env python3
"""
自适应无监督训练
自动发现亚基因组数量和结构
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from adaptive_unsupervised_model import (
    AdaptiveUnsupervisedEncoder,
    AdaptiveLosses
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OMP_NUM_THREADS"] = "20"
torch.set_num_threads(20)


def add_noise_augmentation(x, noise_level=0.03):
    """高斯噪声增强"""
    noise = torch.randn_like(x) * noise_level
    return x + noise


def auto_determine_clusters(z, chrom_ids, method='silhouette', max_k=10):
    """
    自动确定最佳聚类数量
    方法：
    1. silhouette: 遍历k，选择轮廓系数最高的
    2. elbow: 肘部法则
    3. gap: Gap统计量
    """
    unique_chroms = sorted(set(chrom_ids))

    # 计算每个染色体的质心
    centroids = []
    for chrom in unique_chroms:
        mask = np.array(chrom_ids) == chrom
        centroid = z[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    if method == 'silhouette':
        best_k = 2
        best_score = -1

        for k in range(2, min(max_k, len(centroids))):
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clusterer.fit_predict(centroids)

            if len(np.unique(labels)) < 2:
                continue

            try:
                score = silhouette_score(centroids, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue

        return best_k, best_score

    elif method == 'distance_threshold':
        # 使用层次聚类的距离阈值自动确定
        dist_matrix = squareform(pdist(centroids, metric='euclidean'))
        linkage_matrix = linkage(dist_matrix, method='ward')

        # 自适应阈值：取距离的75分位数
        threshold = np.percentile(linkage_matrix[:, 2], 75)
        labels = fcluster(linkage_matrix, threshold, criterion='distance')

        n_clusters = len(np.unique(labels))
        return n_clusters, threshold

    else:
        # 默认返回染色体数的1/3到1/5
        estimated_k = max(2, len(centroids) // 4)
        return estimated_k, 0.0


class AdaptiveTrainer:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.losses = AdaptiveLosses()

        self.best_score = -1.0
        self.best_state = None
        self.patience = 0

        # 三阶段训练
        self.phase_configs = {
            1: {  # 阶段1: 重构 + 平滑
                'epochs': args.epochs // 3,
                'weights': {
                    'recon': 1.0,
                    'fm': 0.3,
                    'diversity': 0.0,
                    'smoothness': 0.5,
                    'augment': 0.0,
                    'spread': 0.0
                }
            },
            2: {  # 阶段2: 增加多样性和一致性
                'epochs': args.epochs // 3,
                'weights': {
                    'recon': 0.7,
                    'fm': 0.3,
                    'diversity': 0.5,
                    'smoothness': 0.8,
                    'augment': 0.5,
                    'spread': 0.3
                }
            },
            3: {  # 阶段3: 强化结构
                'epochs': args.epochs // 3,
                'weights': {
                    'recon': 0.5,
                    'fm': 0.2,
                    'diversity': 0.8,
                    'smoothness': 1.0,
                    'augment': 0.8,
                    'spread': 0.5
                }
            }
        }

    def compute_loss(self, x, window_ids, phase=1):
        """计算总损失"""
        weights = self.phase_configs[phase]['weights']

        # 数据增强
        x_aug = add_noise_augmentation(x, noise_level=0.03)

        # 前向传播
        recon, z = self.model(x)
        recon_aug, z_aug = self.model(x_aug)

        # 各项损失
        l_recon = self.losses.reconstruction_loss(recon, x)
        l_fm = self.losses.flow_matching_loss(self.model, z)
        l_diversity = self.losses.diversity_loss(z)
        l_smoothness = self.losses.local_smoothness_loss(z, window_ids)
        l_augment = self.losses.augmentation_consistency_loss(z, z_aug)
        l_spread = self.losses.spread_loss(z)

        # 加权总损失
        total = (weights['recon'] * l_recon +
                weights['fm'] * l_fm +
                weights['diversity'] * l_diversity +
                weights['smoothness'] * l_smoothness +
                weights['augment'] * l_augment +
                weights['spread'] * l_spread)

        return total, {
            'recon': l_recon.item(),
            'fm': l_fm.item(),
            'diversity': l_diversity.item(),
            'smoothness': l_smoothness.item(),
            'augment': l_augment.item(),
            'spread': l_spread.item()
        }

    def evaluate(self, X_tensor, window_ids):
        """评估并自动确定聚类"""
        self.model.eval()
        chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]

        with torch.no_grad():
            _, z_all = self.model(X_tensor)
            z_np = z_all.numpy()

        # 自动确定聚类数量
        n_clusters, cluster_score = auto_determine_clusters(
            z_np, chrom_ids,
            method='silhouette',
            max_k=min(10, len(set(chrom_ids)))
        )

        # 使用确定的聚类数进行聚类
        unique_chroms = sorted(set(chrom_ids))
        centroids = []
        for chrom in unique_chroms:
            mask = np.array(chrom_ids) == chrom
            centroid = z_np[mask].mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(centroids)

        # 染色体 → 聚类映射
        chrom_to_cluster = {chrom: cluster_labels[i]
                           for i, chrom in enumerate(unique_chroms)}

        # 计算轮廓系数
        try:
            sil = silhouette_score(centroids, cluster_labels)
        except:
            sil = 0.0

        score = sil

        return score, n_clusters, chrom_to_cluster, z_np

    def train(self, X_tensor, window_ids):
        print("\n" + "="*80)
        print("ADAPTIVE UNSUPERVISED TRAINING")
        print("Automatically discovering subgenome structure...")
        print("="*80)

        for phase in [1, 2, 3]:
            config = self.phase_configs[phase]
            print(f"\n{'='*80}")
            print(f"PHASE {phase}/3: {config['epochs']} epochs")
            print(f"Loss weights: {config['weights']}")
            print(f"{'='*80}")

            for epoch in range(config['epochs']):
                self.model.train()

                # 随机批次
                idx = torch.randperm(X_tensor.size(0))[:self.args.batch_size]
                batch_x = X_tensor[idx]
                batch_ids = [window_ids[i] for i in idx.tolist()]

                # 计算损失
                loss, loss_dict = self.compute_loss(batch_x, batch_ids, phase)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 学习率调度
                self.scheduler.step(loss.item())

                # 评估
                if epoch % 50 == 0:
                    score, n_clusters, chrom_map, _ = self.evaluate(X_tensor, window_ids)

                    print(f"[P{phase}] Epoch {epoch:4d} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Recon: {loss_dict['recon']:.4f} | "
                          f"Smooth: {loss_dict['smoothness']:.4f} | "
                          f"Sil: {score:.4f} | "
                          f"K: {n_clusters} | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                    # 保存最佳
                    if score > self.best_score:
                        self.best_score = score
                        self.best_state = self.model.state_dict()
                        self.best_n_clusters = n_clusters
                        self.best_chrom_map = chrom_map
                        self.patience = 0
                        print(f"  → New best! Silhouette: {score:.4f}, K: {n_clusters}")

                        # 打印聚类分配
                        if epoch % 200 == 0:
                            print("  Chromosome → Subgenome:")
                            for chrom, cluster in sorted(chrom_map.items()):
                                print(f"    {chrom} → Subgenome {cluster}")
                    else:
                        self.patience += 1

                # Early stopping
                if phase == 3 and self.patience >= self.args.early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        return self.best_state, self.best_score, self.best_n_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pickle', required=True)
    parser.add_argument('--output_matrix', required=True)
    # ── 新增：亚基因组分配 JSON 输出路径（可选，不传则不输出）──
    parser.add_argument('--output_subgenome_json', default=None,
                        help="亚基因组分配结果 JSON 输出路径（供后续脚本读取）")

    # 模型参数
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--use_mhc', action='store_true', default=True)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=18000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--early_stop_patience', type=int, default=500)

    args = parser.parse_args()

    # 加载数据
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    with open(args.input_pickle, 'rb') as f:
        data = pickle.load(f)

    window_ids = sorted([k for k in data.keys() if isinstance(data[k], np.ndarray)])
    X = np.vstack([data[wid] for wid in window_ids])

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X)

    chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]
    unique_chroms = sorted(set(chrom_ids))

    print(f"Windows: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Chromosomes: {len(unique_chroms)} → {unique_chroms}")
    print(f"Clustering: ADAPTIVE (auto-determined)")

    # 模型
    print("\n" + "="*80)
    print("MODEL INITIALIZATION")
    print("="*80)

    model = AdaptiveUnsupervisedEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_streams=args.n_streams,
        n_layers=args.n_layers,
        use_mhc=args.use_mhc
    )

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Use mHC: {args.use_mhc}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )

    # 训练
    trainer = AdaptiveTrainer(model, optimizer, scheduler, args)
    best_state, best_score, best_k = trainer.train(X_tensor, window_ids)

    # 最终评估
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    model.load_state_dict(best_state)
    score, n_clusters, chrom_map, z_all = trainer.evaluate(X_tensor, window_ids)

    print(f"\nBest Silhouette: {best_score:.4f}")
    print(f"Discovered Subgenomes: {n_clusters}")
    print("\nChromosome → Subgenome Assignment:")

    # 按亚基因组分组显示（key 由实际聚类 id 决定，完全动态）
    subgenome_groups = {}
    for chrom, cluster in sorted(chrom_map.items()):
        key = f"subgenome_{cluster}"   # e.g. "subgenome_0", "subgenome_1", ...
        if key not in subgenome_groups:
            subgenome_groups[key] = []
        subgenome_groups[key].append(chrom)

    for sg_key in sorted(subgenome_groups.keys()):
        chroms = subgenome_groups[sg_key]
        print(f"\n  {sg_key}: {', '.join(chroms)}")

    # ── 新增：输出亚基因组分配 JSON ──
    if args.output_subgenome_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_subgenome_json)), exist_ok=True)
        with open(args.output_subgenome_json, 'w') as f:
            json.dump(subgenome_groups, f, indent=2)
        print(f"\nSubgenome assignment saved to: {args.output_subgenome_json}")

    # 计算距离矩阵（block 级别）
    print("\n" + "="*80)
    print("SAVING BLOCK DISTANCE MATRIX")
    print("="*80)

    # 用 block 级别的 z 计算 block 间距离（与原脚本一致）
    z_df = pd.DataFrame(z_all)
    dist_block = squareform(pdist(z_df.values, metric='euclidean'))
    output_df = pd.DataFrame(dist_block, index=window_ids, columns=window_ids)
    output_df.to_csv(args.output_matrix, sep='\t')

    print(f"Distance matrix shape: {output_df.shape}")
    print(f"Block IDs: {len(window_ids)}")
    print(f"Distance matrix saved to: {args.output_matrix}")
    print(f"  - Rows/Columns: {len(window_ids)} blocks")
    print(f"  - Format: block IDs (e.g., Chr1A_Spontaneum_1)")
    print("="*80)


if __name__ == "__main__":
    main()
