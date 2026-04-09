#!/usr/bin/env python3
"""
自适应无监督训练
自动发现亚基因组数量和结构
"""
import os
import sys
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

# FIX: Force line-buffered stdout so every print() appears immediately in PBS
#      logs / piped output. Without this, output is block-buffered and nothing
#      appears until the buffer fills (looks like the job is "frozen").
sys.stdout.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OMP_NUM_THREADS"] = "20"
torch.set_num_threads(20)


def add_noise_augmentation(x, noise_level=0.03):
    """高斯噪声增强"""
    noise = torch.randn_like(x) * noise_level
    return x + noise


def auto_determine_clusters(z, chrom_ids, method='silhouette', max_k=10):
    unique_chroms = sorted(set(chrom_ids))

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
        dist_matrix = squareform(pdist(centroids, metric='euclidean'))
        linkage_matrix = linkage(dist_matrix, method='ward')

        threshold = np.percentile(linkage_matrix[:, 2], 75)
        labels = fcluster(linkage_matrix, threshold, criterion='distance')

        n_clusters = len(np.unique(labels))
        return n_clusters, threshold

    else:
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

        self.phase_configs = {
            1: {
                'epochs': args.epochs // 3,
                'weights': {
                    'recon': 1.0, 'fm': 0.3, 'diversity': 0.0,
                    'smoothness': 0.5, 'augment': 0.0, 'spread': 0.0
                }
            },
            2: {
                'epochs': args.epochs // 3,
                'weights': {
                    'recon': 0.7, 'fm': 0.3, 'diversity': 0.5,
                    'smoothness': 0.8, 'augment': 0.5, 'spread': 0.3
                }
            },
            3: {
                'epochs': args.epochs // 3,
                'weights': {
                    'recon': 0.5, 'fm': 0.2, 'diversity': 0.8,
                    'smoothness': 1.0, 'augment': 0.8, 'spread': 0.5
                }
            }
        }

    def compute_loss(self, x, window_ids, phase=1):
        weights = self.phase_configs[phase]['weights']

        x_aug = add_noise_augmentation(x, noise_level=0.03)

        recon, z = self.model(x)
        recon_aug, z_aug = self.model(x_aug)

        l_recon      = self.losses.reconstruction_loss(recon, x)
        l_fm         = self.losses.flow_matching_loss(self.model, z)
        l_diversity  = self.losses.diversity_loss(z)
        l_smoothness = self.losses.local_smoothness_loss(z, window_ids)
        l_augment    = self.losses.augmentation_consistency_loss(z, z_aug)
        l_spread     = self.losses.spread_loss(z)

        total = (weights['recon']      * l_recon      +
                 weights['fm']         * l_fm         +
                 weights['diversity']  * l_diversity  +
                 weights['smoothness'] * l_smoothness +
                 weights['augment']    * l_augment    +
                 weights['spread']     * l_spread)

        return total, {
            'recon':      l_recon.item(),
            'fm':         l_fm.item(),
            'diversity':  l_diversity.item(),
            'smoothness': l_smoothness.item(),
            'augment':    l_augment.item(),
            'spread':     l_spread.item()
        }

    def evaluate(self, X_tensor, window_ids):
        self.model.eval()
        chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]

        with torch.no_grad():
            _, z_all = self.model(X_tensor)
            z_np = z_all.numpy()

        n_clusters, cluster_score = auto_determine_clusters(
            z_np, chrom_ids,
            method='silhouette',
            max_k=min(10, len(set(chrom_ids)))
        )

        unique_chroms = sorted(set(chrom_ids))
        centroids = []
        for chrom in unique_chroms:
            mask = np.array(chrom_ids) == chrom
            centroid = z_np[mask].mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(centroids)

        chrom_to_cluster = {chrom: cluster_labels[i]
                            for i, chrom in enumerate(unique_chroms)}

        try:
            sil = silhouette_score(centroids, cluster_labels)
        except:
            sil = 0.0

        return sil, n_clusters, chrom_to_cluster, z_np

    def train(self, X_tensor, window_ids):
        print("\n" + "=" * 80)
        print("ADAPTIVE UNSUPERVISED TRAINING")
        print("Automatically discovering subgenome structure...")
        print("=" * 80)

        # ── Track cumulative epoch count for global progress display ──────────
        total_epochs   = self.args.epochs
        elapsed_epochs = 0

        for phase in [1, 2, 3]:
            config       = self.phase_configs[phase]
            phase_epochs = config['epochs']

            print(f"\n{'=' * 80}")
            print(f"PHASE {phase}/3  |  {phase_epochs} epochs  "
                  f"(global {elapsed_epochs + 1} – {elapsed_epochs + phase_epochs} / {total_epochs})")
            print(f"Loss weights: {config['weights']}")
            print(f"{'=' * 80}")

            # ── Silhouette is expensive; cache it and refresh every 50 epochs ─
            cached_sil   = 0.0
            cached_k     = 2

            for epoch in range(phase_epochs):
                global_epoch = elapsed_epochs + epoch + 1
                self.model.train()

                idx      = torch.randperm(X_tensor.size(0))[:self.args.batch_size]
                batch_x  = X_tensor[idx]
                batch_ids = [window_ids[i] for i in idx.tolist()]

                loss, loss_dict = self.compute_loss(batch_x, batch_ids, phase)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step(loss.item())

                # FIX: Evaluate silhouette every 50 epochs (expensive operation).
                #      For every single epoch print a compact one-liner so the
                #      user can always see training is alive and progressing.
                if epoch % 50 == 0:
                    score, n_clusters, chrom_map, _ = self.evaluate(X_tensor, window_ids)
                    cached_sil = score
                    cached_k   = n_clusters

                    # Save best state
                    if score > self.best_score:
                        self.best_score     = score
                        self.best_state     = self.model.state_dict()
                        self.best_n_clusters = n_clusters
                        self.best_chrom_map  = chrom_map
                        self.patience        = 0
                        best_marker = " ★ NEW BEST"
                    else:
                        self.patience += 1
                        best_marker = ""

                    # Every 200 epochs also print chromosome → subgenome map
                    if epoch % 200 == 0 and epoch > 0:
                        print("  Chromosome → Subgenome:")
                        for chrom, cluster in sorted(chrom_map.items()):
                            print(f"    {chrom} → Subgenome {cluster}")
                else:
                    best_marker = ""

                # ── Per-epoch one-liner (always printed, always flushed) ───────
                # Format:
                #   [P1 | G  1234/100000 |  12.3%]  Loss=0.3421  Rc=0.2100
                #   Sm=0.0800  Sil=0.412(50ep)  K=3  LR=0.000300
                pct = global_epoch / total_epochs * 100
                print(
                    f"[P{phase} | G{global_epoch:6d}/{total_epochs} | {pct:5.1f}%]  "
                    f"Loss={loss.item():.4f}  "
                    f"Rc={loss_dict['recon']:.4f}  "
                    f"Sm={loss_dict['smoothness']:.4f}  "
                    f"Div={loss_dict['diversity']:.4f}  "
                    f"Sil={cached_sil:.4f}(50ep)  "
                    f"K={cached_k}  "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                    f"{best_marker}",
                    flush=True   # FIX: explicit flush ensures PBS/nohup logs update immediately
                )

                # Early stopping (phase 3 only)
                if phase == 3 and self.patience >= self.args.early_stop_patience:
                    print(f"\n[Early stopping] patience={self.args.early_stop_patience} "
                          f"reached at global epoch {global_epoch}",
                          flush=True)
                    break

            elapsed_epochs += phase_epochs

        return self.best_state, self.best_score, self.best_n_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pickle',  required=True)
    # FIX: argument name was --output_tsv in the shell script but the code
    #      referenced args.output_matrix → NameError at the very end.
    #      Accept both names; output_tsv is canonical.
    parser.add_argument('--output_tsv',    required=True,
                        dest='output_tsv',
                        help="Output distance matrix TSV path")
    parser.add_argument('--output_subgenome_json', default=None,
                        help="亚基因组分配结果 JSON 输出路径（可选）")

    # 模型参数
    parser.add_argument('--input_dim',  type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--n_streams',  type=int, default=4)
    parser.add_argument('--n_layers',   type=int, default=3)
    parser.add_argument('--use_mhc',    action='store_true', default=True)

    # 训练参数
    parser.add_argument('--epochs',               type=int,   default=18000)
    parser.add_argument('--lr',                   type=float, default=0.0003)
    parser.add_argument('--batch_size',           type=int,   default=512)
    parser.add_argument('--early_stop_patience',  type=int,   default=500)

    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80, flush=True)

    with open(args.input_pickle, 'rb') as f:
        data = pickle.load(f)

    window_ids = sorted([k for k in data.keys() if isinstance(data[k], np.ndarray)])
    X = np.vstack([data[wid] for wid in window_ids])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X)

    chrom_ids     = [wid.rpartition('_')[0] for wid in window_ids]
    unique_chroms = sorted(set(chrom_ids))

    print(f"Windows     : {X.shape[0]}")
    print(f"Features    : {X.shape[1]}")
    print(f"Chromosomes : {len(unique_chroms)} → {unique_chroms}")
    print(f"Clustering  : ADAPTIVE (auto-determined)", flush=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80, flush=True)

    model = AdaptiveUnsupervisedEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_streams=args.n_streams,
        n_layers=args.n_layers,
        use_mhc=args.use_mhc
    )

    print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Use mHC      : {args.use_mhc}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = AdaptiveTrainer(model, optimizer, scheduler, args)
    best_state, best_score, best_k = trainer.train(X_tensor, window_ids)

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80, flush=True)

    model.load_state_dict(best_state)
    score, n_clusters, chrom_map, z_all = trainer.evaluate(X_tensor, window_ids)

    print(f"\nBest Silhouette  : {best_score:.4f}")
    print(f"Discovered K     : {n_clusters}")
    print("\nChromosome → Subgenome Assignment:")

    subgenome_groups: dict = {}
    for chrom, cluster in sorted(chrom_map.items()):
        key = f"subgenome_{cluster}"
        subgenome_groups.setdefault(key, []).append(chrom)

    for sg_key in sorted(subgenome_groups.keys()):
        print(f"\n  {sg_key}: {', '.join(subgenome_groups[sg_key])}")

    if args.output_subgenome_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_subgenome_json)),
                    exist_ok=True)
        with open(args.output_subgenome_json, 'w') as f:
            json.dump(subgenome_groups, f, indent=2)
        print(f"\nSubgenome assignment saved to: {args.output_subgenome_json}",
              flush=True)

    # ── Save distance matrix ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING BLOCK DISTANCE MATRIX")
    print("=" * 80, flush=True)

    z_df       = pd.DataFrame(z_all)
    dist_block = squareform(pdist(z_df.values, metric='euclidean'))
    output_df  = pd.DataFrame(dist_block, index=window_ids, columns=window_ids)

    # FIX: was args.output_matrix (undefined) → use args.output_tsv
    output_df.to_csv(args.output_tsv, sep='\t')

    print(f"Distance matrix shape : {output_df.shape}")
    print(f"Block IDs             : {len(window_ids)}")
    print(f"Distance matrix saved : {args.output_tsv}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
