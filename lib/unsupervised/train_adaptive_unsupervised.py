#!/usr/bin/env python3
"""
自适应无监督训练 —— v1.2 CPU 数据并行版

设计原则:
  1. 零硬编码: 线程/进程数全部通过 CLI 或 env 配置
  2. 向后兼容: 不传 --num_workers 时行为与 v1.2 单进程完全一致
  3. CPU 数据并行: 多个 MKL 引擎各自跑自己的 sweet-spot, 通过 gloo
     backend 的 all_reduce 同步梯度, 等价于 effective_batch_size =
     num_workers × batch_size 的 DDP 训练
  4. NUMA 友好: 每个 worker 自动绑定到连续的 CPU 核子集, 减少跨 socket 抖动

用法:
  # 单进程 (默认, 等价 v1.2):
  python train_adaptive_unsupervised.py --input_pickle X --output_tsv Y ...

  # 4 worker × 16 线程 (= 64 核, 适合 2-socket AMD EPYC):
  python train_adaptive_unsupervised.py --num_workers 4 --num_threads 16 ...

  # 自动线程分配 (cpu_count / num_workers):
  python train_adaptive_unsupervised.py --num_workers 4 --num_threads 0 ...

  # 通过环境变量 (shell 脚本里方便):
  KGP_NUM_WORKERS=4 KGP_NUM_THREADS=16 python train_adaptive_unsupervised.py ...

注意事项:
  - 使用 --num_workers N 时, 有效 batch_size = N × --batch_size
    训练动力学等价于单进程 + N 倍 batch, 步数不变但每步更稳
  - 如果 num_workers × num_threads > cpu_count, 会打印警告 (过订阅)
    但允许运行 (某些场景 IO-bound 过订阅可以更快)
"""

# ═══════════════════════════════════════════════════════════════════════════
#  阶段 0: 早期解析 (必须在 import torch 之前)
# ═══════════════════════════════════════════════════════════════════════════
import os
import sys


def _detect_cpu_count() -> int:
    """当前进程可用的 CPU 数量 (respects taskset/numactl)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        import multiprocessing
        return multiprocessing.cpu_count()


def _early_parse_parallelism():
    """
    在 argparse 之前先抓出 --num_workers / --num_threads,
    因为必须在 import torch 前设置 OMP/MKL 环境变量。

    优先级: CLI arg > KGP_NUM_* env > auto-detect。
    """
    nw = None
    nt = None
    argv = sys.argv
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--num_workers' and i + 1 < len(argv):
            nw = int(argv[i + 1]); i += 2; continue
        if a.startswith('--num_workers='):
            nw = int(a.split('=', 1)[1]); i += 1; continue
        if a == '--num_threads' and i + 1 < len(argv):
            nt = int(argv[i + 1]); i += 2; continue
        if a.startswith('--num_threads='):
            nt = int(a.split('=', 1)[1]); i += 1; continue
        i += 1

    if nw is None:
        nw = int(os.environ.get('KGP_NUM_WORKERS', '1'))
    if nt is None:
        env_nt = os.environ.get('KGP_NUM_THREADS')
        # 默认 16 线程/worker. 这是 argparse 意义上的 default,
        # 不是硬编码上限 —— CLI --num_threads / env KGP_NUM_THREADS 随时覆盖。
        # 选 16 的理由: AMD Zen3 / Intel Ice Lake 上实测 sweet spot, 能跑满
        # 单 CCX 的 L3 cache 又不触发跨 NUMA penalty.
        nt = int(env_nt) if env_nt is not None else 16

    nw = max(1, nw)

    # nt == 0 → auto (cpu_count / num_workers); nt < 0 → all cores
    total = _detect_cpu_count()
    if nt == 0:
        nt = max(1, total // nw)
    elif nt < 0:
        nt = total

    return nw, nt, total


_NUM_WORKERS, _NUM_THREADS, _CPU_TOTAL = _early_parse_parallelism()

if _NUM_WORKERS * _NUM_THREADS > _CPU_TOTAL:
    print(
        f"[WARN] Oversubscription: {_NUM_WORKERS} workers × {_NUM_THREADS} "
        f"threads = {_NUM_WORKERS * _NUM_THREADS} > {_CPU_TOTAL} CPUs available. "
        f"This may hurt performance due to context switching.",
        file=sys.stderr
    )

# ── 设置环境变量 (在 import torch 之前!) ──
_NT_STR = str(_NUM_THREADS)
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = _NT_STR
os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
os.environ.setdefault('MKL_DEBUG_CPU_TYPE', '5')   # AMD-friendly
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ═══════════════════════════════════════════════════════════════════════════
#  阶段 1: 正常 imports
# ═══════════════════════════════════════════════════════════════════════════
import json
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from adaptive_unsupervised_model import (
    AdaptiveUnsupervisedEncoder,
    AdaptiveLosses,
)

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CPU 亲和性 & torch runtime 配置
# ═══════════════════════════════════════════════════════════════════════════
def _set_worker_cpu_affinity(rank: int, world_size: int, threads_per_worker: int):
    """
    将当前进程绑定到 CPU 列表的一个连续子集。

    在 2-socket × 8-NUMA 的 AMD EPYC 上, 连续 CPU 编号往往对应同一个 NUMA,
    所以这个绑定同时起到了 NUMA locality 的作用。
    """
    if not hasattr(os, 'sched_setaffinity'):
        return None
    try:
        all_cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None

    total = len(all_cpus)
    start = rank * threads_per_worker
    end = start + threads_per_worker

    if start >= total:
        # 过订阅: rank 超出核数, 不做绑定
        return None

    end = min(end, total)
    my_cpus = set(all_cpus[start:end])
    try:
        os.sched_setaffinity(0, my_cpus)
    except OSError:
        return None
    return sorted(my_cpus)


def _configure_torch_runtime(num_threads: int):
    """在本进程内配置 torch 的线程和 JIT fusion."""
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(2, num_threads // 4))
    torch.backends.mkldnn.enabled = True
    try:
        torch.jit.enable_onednn_fusion(True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  训练辅助
# ═══════════════════════════════════════════════════════════════════════════
def add_noise_augmentation(x, noise_level=0.03):
    noise = torch.randn_like(x) * noise_level
    return x + noise


def auto_determine_clusters(z, chrom_ids, method='silhouette', max_k=10):
    unique_chroms = sorted(set(chrom_ids))
    centroids = []
    for chrom in unique_chroms:
        mask = np.array(chrom_ids) == chrom
        centroids.append(z[mask].mean(axis=0))
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
            except Exception:
                continue
        return best_k, best_score
    elif method == 'distance_threshold':
        dist_matrix = squareform(pdist(centroids, metric='euclidean'))
        linkage_matrix = linkage(dist_matrix, method='ward')
        threshold = np.percentile(linkage_matrix[:, 2], 75)
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        return len(np.unique(labels)), threshold
    else:
        return max(2, len(centroids) // 4), 0.0


def _unwrap(model):
    """取出 DDP 包装下的裸模型."""
    return model.module if hasattr(model, 'module') else model


# ═══════════════════════════════════════════════════════════════════════════
#  AdaptiveTrainer — rank 感知
# ═══════════════════════════════════════════════════════════════════════════
class AdaptiveTrainer:
    """
    训练器。rank=0 为 main process:
      - 负责 evaluate / silhouette 计算 / best-state 跟踪 / 日志输出
      - 早停决策 → broadcast 给其他 ranks

    其他 rank 只做 forward/backward; 梯度 all_reduce 由 DDP 在 backward 里
    自动完成。
    """
    def __init__(self, model, optimizer, scheduler, args,
                 rank: int = 0, world_size: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.losses = AdaptiveLosses()
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        self.best_score = -1.0
        self.best_state = None
        self.best_n_clusters = 2
        self.best_chrom_map = {}
        self.patience = 0

        self.phase_configs = {
            1: {
                'epochs': args.epochs // 3,
                'weights': {'recon': 1.0, 'fm': 0.3, 'diversity': 0.0,
                            'smoothness': 0.5, 'augment': 0.0, 'spread': 0.0}
            },
            2: {
                'epochs': args.epochs // 3,
                'weights': {'recon': 0.7, 'fm': 0.3, 'diversity': 0.5,
                            'smoothness': 0.8, 'augment': 0.5, 'spread': 0.3}
            },
            3: {
                'epochs': args.epochs // 3,
                'weights': {'recon': 0.5, 'fm': 0.2, 'diversity': 0.8,
                            'smoothness': 1.0, 'augment': 0.8, 'spread': 0.5}
            }
        }

    def _log(self, *a, **kw):
        if self.is_main:
            print(*a, **kw)

    def compute_loss(self, x, window_ids, phase=1):
        weights = self.phase_configs[phase]['weights']

        x_aug = add_noise_augmentation(x, noise_level=0.03)

        recon, z = self.model(x)
        recon_aug, z_aug = self.model(x_aug)

        # flow_matching_loss 需要调用 velocity_net; DDP 包装下走 _unwrap
        underlying = _unwrap(self.model)

        l_recon      = self.losses.reconstruction_loss(recon, x)
        l_fm         = self.losses.flow_matching_loss(underlying, z)
        l_diversity  = self.losses.diversity_loss(z, window_ids)
        l_smoothness = self.losses.local_smoothness_loss(z, window_ids)
        l_augment    = self.losses.augmentation_consistency_loss(z, z_aug)
        l_spread     = self.losses.spread_loss(z, window_ids)

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
            'spread':     l_spread.item(),
        }

    def evaluate(self, X_tensor, window_ids):
        """只在 rank 0 调用。"""
        underlying = _unwrap(self.model)
        underlying.eval()
        chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]

        with torch.inference_mode():
            _, z_all = underlying(X_tensor)
            z_np = z_all.numpy()

        n_clusters, _ = auto_determine_clusters(
            z_np, chrom_ids,
            method='silhouette',
            max_k=min(10, len(set(chrom_ids)))
        )

        unique_chroms = sorted(set(chrom_ids))
        centroids = np.array([
            z_np[np.array(chrom_ids) == chrom].mean(axis=0)
            for chrom in unique_chroms
        ])

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(centroids)
        chrom_to_cluster = {chrom: cluster_labels[i]
                            for i, chrom in enumerate(unique_chroms)}
        try:
            sil = silhouette_score(centroids, cluster_labels)
        except Exception:
            sil = 0.0
        return sil, n_clusters, chrom_to_cluster, z_np

    def _broadcast_stop_flag(self, should_stop: bool) -> bool:
        """rank 0 决定是否早停, 广播到所有 ranks."""
        if self.world_size <= 1:
            return should_stop
        flag = torch.tensor([1 if should_stop else 0], dtype=torch.int32)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def train(self, X_tensor, window_ids):
        self._log("\n" + "=" * 80)
        self._log("ADAPTIVE UNSUPERVISED TRAINING")
        self._log(f"World size    : {self.world_size} worker(s)")
        self._log(f"Threads/wkr   : {torch.get_num_threads()} (intra) / "
                  f"{torch.get_num_interop_threads()} (inter)")
        self._log(f"Effective batch: {self.world_size} × {self.args.batch_size} "
                  f"= {self.world_size * self.args.batch_size}")
        self._log("=" * 80, flush=True)

        total_epochs = self.args.epochs
        elapsed = 0

        for phase in [1, 2, 3]:
            config = self.phase_configs[phase]
            phase_epochs = config['epochs']

            self._log(f"\n{'=' * 80}")
            self._log(f"PHASE {phase}/3  |  {phase_epochs} epochs  "
                      f"(global {elapsed + 1} – {elapsed + phase_epochs} / {total_epochs})")
            self._log(f"Loss weights: {config['weights']}")
            self._log(f"{'=' * 80}", flush=True)

            cached_sil = 0.0
            cached_k = 2

            for epoch in range(phase_epochs):
                global_epoch = elapsed + epoch + 1
                self.model.train()

                # ── 每个 rank 用不同 seed 采样不同 micro-batch ──
                g = torch.Generator()
                g.manual_seed(global_epoch * 10007 + self.rank * 31 + phase)
                idx = torch.randperm(X_tensor.size(0), generator=g)[:self.args.batch_size]
                batch_x = X_tensor[idx]
                batch_ids = [window_ids[i] for i in idx.tolist()]

                loss, loss_dict = self.compute_loss(batch_x, batch_ids, phase)

                self.optimizer.zero_grad()
                loss.backward()     # DDP 在此 all-reduce 梯度
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step(loss.item())

                # ── evaluate 只在 rank 0, 每 50 epoch 一次 ──
                if self.is_main and (epoch % 50 == 0):
                    score, n_clusters, chrom_map, _ = self.evaluate(X_tensor, window_ids)
                    cached_sil = score
                    cached_k = n_clusters

                    if score > self.best_score:
                        self.best_score = score
                        # 保存裸模型 state_dict (去掉 DDP "module." 前缀)
                        self.best_state = {
                            k: v.detach().clone()
                            for k, v in _unwrap(self.model).state_dict().items()
                        }
                        self.best_n_clusters = n_clusters
                        self.best_chrom_map = chrom_map
                        self.patience = 0
                        best_marker = " ★ NEW BEST"
                    else:
                        self.patience += 1
                        best_marker = ""

                    if epoch % 200 == 0 and epoch > 0:
                        self._log("  Chromosome → Subgenome:")
                        for chrom, cluster in sorted(chrom_map.items()):
                            self._log(f"    {chrom} → Subgenome {cluster}")
                else:
                    best_marker = ""

                # ── 每步在 rank 0 打一行进度 ──
                if self.is_main:
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
                        flush=True
                    )

                # ── 早停: rank 0 决定, 广播给所有 rank ──
                if phase == 3:
                    should_stop = (self.is_main and
                                   self.patience >= self.args.early_stop_patience)
                    if self._broadcast_stop_flag(should_stop):
                        self._log(f"\n[Early stopping] patience="
                                  f"{self.args.early_stop_patience} reached at "
                                  f"global epoch {global_epoch}", flush=True)
                        break

            elapsed += phase_epochs

        return self.best_state, self.best_score, self.best_n_clusters


# ═══════════════════════════════════════════════════════════════════════════
#  _run_training — 单进程和多进程共用的入口
# ═══════════════════════════════════════════════════════════════════════════
def _run_training(rank: int, world_size: int, args):
    is_main = (rank == 0)

    _configure_torch_runtime(args.num_threads)

    # ── 加载数据 ──
    if is_main:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80, flush=True)

    with open(args.input_pickle, 'rb') as f:
        data = pickle.load(f)

    window_ids = sorted([k for k in data.keys() if isinstance(data[k], np.ndarray)])
    X = np.vstack([data[wid] for wid in window_ids])
    X = StandardScaler().fit_transform(X)
    X_tensor = torch.FloatTensor(X)

    chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]
    unique_chroms = sorted(set(chrom_ids))

    if is_main:
        print(f"Windows     : {X.shape[0]}")
        print(f"Features    : {X.shape[1]}")
        print(f"Chromosomes : {len(unique_chroms)} → {unique_chroms}")
        print(f"Workers     : {world_size}")
        print(f"Threads/wkr : {args.num_threads}", flush=True)

    # ── 模型构建 (所有 rank 同一随机种子, DDP 会再做一次广播) ──
    torch.manual_seed(args.seed)
    model = AdaptiveUnsupervisedEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_streams=args.n_streams,
        n_layers=args.n_layers,
        use_mhc=args.use_mhc,
    )

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False
        )

    if is_main:
        n_params = sum(p.numel() for p in _unwrap(model).parameters())
        print(f"Total params : {n_params:,}")
        print(f"Use mHC      : {args.use_mhc}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )

    trainer = AdaptiveTrainer(model, optimizer, scheduler, args,
                              rank=rank, world_size=world_size)
    best_state, best_score, best_k = trainer.train(X_tensor, window_ids)

    if world_size > 1:
        dist.barrier()

    # ── 最终 eval + 保存: 只在 rank 0 ──
    if not is_main:
        return

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80, flush=True)

    underlying = _unwrap(model)
    if best_state is not None:
        underlying.load_state_dict(best_state)

    score, n_clusters, chrom_map, z_all = trainer.evaluate(X_tensor, window_ids)

    print(f"\nBest Silhouette  : {best_score:.4f}")
    print(f"Discovered K     : {n_clusters}")
    print("\nChromosome → Subgenome Assignment:")

    subgenome_groups = {}
    for chrom, cluster in sorted(chrom_map.items()):
        subgenome_groups.setdefault(f"subgenome_{cluster}", []).append(chrom)
    for sg in sorted(subgenome_groups.keys()):
        print(f"\n  {sg}: {', '.join(subgenome_groups[sg])}")

    if args.output_subgenome_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_subgenome_json)),
                    exist_ok=True)
        with open(args.output_subgenome_json, 'w') as f:
            json.dump(subgenome_groups, f, indent=2)
        print(f"\nSubgenome assignment saved to: {args.output_subgenome_json}",
              flush=True)

    print("\n" + "=" * 80)
    print("SAVING BLOCK DISTANCE MATRIX")
    print("=" * 80, flush=True)

    z_df = pd.DataFrame(z_all)
    dist_block = squareform(pdist(z_df.values, metric='euclidean'))
    output_df = pd.DataFrame(dist_block, index=window_ids, columns=window_ids)
    output_df.to_csv(args.output_tsv, sep='\t')

    print(f"Distance matrix shape : {output_df.shape}")
    print(f"Block IDs             : {len(window_ids)}")
    print(f"Distance matrix saved : {args.output_tsv}")
    print("=" * 80, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  多进程 worker 入口
# ═══════════════════════════════════════════════════════════════════════════
def _worker_entry(rank: int, args):
    """每个 mp.spawn 出来的子进程从这里开始。"""
    # 子进程中重新设置环境 (spawn 不继承父进程的 env 修改)
    nt = args.num_threads
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = str(nt)
    os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
    os.environ.setdefault('OMP_DYNAMIC', 'FALSE')

    # CPU 亲和性绑定
    my_cpus = _set_worker_cpu_affinity(rank, args.num_workers, nt)

    # ── DDP 初始化: file-based rendezvous (gloo backend) ──────────────────
    # 使用文件 rendezvous 而非 TCP port, 避免多个 qsub 任务同时跑时的端口冲突。
    # 父进程已经在 main() 里设好了 KGP_RENDEZVOUS, 所有子进程通过 env 继承。
    rendezvous = os.environ.get('KGP_RENDEZVOUS')
    if not rendezvous:
        # fallback (不应走到这里, 但以防万一)
        import tempfile
        rendezvous = os.path.join(
            tempfile.gettempdir(),
            f'kgp_rdzv_{os.getppid()}_{os.getpid()}.lock'
        )

    rendezvous = os.path.abspath(rendezvous)
    init_method = f'file://{rendezvous}'

    dist.init_process_group(
        backend='gloo',
        init_method=init_method,
        rank=rank,
        world_size=args.num_workers,
    )

    print(f"[Worker {rank}/{args.num_workers}] "
          f"bound to CPUs: {my_cpus if my_cpus else 'unbound'}  "
          f"rdzv={os.path.basename(rendezvous)}",
          flush=True)

    try:
        _run_training(rank=rank, world_size=args.num_workers, args=args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════
def _build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── IO ──
    p.add_argument('--input_pickle', required=True)
    p.add_argument('--output_tsv', required=True, dest='output_tsv')
    p.add_argument('--output_subgenome_json', default=None)

    # ── 模型 ──
    p.add_argument('--input_dim',  type=int, default=1024)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--latent_dim', type=int, default=20)
    p.add_argument('--n_streams',  type=int, default=4)
    p.add_argument('--n_layers',   type=int, default=3)
    p.add_argument('--use_mhc',    action='store_true', default=True)

    # ── 训练 ──
    p.add_argument('--epochs',              type=int,   default=18000)
    p.add_argument('--lr',                  type=float, default=0.0003)
    p.add_argument('--batch_size',          type=int,   default=512,
                   help="Per-worker micro-batch size. "
                        "Effective batch = num_workers × batch_size")
    p.add_argument('--early_stop_patience', type=int,   default=500)
    p.add_argument('--seed',                type=int,   default=42)

    # ── CPU 并行 (关键新参数, 零硬编码) ──
    p.add_argument('--num_workers', type=int,
                   default=_NUM_WORKERS,
                   help="Number of data-parallel worker processes. "
                        "1 = single-process (v1.2 compatible). "
                        "Default from env KGP_NUM_WORKERS, else 1.")
    p.add_argument('--num_threads', type=int,
                   default=_NUM_THREADS,
                   help="MKL/OMP threads per worker. "
                        "0 = auto (cpu_count / num_workers), "
                        "-1 = use all available CPUs. "
                        "Default from env KGP_NUM_THREADS, else auto.")

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # 兜底: 确保最终值合法
    args.num_workers = max(1, args.num_workers)
    if args.num_threads <= 0:
        args.num_threads = max(1, _CPU_TOTAL // args.num_workers)

    print("=" * 80)
    print(f"Parallelism config:")
    print(f"  CPU cores available : {_CPU_TOTAL}")
    print(f"  Worker processes    : {args.num_workers}")
    print(f"  Threads per worker  : {args.num_threads}")
    print(f"  Total thread budget : {args.num_workers * args.num_threads}")
    print("=" * 80, flush=True)

    if args.num_workers > 1:
        # ── Rendezvous 文件管理: 每个 python 进程都是独立的 rendezvous ──
        # 优先级: KGP_RENDEZVOUS env > auto-pick (pid + 纳秒时间戳)。
        # shell 脚本建议设 KGP_RENDEZVOUS=<process_dir>/.kgp_rdzv_${PBS_JOBID:-$$}
        # 以便和 qsub 调度器联动; 不设也绝对不会冲突。
        import time
        rdz_path = os.environ.get('KGP_RENDEZVOUS')
        if not rdz_path:
            import tempfile
            rdz_path = os.path.join(
                tempfile.gettempdir(),
                f'kgp_rdzv_{os.getpid()}_{time.time_ns()}.lock'
            )
            os.environ['KGP_RENDEZVOUS'] = rdz_path
        rdz_path = os.path.abspath(rdz_path)
        os.environ['KGP_RENDEZVOUS'] = rdz_path

        # 清理可能的 stale 文件 (之前任务崩溃遗留的)
        if os.path.exists(rdz_path):
            try:
                os.unlink(rdz_path)
            except OSError:
                pass

        print(f"DDP rendezvous file : {rdz_path}", flush=True)

        try:
            mp.spawn(
                _worker_entry,
                args=(args,),
                nprocs=args.num_workers,
                join=True,
            )
        finally:
            # 训练结束后清理 rendezvous 文件
            if os.path.exists(rdz_path):
                try:
                    os.unlink(rdz_path)
                except OSError:
                    pass
    else:
        # 单进程 —— 与 v1.2 行为完全一致, 不需要 rendezvous
        _run_training(rank=0, world_size=1, args=args)


if __name__ == '__main__':
    main()
