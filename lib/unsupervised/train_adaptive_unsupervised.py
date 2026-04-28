#!/usr/bin/env python3
"""
自适应无监督训练 —— v1.3.1 CPU 数据并行版

v1.2 设计原则 (保留):
  1. 零硬编码: 线程/进程数全部通过 CLI 或 env 配置
  2. 向后兼容: 不传 --num_workers 时行为与 v1.2 单进程完全一致
  3. CPU 数据并行: 多个 MKL 引擎各自跑自己的 sweet-spot, 通过 gloo
     backend 的 all_reduce 同步梯度
  4. NUMA 友好: 每个 worker 自动绑定到连续的 CPU 核子集

v1.3 (2026-04-24) 内部优化 (CLI 接口不变):
  - patience 按 epoch 计数 (旧版按 evaluate 次数, 等效要 25000 epoch 无改善
    才停, 默认 18000 epoch 训练永不触发)
  - set_num_interop_threads 容错
  - 惰性计算权重为 0 的损失
  - best_state 同步落盘 (best_checkpoint.pt)
  - 标准化换成 torch 原生

v1.3 (2026-04-27) 关键 BUG-FIX —— 与 model.py 配套:
  - hyena 死代码导致的 70+ 小时空转 (model.py 已修复主因)
  - find_unused_parameters: True → False
  - 默认关掉 TORCH_DISTRIBUTED_DEBUG (528 MB 日志噪声)

═══════════════════════════════════════════════════════════════════════════
v1.3.1 (2026-04-28) 训练动力学修复 —— 4 个真 BUG:
═══════════════════════════════════════════════════════════════════════════

  ── 现象 ────────────────────────────────────────────────────────────────
    EPOCHS=100000, LR=0.0001, early_stop_patience=1000 配置下:
      * 训练跑到 8000 epoch 还在 Phase 1 (因为 Phase 1 = 33333 epoch)
      * Pat=7949/1000 但训练不停
      * LR 已经掉到 1e-6 地板 (从 epoch ~3000 开始)
      * K=2 + 几乎所有染色体都进 Subgenome 0
      * best Sil 是 epoch 51 的 0.5674, 之后再没改善过
    用户反馈: "之前也是很高的初始 epoch, 但是早停会在 loss 不怎么动的
    时候就退出啊", 即旧版本至少早停是有用的。

  ── 根因 (4 个独立 bug) ─────────────────────────────────────────────────

    BUG 1: 早停被锁在 Phase 3
      旧 train() 里:
          if phase == 3:
              should_stop = (... patience >= early_stop_patience)
              if self._broadcast_stop_flag(should_stop):
                  break
      Phase 1/2 不管 patience 怎么涨都不停。EPOCHS=100000 时 Phase 1 占
      33333 epoch, 即使 Phase 1 的 recon loss 在 epoch 1000 就已经收敛,
      也要继续白跑 32000 epoch 才会进 Phase 2。

    BUG 2: patience 只在 silhouette 改善时清零
      Phase 1 的 loss 权重: diversity=0.0, spread=0.0, augment=0.0
      ——根本没有任何 loss 项推染色体分簇。silhouette 在 epoch 51 凭运气
      达到 0.5674, 之后 reconstruction 把 latent 摊平, silhouette 单调
      下降到 ~0.49 再也回不去。patience 从此一路涨到天荒地老。
      用户的预期 ("loss 不动就停") 才是对的: Phase 1 该看 loss, 不该
      看 silhouette。

    BUG 3: ReduceLROnPlateau 跨 phase 共享, LR 在 Phase 1 就死了
      scheduler 配置: factor=0.5, patience=200, min_lr=1e-6.
      Phase 1 的 recon loss 在 epoch 1000 收敛后, scheduler 每 200 epoch
      没改善就把 LR 砍半, 大约 epoch 3000 触底 1e-6。
      等 Phase 2 在 epoch 33334 启动、diversity/spread 突然加进来时,
      LR 已经死透, 模型根本爬不出 Phase 1 找到的局部解。Phase 3 同理。
      EPOCHS=18000 时这个问题被掩盖 (Phase 1 才 6000 epoch, LR 来不及
      死透), 一拉长就暴露。

    BUG 4: Phase 2/3 开头从 Phase 1 末态出发
      旧版本 phase 切换时: 模型继续从上一 phase 最后一步的状态开始。
      但 Phase 1 末态可能是个 reconstruction-only 摊平的烂解, 比 Phase 1
      早期 silhouette 0.5674 的状态差得多。从烂解出发, Phase 2 的
      diversity/spread 即使 LR 充足也很难恢复。

  ── 修复 ────────────────────────────────────────────────────────────────

    FIX 1: 早停在所有 phase 都能触发, 但语义分两种
      * Phase 1/2: patience 超阈值 → 推进到下一 phase (而不是终止)
      * Phase 3: patience 超阈值 → 终止整个训练
      用户原本的用法 ("loss 不动就退出") 在 Phase 3 完全保留;
      Phase 1/2 升级为更聪明的 "loss 不动就快进", 不浪费 epoch。

    FIX 2: patience 改成 loss-based + silhouette-based 双触发清零
      * 每步 patience += 1
      * 当前 loss 低于 phase 内历史最低 → patience = 0
      * silhouette 高于全局历史最高 → patience = 0
      用户 "loss 不怎么动就停" 的直觉直接落实成代码。

    FIX 3: 每个 phase 开头重置 LR + 重建 scheduler
      * lr 拉回 args.lr (Phase 2/3 也有完整 LR 预算)
      * scheduler 重建: 全新的 plateau patience 计数从 0 开始
      防止 Phase 1 把 LR 砍光后留给 Phase 2/3 一个死局。

    FIX 4: Phase 2/3 开头 restore best_state
      从已知最好的 silhouette 点出发, 而不是从 Phase 1 末态烂解出发。
      DDP 模式下 rank 0 加载 + broadcast 到其他 rank。

  ── 行为差异 (相对 v1.3) ────────────────────────────────────────────────
    * CLI 完全兼容, 无新参数
    * --early_stop_patience 默认 500, 含义: "phase 内连续 X epoch
      loss/silhouette 都不改善就推进 phase / 终止"
    * EPOCHS=100000 现在是合理的 "上限", 实际训练通常在每个 phase
      跑 1000-3000 epoch 就推进, 总用时大幅缩短
    * EPOCHS=18000 (v1.2 默认) 行为变化不大: Phase 1 该早完早完, Phase 3
      早停准确触发
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
        nt = int(env_nt) if env_nt is not None else 16

    nw = max(1, nw)

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

_NT_STR = str(_NUM_THREADS)
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = _NT_STR
os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
os.environ.setdefault('MKL_DEBUG_CPU_TYPE', '5')
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
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
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
        return None

    end = min(end, total)
    my_cpus = set(all_cpus[start:end])
    try:
        os.sched_setaffinity(0, my_cpus)
    except OSError:
        return None
    return sorted(my_cpus)


def _configure_torch_runtime(num_threads: int):
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(2, num_threads // 4))
    except RuntimeError:
        pass
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


def _standardize_inplace(X_np: np.ndarray) -> torch.Tensor:
    X_tensor = torch.from_numpy(np.ascontiguousarray(X_np)).float()
    mean = X_tensor.mean(dim=0, keepdim=True)
    std  = X_tensor.std(dim=0, unbiased=False, keepdim=True).clamp_(min=1e-8)
    X_tensor.sub_(mean).div_(std)
    return X_tensor


def _build_scheduler(optimizer):
    """
    v1.3.1: scheduler 抽出独立工厂函数, 因为现在每个 phase 开头都要重建一次。
    factor=0.5 / patience=200 / min_lr=1e-6 与 v1.2/v1.3 完全一致, 接口不变。
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )


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

    v1.3.1 状态机:
      - self.patience  : 上次 "改善" 距今的 epoch 数 (rank 0 维护)
        改善定义 = (loss 创 phase 内新低) OR (silhouette 创全局新高)
      - self.best_loss_in_phase : 当前 phase 的最低 total loss (用于 patience 判定)
        每个 phase 开头重置为 inf
      - self.best_score / self.best_state : 全局最佳 silhouette 及对应权重
        跨 phase 保留, Phase 2/3 开头会 restore best_state
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

        # 全局最佳: 跨 phase 保留, 训练结束后返回
        self.best_score = -1.0
        self.best_state = None
        self.best_n_clusters = 2
        self.best_chrom_map = {}

        # phase 内 loss 跟踪 (v1.3.1 新增)
        self.best_loss_in_phase = float('inf')
        self.patience = 0

        # checkpoint 路径
        self._checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(args.input_pickle)),
            'best_checkpoint.pt'
        )

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
                'epochs': args.epochs - 2 * (args.epochs // 3),
                'weights': {'recon': 0.5, 'fm': 0.2, 'diversity': 0.8,
                            'smoothness': 1.0, 'augment': 0.8, 'spread': 0.5}
            }
        }

    def _log(self, *a, **kw):
        if self.is_main:
            print(*a, **kw)

    def _save_checkpoint(self, global_epoch: int, phase: int, score: float,
                         n_clusters: int, chrom_map: dict):
        if not self.is_main or self.best_state is None:
            return
        tmp_path = self._checkpoint_path + '.tmp'
        try:
            torch.save({
                'state_dict'   : self.best_state,
                'score'        : float(score),
                'n_clusters'   : int(n_clusters),
                'chrom_map'    : dict(chrom_map),
                'epoch'        : int(global_epoch),
                'phase'        : int(phase),
                'version'      : '1.3.1',
            }, tmp_path)
            os.replace(tmp_path, self._checkpoint_path)
        except Exception as e:
            self._log(f"  [WARN] checkpoint save failed: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def _broadcast_model_state(self):
        """
        v1.3.1: phase 切换时, rank 0 把 best_state load 进 self.model 后,
        需要把权重广播给其他 rank, 否则 DDP 各 rank 的模型就分歧了。

        DDP 在 backward 时只 all_reduce 梯度, 不 sync 权重。所以手动 load
        state_dict 后必须显式 broadcast。
        """
        if self.world_size <= 1:
            return
        for p in self.model.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.model.buffers():
            dist.broadcast(b.data, src=0)

    def compute_loss(self, x, window_ids, phase=1):
        """
        统一前向 + 6 loss 计算 (DDP-safe 版)。详见 v1.3 注释。
        v1.3.1 没有改这个函数, 只是它的返回值 total 现在会被外面用来
        驱动 patience 状态机。
        """
        weights = self.phase_configs[phase]['weights']

        x_aug = add_noise_augmentation(x, noise_level=0.03)
        t   = torch.rand(x.size(0), 1, device=x.device)
        z_0 = torch.randn(x.size(0), self.args.latent_dim, device=x.device)

        out = self.model(x, x_aug=x_aug, t=t, z_0=z_0)
        recon  = out['recon']
        z      = out['z']
        z_aug  = out['z_aug']
        pred_v = out['pred_v']

        l_recon   = F.mse_loss(recon, x)
        target_v  = z - z_0
        l_fm      = F.mse_loss(pred_v, target_v)
        l_augment = F.mse_loss(z, z_aug)

        _zero = self.losses._zero_like(z)

        l_diversity  = (self.losses.diversity_loss(z, window_ids)
                        if weights['diversity']  > 0 else _zero)
        l_smoothness = (self.losses.local_smoothness_loss(z, window_ids)
                        if weights['smoothness'] > 0 else _zero)
        l_spread     = (self.losses.spread_loss(z, window_ids)
                        if weights['spread']     > 0 else _zero)

        items = [
            ('recon',      weights['recon'],      l_recon),
            ('fm',         weights['fm'],         l_fm),
            ('diversity',  weights['diversity'],  l_diversity),
            ('smoothness', weights['smoothness'], l_smoothness),
            ('augment',    weights['augment'],    l_augment),
            ('spread',     weights['spread'],     l_spread),
        ]
        total = None
        for _name, w, val in items:
            if w == 0.0:
                continue
            term = w * val
            total = term if total is None else (total + term)

        if total is None:
            total = 0.0 * l_recon

        anchor = 1e-30 * (recon.sum() + z.sum() + z_aug.sum() + pred_v.sum())
        total = total + anchor

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

    def _reset_phase_state(self, phase: int):
        """
        v1.3.1: phase 边界状态重置。在每个 phase 进入 epoch 循环前调用。

        操作:
          1. (phase >= 2) restore best_state: 从已知最好 silhouette 点出发
          2. (phase >= 2) LR 拉回 args.lr
          3. (phase >= 2) 重建 ReduceLROnPlateau scheduler
          4. patience / best_loss_in_phase 清零

        DDP 安全性: rank 0 load_state_dict 后必须 broadcast 给其他 rank,
        否则 DDP backward 时各 rank 模型分歧 → 梯度 all_reduce 后行为不
        确定 (虽然不会立刻报错)。
        """
        if phase >= 2:
            if self.is_main and self.best_state is not None:
                _unwrap(self.model).load_state_dict(self.best_state)
                self._log(f"  ↻ Restored best model from history "
                          f"(Sil={self.best_score:.4f})")
            elif self.is_main:
                self._log(f"  ↻ No best_state yet (Phase 1 found nothing); "
                          f"continuing from current weights")

            # DDP: 把 rank 0 的权重同步给其他 rank
            self._broadcast_model_state()

            # v1.3.1: best_state 恢复后 AdamW 动量是 Phase 末态的脏数据,
            # 指向旧权重的梯度方向。清零 exp_avg/exp_avg_sq 让新 phase 从干净动量出发。
            if self.is_main and self.best_state is not None:
                for state in self.optimizer.state.values():
                    for k in list(state.keys()):
                        if isinstance(state[k], torch.Tensor):
                            state[k].zero_()

            # LR 重置 (所有 rank 都做, 因为 optimizer state 是 per-rank 的)
            new_lr = self.args.lr
            for g in self.optimizer.param_groups:
                g['lr'] = new_lr
            # scheduler 重建: 全新 plateau patience
            self.scheduler = _build_scheduler(self.optimizer)
            self._log(f"  ↻ LR reset to {new_lr:.6f}, scheduler rebuilt")

        if self.is_main:
            self.patience = 0
            self.best_loss_in_phase = float('inf')

    def train(self, X_tensor, window_ids):
        self._log("\n" + "=" * 80)
        self._log("ADAPTIVE UNSUPERVISED TRAINING  (v1.3.1)")
        self._log(f"World size    : {self.world_size} worker(s)")
        self._log(f"Threads/wkr   : {torch.get_num_threads()} (intra) / "
                  f"{torch.get_num_interop_threads()} (inter)")
        self._log(f"Effective batch: {self.world_size} × {self.args.batch_size} "
                  f"= {self.world_size * self.args.batch_size}")
        self._log(f"Checkpoint    : {self._checkpoint_path}")
        self._log(f"Patience      : {self.args.early_stop_patience} epochs "
                  f"(per-phase, loss-based + silhouette-based)")
        self._log(f"Early stop    : Phase 1/2 → advance phase, Phase 3 → terminate")
        self._log("=" * 80, flush=True)

        total_epochs = self.args.epochs
        elapsed = 0
        terminate_training = False

        for phase in [1, 2, 3]:
            config = self.phase_configs[phase]
            phase_epochs = config['epochs']

            self._log(f"\n{'=' * 80}")
            self._log(f"PHASE {phase}/3  |  {phase_epochs} epochs  "
                      f"(global {elapsed + 1} – {elapsed + phase_epochs} / {total_epochs})")
            self._log(f"Loss weights: {config['weights']}")
            self._log(f"{'=' * 80}", flush=True)

            # ── v1.3.1: phase 进入前重置 LR / scheduler / patience ──
            self._reset_phase_state(phase)

            cached_sil = 0.0
            cached_k = 2
            epochs_run_in_phase = 0

            for epoch in range(phase_epochs):
                global_epoch = elapsed + epoch + 1
                epochs_run_in_phase = epoch + 1
                self.model.train()

                g = torch.Generator()
                g.manual_seed(global_epoch * 10007 + self.rank * 31 + phase)
                idx = torch.randperm(X_tensor.size(0), generator=g)[:self.args.batch_size]
                batch_x = X_tensor[idx]
                batch_ids = [window_ids[i] for i in idx.tolist()]

                loss, loss_dict = self.compute_loss(batch_x, batch_ids, phase)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step(loss.item())

                # ── v1.3.1: patience 状态机 (rank 0 维护) ──
                # 每 epoch 默认 +1; loss 创 phase 内新低 → 清零
                if self.is_main:
                    self.patience += 1
                    cur_loss = loss.item()
                    if cur_loss < self.best_loss_in_phase:
                        self.best_loss_in_phase = cur_loss
                        self.patience = 0   # loss 改善: 重置

                # ── evaluate 只在 rank 0, 每 50 epoch 一次 ──
                best_marker = ""
                if self.is_main and (epoch % 50 == 0):
                    score, n_clusters, chrom_map, _ = self.evaluate(X_tensor, window_ids)
                    cached_sil = score
                    cached_k = n_clusters

                    if score > self.best_score:
                        self.best_score = score
                        self.best_state = {
                            k: v.detach().clone()
                            for k, v in _unwrap(self.model).state_dict().items()
                        }
                        self.best_n_clusters = n_clusters
                        self.best_chrom_map = chrom_map
                        self.patience = 0     # silhouette 改善: 也重置
                        best_marker = " ★ NEW BEST"

                        self._save_checkpoint(
                            global_epoch=global_epoch,
                            phase=phase,
                            score=score,
                            n_clusters=n_clusters,
                            chrom_map=chrom_map,
                        )

                    if epoch % 200 == 0 and epoch > 0:
                        self._log("  Chromosome → Subgenome (snapshot, not final):")
                        for chrom, cluster in sorted(chrom_map.items()):
                            self._log(f"    {chrom} → Subgenome {cluster}")

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
                        f"Pat={self.patience:4d}/{self.args.early_stop_patience}  "
                        f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                        f"{best_marker}",
                        flush=True
                    )

                # ── v1.3.1: 早停在所有 phase 都能触发 ──
                # Phase 1/2 → 推进到下一 phase
                # Phase 3   → 终止整个训练
                should_stop = (self.is_main and
                               self.patience >= self.args.early_stop_patience)
                stop_flag = self._broadcast_stop_flag(should_stop)

                if stop_flag:
                    if phase < 3:
                        self._log(f"\n[Phase {phase} stalled] no loss/silhouette "
                                  f"improvement for {self.args.early_stop_patience} "
                                  f"epochs at global epoch {global_epoch}. "
                                  f"Advancing to Phase {phase + 1} early "
                                  f"(skipping {phase_epochs - epochs_run_in_phase} "
                                  f"remaining epochs of Phase {phase}).",
                                  flush=True)
                        break
                    else:
                        self._log(f"\n[Early stopping] no loss/silhouette "
                                  f"improvement for {self.args.early_stop_patience} "
                                  f"epochs in Phase 3 at global epoch {global_epoch}.",
                                  flush=True)
                        terminate_training = True
                        break

            elapsed += epochs_run_in_phase

            if terminate_training:
                break

        return self.best_state, self.best_score, self.best_n_clusters


# ═══════════════════════════════════════════════════════════════════════════
#  _run_training — 单进程和多进程共用的入口
# ═══════════════════════════════════════════════════════════════════════════
def _run_training(rank: int, world_size: int, args):
    is_main = (rank == 0)

    _configure_torch_runtime(args.num_threads)

    if is_main:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80, flush=True)

    with open(args.input_pickle, 'rb') as f:
        data = pickle.load(f)

    window_ids = sorted([k for k in data.keys() if isinstance(data[k], np.ndarray)])
    X = np.vstack([data[wid] for wid in window_ids])

    X_tensor = _standardize_inplace(X)
    del X

    chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]
    unique_chroms = sorted(set(chrom_ids))

    if is_main:
        print(f"Windows     : {X_tensor.shape[0]}")
        print(f"Features    : {X_tensor.shape[1]}")
        print(f"Chromosomes : {len(unique_chroms)} → {unique_chroms}")
        print(f"Workers     : {world_size}")
        print(f"Threads/wkr : {args.num_threads}", flush=True)

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
        # find_unused_parameters=False (v1.3 hyena fix 之后所有参数都参与回流)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=False,
        )

    if is_main:
        n_params = sum(p.numel() for p in _unwrap(model).parameters())
        print(f"Total params : {n_params:,}")
        print(f"Use mHC      : {args.use_mhc}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = _build_scheduler(optimizer)

    trainer = AdaptiveTrainer(model, optimizer, scheduler, args,
                              rank=rank, world_size=world_size)
    best_state, best_score, best_k = trainer.train(X_tensor, window_ids)

    if world_size > 1:
        dist.barrier()

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
    print(f"Best checkpoint saved : {trainer._checkpoint_path}")
    print("=" * 80, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  多进程 worker 入口
# ═══════════════════════════════════════════════════════════════════════════
def _worker_entry(rank: int, args):
    nt = args.num_threads
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = str(nt)
    os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
    os.environ.setdefault('OMP_DYNAMIC', 'FALSE')

    my_cpus = _set_worker_cpu_affinity(rank, args.num_workers, nt)

    rendezvous = os.environ.get('KGP_RENDEZVOUS')
    if not rendezvous:
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

    p.add_argument('--input_pickle', required=True)
    p.add_argument('--output_tsv', required=True, dest='output_tsv')
    p.add_argument('--output_subgenome_json', default=None)

    p.add_argument('--input_dim',  type=int, default=1024)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--latent_dim', type=int, default=20)
    p.add_argument('--n_streams',  type=int, default=4)
    p.add_argument('--n_layers',   type=int, default=3)
    p.add_argument('--use_mhc',    action='store_true', default=True)

    p.add_argument('--epochs',              type=int,   default=18000)
    p.add_argument('--lr',                  type=float, default=0.0003)
    p.add_argument('--batch_size',          type=int,   default=512,
                   help="Per-worker micro-batch size. "
                        "Effective batch = num_workers × batch_size")
    p.add_argument('--early_stop_patience', type=int,   default=500,
                   help="v1.3.1: per-phase patience in epochs. "
                        "Counts epochs since last loss minimum (within phase) "
                        "or silhouette maximum (global). On exceed: "
                        "Phase 1/2 advances to next phase, Phase 3 terminates.")
    p.add_argument('--seed',                type=int,   default=42)

    p.add_argument('--num_workers', type=int,
                   default=_NUM_WORKERS,
                   help="Number of data-parallel worker processes. "
                        "1 = single-process. "
                        "Default from env KGP_NUM_WORKERS, else 1.")
    p.add_argument('--num_threads', type=int,
                   default=_NUM_THREADS,
                   help="MKL/OMP threads per worker. "
                        "0 = auto, -1 = use all available CPUs.")

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

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
            if os.path.exists(rdz_path):
                try:
                    os.unlink(rdz_path)
                except OSError:
                    pass
    else:
        _run_training(rank=0, world_size=1, args=args)


if __name__ == '__main__':
    main()

