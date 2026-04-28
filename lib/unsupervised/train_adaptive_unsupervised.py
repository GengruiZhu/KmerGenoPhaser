#!/usr/bin/env python3
"""
自适应无监督训练 —— v1.3 CPU 数据并行版

v1.2 设计原则 (保留):
  1. 零硬编码: 线程/进程数全部通过 CLI 或 env 配置
  2. 向后兼容: 不传 --num_workers 时行为与 v1.2 单进程完全一致
  3. CPU 数据并行: 多个 MKL 引擎各自跑自己的 sweet-spot, 通过 gloo
     backend 的 all_reduce 同步梯度
  4. NUMA 友好: 每个 worker 自动绑定到连续的 CPU 核子集

v1.3 (2026-04-24) 内部优化 (CLI 接口不变, 与 v1.2 完全兼容):

  [BUG-FIX] Phase 3 早停语义对齐论文
    - v1.2: patience 在每次 evaluate (每 50 epoch) 时累加, default=500
      导致实际需要 500 × 50 = 25000 epoch 无提升才触发,
      而默认 total_epochs=18000, 所以早停永远不会发生。
    - v1.3: patience 按 epoch 计数, 在每次 evaluate 改善时清零。
      default 500 = "连续 500 epoch 无提升才停", 匹配论文 §4.5.5 描述。

  [ROBUSTNESS] set_num_interop_threads 容错
    - v1.2: 直接调用, 在 import 顺序下偶尔抛 RuntimeError 导致启动失败。
    - v1.3: try/except 包裹, 已初始化时静默跳过。

  [PERF] 惰性计算权重为 0 的损失
    - v1.2: Phase 1/2 中 weight=0 的 diversity/smoothness/spread 仍被计算,
      在大批量下浪费时间。
    - v1.3: 仅在对应权重 > 0 时计算, 否则使用图连通的 zero-like 张量填位。
      DDP anchor 仍然保留所有输出张量, 不影响 reducer ready-flag 一致性。

  [CHECKPOINTING] best_state 同步落盘
    - v1.2: 最优权重仅保存在 self.best_state 内存中, 训练崩溃即丢失。
    - v1.3: 每次 "NEW BEST" 时, rank 0 原子写入
        <dirname(input_pickle)>/best_checkpoint.pt
      下次重跑可以手动加载 (本版本不做自动恢复, 保持 CLI 不变)。
      写入失败不影响训练主循环, 仅打印警告。

  [PERF] 标准化换成 torch 原生
    - v1.2: sklearn.StandardScaler 单线程 + float64 中间态。
    - v1.3: torch.mean/std (unbiased=False 匹配 sklearn 行为),
      完全在 float32 上运算, 省一次大矩阵的 numpy↔torch 转换。

v1.3 (2026-04-27) 关键 BUG-FIX 收尾 —— 与 model.py 配套:

  [BUG-FIX] hyena 死代码导致 70+ 小时空转 (model.py 已修复主因)
    - 现象: DDP 日志报告所有 hyena_blocks 参数 unused, 训练 70 小时
      silhouette 不动。根因是 ManifoldConstrainedResidual.forward
      的入参 x 没被使用 —— 详见 model.py 文件头注释。
    - 此处的配套修改: 将 DDP 包装的 find_unused_parameters 从 True 改
      为 False。修复后所有参数都真实参与梯度回流, 不再需要 DDP 每步
      扫一遍参数表来找 "unused", backward 提速约 5–10%。
    - DDP anchor (1e-30 量级的 z+z_aug+pred_v 求和) 保留: 它的作用是
      在 augment weight=0 (Phase 1) 时把 z_aug 的子图也连进 total loss,
      防止 phase 切换时 DDP 看到的图结构不一致。anchor 数值上等于 0,
      不影响优化方向。

  [LOG-NOISE FIX] 默认关掉 TORCH_DISTRIBUTED_DEBUG
    - 之前的 528 MB 日志大头来自 shell 脚本里 export 的
      TORCH_DISTRIBUTED_DEBUG=DETAIL + TORCH_CPP_LOG_LEVEL=INFO,
      gloo 每次 collective 都打 ProcessGroupWrapper 指纹 + Reducer
      detail, 严重拖慢 allreduce 并把磁盘 I/O 变成瓶颈。
    - 配套的 shell 脚本已把这两行改为 OFF/ERROR, 此处仅在文档中记录。
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
    """
    在本进程内配置 torch 的线程和 JIT fusion。

    v1.3: set_num_interop_threads 只能在任何 op dispatch 之前调用一次。
    部分 import 顺序下 PyTorch 内部已经初始化, 直接调用会抛
    RuntimeError。用 try/except 包裹, 已初始化时静默继续。
    """
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(2, num_threads // 4))
    except RuntimeError:
        # interop thread pool 已经初始化; 保留当前值即可。
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
    """
    v1.3: torch 原生标准化, 替代 sklearn.StandardScaler.

    匹配 sklearn 的行为:
      - 按列 (feature-wise) 减均值除标准差
      - std 使用 population std (ddof=0), 与 sklearn 默认一致
      - std ~= 0 的列 clamp 到 1e-8, 避免除零 (sklearn 默认 with_std=True 也会这样)

    省掉一次大矩阵的 float64 中间态, 并在 float32 上完成整个过程。
    """
    X_tensor = torch.from_numpy(np.ascontiguousarray(X_np)).float()
    mean = X_tensor.mean(dim=0, keepdim=True)
    std  = X_tensor.std(dim=0, unbiased=False, keepdim=True).clamp_(min=1e-8)
    X_tensor.sub_(mean).div_(std)
    return X_tensor


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

    v1.3 变化 (外部接口不变):
      - self.patience 按 epoch 累加, 由 evaluate 在改善时清零
        (v1.2 是按 evaluate 次数累加, 导致早停永远不触发)
      - _save_checkpoint 将每次 best_state 同步落盘
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

        # v1.3: checkpoint 路径跟 input_pickle 同目录 (PROCESS_DIR)
        # shell 脚本里 PROCESS_DIR = <work_dir>/process/<species_name>/,
        # FEATURES_PKL 就在这个目录下, 所以 dirname 取对。
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
                'epochs': args.epochs // 3,
                'weights': {'recon': 0.5, 'fm': 0.2, 'diversity': 0.8,
                            'smoothness': 1.0, 'augment': 0.8, 'spread': 0.5}
            }
        }

    def _log(self, *a, **kw):
        if self.is_main:
            print(*a, **kw)

    def _save_checkpoint(self, global_epoch: int, phase: int, score: float,
                         n_clusters: int, chrom_map: dict):
        """
        v1.3: 原子写入 best checkpoint。
        失败不影响训练; 仅打印 warning, 内存中的 best_state 仍可用。
        """
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
                'version'      : '1.3',
            }, tmp_path)
            os.replace(tmp_path, self._checkpoint_path)
        except Exception as e:
            self._log(f"  [WARN] checkpoint save failed: {e}")
            # 清理可能留下的 .tmp
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def compute_loss(self, x, window_ids, phase=1):
        """
        统一前向 + 6 loss 计算 (DDP-safe 版)。

        关键点:
          1. 只对 self.model 调用一次 (确保 DDP 的 reducer 只处理一次 forward)。
          2. 所有 loss 张量必须"连在计算图上"——即使数学上为 0, 也要让
             autograd 能 traverse 到它。退化情况下 (如 batch 里只有单染色体)
             用 z.sum() * 0.0 而非 torch.tensor(0.0)。
          3. 权重为 0 的 loss **跳过** (不乘 0), 因为 `0 * tensor` 会被
             autograd 剪枝, 导致该 loss 贡献的子图 (如 z_aug 路径) 在
             不同 rank 间可能被剪掉/保留不一致, 触发
             "Expected to have finished reduction in prior iteration" 错误。
          4. 末尾加一个 tiny anchor = 1e-30 * sum(recon+z+z_aug+pred_v),
             确保所有输出张量每步都连到 total loss, 即使其他所有 loss
             的权重都是 0 也能保持 DDP 的参数 ready-flag 一致性。

        v1.3: 对于 cheap 的 pure-z-domain 损失 (diversity / smoothness /
        spread), 当权重为 0 时直接填 _zero_like(z) 占位, 跳过实际计算。
        augment loss 是 z vs z_aug 的 MSE, 与 z_aug 的 encoder forward
        本来就是要跑的 (anchor 依赖), 因此不跳, 直接算 (基本不花时间)。

        v1.3 (2026-04-27) 备注: 在 mHC.forward 修复 + find_unused=False 之后,
        anchor 不再是 "防 DDP 出错" 必需, 但仍保留——它的边际成本 ~0,
        而且让计算图在 phase 切换 (augment weight 从 0 变成 0.5 等) 时
        始终保持同样的输出依赖结构, 给将来扩展留余量。
        """
        weights = self.phase_configs[phase]['weights']

        x_aug = add_noise_augmentation(x, noise_level=0.03)
        t   = torch.rand(x.size(0), 1, device=x.device)
        z_0 = torch.randn(x.size(0), self.args.latent_dim, device=x.device)

        # ── 单次 DDP-routed forward ──
        out = self.model(x, x_aug=x_aug, t=t, z_0=z_0)
        recon  = out['recon']
        z      = out['z']
        z_aug  = out['z_aug']
        pred_v = out['pred_v']

        # ── 必算项: recon, fm, augment ──
        l_recon   = F.mse_loss(recon, x)
        target_v  = z - z_0
        l_fm      = F.mse_loss(pred_v, target_v)
        l_augment = F.mse_loss(z, z_aug)   # z_aug 本来就算了, 这里再跑一个 MSE 几乎免费

        # ── 惰性项: 仅在权重 > 0 时真算, 否则占位 ──
        # v1.3 优化: Phase 1 里 diversity/spread 权重都是 0, 跳过
        # window_ids 解析 + centroid 聚合 + pairwise distance 的开销。
        _zero = self.losses._zero_like(z)

        l_diversity  = (self.losses.diversity_loss(z, window_ids)
                        if weights['diversity']  > 0 else _zero)
        l_smoothness = (self.losses.local_smoothness_loss(z, window_ids)
                        if weights['smoothness'] > 0 else _zero)
        l_spread     = (self.losses.spread_loss(z, window_ids)
                        if weights['spread']     > 0 else _zero)

        # ── 聚合: 权重为 0 的项跳过, 不乘 0 (避免 autograd 剪枝) ──
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
            # 所有权重都是 0 (不应发生, 但防御性处理)
            total = 0.0 * l_recon  # 连图的 0

        # ── DDP anchor: 让所有 forward 输出每步都连到 total ──
        # 1e-30 量级远低于任何真实 loss 的噪声, 数值上等于什么都没加,
        # 但 autograd 会为此保留从每个输出到参数的完整 backward 路径,
        # 从而保证 DDP 在每个 rank 上看到完全一致的 ready-flag 序列。
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

    def train(self, X_tensor, window_ids):
        self._log("\n" + "=" * 80)
        self._log("ADAPTIVE UNSUPERVISED TRAINING  (v1.3)")
        self._log(f"World size    : {self.world_size} worker(s)")
        self._log(f"Threads/wkr   : {torch.get_num_threads()} (intra) / "
                  f"{torch.get_num_interop_threads()} (inter)")
        self._log(f"Effective batch: {self.world_size} × {self.args.batch_size} "
                  f"= {self.world_size * self.args.batch_size}")
        self._log(f"Checkpoint    : {self._checkpoint_path}")
        self._log(f"Patience (ep) : {self.args.early_stop_patience}  "
                  f"(v1.3: 按 epoch 计数, 改善即清零)")
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

            # Phase 切换时, 重置 patience: 新阶段的 loss 形貌可能和旧阶段完全
            # 不一样, 继承旧的 patience 会导致 Phase 3 开始时立刻触发早停。
            if self.is_main:
                self.patience = 0

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

                # ── v1.3: patience 每个 epoch 累加 (rank 0 独家维护) ──
                if self.is_main:
                    self.patience += 1

                # ── evaluate 只在 rank 0, 每 50 epoch 一次 ──
                best_marker = ""
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
                        self.patience = 0      # v1.3: 改善即清零
                        best_marker = " ★ NEW BEST"

                        # v1.3: 同步落盘, 崩溃保护
                        self._save_checkpoint(
                            global_epoch=global_epoch,
                            phase=phase,
                            score=score,
                            n_clusters=n_clusters,
                            chrom_map=chrom_map,
                        )

                    if epoch % 200 == 0 and epoch > 0:
                        self._log("  Chromosome → Subgenome:")
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

                # ── 早停: rank 0 决定, 广播给所有 rank ──
                if phase == 3:
                    should_stop = (self.is_main and
                                   self.patience >= self.args.early_stop_patience)
                    if self._broadcast_stop_flag(should_stop):
                        self._log(f"\n[Early stopping] no improvement for "
                                  f"{self.args.early_stop_patience} epochs "
                                  f"(reached at global epoch {global_epoch})",
                                  flush=True)
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

    # v1.3: torch 原生标准化, 替代 sklearn.StandardScaler
    # 结果与 sklearn 数值上在 float32 精度内一致 (unbiased=False matches ddof=0).
    X_tensor = _standardize_inplace(X)
    del X   # free the numpy buffer

    chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]
    unique_chroms = sorted(set(chrom_ids))

    if is_main:
        print(f"Windows     : {X_tensor.shape[0]}")
        print(f"Features    : {X_tensor.shape[1]}")
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
        # ── DDP 包装 ─────────────────────────────────────────────────────
        # v1.3 (2026-04-27) FIX: find_unused_parameters: True → False
        #
        # 旧注释:
        #   v1.2 时打开 find_unused_parameters=True, 用来容忍
        #   "compute_loss 在权重为 0 时跳过 loss 项" 导致部分子图不
        #   参与 backward 的情况。
        #
        # 新原因:
        #   实际上之前真正 unused 的不是被跳过的 loss 项, 而是 hyena_blocks
        #   的所有 80 个参数 —— ManifoldConstrainedResidual.forward 没用
        #   入参 x, hyena 的输出在 _encode 里被当场丢弃。已在 model.py
        #   修复 (mHC.forward 把 x 加到所有 stream 上)。
        #
        # 现在所有参数每步都收到梯度, find_unused_parameters=False:
        #   * 省掉每步 ~5–10% 的 backward 开销 (DDP 不再扫参数表);
        #   * compute_loss 里的 1e-30 anchor 仍保留, 用来确保 z_aug 路径
        #     在 augment weight=0 (Phase 1) 时也连进 total loss 的图,
        #     避免 phase 切换时图结构出现 rank 间差异。anchor 数值为 0,
        #     不影响优化方向。
        #
        # 如果将来你再扩 forward 让某些参数确实变 conditionally-unused
        # (e.g. 不同 phase 用不同 head), 把这里改回 True 即可。
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=False,
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
    print(f"Best checkpoint saved : {trainer._checkpoint_path}")
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
    p.add_argument('--early_stop_patience', type=int,   default=500,
                   help="v1.3: counted in epochs (not eval steps). "
                        "Stops when no silhouette improvement for this "
                        "many consecutive epochs.")
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
