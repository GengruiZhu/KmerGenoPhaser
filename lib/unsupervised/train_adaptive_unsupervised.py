#!/usr/bin/env python3
"""
自适应无监督训练 —— v1.3 CPU + GPU 自适应版

v1.2 设计原则 (保留):
  1. 零硬编码: 线程/进程数全部通过 CLI 或 env 配置
  2. 向后兼容: 不传 --num_workers 时行为与 v1.1 单进程完全一致
  3. CPU 数据并行: 多个 MKL 引擎各自跑自己的 sweet-spot, 通过 gloo
     backend 的 all_reduce 同步梯度
  4. NUMA 友好: 每个 worker 自动绑定到连续的 CPU 核子集

v1.3 (2026-04-24) 内部优化 + GPU 支持:

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
      best_state 写入前搬到 CPU, 省 GPU 显存 + 跨机器可加载。

  [PERF] 标准化换成 torch 原生
    - v1.2: sklearn.StandardScaler 单线程 + float64 中间态。
    - v1.3: torch.mean/std (unbiased=False 匹配 sklearn 行为),
      完全在 float32 上运算, 省一次大矩阵的 numpy↔torch 转换。

  [DEVICE] --device {cpu,cuda,gpu,auto} (默认 auto)
    - cpu : 强制 CPU, 等同 v1.2 行为
    - cuda: 强制 GPU, 无可用 CUDA 则报错 ('gpu' 是 'cuda' 的别名)
    - auto: 有 CUDA 则用 GPU, 否则 CPU
    模型文件 (adaptive_unsupervised_model.py) 本身是 device-agnostic,
    无需为 GPU 重新实现。

  [PRECISION] --precision {fp32,bf16,auto} (默认 auto)
    - fp32: 全精度 (CPU / 老 GPU 强制走此路径)
    - bf16: 启用 torch.autocast(bfloat16)。无 GradScaler (bf16 不需要)
    - auto: A100+ / H100 (compute capability >= 8.0) 自动开 bf16, 其余 fp32
    bf16 只影响 forward + loss 计算, 参数/优化器状态仍是 fp32。

  [MULTI-GPU]
    - 单机多卡: --num_workers=N (或 0 → auto 检测 cuda.device_count())
      每个 worker 绑定一张 GPU (torch.cuda.set_device(local_rank)).
      DDP backend 从 gloo 切到 nccl, 并传 device_ids=[local_rank]。
    - 多机多卡: 通过 torchrun (或 torch.distributed.launch) 启动。
      检测到 env LOCAL_RANK + WORLD_SIZE 时自动跳过内部 mp.spawn,
      直接使用 torchrun 设置的分布式环境。PBS/SLURM 都 OK。

  [CPU 行为不变]
    不传 --device (或 --device cpu / --device auto 但无 GPU) 时,
    行为字节级等同 v1.2: 同样的 gloo backend、CPU 亲和性绑定、
    线程配置、checkpoint 路径。
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
    在 argparse 之前先抓出 --num_workers / --num_threads / --device /
    --precision, 因为必须在 import torch 前设置 OMP/MKL 环境变量 +
    决定要不要屏蔽 CUDA_VISIBLE_DEVICES。

    优先级: CLI arg > KGP_* env > auto-detect。

    v1.3 增加:
      --device {cpu,cuda,gpu,auto}    (默认 auto; 'gpu' 是 'cuda' 的别名)
      --precision {fp32,bf16,auto}    (默认 auto)
    """
    nw = None
    nt = None
    dev = None
    prec = None
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
        if a == '--device' and i + 1 < len(argv):
            dev = argv[i + 1].lower(); i += 2; continue
        if a.startswith('--device='):
            dev = a.split('=', 1)[1].lower(); i += 1; continue
        if a == '--precision' and i + 1 < len(argv):
            prec = argv[i + 1].lower(); i += 2; continue
        if a.startswith('--precision='):
            prec = a.split('=', 1)[1].lower(); i += 1; continue
        i += 1

    if nw is None:
        nw = int(os.environ.get('KGP_NUM_WORKERS', '0'))  # 0 = auto resolve later
    if nt is None:
        env_nt = os.environ.get('KGP_NUM_THREADS')
        nt = int(env_nt) if env_nt is not None else 16
    if dev is None:
        dev = os.environ.get('KGP_DEVICE', 'auto').lower()
    if prec is None:
        prec = os.environ.get('KGP_PRECISION', 'auto').lower()

    # 规范化别名
    if dev == 'gpu':
        dev = 'cuda'
    if dev not in ('cpu', 'cuda', 'auto'):
        print(f"[WARN] unknown --device '{dev}', falling back to 'auto'",
              file=sys.stderr)
        dev = 'auto'
    if prec not in ('fp32', 'bf16', 'auto'):
        print(f"[WARN] unknown --precision '{prec}', falling back to 'auto'",
              file=sys.stderr)
        prec = 'auto'

    # torchrun / torch.distributed.launch 启动时, WORLD_SIZE 由 launcher 给,
    # 这种情况下我们不自己 mp.spawn, 也不应该让 nw 覆盖 WORLD_SIZE。
    _torchrun = 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if _torchrun:
        nw = int(os.environ['WORLD_SIZE'])  # 由 launcher 决定

    # CPU 模式下仍然沿用 v1.3 行为: nw <= 0 → 1
    # CUDA/auto 下 nw <= 0 保留 0, 延迟到 import torch 后用 cuda.device_count() 决定
    if dev == 'cpu':
        nw = max(1, nw)

    total = _detect_cpu_count()
    # CPU 模式或已确定 nw 的情况下, 计算线程分配
    if nt == 0 and nw > 0:
        nt = max(1, total // nw)
    elif nt < 0:
        nt = total
    else:
        nt = max(1, nt)

    return nw, nt, total, dev, prec, _torchrun


_NUM_WORKERS, _NUM_THREADS, _CPU_TOTAL, _DEVICE_REQ, _PRECISION_REQ, _TORCHRUN = \
    _early_parse_parallelism()

if _DEVICE_REQ == 'cpu' and _NUM_WORKERS * _NUM_THREADS > _CPU_TOTAL:
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

# v1.3: 只在明确 CPU 模式下屏蔽 CUDA。auto/cuda 模式保留用户 (或 PBS/SLURM)
# 预设的 CUDA_VISIBLE_DEVICES, 不去干扰。
if _DEVICE_REQ == 'cpu':
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
#  v1.3: device / precision 解析
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_device(device_req: str) -> str:
    """
    把 CLI 的 --device 值解析成 'cpu' 或 'cuda'。
      - 'cpu'  → 'cpu'
      - 'cuda' → 'cuda' (无 CUDA 时报错)
      - 'auto' → 'cuda' if available else 'cpu'
    """
    if device_req == 'cpu':
        return 'cpu'
    if device_req == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda 但 torch.cuda.is_available() 是 False。"
                "请检查 (a) pytorch 是否是 CUDA 版本, "
                "(b) CUDA_VISIBLE_DEVICES / nvidia-smi, "
                "(c) 或者改用 --device auto / --device cpu。"
            )
        return 'cuda'
    # auto
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _detect_bf16_supported() -> bool:
    """A100+ / H100 (compute capability >= 8.0) 支持原生 bf16 矩阵乘。"""
    if not torch.cuda.is_available():
        return False
    try:
        if not torch.cuda.is_bf16_supported():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


def _resolve_precision(precision_req: str, device: str) -> str:
    """
    把 CLI 的 --precision 值解析成 'fp32' 或 'bf16'。
      - CPU 强制 fp32
      - fp32 → fp32
      - bf16 → bf16 (不支持时降级 fp32 并警告)
      - auto → bf16 if A100+ else fp32
    """
    if device != 'cuda':
        if precision_req == 'bf16':
            print("[WARN] --precision bf16 在 CPU 上无效, 降级为 fp32。",
                  file=sys.stderr)
        return 'fp32'
    if precision_req == 'fp32':
        return 'fp32'
    if precision_req == 'bf16':
        if not _detect_bf16_supported():
            print("[WARN] --precision bf16 但 GPU 不支持 (需 compute capability "
                  ">= 8.0 / A100+ 级别), 降级为 fp32。", file=sys.stderr)
            return 'fp32'
        return 'bf16'
    # auto
    return 'bf16' if _detect_bf16_supported() else 'fp32'


def _resolve_world_size(device: str, user_nw: int) -> int:
    """
    根据 device 和用户指定的 --num_workers 决定最终 world_size。
      - CPU: user_nw 原样 (>=1)
      - CUDA: user_nw<=0 → 使用全部可见 GPU; user_nw>N_gpu → 截断到 N_gpu
      - torchrun 启动时外部已经定好 WORLD_SIZE, 此时 user_nw == WORLD_SIZE
    """
    if device == 'cpu':
        return max(1, user_nw)
    # cuda
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("device=cuda 但 torch.cuda.device_count()==0")
    if user_nw <= 0:
        return n_gpu
    if user_nw > n_gpu:
        print(f"[WARN] --num_workers={user_nw} > 可见 GPU 数 ({n_gpu}), "
              f"截断为 {n_gpu}。", file=sys.stderr)
        return n_gpu
    return user_nw


# torchrun / torch.distributed.launch 检测
def _is_torchrun_launched() -> bool:
    return _TORCHRUN


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


def _configure_torch_runtime(num_threads: int, device: str = 'cpu'):
    """
    在本进程内配置 torch 的线程和 JIT fusion。

    v1.3: set_num_interop_threads 只能在任何 op dispatch 之前调用一次。
    部分 import 顺序下 PyTorch 内部已经初始化, 直接调用会抛
    RuntimeError。用 try/except 包裹, 已初始化时静默继续。

    v1.3: GPU 模式下不设 CPU 线程数 (PyTorch 默认配置即可), 也不需要
    mkldnn / onednn_fusion 这些 CPU 后端的优化开关。
    """
    if device == 'cuda':
        # GPU 模式: 显式开 TF32 (Ampere+), 以及 cudnn benchmark
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True   # 动态 shape 少, 受益
        except Exception:
            pass
        return

    # CPU 模式: 沿用 v1.3 配置
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

    v1.3 新增:
      - device / use_bf16 字段, 用于 GPU 路径 + 自动混合精度。
        CPU 模式下 device='cpu', use_bf16=False, 行为与 v1.3 一致。
    """
    def __init__(self, model, optimizer, scheduler, args,
                 rank: int = 0, world_size: int = 1,
                 device: str = 'cpu', use_bf16: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.losses = AdaptiveLosses()
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.device = device
        self.use_bf16 = use_bf16
        # torch.device 对象, 给 broadcast / tensor 创建用
        if device == 'cuda':
            # local_rank == rank (单机多卡) 或 int(os.environ['LOCAL_RANK']) (多机)
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            self._torch_device = torch.device(f'cuda:{local_rank}')
        else:
            self._torch_device = torch.device('cpu')

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

        v1.3: 当 self.use_bf16=True 时, 用 torch.autocast('cuda', bfloat16)
        包裹 forward + loss 计算。bf16 不需要 GradScaler (动态范围等同 fp32)。
        参数 / 优化器状态仍是 fp32, 只有 activation / 部分 matmul 降精度。
        """
        weights = self.phase_configs[phase]['weights']

        x_aug = add_noise_augmentation(x, noise_level=0.03)
        t   = torch.rand(x.size(0), 1, device=x.device)
        z_0 = torch.randn(x.size(0), self.args.latent_dim, device=x.device)

        # v1.3: autocast 上下文。CPU / fp32 时是 nullcontext, 开销为零。
        if self.use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda',
                                          dtype=torch.bfloat16)
        else:
            import contextlib
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
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
        """只在 rank 0 调用。

        v1.3: z_all 在 GPU 训练时是 cuda tensor, 必须先 .cpu() 才能 .numpy()。
        """
        underlying = _unwrap(self.model)
        underlying.eval()
        chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]

        with torch.inference_mode():
            _, z_all = underlying(X_tensor)
            z_np = z_all.detach().cpu().numpy()

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
        """rank 0 决定是否早停, 广播到所有 ranks.

        v1.3: NCCL backend 要求 tensor 在 GPU 上; gloo 则要求在 CPU。
        用 self._torch_device 统一处理。
        """
        if self.world_size <= 1:
            return should_stop
        flag = torch.tensor([1 if should_stop else 0], dtype=torch.int32,
                            device=self._torch_device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def train(self, X_tensor, window_ids):
        self._log("\n" + "=" * 80)
        self._log("ADAPTIVE UNSUPERVISED TRAINING  (v1.3)")
        self._log(f"Device        : {self.device}"
                  + (f" (bf16 autocast)" if self.use_bf16 else " (fp32)"))
        self._log(f"World size    : {self.world_size} worker(s)")
        if self.device == 'cpu':
            self._log(f"Threads/wkr   : {torch.get_num_threads()} (intra) / "
                      f"{torch.get_num_interop_threads()} (inter)")
        else:
            self._log(f"Torch device  : {self._torch_device}")
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
                        # v1.3: 显式 .cpu(), 这样:
                        #   (a) 不占 GPU 显存 (有 n_epoch 次 NEW BEST 就有 n 份)
                        #   (b) 存到磁盘后跨机器加载不强制要求相同 GPU
                        self.best_state = {
                            k: v.detach().cpu().clone()
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

    # v1.3: device / precision 解析 (_DEVICE_REQ 已经在 _early_parse 里拿到,
    # 但需要先 import torch 完成后再判断 cuda 可用性, 所以这里再解析一次)
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    use_bf16 = (precision == 'bf16')

    _configure_torch_runtime(args.num_threads, device=device)

    # GPU 模式下绑定本进程到特定 GPU (DDP NCCL 要求)
    # torchrun 时 LOCAL_RANK 由 launcher 设置; 单机 mp.spawn 时 rank == local_rank
    if device == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        torch_device = torch.device(f'cuda:{local_rank}')
    else:
        torch_device = torch.device('cpu')

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

    # v1.3: 搬到目标 device。全部窗口一次性上卡是合理的 (features ~1364 dim, 窗口
    # 数 ~1e3-1e4, 总大小 <100 MB 级别, 远小于任何 GPU 的显存)。
    X_tensor = X_tensor.to(torch_device)

    chrom_ids = [wid.rpartition('_')[0] for wid in window_ids]
    unique_chroms = sorted(set(chrom_ids))

    if is_main:
        print(f"Windows     : {X_tensor.shape[0]}")
        print(f"Features    : {X_tensor.shape[1]}")
        print(f"Chromosomes : {len(unique_chroms)} → {unique_chroms}")
        print(f"Workers     : {world_size}")
        print(f"Device      : {torch_device}")
        print(f"Precision   : {precision}")
        if device == 'cpu':
            print(f"Threads/wkr : {args.num_threads}", flush=True)
        else:
            try:
                print(f"GPU name    : {torch.cuda.get_device_name(local_rank)}",
                      flush=True)
            except Exception:
                pass

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
    # v1.3: 模型搬到 device (包括 nn.Parameter 中的 alpha_pre/post/res 等标量)
    model = model.to(torch_device)

    if world_size > 1:
        # find_unused_parameters=True:
        # 我们的 compute_loss 为了避免 autograd 对 0 权重项做剪枝,
        # 采用 "if w==0: continue" 跳过这些 loss。结果就是某些 iteration
        # 里, 特定 loss 贡献的子图 (比如 z_aug 分支) 可能不参与 backward。
        # 这在单进程下无害, 但 DDP 默认要求"每步所有参数都有 grad"。
        #
        # 打开 find_unused_parameters=True 让 DDP 每步扫描实际收到梯度
        # 的参数, 容忍动态图。代价约 5-10% backward 时间, 但换取正确性。
        #
        # 虽然 DDP anchor 理论上可以连接所有输出张量, 但 1e-30 量级
        # 在 float32 下可能下溢, 保险起见保留此旗标。
        ddp_kwargs = {'find_unused_parameters': True}
        if device == 'cuda':
            # NCCL 要求显式 device_ids; 不传会导致每个 rank 都把模型复制到
            # cuda:0 上造成显存爆炸 + 慢。
            ddp_kwargs['device_ids'] = [local_rank]
            ddp_kwargs['output_device'] = local_rank
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    if is_main:
        n_params = sum(p.numel() for p in _unwrap(model).parameters())
        print(f"Total params : {n_params:,}")
        print(f"Use mHC      : {args.use_mhc}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )

    trainer = AdaptiveTrainer(model, optimizer, scheduler, args,
                              rank=rank, world_size=world_size,
                              device=device, use_bf16=use_bf16)
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
    """每个 mp.spawn 出来的子进程从这里开始 (单机多进程路径)。

    v1.3: 按 args.device 分支:
      - cpu:  gloo backend + CPU 亲和性绑定, 行为同 v1.3
      - cuda: nccl backend + cuda.set_device(local_rank), 跳过 CPU 亲和性
    """
    # 子进程中重新设置环境 (spawn 不继承父进程的 env 修改)
    nt = args.num_threads
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = str(nt)
    os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
    os.environ.setdefault('OMP_DYNAMIC', 'FALSE')

    # v1.3: 把 LOCAL_RANK 写进 env, 下游 _run_training 和 AdaptiveTrainer 会读
    os.environ['LOCAL_RANK'] = str(rank)

    # 先粗略解析 device (此处还没 import torch, 但 _DEVICE_REQ 已在模块 level)
    # 最终以 _resolve_device() (需要 torch.cuda 就绪) 为准。
    dev_req = args.device

    if dev_req == 'cpu' or (dev_req == 'auto' and not torch.cuda.is_available()):
        my_cpus = _set_worker_cpu_affinity(rank, args.num_workers, nt)
        backend = 'gloo'
        worker_tag = f"CPUs: {my_cpus if my_cpus else 'unbound'}"
    else:
        # CUDA 路径: 绑 GPU, 跳过 CPU 亲和性 (让 GPU 自己调度)
        torch.cuda.set_device(rank)
        backend = 'nccl'
        worker_tag = f"cuda:{rank} ({torch.cuda.get_device_name(rank)})"

    # ── DDP 初始化: file-based rendezvous ─────────────────────────────────
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
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=args.num_workers,
    )

    print(f"[Worker {rank}/{args.num_workers}] "
          f"backend={backend}  {worker_tag}  "
          f"rdzv={os.path.basename(rendezvous)}",
          flush=True)

    try:
        _run_training(rank=rank, world_size=args.num_workers, args=args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _torchrun_entry(args):
    """
    v1.3: 多机多卡入口 (torchrun / torch.distributed.launch 启动)。
    launcher 已经设好 RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT,
    我们不做 mp.spawn, 直接 init_process_group。
    """
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if args.device == 'cpu' or (args.device == 'auto' and
                                 not torch.cuda.is_available()):
        backend = 'gloo'
        worker_tag = f"CPU"
    else:
        torch.cuda.set_device(local_rank)
        backend = 'nccl'
        worker_tag = f"cuda:{local_rank}"

    # torchrun 的 init_method 默认就是 env:// (MASTER_ADDR/MASTER_PORT)
    dist.init_process_group(backend=backend, init_method='env://',
                            rank=rank, world_size=world_size)

    print(f"[torchrun {rank}/{world_size}] backend={backend} {worker_tag}",
          flush=True)

    try:
        _run_training(rank=rank, world_size=world_size, args=args)
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

    # ── CPU 并行 (v1.3 新参数, 零硬编码) ──
    p.add_argument('--num_workers', type=int,
                   default=_NUM_WORKERS,
                   help="Number of data-parallel worker processes. "
                        "CPU: 1 = single-process (v1.2 compatible). "
                        "CUDA: 0 or <=0 auto-detects as cuda.device_count(). "
                        "Default from env KGP_NUM_WORKERS, else auto.")
    p.add_argument('--num_threads', type=int,
                   default=_NUM_THREADS,
                   help="MKL/OMP threads per worker (CPU mode only). "
                        "0 = auto (cpu_count / num_workers), "
                        "-1 = use all available CPUs. "
                        "Default from env KGP_NUM_THREADS, else auto.")

    # ── v1.3 新参数: device & precision ──
    p.add_argument('--device', type=str,
                   default=_DEVICE_REQ,
                   choices=['cpu', 'cuda', 'gpu', 'auto'],
                   help="Compute device. "
                        "cpu = force CPU (v1.3 behavior). "
                        "cuda (or gpu) = force GPU, fail if unavailable. "
                        "auto = use GPU if available, else CPU. "
                        "Default from env KGP_DEVICE, else 'auto'.")
    p.add_argument('--precision', type=str,
                   default=_PRECISION_REQ,
                   choices=['fp32', 'bf16', 'auto'],
                   help="Numerical precision for forward/loss. "
                        "fp32 = full precision (CPU-only always). "
                        "bf16 = torch.autocast(bfloat16), no GradScaler. "
                        "auto = bf16 on A100+/H100 (compute capability >= 8.0), "
                        "else fp32. Default from env KGP_PRECISION, else 'auto'.")

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # ── v1.3: 先解析 device (此时 torch 已经 import, 可以判断 cuda 可用性) ──
    # 规范化 'gpu' → 'cuda' (argparse 已经限制 choices 为 {cpu, cuda, gpu, auto})
    if args.device == 'gpu':
        args.device = 'cuda'

    # 解析最终 device (auto → cpu/cuda)
    final_device = _resolve_device(args.device)
    # 把解析后的真实 device 写回 args, 子进程能直接用
    args.device = final_device

    # ── v1.3: world_size 决策 ──
    # torchrun 启动时 WORLD_SIZE 由 launcher 给定, 我们尊重它。
    # 否则用 _resolve_world_size 根据 device 计算 (GPU 下默认 = device_count).
    if _is_torchrun_launched():
        world_size = int(os.environ['WORLD_SIZE'])
        args.num_workers = world_size
        print(f"[torchrun mode] Launcher-provided WORLD_SIZE={world_size}",
              flush=True)
    else:
        world_size = _resolve_world_size(final_device, args.num_workers)
        args.num_workers = world_size

    # num_threads 补全 (CPU 模式才有意义)
    if final_device == 'cpu' and args.num_threads <= 0:
        args.num_threads = max(1, _CPU_TOTAL // max(1, args.num_workers))

    print("=" * 80)
    print(f"Parallelism config (v1.3):")
    print(f"  Device              : {final_device}")
    print(f"  Precision requested : {args.precision}  "
          f"(resolved at _run_training per worker)")
    if final_device == 'cuda':
        n_gpu = torch.cuda.device_count()
        print(f"  CUDA devices        : {n_gpu}")
        for i in range(n_gpu):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    print(f"  Worker processes    : {args.num_workers}")
    if final_device == 'cpu':
        print(f"  CPU cores available : {_CPU_TOTAL}")
        print(f"  Threads per worker  : {args.num_threads}")
        print(f"  Total thread budget : {args.num_workers * args.num_threads}")
    print(f"  torchrun launched   : {_is_torchrun_launched()}")
    print("=" * 80, flush=True)

    # ── torchrun 路径: 不 spawn, 直接进 _torchrun_entry ──
    if _is_torchrun_launched():
        _torchrun_entry(args)
        return

    # ── 单机路径 ──
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
        # CPU 单进程: 直接跑
        # GPU 单卡: 也走这里, _run_training 内部会搬 model/tensor 到 cuda:0
        _run_training(rank=0, world_size=1, args=args)


if __name__ == '__main__':
    main()
