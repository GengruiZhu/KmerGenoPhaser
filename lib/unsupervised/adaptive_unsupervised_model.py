#!/usr/bin/env python3
"""
自适应无监督基因组编码器
自动发现亚基因组数量和结构

v1.2 内容 (保留):
  - mHC 的 layer_fn 使用 TransformBlock (FC+LN+GELU)
  - Decoder 使用对称 mHC 架构 (论文 §3.3.3)
  - 6 个 loss 与论文公式对齐
  - 三投影融合 (3 次 GEMM → 1 次 GEMM)

v1.2 DDP 兼容性修复 (2026-04-22):
  - 摘掉 _sinkhorn_knopp / _fused_mhc_update 上的 @torch.jit.script。
    原因: PyTorch 2.6 在 DDP 包装下, JIT script 化的函数 + autograd
    backward 组合会触发 "Schema not found for node" 错误。
  - 函数保留纯 Python 实现, 在 MKL/oneDNN 下性能仍然可观。
  - 该修改与单进程模式 (num_workers=1) 完全兼容, 不改变数值行为。

v1.3 (2026-04-24) 数值稳定性改进:
  - _sinkhorn_knopp 改为 log-domain 迭代: 用 logsumexp 做行/列归一化,
    最后一次性 exp。避免 torch.exp(alpha_res * logits) 在 alpha_res
    训练中增大时产生的数值过冲。
  - 数学上与 v1.2 完全等价 (都是 Sinkhorn 投影到 Birkhoff polytope),
    在 float32 下结果差异量级 ~1e-7, 不影响训练动力学。
  - 接口 (函数签名/返回形状) 与 v1.2 完全一致, 调用方无需修改。

v1.3 (2026-04-27) 关键 BUG-FIX —— hyena_blocks 死代码问题:
  ── 现象 ────────────────────────────────────────────────────────────────
    DDP 日志里 hyena_blocks.{0..N-1} 的全部 80 个参数都被报告
    "is marked as unused", 配合 find_unused_parameters=True
    每步多扫 5–10% 时间, 但更严重的是: 模型主体的 8 层 Hyena 卷积
    根本没在学习, encoder 实际上只有 mHC 在更新。
  ── 根因 ────────────────────────────────────────────────────────────────
    ManifoldConstrainedResidual.forward(self, x, x_streams, layer_fn)
    形参 `x` 仅用于 `batch = x.size(0)` 取 batch size, 整个
    forward 体内再无第二次出现 —— 完全没接进计算图。
    而 _encode 里:
        h = hyena_block(h)              # hyena 算了一遍
        h_flat = h.squeeze(1)
        h_flat, streams = mhc(h_flat, streams, ...)  # h_flat 被无视
        h = h_flat.unsqueeze(1)         # 新 h 来自 streams.mean(), 与 hyena 无关
    hyena 的输出彻底丢失, 它的 weight 自然收不到梯度。
  ── 修复 ────────────────────────────────────────────────────────────────
    在 ManifoldConstrainedResidual.forward 开头将 x 广播加到每个 stream:
        x_streams = x_streams + x.unsqueeze(1)
    数学上的解释: 把 hyena 的输出当作所有 stream 的公共偏置,
    与论文 §3.3.2 mHC 的 "input mixing" 概念一致, 让 hyena 的特征
    真正进入 mHC 的更新方程。修复后:
      * 80 个 hyena 参数全部进入梯度回流路径
      * find_unused_parameters 可以关掉 (见 train script)
      * 数值上, 因为加法是 stream-symmetric 的, 不破坏 mHC 的
        manifold-constrained residual 性质
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 核心数值函数 (纯 Python, DDP 安全) =======
#
# 以前这两个函数带 @torch.jit.script 装饰器。但在 PyTorch 2.6 下,
# scripted 函数 + DDP + autograd backward 的组合会触发 JIT 内部的
# schema lookup 失败。去掉装饰器即彻底规避该问题。

def _sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    论文 Eq.9 的 Sinkhorn-Knopp 迭代, 20 次行/列交替归一化。

    v1.3: 改为 log-domain 实现。
      - 原 v1.2: m = exp(logits); 然后反复 m /= m.sum(-1) 和 m /= m.sum(-2)
      - v1.3:   log_m = logits; 反复 log_m -= logsumexp(log_m, -1); 最后 exp
      - 数学等价, float32 下数值更稳 (避免 exp 过冲), 返回形状/语义不变。

    输入 logits 形如 (B, n, n), 未归一化。
    输出一个近似双随机矩阵, 行列和 ≈ 1。
    """
    log_m = logits
    for _ in range(n_iters):
        # row normalize:   log(M_ij / sum_j M_ij) = log M_ij - logsumexp_j log M_ij
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
        # col normalize
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
    return torch.exp(log_m)


def _fused_mhc_update(
    x_streams: torch.Tensor,   # (B, n, C)
    h_res: torch.Tensor,       # (B, n, n)
    out: torch.Tensor,         # (B, C)      layer_fn 的输出
    h_post: torch.Tensor,      # (B, n)
) -> torch.Tensor:
    """
    论文 Eq.3:  X_{l+1} = H_res · X_l + H_post^T · F(·)
    融合矩阵乘 + 广播乘 + 加法。返回 (B, n, C) 的更新后 streams。
    """
    mixed = torch.matmul(h_res, x_streams)                # (B, n, C)
    update = out.unsqueeze(1) * h_post.unsqueeze(-1)      # (B, n, C)
    return mixed + update


# ==================== mHC 层内变换块 (论文 Eq.9 的 F(·, W_l)) ===========
class TransformBlock(nn.Module):
    """F(·, W_l):  Linear → LayerNorm → GELU → Dropout"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


# ==================== mHC残差 (融合投影版) ======================
class ManifoldConstrainedResidual(nn.Module):
    """
    Manifold-Constrained Hyper-Connection 单层 (论文 §3.3.2)。

    优化:
      - h_pre / h_post / h_res 三个独立 Linear 融合为单个 fused_proj,
        输出 (2n + n²) 维, 内部切片取三组 logits。
      - Sinkhorn 迭代和末尾融合更新用纯 Python 函数实现
        (DDP + autograd 兼容)。

    v1.3 (2026-04-27): forward 开头加入 `x_streams += x.unsqueeze(1)`
    让 layer 的输入 (encoder 里就是 hyena_block 的输出) 真正进入计算图。
    详见文件头 docstring 的 BUG-FIX 段落。
    """
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams

        fused_out_dim = 2 * n_streams + n_streams * n_streams
        self.fused_proj = nn.Linear(dim * n_streams, fused_out_dim, bias=True)

        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.sinkhorn_iters = 20
        self.norm = nn.LayerNorm(dim)

        self._n = n_streams
        self._slice_pre = slice(0, n_streams)
        self._slice_post = slice(n_streams, 2 * n_streams)
        self._slice_res = slice(2 * n_streams, 2 * n_streams + n_streams * n_streams)

    def forward(self, x, x_streams, layer_fn):
        batch = x.size(0)
        if x_streams is None:
            x_streams = x.unsqueeze(1).repeat(1, self.n_streams, 1)

        # ★ v1.3 FIX (2026-04-27):
        # 把 layer 的输入 x (encoder 路径里就是 hyena_block 的输出) 广播
        # 加到所有 stream。修复前 x 仅用于 size(0), 上游卷积/Hyena 完全
        # 不在梯度路径里, DDP 报告 hyena_blocks.* 全部 unused, 模型核心
        # 等于没在训练。
        # 选择 "加法注入" 而非"替换 stream-0"是因为:
        #   1) 保留 mHC 原本的 stream 状态, 不破坏 manifold-constrained
        #      residual 性质;
        #   2) 数值上是 O(1) 的 elementwise 操作, 几乎零开销;
        #   3) 保持所有 stream 对称, fused_proj 的对称性也保留。
        x_streams = x_streams + x.unsqueeze(1)

        x_flat = x_streams.reshape(batch, -1)
        x_norm = F.layer_norm(x_flat, [x_flat.size(-1)])

        fused_logits = self.fused_proj(x_norm)                   # (B, 2n + n²)
        h_pre_logits  = fused_logits[:, self._slice_pre]
        h_post_logits = fused_logits[:, self._slice_post]
        h_res_logits  = fused_logits[:, self._slice_res].reshape(
            batch, self._n, self._n
        )

        h_pre = torch.sigmoid(self.alpha_pre * h_pre_logits)
        h_post = 2.0 * torch.sigmoid(self.alpha_post * h_post_logits)
        h_res = _sinkhorn_knopp(
            self.alpha_res * h_res_logits, self.sinkhorn_iters
        )

        x_agg = (h_pre.unsqueeze(-1) * x_streams).sum(dim=1)      # (B, C)
        out = layer_fn(self.norm(x_agg))                          # (B, C)
        updated_streams = _fused_mhc_update(x_streams, h_res, out, h_post)

        return updated_streams.mean(dim=1), updated_streams


# ==================== Hyena卷积 =================================
class HyenaConvBlock(nn.Module):
    def __init__(self, dim, short_kernel=5, long_kernel=15, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv_short = nn.Conv1d(dim, dim, kernel_size=short_kernel,
                                    padding=short_kernel // 2, groups=dim)
        self.conv_long = nn.Conv1d(dim, dim, kernel_size=long_kernel,
                                   padding=long_kernel // 2, groups=dim)
        self.gate_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x_t = x.transpose(1, 2)
        short_feat = self.conv_short(x_t)
        long_feat = self.conv_long(x_t)
        gate_input = (short_feat + long_feat).transpose(1, 2)
        gate, value = self.gate_proj(gate_input).chunk(2, dim=-1)
        out = value * torch.sigmoid(gate)
        out = self.dropout(self.out_proj(out))
        return res + out


# ==================== 多尺度特征提取 ============================
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
            prev_dim = h_dim
        self.layers = nn.ModuleList(layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


# ==================== 自适应无监督编码器 ========================
class AdaptiveUnsupervisedEncoder(nn.Module):
    """
    接口与 v1.2 完全相同:
      - __init__(input_dim, hidden_dim, latent_dim, n_streams, n_layers, use_mhc)
      - forward(x) → (recon, z_norm)
      - predict_velocity(z_t, t) → v
    """
    def __init__(self, input_dim=1024, hidden_dim=256, latent_dim=16,
                 n_streams=4, n_layers=3, use_mhc=True):
        super().__init__()
        self.use_mhc = use_mhc
        self.n_streams = n_streams
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.feature_extractor = MultiScaleFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=[hidden_dim * 2, hidden_dim, hidden_dim // 2]
        )

        total_feat_dim = hidden_dim * 2 + hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.hyena_blocks = nn.ModuleList([
            HyenaConvBlock(hidden_dim, short_kernel=5, long_kernel=15, dropout=0.1)
            for _ in range(n_layers)
        ])

        if use_mhc:
            self.mhc_layers = nn.ModuleList([
                ManifoldConstrainedResidual(hidden_dim, n_streams=n_streams)
                for _ in range(n_layers)
            ])
            self.enc_transform_blocks = nn.ModuleList([
                TransformBlock(hidden_dim, dropout=0.1)
                for _ in range(n_layers)
            ])

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Softplus()
        )

        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        self.dec_expand = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        if use_mhc:
            self.dec_mhc_layers = nn.ModuleList([
                ManifoldConstrainedResidual(hidden_dim, n_streams=n_streams)
                for _ in range(n_layers)
            ])
            self.dec_transform_blocks = nn.ModuleList([
                TransformBlock(hidden_dim, dropout=0.1)
                for _ in range(n_layers)
            ])

        self.dec_project = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def _encode(self, x):
        """
        Encoder-only 路径 (feature extractor + Hyena + mHC + bottleneck)。
        返回 (z, z_norm), 不走 decoder。供 augmentation consistency loss 使用,
        省一次 decoder forward。

        v1.3 (2026-04-27): 因为 mHC.forward 已修复为使用入参 x, hyena_block(h)
        的输出 h 现在会真正经由 mHC 注入到 streams 上, hyena 参数从此进入
        梯度回流路径。
        """
        multi_feats = self.feature_extractor(x)
        fused = self.fusion(torch.cat(multi_feats, dim=-1))
        h = fused.unsqueeze(1)

        if self.use_mhc:
            streams = fused.unsqueeze(1).repeat(1, self.n_streams, 1)

        for i, hyena_block in enumerate(self.hyena_blocks):
            h = hyena_block(h)
            if self.use_mhc:
                h_flat = h.squeeze(1)
                # 修复后, h_flat 不再被 mHC 丢弃: mHC 内部会把 h_flat
                # 加到每个 stream 上, 让 hyena 参数收到梯度。
                h_flat, streams = self.mhc_layers[i](
                    h_flat, streams, self.enc_transform_blocks[i]
                )
                h = h_flat.unsqueeze(1)

        z = self.bottleneck(h.squeeze(1))
        z_norm = F.normalize(z, p=2, dim=1)
        return z, z_norm

    def _decode(self, z):
        """
        Decoder-only 路径 (expand + 对称 mHC + project)。

        v1.3 (2026-04-27): dec_mhc_layers 同样受益于 mHC.forward 的修复 ——
        每层 mHC 现在会读取上一层的输出 d, 而不是只在初始 streams 上反复
        循环。decoder 也从此真正成为 N 层的对称结构。
        """
        d = self.dec_expand(z)
        if self.use_mhc:
            d_streams = d.unsqueeze(1).repeat(1, self.n_streams, 1)
            for i in range(self.n_layers):
                d, d_streams = self.dec_mhc_layers[i](
                    d, d_streams, self.dec_transform_blocks[i]
                )
        return self.dec_project(d)

    def forward(self, x, x_aug=None, t=None, z_0=None):
        """
        统一前向。支持两种调用模式:

        1) 传统/推理模式 (x_aug/t/z_0 均为 None):
             returns (recon, z_norm)
           与 v1.1/v1.2 行为完全一致, 保证下游代码 (evaluate() 等) 无需改动。

        2) 训练模式 (传入 x_aug, t, z_0):
             returns dict {
                'recon'   : (B, input_dim)    from x
                'z'       : (B, latent_dim)   from x
                'z_aug'   : (B, latent_dim)   from x_aug, encoder-only
                'pred_v'  : (B, latent_dim)   velocity_net(interp(z_0, z), t)
             }
           所有张量都在一次 DDP-wrapped forward 里产生, 反向时
           DDP reducer 只处理一个一致的梯度图, 消除 "marked ready twice"
           及 "finished reduction in prior iteration" 类错误。

        训练模式下 l_recon 用 recon; l_fm 用 pred_v 和 (z - z_0);
        l_aug 用 z 和 z_aug; l_diversity/smoothness/spread 用 z。
        """
        # ── 主路径: encode + decode ──
        z_raw, z_norm = self._encode(x)
        recon = self._decode(z_raw)

        # ── 传统调用: 保持 (recon, z_norm) 返回值 ──
        if x_aug is None and t is None and z_0 is None:
            return recon, z_norm

        out = {'recon': recon, 'z': z_norm}

        # ── 增强路径: 只跑 encoder, 不必再跑 decoder (consistency 只看 z) ──
        if x_aug is not None:
            _, z_aug = self._encode(x_aug)
            out['z_aug'] = z_aug

        # ── flow matching: velocity 预测 ──
        if t is not None and z_0 is not None:
            z_t = (1.0 - t) * z_0 + t * z_norm
            pred_v = self.velocity_net(torch.cat([z_t, t], dim=1))
            out['pred_v'] = pred_v

        return out

    def predict_velocity(self, z_t, t):
        """保留旧接口, 供单进程推理使用。DDP 训练路径应走 forward()."""
        return self.velocity_net(torch.cat([z_t, t], dim=1))


# ==================== 无监督损失函数 ============================
class AdaptiveLosses:
    """
    自适应无监督损失 —— v1.2 全部对齐论文公式。

    DDP-safe 注意事项:
      退化情况 (e.g. batch 里只有 1 个 chromosome) 返回 `z.sum() * 0.0`
      而非 `torch.tensor(0.0)`。前者保持计算图连通 (grad 值为 0 但
      autograd 能 traverse), 后者是常量会让 DDP 不同 rank 的计算图
      不对称, 触发 "marked ready twice" 或 "prior iteration reduction"
      类错误。
    """

    @staticmethod
    def _zero_like(z):
        """Graph-connected zero scalar: grad flows through with value 0."""
        return z.sum() * 0.0

    @staticmethod
    def reconstruction_loss(recon, x):
        return F.mse_loss(recon, x)

    @staticmethod
    def flow_matching_loss(model, z):
        """保留旧接口 (单进程模式可用); DDP 训练路径走 forward() 里的 pred_v."""
        t = torch.rand(z.size(0), 1)
        z_0 = torch.randn_like(z)
        z_t = (1 - t) * z_0 + t * z
        pred_v = model.predict_velocity(z_t, t)
        target_v = z - z_0
        return F.mse_loss(pred_v, target_v)

    @staticmethod
    def diversity_loss(z, window_ids):
        chrom_groups = {}
        for i, wid in enumerate(window_ids):
            chrom = wid.rpartition('_')[0]
            chrom_groups.setdefault(chrom, []).append(i)

        centroids = []
        for chrom in sorted(chrom_groups.keys()):
            idxs = chrom_groups[chrom]
            centroid = z[idxs].mean(dim=0)
            centroids.append(centroid)

        if len(centroids) < 2:
            return AdaptiveLosses._zero_like(z)

        centroids_t = torch.stack(centroids)
        K = centroids_t.size(0)
        diffs = centroids_t.unsqueeze(0) - centroids_t.unsqueeze(1)
        dists = diffs.norm(p=2, dim=-1)
        mask = 1 - torch.eye(K, device=z.device)
        mean_dist = (dists * mask).sum() / (K * (K - 1) + 1e-8)
        return -mean_dist

    @staticmethod
    def local_smoothness_loss(z, window_ids):
        losses = []
        chrom_groups = {}
        for i, wid in enumerate(window_ids):
            chrom = wid.rpartition('_')[0]
            chrom_groups.setdefault(chrom, []).append(i)

        for chrom, indices in chrom_groups.items():
            if len(indices) < 2:
                continue
            indices = sorted(indices)
            for j in range(len(indices) - 1):
                idx1, idx2 = indices[j], indices[j + 1]
                diff = z[idx1] - z[idx2]
                losses.append((diff * diff).sum())

        if losses:
            return torch.stack(losses).mean()
        else:
            return AdaptiveLosses._zero_like(z)

    @staticmethod
    def augmentation_consistency_loss(z1, z2):
        return F.mse_loss(z1, z2)

    @staticmethod
    def spread_loss(z, window_ids):
        chrom_groups = {}
        for i, wid in enumerate(window_ids):
            chrom = wid.rpartition('_')[0]
            chrom_groups.setdefault(chrom, []).append(i)

        centroids = []
        for chrom in sorted(chrom_groups.keys()):
            idxs = chrom_groups[chrom]
            centroid = z[idxs].mean(dim=0)
            centroids.append(centroid)

        if len(centroids) < 2:
            return AdaptiveLosses._zero_like(z)

        centroids_t = torch.stack(centroids)
        var_total = centroids_t.var(dim=0).sum()
        return -var_total
