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
    backward 组合会触发 "Schema not found for node" 错误 (具体是
    TorchScript 在 backward 里插入的 _size_if_not_equal 辅助算子
    找不到 schema)。
  - 函数保留纯 Python 实现, 在 MKL/oneDNN 下性能仍然可观。
  - 该修改与单进程模式 (num_workers=1) 完全兼容, 不改变数值行为。
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
    输入 logits 是未归一化的原始矩阵 (batch, n, n)。
    """
    eps = 1e-8
    m = torch.exp(logits)
    for _ in range(n_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)   # row normalize
        m = m / (m.sum(dim=-2, keepdim=True) + eps)   # col normalize
    return m


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

    def forward(self, x):
        multi_feats = self.feature_extractor(x)
        fused = self.fusion(torch.cat(multi_feats, dim=-1))
        h = fused.unsqueeze(1)

        if self.use_mhc:
            streams = fused.unsqueeze(1).repeat(1, self.n_streams, 1)

        for i, hyena_block in enumerate(self.hyena_blocks):
            h = hyena_block(h)
            if self.use_mhc:
                h_flat = h.squeeze(1)
                h_flat, streams = self.mhc_layers[i](
                    h_flat, streams, self.enc_transform_blocks[i]
                )
                h = h_flat.unsqueeze(1)

        z = self.bottleneck(h.squeeze(1))
        z_norm = F.normalize(z, p=2, dim=1)

        d = self.dec_expand(z)

        if self.use_mhc:
            d_streams = d.unsqueeze(1).repeat(1, self.n_streams, 1)
            for i in range(self.n_layers):
                d, d_streams = self.dec_mhc_layers[i](
                    d, d_streams, self.dec_transform_blocks[i]
                )

        recon = self.dec_project(d)
        return recon, z_norm

    def predict_velocity(self, z_t, t):
        return self.velocity_net(torch.cat([z_t, t], dim=1))


# ==================== 无监督损失函数 ============================
class AdaptiveLosses:
    """自适应无监督损失 —— v1.2 全部对齐论文公式"""

    @staticmethod
    def reconstruction_loss(recon, x):
        return F.mse_loss(recon, x)

    @staticmethod
    def flow_matching_loss(model, z):
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
            return torch.tensor(0.0, device=z.device)

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
            return torch.tensor(0.0, device=z.device)

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
            return torch.tensor(0.0, device=z.device)

        centroids_t = torch.stack(centroids)
        var_total = centroids_t.var(dim=0).sum()
        return -var_total
