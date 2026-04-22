#!/usr/bin/env python3
"""
自适应无监督基因组编码器
自动发现亚基因组数量和结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== Sinkhorn-Knopp ====================
class SinkhornKnopp(nn.Module):
    def __init__(self, n_iter=20, epsilon=1e-8):
        super().__init__()
        self.n_iter = n_iter
        self.epsilon = epsilon

    def forward(self, H):
        M = torch.exp(H)
        for _ in range(self.n_iter):
            M = M / (M.sum(dim=-1, keepdim=True) + self.epsilon)
            M = M / (M.sum(dim=-2, keepdim=True) + self.epsilon)
        return M


# ==================== mHC 层内变换块 (论文 Eq.9 的 F(·, W_l)) ===========
class TransformBlock(nn.Module):
    """
    F(·, W_l):  Linear → LayerNorm → GELU → Dropout
    论文描述为 "fully connected transformation with batch normalisation
    and ReLU activation"。此处用 LayerNorm + GELU 替代 BN + ReLU，
    因为 LN 对小 batch 更稳定，GELU 与本代码库风格一致。
    """
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


# ==================== mHC残差（无状态）====================
class ManifoldConstrainedResidual(nn.Module):
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams

        self.h_pre_proj = nn.Linear(dim * n_streams, n_streams, bias=True)
        self.h_post_proj = nn.Linear(dim * n_streams, n_streams, bias=True)
        self.h_res_proj = nn.Linear(dim * n_streams, n_streams * n_streams, bias=True)

        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.sinkhorn = SinkhornKnopp(n_iter=20)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, x_streams, layer_fn):
        batch = x.size(0)
        if x_streams is None:
            x_streams = x.unsqueeze(1).repeat(1, self.n_streams, 1)

        x_flat = x_streams.reshape(batch, -1)
        x_norm = F.layer_norm(x_flat, [x_flat.size(-1)])

        h_pre = torch.sigmoid(self.alpha_pre * self.h_pre_proj(x_norm))
        x_agg = torch.sum(h_pre.unsqueeze(-1) * x_streams, dim=1)

        # ── 论文 Eq.9:  F(H_pre · X, W_l)  ──────────────────────────
        out = layer_fn(self.norm(x_agg))

        h_post = 2 * torch.sigmoid(self.alpha_post * self.h_post_proj(x_norm))
        h_res_logits = self.alpha_res * self.h_res_proj(x_norm)
        h_res = self.sinkhorn(h_res_logits.reshape(batch, self.n_streams, self.n_streams))

        mixed_streams = torch.matmul(h_res, x_streams)
        updated_streams = mixed_streams + out.unsqueeze(1) * h_post.unsqueeze(-1)

        return updated_streams.mean(dim=1), updated_streams


# ==================== Hyena卷积 ====================
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


# ==================== 多尺度特征提取 ====================
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


# ==================== 自适应无监督编码器 ====================
class AdaptiveUnsupervisedEncoder(nn.Module):
    """
    完全自适应的无监督编码器
    不需要预设聚类数量，自动发现亚基因组结构

    v1.2 修正：
      - mHC 的 layer_fn 使用 TransformBlock (FC+LN+GELU)，
        而非 v1.1 中的 identity lambda。
      - Decoder 使用对称 mHC 架构 (论文 §3.3.3)。
    """
    def __init__(self, input_dim=1024, hidden_dim=256, latent_dim=16,
                 n_streams=4, n_layers=3, use_mhc=True):
        super().__init__()
        self.use_mhc = use_mhc
        self.n_streams = n_streams
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 多尺度特征
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

        # Hyena卷积
        self.hyena_blocks = nn.ModuleList([
            HyenaConvBlock(hidden_dim, short_kernel=5, long_kernel=15, dropout=0.1)
            for _ in range(n_layers)
        ])

        # ── Encoder mHC ─────────────────────────────────────────────
        if use_mhc:
            self.mhc_layers = nn.ModuleList([
                ManifoldConstrainedResidual(hidden_dim, n_streams=n_streams)
                for _ in range(n_layers)
            ])
            # v1.2: 每层 mHC 配一个 TransformBlock 作为 layer_fn
            self.enc_transform_blocks = nn.ModuleList([
                TransformBlock(hidden_dim, dropout=0.1)
                for _ in range(n_layers)
            ])

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Softplus()
        )

        # Flow Matching
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # ── Decoder: 对称 mHC 网络 (论文 §3.3.3) ────────────────────
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
        batch = x.size(0)

        # ── Encoder ──────────────────────────────────────────────────
        # 特征提取
        multi_feats = self.feature_extractor(x)
        fused = self.fusion(torch.cat(multi_feats, dim=-1))
        h = fused.unsqueeze(1)

        # 编码
        if self.use_mhc:
            streams = fused.unsqueeze(1).repeat(1, self.n_streams, 1)

        for i, hyena_block in enumerate(self.hyena_blocks):
            h = hyena_block(h)
            if self.use_mhc:
                h_flat = h.squeeze(1)
                # v1.2: 传入 TransformBlock 而非 identity lambda
                h_flat, streams = self.mhc_layers[i](
                    h_flat, streams, self.enc_transform_blocks[i]
                )
                h = h_flat.unsqueeze(1)

        # 潜在空间
        z = self.bottleneck(h.squeeze(1))
        z_norm = F.normalize(z, p=2, dim=1)

        # ── Decoder (对称 mHC) ───────────────────────────────────────
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


# ==================== 无监督损失函数 ====================
class AdaptiveLosses:
    """
    自适应无监督损失 —— v1.2 全部修正为论文公式

    所有需要 window_ids 的损失函数现在接受 window_ids 参数，
    通过 window_id 的前缀提取染色体名，按染色体计算质心。
    """

    @staticmethod
    def reconstruction_loss(recon, x):
        """论文 Eq.5:  L_recon = (1/B) Σ ||x_i - x̂_i||²"""
        return F.mse_loss(recon, x)

    @staticmethod
    def flow_matching_loss(model, z):
        """论文 Eq.6:  Flow-matching regularisation"""
        t = torch.rand(z.size(0), 1)
        z_0 = torch.randn_like(z)
        z_t = (1 - t) * z_0 + t * z
        pred_v = model.predict_velocity(z_t, t)
        target_v = z - z_0
        return F.mse_loss(pred_v, target_v)

    @staticmethod
    def diversity_loss(z, window_ids):
        """
        论文 Eq.7:
          L_div = -(1 / (K(K-1))) Σ_{a≠b} ||z̄_a - z̄_b||_2

        使用每条染色体的质心作为 "cluster centroid"，
        最大化质心之间的平均 L2 距离，防止模式崩溃。
        """
        # ── 按染色体聚合质心 ──
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

        centroids_t = torch.stack(centroids)          # (K, d_z)
        K = centroids_t.size(0)

        # 计算所有质心对的 L2 距离
        diffs = centroids_t.unsqueeze(0) - centroids_t.unsqueeze(1)  # (K, K, d_z)
        dists = diffs.norm(p=2, dim=-1)                               # (K, K)

        # 去掉对角线，取平均，取负（最小化此 loss = 最大化距离）
        mask = 1 - torch.eye(K, device=z.device)
        mean_dist = (dists * mask).sum() / (K * (K - 1) + 1e-8)

        return -mean_dist

    @staticmethod
    def local_smoothness_loss(z, window_ids):
        """
        论文 Eq.8:
          L_smooth = (1/|A|) Σ_{(w, w+1)∈A} ||z_w - z_{w+1}||²

        同一染色体的相邻窗口应该有相似的表示。
        使用 L2² 而非余弦距离。
        """
        losses = []

        # 按染色体分组
        chrom_groups = {}
        for i, wid in enumerate(window_ids):
            chrom = wid.rpartition('_')[0]
            chrom_groups.setdefault(chrom, []).append(i)

        # 计算相邻窗口的 L2² 距离
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
        """
        论文 Eq.10:
          L_aug = (1/B) Σ ||z_i - z_i^noisy||²

        数据增强一致性，使用 L2² 而非余弦距离。
        """
        return F.mse_loss(z1, z2)

    @staticmethod
    def spread_loss(z, window_ids):
        """
        论文 Eq.11:
          L_spread = -Var({z̄_{chr_j}})

        最大化每条染色体质心的方差，防止所有染色体坍缩到同一点。
        """
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

        centroids_t = torch.stack(centroids)   # (N_chr, d_z)
        # 计算所有维度上的总方差
        var_total = centroids_t.var(dim=0).sum()

        return -var_total
