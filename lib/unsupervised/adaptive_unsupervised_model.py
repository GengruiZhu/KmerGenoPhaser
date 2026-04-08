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
                                    padding=short_kernel//2, groups=dim)
        self.conv_long = nn.Conv1d(dim, dim, kernel_size=long_kernel, 
                                   padding=long_kernel//2, groups=dim)
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
    """
    def __init__(self, input_dim=1024, hidden_dim=256, latent_dim=16, 
                 n_streams=4, n_layers=3, use_mhc=True):
        super().__init__()
        self.use_mhc = use_mhc
        self.n_streams = n_streams
        self.latent_dim = latent_dim
        
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
        
        # mHC
        if use_mhc:
            self.mhc_layers = nn.ModuleList([
                ManifoldConstrainedResidual(hidden_dim, n_streams=n_streams)
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
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch = x.size(0)
        
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
                h_flat, streams = self.mhc_layers[i](h_flat, streams, lambda y: y)
                h = h_flat.unsqueeze(1)
        
        # 潜在空间
        z = self.bottleneck(h.squeeze(1))
        z_norm = F.normalize(z, p=2, dim=1)
        
        # 解码
        recon = self.decoder(z)
        
        return recon, z_norm
    
    def predict_velocity(self, z_t, t):
        return self.velocity_net(torch.cat([z_t, t], dim=1))


# ==================== 无监督损失函数 ====================
class AdaptiveLosses:
    """自适应无监督损失"""
    
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
    def diversity_loss(z):
        """
        鼓励特征多样性
        防止所有样本坍缩到同一点
        """
        z_norm = F.normalize(z, p=2, dim=1)
        
        # 计算批次内的平均相似度
        sim_matrix = torch.mm(z_norm, z_norm.t())
        
        # 去掉对角线（自己和自己）
        mask = 1 - torch.eye(z.size(0), device=z.device)
        sim_off_diag = sim_matrix * mask
        
        # 希望平均相似度适中（不要太高也不要太低）
        # 太高 = 模式崩溃，太低 = 特征无意义
        target_sim = 0.3  # 经验值
        avg_sim = sim_off_diag.sum() / (mask.sum() + 1e-8)
        
        return F.mse_loss(avg_sim, torch.tensor(target_sim, device=z.device))
    
    @staticmethod
    def local_smoothness_loss(z, window_ids):
        """
        同一染色体的相邻窗口应该有相似的表示
        window_ids: 形如 ['Chr1A_00001', 'Chr1A_00002', ...]
        """
        losses = []
        
        # 按染色体分组
        chrom_groups = {}
        for i, wid in enumerate(window_ids):
            chrom = wid.rpartition('_')[0]
            if chrom not in chrom_groups:
                chrom_groups[chrom] = []
            chrom_groups[chrom].append(i)
        
        # 计算相邻窗口的相似度
        for chrom, indices in chrom_groups.items():
            if len(indices) < 2:
                continue
            
            indices = sorted(indices)
            for i in range(len(indices) - 1):
                idx1, idx2 = indices[i], indices[i+1]
                z1 = F.normalize(z[idx1:idx1+1], p=2, dim=1)
                z2 = F.normalize(z[idx2:idx2+1], p=2, dim=1)
                
                # 相邻窗口应该相似（余弦相似度接近1）
                sim = (z1 * z2).sum()
                losses.append(1 - sim)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=z.device)
    
    @staticmethod
    def augmentation_consistency_loss(z1, z2):
        """数据增强一致性"""
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        sim = (z1_norm * z2_norm).sum(dim=1)
        return 1 - sim.mean()
    
    @staticmethod
    def spread_loss(z):
        """
        鼓励特征在潜在空间中均匀分布
        使用方差作为度量
        """
        # 每个维度的方差应该接近某个目标值
        var_per_dim = z.var(dim=0)
        target_var = 0.5  # 经验值
        return F.mse_loss(var_per_dim, 
                         torch.full_like(var_per_dim, target_var))
