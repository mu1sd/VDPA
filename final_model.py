# vdpa_clean.py
# Final minimal implementation for the paper:
#   STK (Soft Top-K) + LGA (Learnable Global Attention)
#   VDFC (Variance-Driven Fusion Controller)
#   ALP (Adaptive Local Pooling, dynamic class token)
#   VDPA block + ViT-like backbone and MLP head
#
# PyTorch >= 1.10

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
class PatchEmbed(nn.Module):
    """ViT-style patch embedding (no cls token)."""
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, N, C']
        x = self.proj(x)                       # [B, C', H/P, W/P]
        x = x.flatten(2).transpose(1, 2)       # [B, N, C']
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------
# ALP: Adaptive Local Pooling
# -----------------------------
def build_grid_neighbors(grid_size: int, k: int = 9) -> torch.Tensor:
    """
    以 3x3 为例（K=9），为每个格点返回邻居序号（含自身），边界自动补齐。
    返回 [T, K]，T=grid_size*grid_size
    """
    T = grid_size * grid_size
    out = []
    for i in range(grid_size):
        for j in range(grid_size):
            idxs = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        idxs.append(ni * grid_size + nj)
            # 边缘不足补中心
            while len(idxs) < k:
                idxs.append(i * grid_size + j)
            out.append(idxs[:k])
    return torch.tensor(out, dtype=torch.long)  # [T, K]


class AdaptiveLocalPooling(nn.Module):
    """
    ALP：对每个 token 的 K 个邻居做余弦相似加权 -> 局部聚合，
    最后对所有 token 做全局平均得到动态 class token。
    X: [B, T, C], neighbor_idx: [T, K]
    return: [B, 1, C]
    """
    def __init__(self, k_neighbors: int = 9, temperature: float = 1.0):
        super().__init__()
        self.k = k_neighbors
        self.temperature = temperature

    def forward(self, X, neighbor_idx):
        B, T, C = X.shape
        idx = neighbor_idx.to(X.device)           # [T, K]
        # 关键：高级索引，一步得到 [B, T, K, C]
        neighbors = X[:, idx, :]                  # [B, T, K, C]

        q = X.unsqueeze(2)                        # [B, T, 1, C]
        sim = F.cosine_similarity(q, neighbors, dim=-1)        # [B, T, K]
        w = F.softmax(sim / self.temperature, dim=-1)          # [B, T, K]
        pooled = (w.unsqueeze(-1) * neighbors).sum(dim=2)      # [B, T, C]
        cls_token = pooled.mean(dim=1, keepdim=True)           # [B, 1, C]
        return cls_token



# -----------------------------
# STK branch (Soft Top-K)
# -----------------------------
class STKBranch(nn.Module):
    """
    输入 q,k,v（多头），输出局部稀疏注意力结果。
    """
    def __init__(self, temperature: float = 0.5, attn_drop: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, scale):
        # q,k,v: [B, H, N, D]
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale    # [B,H,N,N]
        soft_mask = F.softmax(logits / self.temperature, dim=-1) # 低温 → 更尖
        attn = F.softmax(logits * soft_mask, dim=-1)             # 平滑近似 Top-K
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                              # [B,H,N,D]
        return out


# -----------------------------
# LGA branch (Learnable Global Attention)
# -----------------------------
class LGABranch(nn.Module):
    """
    可学习位置权重对 k,v 做加权池化得到全局向量，再与 q 做点积注意力。
    """
    def __init__(self, num_heads: int, max_tokens: int = 1025, attn_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        # per-head 可学习位置权重，沿 token 维做 softmax
        self.pos_weight = nn.Parameter(torch.zeros(1, num_heads, max_tokens, 1))
        nn.init.trunc_normal_(self.pos_weight, std=0.02)
        self.drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, scale, N):
        # 限定到实际 token 数 N
        w = F.softmax(self.pos_weight[:, :, :N, :], dim=2)        # [1,H,N,1]
        k_glb = (k * w).sum(dim=2, keepdim=True)                  # [B,H,1,D]
        v_glb = (v * w).sum(dim=2, keepdim=True)                  # [B,H,1,D]
        attn = torch.matmul(q, k_glb.transpose(-2, -1)) * scale   # [B,H,N,1]
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v_glb)                           # [B,H,N,D]
        return out


# -----------------------------
# VDFC: Variance-Driven Fusion Controller
# -----------------------------
class VDFC(nn.Module):
    """
    用 token 维的通道方差（对 head 平均）作为结构离散度信号，产生样本级 α∈[min,max]。
    """
    def __init__(self, min_alpha: float = 0.31, max_alpha: float = 0.89):
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        # 可学习仿射映射
        self.w = nn.Parameter(torch.tensor(5.0))
        self.b = nn.Parameter(torch.tensor(-2.5))

    def forward(self, q):
        # q: [B,H,N,D]  → 先对 token 取方差，再对 (H,D) 取平均 → [B,1,1,1]
        var_token = q.var(dim=2, unbiased=False)     # [B,H,D]
        s = var_token.mean(dim=(1, 2), keepdim=True) # [B,1]
        gate = torch.sigmoid(self.w * s + self.b)    # [B,1]
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * gate
        return alpha.view(-1, 1, 1, 1)               # broadcast to [B,1,1,1]


# -----------------------------
# VDPA Block
# -----------------------------
class VDPABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0,
                 attn_drop=0.0, drop=0.0, temperature=0.5,
                 max_tokens=1025):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # PreNorm
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        self.stk = STKBranch(temperature=temperature, attn_drop=attn_drop)
        self.lga = LGABranch(num_heads=num_heads, max_tokens=max_tokens, attn_drop=attn_drop)
        self.vdfc = VDFC()

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        """
        x: [B,N,C]
        """
        B, N, C = x.shape
        # ---- Attention ----
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # [B,H,N,D]

        out_loc = self.stk(q, k, v, self.scale)                # [B,H,N,D]
        out_glb = self.lga(q, k, v, self.scale, N)             # [B,H,N,D]
        alpha  = self.vdfc(q)                                  # [B,1,1,1]

        out = alpha * out_loc + (1.0 - alpha) * out_glb        # [B,H,N,D]
        out = out.transpose(1, 2).reshape(B, N, C)             # [B,N,C]
        out = self.proj(out)
        out = self.dropout(out)
        x = x + out

        # ---- MLP ----
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# VDPA Model
# -----------------------------
class VDPA(nn.Module):
    def __init__(self,
                 img_size=512, patch_size=16, in_chans=3,
                 num_classes=2, embed_dim=768, depth=10, num_heads=8,
                 mlp_ratio=4.0, attn_drop=0.0, drop=0.0,
                 k_neighbors=9, alp_temperature=1.0,
                 stk_temperature=0.5):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = self.patch_embed.grid_size
        self.num_patches = grid * grid

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop)

        # 堆叠 VDPA blocks
        self.blocks = nn.ModuleList([
            VDPABlock(embed_dim, num_heads, mlp_ratio,
                      attn_drop=attn_drop, drop=drop,
                      temperature=stk_temperature,
                      max_tokens=self.num_patches+1)
            for _ in range(depth)
        ])

        # ALP 动态 class token
        self.alp = AdaptiveLocalPooling(k_neighbors=k_neighbors, temperature=alp_temperature)
        self.register_buffer("neighbor_idx", build_grid_neighbors(grid, k_neighbors))  # [T,K]

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: [B,3,H,W]  (H=W=img_size)
        """
        x = self.patch_embed(x)                       # [B,N,C]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)                                # [B,N,C]

        # 动态 class token（不额外引入 [CLS]）
        cls_tok = self.alp(self.norm(x), self.neighbor_idx)   # [B,1,C]
        logits = self.head(cls_tok.squeeze(1))                # [B,num_classes]
        return logits


# -----------------------------
# Minimal sanity check
# -----------------------------
if __name__ == "__main__":
    B = 2
    model = VDPA(num_classes=2, depth=10, embed_dim=512, num_heads=8,
                 stk_temperature=0.5, k_neighbors=9)
    dummy = torch.randn(B, 3, 512, 512)
    out = model(dummy)
    print("logits:", out.shape)  # [B,2]
