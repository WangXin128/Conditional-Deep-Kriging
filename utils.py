
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_ctx_tgt_indices_sparse_weighted(
    coords: torch.Tensor,
    n_ctx: int,
    n_tgt: int,
    device: torch.device,
    min_tgt: int = 1,
    hard_tgt_ratio: float = 0.5,     # tgt 里有多少比例来自“稀疏难点池”
    hard_pool_ratio: float = 0.3,    # 取 d_nn 最大的 top 30% 作为难点池
    weight_power: float = 2.0,       # 权重 = (d_nn + eps)^weight_power
    eps: float = 1e-6,
):
    """
    coords: (N,2) 或 (B,N,2) 归一化坐标（建议 [-1,1]）
    返回:
      - 若输入 (N,2): ctx_idx (n_ctx,), tgt_idx (n_tgt,)
      - 若输入 (B,N,2): ctx_idx (B,n_ctx), tgt_idx (B,n_tgt)
    """
    assert coords.dim() in (2, 3)

    def _one_sample(coords_2d: torch.Tensor):
        # coords_2d: (N,2)
        N = coords_2d.shape[0]
        if N <= min_tgt:
            perm = torch.randperm(N, device=device)
            return perm, perm[:0]

        # 先修正 n_tgt / n_ctx，保证有空间
        n_tgt_eff = min(n_tgt, N - 1)
        n_tgt_eff = max(min_tgt, n_tgt_eff)
        n_ctx_eff = min(n_ctx, N - n_tgt_eff)
        n_ctx_eff = max(1, n_ctx_eff)
        # 再保险一次
        n_tgt_eff = min(n_tgt_eff, N - n_ctx_eff)
        n_tgt_eff = max(min_tgt, n_tgt_eff)

        # -------- 1) 计算最近邻距离 d_nn --------
        # (N,N)
        D = torch.cdist(coords_2d, coords_2d)  # 欧氏距离
        D.fill_diagonal_(float("inf"))
        d_nn = D.min(dim=1).values  # (N,)

        # -------- 2) 权重：稀疏越大权重越大 --------
        w = (d_nn + eps).pow(weight_power)
        w = torch.clamp(w, min=eps)

        # -------- 3) 构建“难点池”：d_nn 最大的 top hard_pool_ratio --------
        hard_pool_size = max(1, int(hard_pool_ratio * N))
        actual_k = min(hard_pool_size, d_nn.shape[0])
        hard_idx = torch.topk(d_nn, k=actual_k, largest=True).indices  # (K,)
        # tgt 中有多少来自难点池
        n_tgt_hard = int(round(hard_tgt_ratio * n_tgt_eff))
        n_tgt_hard = max(0, min(n_tgt_hard, hard_pool_size, n_tgt_eff))
        n_tgt_rest = n_tgt_eff - n_tgt_hard

        chosen = torch.zeros(N, dtype=torch.bool, device=device)

        # ---- 3.1) 从难点池里按权重抽 n_tgt_hard ----
        tgt_list = []
        if n_tgt_hard > 0:
            w_hard = w[hard_idx]
            w_hard = w_hard / w_hard.sum()
            sel_in_hard = torch.multinomial(w_hard, num_samples=n_tgt_hard, replacement=False)
            tgt_hard = hard_idx[sel_in_hard]
            tgt_list.append(tgt_hard)
            chosen[tgt_hard] = True

        # ---- 3.2) 从剩余点里按权重抽 n_tgt_rest ----
        if n_tgt_rest > 0:
            rest_idx = (~chosen).nonzero(as_tuple=False).squeeze(-1)
            w_rest = w[rest_idx]
            w_rest = w_rest / w_rest.sum()
            sel_rest = torch.multinomial(w_rest, num_samples=n_tgt_rest, replacement=False)
            tgt_rest = rest_idx[sel_rest]
            tgt_list.append(tgt_rest)
            chosen[tgt_rest] = True

        tgt_idx = torch.cat(tgt_list, dim=0) if len(tgt_list) > 0 else torch.empty(0, device=device, dtype=torch.long)

        # -------- 4) ctx 从剩余点里抽 n_ctx_eff（建议均匀抽即可）--------
        remain_idx = (~chosen).nonzero(as_tuple=False).squeeze(-1)
        if remain_idx.numel() <= n_ctx_eff:
            ctx_idx = remain_idx
        else:
            perm = torch.randperm(remain_idx.numel(), device=device)
            ctx_idx = remain_idx[perm[:n_ctx_eff]]

        return ctx_idx, tgt_idx

    if coords.dim() == 2:
        return _one_sample(coords.to(device))
    else:
        # coords: (B,N,2)
        B = coords.shape[0]
        ctx_all, tgt_all = [], []
        for b in range(B):
            ctx_b, tgt_b = _one_sample(coords[b].to(device))
            ctx_all.append(ctx_b)
            tgt_all.append(tgt_b)
        # 注意：这里假设每个样本最终 ctx/tgt 数量相同（通常成立）
        return torch.stack(ctx_all, dim=0), torch.stack(tgt_all, dim=0)
def pick_ctx_tgt_indices(
    n_total: int,
    n_ctx: int,
    n_tgt: int,
    device: torch.device,
    min_tgt: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly split indices into context and target (holdout).

    IMPORTANT: we must avoid empty target sets, because `.mean()` over an empty tensor
    returns NaN and will poison training.

    Strategy:
      - try to keep `n_tgt` as requested, but clamp so that n_ctx + n_tgt <= n_total
      - guarantee at least `min_tgt` target points (when n_total>min_tgt)
    """
    assert n_total >= 1

    # Degenerate edge-case (unlikely here, but keep safe)
    if n_total <= min_tgt:
        perm = torch.randperm(n_total, device=device)
        return perm, perm[:0]

    # Desired target count, but cannot exceed n_total-1
    n_tgt = int(min(n_tgt, n_total - 1))
    n_tgt = int(max(min_tgt, n_tgt))

    # Ensure context leaves room for targets
    n_ctx = int(min(n_ctx, n_total - n_tgt))
    n_ctx = int(max(1, n_ctx))

    # Recompute target count after final n_ctx
    n_tgt = int(min(n_tgt, n_total - n_ctx))
    n_tgt = int(max(min_tgt, n_tgt))

    # Final safety
    if n_ctx + n_tgt > n_total:
        n_ctx = n_total - n_tgt

    perm = torch.randperm(n_total, device=device)
    ctx_idx = perm[:n_ctx]
    tgt_idx = perm[n_ctx:n_ctx + n_tgt]
    return ctx_idx, tgt_idx


@dataclass
class AvgMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
