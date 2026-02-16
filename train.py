import argparse
import os
import time
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import CD2_RBFKI
from dataset import MagneticNpyDataset, normalize_coords_index_to_unit, make_full_grid_coords
from utils import (
    set_seed,
    pick_ctx_tgt_indices,
    pick_ctx_tgt_indices_sparse_weighted,
)


# -------------------------
# Helpers: build model
# -------------------------
def build_model(args, device: torch.device) -> torch.nn.Module:
    model = CD2_RBFKI(
        d_e=args.d_e,
        d_c=args.d_c,
        hidden=args.hidden,
        rank_r=args.rank_r,
        nugget=args.nugget,
        chunk_q=args.chunk_q,
        ablate_no_L=args.ablate_no_L,
        ablate_fixed_rhop=args.ablate_fixed_rhop,
        fixed_rho=args.fixed_rho,
        fixed_p=args.fixed_p,
        ablate_no_residual=args.ablate_no_residual,
        ablate_fixed_beta=args.ablate_fixed_beta,
        fixed_beta=args.fixed_beta,
    ).to(device)
    return model


# -------------------------
# Line-aware Helpers
# -------------------------
@torch.no_grad()
def build_vertical_line_state_from_coords_idx(coords_idx: torch.Tensor, min_line_pts: int = 20) -> Optional[Dict]:
    assert coords_idx.dim() == 3 and coords_idx.shape[0] == 1 and coords_idx.shape[-1] == 2
    x = coords_idx[0, :, 0].long()
    uniq_x = torch.unique(x)
    groups = []
    for xv in uniq_x.tolist():
        idx = torch.nonzero(x == xv, as_tuple=False).view(-1)
        if idx.numel() >= int(min_line_pts): groups.append(idx)
    if len(groups) < 2: return None
    return {"groups": groups}


@torch.no_grad()
def pick_ctx_tgt_indices_line_holdout(coords_all, line_state, n_ctx, n_tgt, device, holdout_line_ratio=0.25,
                                      line_tgt_ratio=0.7):
    N = coords_all.shape[1]
    all_idx = torch.arange(N, device=device)
    groups = line_state["groups"]
    G = len(groups)
    k_lines = int(max(1, round(G * float(holdout_line_ratio))))
    k_lines = min(k_lines, G)
    perm = torch.randperm(G, device=device)
    sel_g = perm[:k_lines].tolist()
    hold_pool = torch.cat([groups[g].to(device) for g in sel_g], dim=0).unique()
    n_tgt_line = int(round(float(n_tgt) * float(line_tgt_ratio)))
    n_tgt_line = min(n_tgt_line, int(hold_pool.numel()))
    if n_tgt_line > 0:
        perm2 = torch.randperm(hold_pool.numel(), device=device)
        tgt_line = hold_pool[perm2[:n_tgt_line]]
    else:
        tgt_line = torch.empty((0,), dtype=torch.long, device=device)
    mask_rem = torch.ones((N,), dtype=torch.bool, device=device)
    if tgt_line.numel() > 0: mask_rem[tgt_line] = False
    rem_pool = all_idx[mask_rem]
    n_tgt_rest = int(n_tgt) - int(tgt_line.numel())
    n_tgt_rest = max(0, min(n_tgt_rest, int(rem_pool.numel())))
    if n_tgt_rest > 0:
        perm3 = torch.randperm(rem_pool.numel(), device=device)
        tgt_rest = rem_pool[perm3[:n_tgt_rest]]
        tgt = torch.cat([tgt_line, tgt_rest], dim=0)
    else:
        tgt = tgt_line
    tgt = tgt.unique()
    mask_ctx = torch.ones((N,), dtype=torch.bool, device=device)
    if tgt.numel() > 0: mask_ctx[tgt] = False
    ctx_pool = all_idx[mask_ctx]
    n_ctx_eff = int(min(int(n_ctx), int(ctx_pool.numel())))
    if n_ctx_eff <= 0:
        ctx = all_idx[: min(N, max(1, n_ctx))].clone()
    else:
        perm4 = torch.randperm(ctx_pool.numel(), device=device)
        ctx = ctx_pool[perm4[:n_ctx_eff]]
    return ctx.unsqueeze(0), tgt.unsqueeze(0)


def compute_n_ctx_tgt(N: int, ctx_frac: float, tgt_frac: float) -> Tuple[int, int]:
    s = max(1e-12, float(ctx_frac) + float(tgt_frac))
    ctx_frac = float(ctx_frac) / s;
    tgt_frac = float(tgt_frac) / s
    n_tgt = int(round(N * tgt_frac));
    n_tgt = max(1, min(n_tgt, N - 1))
    n_ctx = int(round(N * ctx_frac));
    n_ctx = max(1, min(n_ctx, N - n_tgt))
    if n_ctx + n_tgt > N: n_tgt = max(1, N - n_ctx)
    if n_ctx + n_tgt > N: n_ctx = max(1, N - n_tgt)
    return n_ctx, n_tgt


def sample_ctx_tgt(coords_all, N, args, device, line_state=None):
    n_ctx, n_tgt = compute_n_ctx_tgt(N, args.ctx_frac, args.tgt_frac)
    if args.split == "line":
        if line_state is None:
            return pick_ctx_tgt_indices_sparse_weighted(coords_all, n_ctx, n_tgt, device, args.hard_tgt_ratio,
                                                        args.hard_pool_ratio, args.weight_power)
        return pick_ctx_tgt_indices_line_holdout(coords_all, line_state, n_ctx, n_tgt, device, args.holdout_line_ratio,
                                                 args.line_tgt_ratio)
    if args.split == "weighted":
        return pick_ctx_tgt_indices_sparse_weighted(coords_all, n_ctx, n_tgt, device, args.hard_tgt_ratio,
                                                    args.hard_pool_ratio, args.weight_power)
    ctx_i, tgt_i = pick_ctx_tgt_indices(N, n_ctx, n_tgt, device=device)
    return ctx_i.unsqueeze(0), tgt_i.unsqueeze(0)


# -------------------------
# Metrics & Denormalization
# -------------------------
def calculate_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Calculates MAE, RMSE, MAPE on physical values."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    abs_error = np.abs(y_true_flat - y_pred_flat)
    v_mae = float(np.mean(abs_error))
    v_rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))

    abs_y_true_flat = np.abs(y_true_flat)
    threshold = np.percentile(abs_y_true_flat, 5)
    if threshold < 1e-9: threshold = 1e-9

    valid_mask = abs_y_true_flat > threshold

    if np.sum(valid_mask) > 0:
        v_mape = float(
            np.mean(np.abs((y_true_flat[valid_mask] - y_pred_flat[valid_mask]) / y_true_flat[valid_mask])) * 100.0)
    else:
        v_mape = 0.0

    return v_mae, v_rmse, v_mape


def denorm_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reverse Z-score: x * std + mean"""
    return x * std + mean


@torch.no_grad()
def predict_and_evaluate(model, values_z, coords_idx, gt_z, grid_coords, device, args, mean, std):
    """
    MODIFIED: Now also returns the learned eta value.
    """
    model.eval()

    # 1. 准备预测
    coords_unit = normalize_coords_index_to_unit(coords_idx.float(), grid_size=gt_z.shape[-1])
    q = grid_coords.unsqueeze(0).to(device)

    # Context 采样 (全图预测通常使用所有观测点作为 context)
    N = values_z.shape[1]
    n_ctx = min(args.eval_ctx_points, N)
    coords_ctx = coords_unit[:, :n_ctx, :]
    values_ctx = values_z[:, :n_ctx]

    t0 = time.time()

    # 关键修改: 请求返回 aux 变量
    pred_flat_z, aux = model(coords_ctx, values_ctx, q, return_aux=True)
    pred_time = time.time() - t0

    # 从 aux 中提取 eta
    final_eta = aux['eta'].mean().item()

    pred_2d_z = pred_flat_z.view(gt_z.shape[-2], gt_z.shape[-1])

    # 2. 反归一化到物理空间
    pred_phys = denorm_zscore(pred_2d_z, mean.squeeze(), std.squeeze())
    gt_phys = denorm_zscore(gt_z[0], mean.squeeze(), std.squeeze())

    pred_np = pred_phys.cpu().numpy()
    gt_np = gt_phys.cpu().numpy()

    # 3. 计算物理指标
    mae, rmse, mape = calculate_metrics_np(gt_np, pred_np)

    # 返回所有结果，包括 eta
    return mae, rmse, mape, pred_time, gt_np, pred_np, final_eta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="magnetic_dataset_zscore/exp_B_betaFixed_alpha_sparse")
    parser.add_argument("--split_data", type=str, default="test")
    parser.add_argument("--grid_size", type=int, default=128)

    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Split strategy
    parser.add_argument("--split", type=str, default="line", choices=["line", "weighted", "random"])
    parser.add_argument("--ctx_frac", type=float, default=0.75)
    parser.add_argument("--tgt_frac", type=float, default=0.25)
    parser.add_argument("--line_min_pts", type=int, default=30)
    parser.add_argument("--holdout_line_ratio", type=float, default=0.3)
    parser.add_argument("--line_tgt_ratio", type=float, default=0.9)
    parser.add_argument("--hard_tgt_ratio", type=float, default=0.7)
    parser.add_argument("--hard_pool_ratio", type=float, default=0.4)
    parser.add_argument("--weight_power", type=float, default=2.0)

    # Model
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--d_c", type=int, default=64)
    parser.add_argument("--d_e", type=int, default=64)
    parser.add_argument("--rank_r", type=int, default=16)
    parser.add_argument("--nugget", type=float, default=1e-4)
    parser.add_argument("--chunk_q", type=int, default=1024)

    # Ablations
    parser.add_argument("--ablate_no_L", action="store_true")
    parser.add_argument("--ablate_fixed_rhop", action="store_true")
    parser.add_argument("--fixed_rho", type=float, default=1.0)
    parser.add_argument("--fixed_p", type=float, default=2.0)
    parser.add_argument("--ablate_no_residual", action="store_true")
    parser.add_argument("--ablate_fixed_beta", action="store_true")
    parser.add_argument("--fixed_beta", type=float, default=0.5)

    parser.add_argument("--eval_ctx_points", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="test_zscore_b3_eta")
    parser.add_argument("--vis_count", type=int, default=20)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ds = MagneticNpyDataset(args.data_dir, split=args.split_data, grid_size=args.grid_size, load_minmax=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    grid_coords = make_full_grid_coords(grid_size=args.grid_size, device=device)

    n_total = min(args.samples, len(ds))
    all_results = []
    sum_metrics = np.zeros(3)  # mae, rmse, mape
    sum_time = 0.0
    sum_eta = 0.0

    it = iter(loader)
    for si in range(n_total):
        batch = next(it)

        gt_z = batch["gt"].to(device)
        values_z = batch["values"].to(device)
        coords_idx = batch["coords_idx"].to(device)

        if "gt_mean" in batch:
            mean = batch["gt_mean"].to(device).view(1, 1, 1)
            std = batch["gt_std"].to(device).view(1, 1, 1)
        else:
            print("Warning: gt_mean/std not found in batch. Evaluation will be wrong.")
            mean, std = torch.zeros(1).to(device), torch.ones(1).to(device)

        coords_unit = normalize_coords_index_to_unit(coords_idx.float(), grid_size=args.grid_size)
        B, N, _ = coords_idx.shape

        line_state = None
        if args.split == "line":
            line_state = build_vertical_line_state_from_coords_idx(coords_idx, min_line_pts=args.line_min_pts)

        model = build_model(args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_loss = float('inf')
        best_state = None
        bad_steps = 0

        # ---- Training Loop (In Normalized Space) ----
        model.train()
        for _ in range(1, args.steps + 1):
            ctx_idx, tgt_idx = sample_ctx_tgt(coords_unit, N, args, device, line_state)

            c_ctx = torch.gather(coords_unit, 1, ctx_idx[..., None].expand(-1, -1, 2))
            v_ctx = torch.gather(values_z, 1, ctx_idx)
            c_tgt = torch.gather(coords_unit, 1, tgt_idx[..., None].expand(-1, -1, 2))
            v_tgt = torch.gather(values_z, 1, tgt_idx)

            try:
                pred = model(c_ctx, v_ctx, c_tgt)
                loss = (pred - v_tgt).abs().mean()  # Loss is in Z-space (stable)
            except:
                loss = torch.tensor(float('nan'))

            if not torch.isfinite(loss):
                if best_state: model.load_state_dict(best_state)
                for pg in optimizer.param_groups: pg['lr'] *= 0.5
                bad_steps += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_steps = 0
            else:
                bad_steps += 1
            if args.patience > 0 and bad_steps >= args.patience: break

        if best_state: model.load_state_dict(best_state)

        # ---- Evaluation (Output Physical Metrics) ----
        try:
            mae, rmse, mape, t_pred, gt_np, pred_np, final_eta = predict_and_evaluate(
                model, values_z, coords_idx, gt_z, grid_coords, device, args, mean, std
            )
        except Exception as e:
            print(f"Sample {si} Failed during evaluation: {e}")
            continue

        values_phys = denorm_zscore(values_z, mean.squeeze(), std.squeeze())
        vals_np = values_phys[0].cpu().numpy()
        coords_np = coords_idx[0].cpu().numpy()

        sum_metrics += [mae, rmse, mape]
        sum_time += t_pred
        sum_eta += final_eta

        print(
            f"Sample {si}: MAE={mae:.4f} nT, RMSE={rmse:.4f} nT, MAPE={mape:.2f}%, Eta={final_eta:.4f}, T={t_pred:.3f}s")

        all_results.append({
            "id": si,
            "mae": mae, "rmse": rmse, "mape": mape, "time": t_pred, "eta": final_eta,
            "gt": gt_np,
            "pred": pred_np,
            "input_vals": vals_np,
            "coords": coords_np
        })

    # Summary
    avg = sum_metrics / max(1, n_total)
    avg_time = sum_time / max(1, n_total)
    avg_eta = sum_eta / max(1, n_total)

    print(f"\n==== Summary ====")
    print(f"Samples: {n_total}")
    print(f"Average: MAE={avg[0]:.4f} nT, RMSE={avg[1]:.4f} nT, MAPE={avg[2]:.2f}%")
    print(f"Avg Eta: {avg_eta:.4f}")
    print(f"Avg Time: {avg_time:.4f} s")

    txt_path = os.path.join(args.out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"==== Instance Optimization Summary (Z-Score) ====\n")
        f.write(
            f"AVG_MAE={avg[0]:.6f}\nAVG_RMSE={avg[1]:.6f}\nAVG_MAPE={avg[2]:.6f}\nAVG_ETA={avg_eta:.6f}\nAVG_TIME={avg_time:.6f}\n")
        f.write("id\tMAE\tRMSE\tMAPE\tEta\tTime\n")
        for r in all_results:
            f.write(f"{r['id']}\t{r['mae']:.6f}\t{r['rmse']:.6f}\t{r['mape']:.6f}\t{r['eta']:.6f}\t{r['time']:.6f}\n")

    print(f"\nSaved results to {txt_path}")

    # Visualize
    if args.vis_count > 0:
        try:
            from visualize import Visualizer
            vis_dir = os.path.join(args.out_dir, "vis")
            vis = Visualizer(save_dir=vis_dir)
            vis.visualize_and_save(all_results, vis_count=min(args.vis_count, len(all_results)))
            print(f"Visualizations saved to {vis_dir}")
        except ImportError:
            print("visualize.py not found, skipping visualization.")


if __name__ == "__main__":
    main()