import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "viridis"

MU0_OVER_4PI = 1e-7  # μ0/(4π) in SI (T·m/A)


# ============================================================
# 1) Paper Eq.(9) sphere anomaly on grid -> raw nT
# ============================================================
def _sphere_anomaly_eq9_SI(X, Y, Z, x0, y0, z0, R, J, D, I):
    x1 = X - x0
    y1 = Y - y0
    z1 = Z - z0

    r2 = x1 * x1 + y1 * y1 + z1 * z1 + 1e-18
    r5 = r2 ** 2.5

    # paper: m = J * π * R^3
    m = J * np.pi * (R ** 3)

    sinI = np.sin(I)
    cosI = np.cos(I)
    sin2I = np.sin(2.0 * I)

    cosD = np.cos(D)
    sinD = np.sin(D)
    sin2D = np.sin(2.0 * D)

    term1 = (2.0 * z1 * z1 - x1 * x1 - y1 * y1) * (sinI ** 2)
    term2 = (2.0 * x1 * x1 - y1 * y1 - z1 * z1) * (cosI ** 2) * (cosD ** 2)
    term3 = (2.0 * y1 * y1 - x1 * x1 - z1 * z1) * (cosI ** 2) * (sinD ** 2)
    term4 = -3.0 * x1 * z1 * sin2I * cosD
    term5 = 3.0 * x1 * y1 * (cosI ** 2) * sin2D
    term6 = -3.0 * y1 * z1 * sin2I * sinD

    dT = MU0_OVER_4PI * (m / r5) * (term1 + term2 + term3 + term4 + term5 + term6)
    return dT.astype(np.float64)


def _map_to_positive_range(field_nT, rng,
                           global_min_nT=0.0,
                           global_max_nT=2000.0,
                           span_range_nT=(200.0, 1500.0),
                           zero_start_prob=0.35,
                           min_offset_range_nT=(50.0, 900.0),
                           robust_percentiles=(1.0, 99.0)):
    """
    将 raw ΔT(nT) 映射到全正 [0, global_max_nT]，且每个样本范围不同。
    这是常量缩放+常量平移（不引入背景趋势）。
    """
    f = field_nT.astype(np.float64)

    p_lo, p_hi = robust_percentiles
    lo = float(np.percentile(f, p_lo))
    hi = float(np.percentile(f, p_hi))

    if hi - lo < 1e-9:
        span = float(rng.uniform(span_range_nT[0], span_range_nT[1]))
        if rng.random() < zero_start_prob:
            offset = 0.0
        else:
            offset = float(rng.uniform(min_offset_range_nT[0], min_offset_range_nT[1]))
            offset = min(offset, global_max_nT - span)
        out = np.full_like(f, offset + 0.5 * span)
        return np.clip(out, global_min_nT, global_max_nT).astype(np.float32)

    s = (f - lo) / (hi - lo)
    s = np.clip(s, 0.0, 1.0)

    span = float(rng.uniform(span_range_nT[0], span_range_nT[1]))
    span = min(span, global_max_nT - global_min_nT)

    if rng.random() < zero_start_prob:
        offset = 0.0
    else:
        offset = float(rng.uniform(min_offset_range_nT[0], min_offset_range_nT[1]))
        offset = min(offset, global_max_nT - span)

    out = offset + s * span
    return np.clip(out, global_min_nT, global_max_nT).astype(np.float32)


def generate_field_one_sample(
        rng,
        grid_size=128,
        area_m=10000.0,
        flight_height=2000.0,
        n_sources=(3, 10),
        depth_range=(800.0, 3500.0),
        radius_range=(80.0, 500.0),
        J_range=(0.5, 2.0),
        dir_mode="random",
        align_A=0.0,
        align_I=np.deg2rad(60.0),
        dir_jitter=0.15,
        global_max_nT=2000.0,
):
    """
    生成一个样本的 GT 网格场（nT）：
      - 多球体 Eq.(9) 叠加得到 raw ΔT
      - 再映射为全正且范围变化的 [0, global_max_nT]
    """
    x = np.linspace(-area_m, area_m, grid_size).astype(np.float64)
    y = np.linspace(-area_m, area_m, grid_size).astype(np.float64)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, float(flight_height), dtype=np.float64)

    total = np.zeros_like(X, dtype=np.float64)

    n = int(rng.integers(n_sources[0], n_sources[1] + 1))
    for _ in range(n):
        x0, y0 = rng.uniform(-0.9 * area_m, 0.9 * area_m, size=2)
        depth = float(rng.uniform(depth_range[0], depth_range[1]))
        z0 = -depth

        R = float(rng.uniform(radius_range[0], radius_range[1]))
        J = float(rng.uniform(J_range[0], J_range[1]))  # 正

        if dir_mode == "align":
            D = float(align_A + rng.normal(scale=dir_jitter))
            I = float(align_I + rng.normal(scale=dir_jitter))
        else:
            D = float(rng.uniform(0.0, 2.0 * np.pi))
            I = float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))

        total += _sphere_anomaly_eq9_SI(X, Y, Z, x0, y0, z0, R, J, D, I)

    raw_nT = (total * 1e9).astype(np.float32)
    mapped_nT = _map_to_positive_range(
        raw_nT, rng,
        global_min_nT=0.0,
        global_max_nT=global_max_nT,
        span_range_nT=(200.0, min(1500.0, global_max_nT)),
        zero_start_prob=0.35,
        min_offset_range_nT=(50.0, 900.0),
        robust_percentiles=(1.0, 99.0),
    )
    return mapped_nT


# ============================================================
# 2) [DEPRECATED] MinMax Normalization helpers (Removed)
# ============================================================
# def normalize_to_minus1_1(arr, vmin, vmax): ...
# def denormalize_from_minus1_1(arr_norm, vmin, vmax): ...


# ============================================================
# 3) Vertical grid-aligned sampling with "realism" (still on-grid)
# ============================================================
def sample_vertical_gridlines(
        grid_size,
        alpha_grid,  # int: line spacing in columns
        beta_grid,  # int: along-line spacing in rows
        seed,
        random_phase=True,  # random x offset and y offset but keep spacing fixed
        line_drop_prob=0.10,  # probability to drop an entire line
        dropout_prob=0.15,  # random point dropout on remaining lines
        gap_count=1,  # number of missing segments per line
        gap_len_frac_min=0.05,  # gap length as fraction of points on that line
        gap_len_frac_max=0.22,
        trim_prob=0.70,  # probability to trim both ends
        trim_frac_max=0.15,  # trim length max fraction (each end)
):
    """
    输出 coords: (N,2) int32, coords[:,0]=x, coords[:,1]=y
    规则：
      - 测线竖直：每条线 x 为整数列
      - 沿线采样：y 为整数行，步长 beta_grid
      - 加真实感：trim/gap/dropout/整条线丢失，但点始终在网格上
    """
    rng = np.random.RandomState(seed)

    alpha = int(max(1, alpha_grid))
    beta = int(max(1, beta_grid))

    # x lines: x0, x0+alpha, ...
    if random_phase:
        x0 = int(rng.randint(0, alpha))
    else:
        x0 = 0
    xs = np.arange(x0, grid_size, alpha, dtype=np.int32)

    # maybe drop some lines
    if line_drop_prob > 0 and xs.size > 1:
        keep = rng.rand(xs.size) > float(line_drop_prob)
        if not np.any(keep):
            keep[rng.randint(0, xs.size)] = True
        xs = xs[keep]

    coords = []

    # y sampling positions
    if random_phase:
        y0 = int(rng.randint(0, beta))
    else:
        y0 = 0
    base_ys = np.arange(y0, grid_size, beta, dtype=np.int32)
    if base_ys.size == 0:
        base_ys = np.arange(0, grid_size, 1, dtype=np.int32)

    for x in xs.tolist():
        ys = base_ys.copy()

        # trim ends
        if rng.rand() < float(trim_prob) and ys.size > 10:
            trim_n = int(rng.uniform(0.0, float(trim_frac_max)) * ys.size)
            trim_n2 = int(rng.uniform(0.0, float(trim_frac_max)) * ys.size)
            start = trim_n
            end = ys.size - trim_n2
            if end - start >= 5:
                ys = ys[start:end]

        if ys.size < 5:
            continue

        # gaps: remove contiguous segments along this line
        if int(gap_count) > 0 and ys.size > 20:
            mask = np.ones(ys.size, dtype=bool)
            for _ in range(int(gap_count)):
                frac = rng.uniform(float(gap_len_frac_min), float(gap_len_frac_max))
                glen = max(1, int(frac * ys.size))
                s = rng.randint(0, max(1, ys.size - glen))
                mask[s:s + glen] = False
            ys = ys[mask]

        if ys.size < 5:
            continue

        # point dropout
        if dropout_prob > 0:
            mask = rng.rand(ys.size) > float(dropout_prob)
            ys = ys[mask]

        if ys.size < 3:
            continue

        # add points
        x_arr = np.full((ys.size,), int(x), dtype=np.int32)
        coords.append(np.stack([x_arr, ys.astype(np.int32)], axis=1))

    if len(coords) == 0:
        # fallback random grid points
        N = 300
        x = rng.randint(0, grid_size, size=N).astype(np.int32)
        y = rng.randint(0, grid_size, size=N).astype(np.int32)
        return np.stack([x, y], axis=1).astype(np.int32)

    coords = np.concatenate(coords, axis=0).astype(np.int32)
    # shuffle (still fine; scatter plot doesn't need order)
    coords = coords[rng.permutation(coords.shape[0])]
    return coords


# ============================================================
# 4) Save helpers + preview scatter (axes 0..127)
# ============================================================
def _save_test_only_dataset(out_dir, GT_norm, GT_mean, GT_std, coords_obj, vals_norm_obj):
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    # 1. 保存归一化后的 GT (Z-Score Normalized)
    np.save(os.path.join(test_dir, "GT_test.npy"), GT_norm.astype(np.float32))

    # 2. 保存归一化参数 (Instance-wise Mean/Std)
    np.save(os.path.join(test_dir, "GT_test_mean.npy"), GT_mean.astype(np.float32))
    np.save(os.path.join(test_dir, "GT_test_std.npy"), GT_std.astype(np.float32))

    # 3. 保存归一化后的采样点 (Irregular Points Normalized)
    np.save(os.path.join(test_dir, "irregular_coords_test.npy"), coords_obj, allow_pickle=True)
    np.save(os.path.join(test_dir, "irregular_values_test.npy"), vals_norm_obj, allow_pickle=True)


def _save_preview_png_scatter_only(out_dir, GT_norm, coords_obj, vals_norm_obj, grid_size, sample_ids=(0, 1, 2)):
    ids = [i for i in sample_ids if i < len(coords_obj)]
    rows = int(np.ceil(len(ids) / 3.0))
    cols = min(3, len(ids))
    fig = plt.figure(figsize=(6.5 * cols, 5.8 * rows))

    for k, sid in enumerate(ids):
        ax = plt.subplot(rows, cols, k + 1)

        # 直接使用归一化后的值进行画图
        vals_z = np.asarray(vals_norm_obj[sid], dtype=np.float32)
        coords = np.asarray(coords_obj[sid], dtype=np.int32)

        x = coords[:, 0].astype(np.int32)
        y = coords[:, 1].astype(np.int32)

        sc = ax.scatter(x, y, s=10, c=vals_z, cmap="viridis", linewidths=0.0)
        ax.set_title("Sample {} Scatter (Z-score)\nvals:[{:.2f},{:.2f}]".format(
            sid, float(vals_z.min()), float(vals_z.max())
        ))
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        ax.set_aspect("equal")
        ax.set_xlim(0, grid_size - 1)
        ax.set_ylim(0, grid_size - 1)
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, label="Sigma", fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "preview_samples_zscore.png"), dpi=220)
    plt.close(fig)


# ============================================================
# 5) CLI alpha/beta groups
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Generate 2x3 sampling experiments (test-only), vertical grid-aligned lines.")

    # basic
    p.add_argument("--out_root", type=str, default="magnetic_dataset_zscore", help="output root directory")
    p.add_argument("--num_test", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--area_m", type=float, default=10000.0)
    p.add_argument("--flight_height", type=float, default=2000.0)

    # magnetic params (keep as you used before)
    p.add_argument("--n_sources_min", type=int, default=3)
    p.add_argument("--n_sources_max", type=int, default=10)
    p.add_argument("--depth_min", type=float, default=800.0)
    p.add_argument("--depth_max", type=float, default=3500.0)
    p.add_argument("--radius_min", type=float, default=80.0)
    p.add_argument("--radius_max", type=float, default=500.0)
    p.add_argument("--J_min", type=float, default=0.5)
    p.add_argument("--J_max", type=float, default=2.0)
    p.add_argument("--global_max_nT", type=float, default=2000.0)

    p.add_argument("--dir_mode", type=str, default="random", choices=["random", "align"])
    p.add_argument("--align_A", type=float, default=0.0)
    p.add_argument("--align_I_deg", type=float, default=60.0)
    p.add_argument("--dir_jitter", type=float, default=0.15)

    # realism knobs (sampling)
    p.add_argument("--random_phase", action="store_true", help="random x/y phase (still fixed spacing), default False")
    p.set_defaults(random_phase=True)

    p.add_argument("--line_drop_prob", type=float, default=0.0, help="probability to drop an entire vertical line")
    p.add_argument("--dropout_prob", type=float, default=0.15, help="random point dropout on remaining points")
    p.add_argument("--gap_count", type=int, default=0, help="number of missing segments per line")
    p.add_argument("--gap_len_frac_min", type=float, default=0.05)
    p.add_argument("--gap_len_frac_max", type=float, default=0.22)
    p.add_argument("--trim_prob", type=float, default=0.0)
    p.add_argument("--trim_frac_max", type=float, default=0.15)

    # ---- Experiment A: fixed alpha, vary beta ----
    p.add_argument("--A_alpha_grid", type=int, default=15, help="A groups: fixed alpha (columns)")
    p.add_argument("--A_beta_dense", type=int, default=2, help="A dense beta (rows step)")
    p.add_argument("--A_beta_medium", type=int, default=4)
    p.add_argument("--A_beta_sparse", type=int, default=8)

    # ---- Experiment B: fixed beta, vary alpha ----
    p.add_argument("--B_beta_grid", type=int, default=4, help="B groups: fixed beta (rows step)")
    p.add_argument("--B_alpha_dense", type=int, default=10)
    p.add_argument("--B_alpha_medium", type=int, default=15)
    p.add_argument("--B_alpha_sparse", type=int, default=20)

    return p


# ============================================================
# 6) Generate 6 datasets
# ============================================================
def generate_sampling_experiments_test_only(args):
    os.makedirs(args.out_root, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- generate GT once (shared across 6 groups) ----
    # 这里的 GT_raw 存储原始物理数值 (nT)
    GT_raw = np.zeros((args.num_test, args.grid_size, args.grid_size), dtype=np.float32)

    print("Generating {} GT fields (shared across 6 groups)...".format(args.num_test))
    for i in range(args.num_test):
        sample_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1)))
        field_nT = generate_field_one_sample(
            rng=sample_rng,
            grid_size=args.grid_size,
            area_m=args.area_m,
            flight_height=args.flight_height,
            n_sources=(args.n_sources_min, args.n_sources_max),
            depth_range=(args.depth_min, args.depth_max),
            radius_range=(args.radius_min, args.radius_max),
            J_range=(args.J_min, args.J_max),
            dir_mode=args.dir_mode,
            align_A=args.align_A,
            align_I=np.deg2rad(args.align_I_deg),
            dir_jitter=args.dir_jitter,
            global_max_nT=args.global_max_nT,
        )
        GT_raw[i] = field_nT

        if (i + 1) % max(1, args.num_test // 10) == 0:
            print("  [{}/{}] GT generated".format(i + 1, args.num_test))

    # ---- group definitions (2x3) ----
    groups = [
        ("exp_A_alphaFixed_beta_dense", args.A_alpha_grid, args.A_beta_dense),
        ("exp_A_alphaFixed_beta_medium", args.A_alpha_grid, args.A_beta_medium),
        ("exp_A_alphaFixed_beta_sparse", args.A_alpha_grid, args.A_beta_sparse),

        ("exp_B_betaFixed_alpha_dense", args.B_alpha_dense, args.B_beta_grid),
        ("exp_B_betaFixed_alpha_medium", args.B_alpha_medium, args.B_beta_grid),
        ("exp_B_betaFixed_alpha_sparse", args.B_alpha_sparse, args.B_beta_grid),
    ]

    # ---- for each group: sample coords on-grid, take values from grid ----
    for gname, alpha_grid, beta_grid in groups:
        out_dir = os.path.join(args.out_root, gname)
        os.makedirs(out_dir, exist_ok=True)

        GT_norm_arr = np.zeros_like(GT_raw)
        GT_mean_arr = np.zeros((args.num_test,), dtype=np.float32)
        GT_std_arr = np.zeros((args.num_test,), dtype=np.float32)

        coords_obj = np.empty((args.num_test,), dtype=object)
        vals_norm_obj = np.empty((args.num_test,), dtype=object)
        n_points_list = []

        print("\n[Group] {}  alpha_grid={}, beta_grid={}".format(gname, int(alpha_grid), int(beta_grid)))

        for i in range(args.num_test):
            samp_seed = int(args.seed + 100000 + i * 97)

            # 1. 采样坐标
            coords = sample_vertical_gridlines(
                grid_size=args.grid_size,
                alpha_grid=int(alpha_grid),
                beta_grid=int(beta_grid),
                seed=samp_seed,
                random_phase=bool(args.random_phase),
                line_drop_prob=float(args.line_drop_prob),
                dropout_prob=float(args.dropout_prob),
                gap_count=int(args.gap_count),
                gap_len_frac_min=float(args.gap_len_frac_min),
                gap_len_frac_max=float(args.gap_len_frac_max),
                trim_prob=float(args.trim_prob),
                trim_frac_max=float(args.trim_frac_max),
            )

            # 2. 从 Raw GT 中获取采样点数值
            field_nT = GT_raw[i]
            vals_nT = field_nT[coords[:, 1], coords[:, 0]].astype(np.float32)

            # 3. 【关键】计算基于采样点的 Z-Score 统计量
            sample_mean = np.mean(vals_nT)
            sample_std = np.std(vals_nT)
            if sample_std < 1e-8:
                sample_std = 1.0  # 避免除以0

            # 4. 执行 Instance-wise Z-Score 归一化
            # 对采样点归一化
            vals_z = (vals_nT - sample_mean) / sample_std

            # 对整张 GT 图归一化 (使用相同的采样点统计量，保证一致性)
            gt_z = (field_nT - sample_mean) / sample_std

            # 5. 存储
            GT_norm_arr[i] = gt_z
            GT_mean_arr[i] = sample_mean
            GT_std_arr[i] = sample_std

            coords_obj[i] = coords.astype(np.int32)
            vals_norm_obj[i] = vals_z.astype(np.float32)

            n_points_list.append(int(coords.shape[0]))

        _save_test_only_dataset(out_dir, GT_norm_arr, GT_mean_arr, GT_std_arr, coords_obj, vals_norm_obj)

        meta = {
            "group": gname,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test": int(args.num_test),
            "grid_size": int(args.grid_size),
            "alpha_grid": int(alpha_grid),
            "beta_grid": int(beta_grid),
            "normalization": "z-score (instance-wise, based on sampled points)",
            "sampling_params": {
                "direction": "vertical (x fixed, y varies)",
                "random_phase": bool(args.random_phase),
                "line_drop_prob": float(args.line_drop_prob),
                "dropout_prob": float(args.dropout_prob),
                "gap_count": int(args.gap_count),
                "gap_len_frac_min": float(args.gap_len_frac_min),
                "gap_len_frac_max": float(args.gap_len_frac_max),
                "trim_prob": float(args.trim_prob),
                "trim_frac_max": float(args.trim_frac_max),
                "values_from": "GT grid (field[y,x])",
                "coords_integer_grid": True
            },
            "points_count_stats": {
                "min": int(np.min(n_points_list)),
                "max": int(np.max(n_points_list)),
                "mean": float(np.mean(n_points_list)),
                "median": float(np.median(n_points_list)),
                "p10": float(np.percentile(n_points_list, 10)),
                "p90": float(np.percentile(n_points_list, 90)),
            },
        }

        with open(os.path.join(out_dir, "sampling_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        _save_preview_png_scatter_only(out_dir, GT_norm_arr, coords_obj, vals_norm_obj, args.grid_size,
                                       sample_ids=(0, 1, 2))

        print("  Saved: {}".format(out_dir))
        print("  Points stats: {}".format(meta["points_count_stats"]))
        print("  Preview: {}".format(os.path.join(out_dir, "preview_samples_zscore.png")))

    print("\n✅ All 6 test-only datasets saved under: {}/".format(args.out_root))


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_sampling_experiments_test_only(args)