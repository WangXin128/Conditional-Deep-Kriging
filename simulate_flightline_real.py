import os
import struct
import zlib
from pathlib import Path

import numpy as np
import scipy.interpolate


def load_geosoft_grd(grd_path: str):
    """
    Read a compressed Geosoft/Oasis Montaj GRD file.
    Returns:
    z: (NV, NE) float64 array, with NoData values converted to NaN
    header: dict (containing resolution information such as DE/DV, etc.)
    """
    p = Path(grd_path)
    with p.open("rb") as f:
        h = f.read(512)
        ES, SF, NE, NV, KX = struct.unpack("<5i", h[0:20])
        DE, DV, X0, Y0, ROT = struct.unpack("<5d", h[20:60])
        ZBASE, ZMULT = struct.unpack("<2d", h[60:76])

        header = dict(
            ES=ES, SF=SF, NE=NE, NV=NV, KX=KX,
            DE=DE, DV=DV, X0=X0, Y0=Y0, ROT=ROT,
            ZBASE=ZBASE, ZMULT=ZMULT
        )

        comp_hdr = f.read(16)
        magic, ver, nblocks, strip_h = struct.unpack("<4i", comp_hdr)

        offsets = struct.unpack("<" + "q" * nblocks, f.read(8 * nblocks))
        sizes = struct.unpack("<" + "i" * nblocks, f.read(4 * nblocks))

        raw_parts = []
        for off, size in zip(offsets, sizes):
            f.seek(off)
            block = f.read(size)
            raw_parts.append(zlib.decompress(block[16:]))

    raw = b"".join(raw_parts)
    z = np.frombuffer(raw, dtype="<f4").reshape((header["NV"], header["NE"])).astype(np.float64)

    # NoData sentinel -> NaN
    z[z <= -1e31] = np.nan

    z = z / header["ZMULT"] + header["ZBASE"]
    return z, header


def find_all_valid_windows(z: np.ndarray, S: int):
    """
    Returns the top-left coordinates (r0, c0) of all S×S windows that contain no NaN values.
    """
    H, W = z.shape
    if S > H or S > W:
        raise ValueError(f"tile_size={S} 太大，格网只有 {H}×{W}")

    valid = np.isfinite(z).astype(np.int32)
    ii = np.zeros((H + 1, W + 1), dtype=np.int32)
    ii[1:, 1:] = valid.cumsum(0).cumsum(1)

    win = ii[S:, S:] - ii[:-S, S:] - ii[S:, :-S] + ii[:-S, :-S]
    coords = np.argwhere(win == S * S)

    if len(coords) == 0:
        raise ValueError(f"找不到完全无 NaN 的 {S}×{S} 块（可把 S 调小）")

    return coords


def pick_tile_max_std(z: np.ndarray, coords: np.ndarray, S: int, sample_k: int = 2000, seed: int = 42):

    rng = np.random.default_rng(seed)

    if len(coords) > sample_k:
        idx = rng.choice(len(coords), size=sample_k, replace=False)
        cand = coords[idx]
    else:
        cand = coords

    best = None
    best_std = -1.0
    for r0, c0 in cand:
        tile = z[r0:r0 + S, c0:c0 + S]
        s = float(np.nanstd(tile))
        if s > best_std:
            best_std = s
            best = (int(r0), int(c0))

    return best, best_std


def sample_flightlines_fixed_offset(
    S=100,
    n_points=10000,
    line_spacing=10,
    point_spacing=2,
    start_x=0,
    dropout_prob=0.05,
    seed=42
):
    """
    Generate vertical survey lines with a fixed starting offset (start_x) and introduce random point dropouts.
    Returns coords: an (N, 2) array with columns ordered as (x, y).
    """
    rng = np.random.default_rng(seed)

    line_spacing = int(max(1, line_spacing))
    point_spacing = int(max(1, point_spacing))
    start_x = start_x % line_spacing

    xs = np.arange(start_x, S, line_spacing, dtype=np.int32)
    if xs.size == 0:
        xs = np.array([start_x], dtype=np.int32)

    ys = np.arange(0, S, point_spacing, dtype=np.int32)
    if ys.size == 0:
        ys = np.array([0], dtype=np.int32)

    coords = []
    for x in xs.tolist():
        xi = np.full((ys.size,), int(x), dtype=np.int32)
        yi = ys.copy()

        if dropout_prob > 0:
            mask = rng.random(size=yi.size) > dropout_prob
            if mask.sum() < 2:
                keep_n = min(2, yi.size)
                mask[:keep_n] = True
            yi = yi[mask]
            xi = xi[mask]

        if len(yi) > 0:
            coords.append(np.stack([xi, yi], axis=1))

    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    coords = np.concatenate(coords, axis=0)
    coords = np.unique(coords, axis=0)

    if coords.shape[0] > n_points:
        idx = rng.choice(coords.shape[0], size=n_points, replace=False)
        coords = coords[idx]

    return coords.astype(np.int32)


def evaluate_sampling_quality(coords_idx, values, gt_grid, S):
    x_norm = coords_idx[:, 0] / (S - 1) * 2 - 1
    y_norm = coords_idx[:, 1] / (S - 1) * 2 - 1

    v_mean = values.mean()
    v_std = values.std() + 1e-6
    vals_norm = (values - v_mean) / v_std

    try:
        rbf = scipy.interpolate.Rbf(x_norm, y_norm, vals_norm, function="thin_plate")
        gx, gy = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S))
        pred_norm = rbf(gx, gy)
        pred = pred_norm * v_std + v_mean
        diff = np.abs(pred - gt_grid)
        return float(np.mean(diff))
    except Exception:
        return float("inf")


def find_best_offset(tile, S, line_spacing, point_spacing, dropout_prob, seed):
    best_mae = float("inf")
    best_coords = None
    best_values = None
    best_offset = -1

    for offset in range(line_spacing):
        coords = sample_flightlines_fixed_offset(
            S=S,
            line_spacing=line_spacing,
            point_spacing=point_spacing,
            start_x=offset,
            dropout_prob=dropout_prob,
            seed=seed + offset
        )

        if coords.shape[0] == 0:
            continue

        vals = tile[coords[:, 1], coords[:, 0]]
        mae = evaluate_sampling_quality(coords, vals, tile, S)

        if mae < best_mae:
            best_mae = mae
            best_coords = coords
            best_values = vals
            best_offset = offset

    if best_coords is None:
        raise RuntimeError("Failed to find a valid offset for flight-line sampling.")

    return best_coords, best_values


def main():
    # ---------- config ----------
    grd_file = "Kluane_Lake_West_mag_res.grd"
    S = 100
    out_root = Path("real_dataset_zscore")
    seed = 42
    point_spacing = 2
    dropout_prob = 0.05

    experiments = [
        ("real_dense", 5),
        ("real_medium", 10),
        ("real_sparse", 15),
    ]
    # ----------------------------

    z, hdr = load_geosoft_grd(grd_file)
    print(f"Loaded GRD: {grd_file}")
    print(f"Grid shape: {z.shape}, resolution: ({hdr['DE']}, {hdr['DV']})")

    coords_windows = find_all_valid_windows(z, S)
    (r0, c0), best_std = pick_tile_max_std(z, coords_windows, S, sample_k=2000, seed=seed)
    tile = z[r0:r0 + S, c0:c0 + S].astype(np.float32)

    for suffix, spacing in experiments:
        save_dir = out_root / suffix / "test"
        os.makedirs(save_dir, exist_ok=True)

        coords_idx, values = find_best_offset(
            tile=tile,
            S=S,
            line_spacing=spacing,
            point_spacing=point_spacing,
            dropout_prob=dropout_prob,
            seed=seed,
        )

        GT_test = tile[None, ...].astype(np.float32)

        irregular_coords = np.empty((1,), dtype=object)
        irregular_vals = np.empty((1,), dtype=object)
        irregular_coords[0] = coords_idx.astype(np.int32)
        irregular_vals[0] = values.astype(np.float32)

        mean_val = float(values.mean())
        std_val = float(values.std())
        means = np.array([mean_val], dtype=np.float32)
        stds = np.array([std_val], dtype=np.float32)

        np.save(save_dir / "GT_test.npy", GT_test)
        np.save(save_dir / "irregular_coords_test.npy", irregular_coords)
        np.save(save_dir / "irregular_values_test.npy", irregular_vals)
        np.save(save_dir / "GT_test_mean.npy", means)
        np.save(save_dir / "GT_test_std.npy", stds)

        print(f"[Saved] {save_dir}")

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()