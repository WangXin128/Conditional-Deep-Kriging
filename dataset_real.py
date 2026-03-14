import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_coords_index_to_unit(coords_idx: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Normalize integer grid coordinates from [0, grid_size-1] to [0, 1].
    coords_idx: (B, N, 2) or (N, 2)
    """
    if grid_size <= 1:
        raise ValueError(f"grid_size must be > 1, got {grid_size}")
    return coords_idx.float() / float(grid_size - 1)


def make_full_grid_coords(grid_size: int, device: torch.device) -> torch.Tensor:
    """
    Generate query coordinates for the entire grid, with shape = (G, 2), where G = grid_size^2.
    The coordinate order is (x, y).
    """
    y = torch.linspace(0, 1, grid_size, device=device)
    x = torch.linspace(0, 1, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    return grid


class MagneticRealDataset(Dataset):

    def __init__(self, data_dir: str, split: str = "test", grid_size: int = 128, load_minmax: bool = True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.grid_size = grid_size
        self.load_minmax = load_minmax
        self.samples: List[Dict] = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        exp_dirs = []
        for child in sorted(self.data_dir.iterdir()):
            if child.is_dir():
                split_dir = child / split
                if split_dir.exists():
                    exp_dirs.append((child.name, split_dir))

        if len(exp_dirs) == 0:
            direct_split = self.data_dir / split
            if direct_split.exists():
                exp_dirs.append((self.data_dir.name, direct_split))

        if len(exp_dirs) == 0:
            raise FileNotFoundError(
                f"No valid dataset folders found under {self.data_dir}. "
                f"Expected subfolders like real_dense/test, real_medium/test, real_sparse/test"
            )

        for exp_name, split_dir in exp_dirs:
            self._load_one_experiment(exp_name, split_dir)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {self.data_dir}")

    def _load_one_experiment(self, exp_name: str, split_dir: Path):
        gt_path = split_dir / "GT_test.npy"
        coords_path = split_dir / "irregular_coords_test.npy"
        values_path = split_dir / "irregular_values_test.npy"

        if not (gt_path.exists() and coords_path.exists() and values_path.exists()):
            return

        gt = np.load(gt_path).astype(np.float32)
        coords_arr = np.load(coords_path, allow_pickle=True)
        values_arr = np.load(values_path, allow_pickle=True)

        if self.load_minmax:
            mean_path = split_dir / "GT_test_mean.npy"
            std_path = split_dir / "GT_test_std.npy"
            if not (mean_path.exists() and std_path.exists()):
                raise FileNotFoundError(f"Missing GT_test_mean.npy or GT_test_std.npy in {split_dir}")
            means = np.load(mean_path).astype(np.float32)
            stds = np.load(std_path).astype(np.float32)
        else:
            means = np.zeros((len(gt),), dtype=np.float32)
            stds = np.ones((len(gt),), dtype=np.float32)

        if not (len(gt) == len(coords_arr) == len(values_arr) == len(means) == len(stds)):
            raise ValueError(
                f"Inconsistent sample counts in {split_dir}: "
                f"GT={len(gt)}, coords={len(coords_arr)}, values={len(values_arr)}, "
                f"means={len(means)}, stds={len(stds)}"
            )

        for i in range(len(gt)):
            gt_i = np.asarray(gt[i], dtype=np.float32)          # (H, W), 物理值
            coords_i = np.asarray(coords_arr[i], dtype=np.int64)  # (K, 2), index
            values_i = np.asarray(values_arr[i], dtype=np.float32)  # (K,), 物理值
            mean_i = np.float32(means[i])
            std_i = np.float32(stds[i])

            if gt_i.ndim != 2:
                raise ValueError(f"GT sample must be 2D, got {gt_i.shape} in {split_dir}")
            if gt_i.shape[0] != self.grid_size or gt_i.shape[1] != self.grid_size:
                raise ValueError(
                    f"GT shape {gt_i.shape} does not match grid_size={self.grid_size} in {split_dir}"
                )
            if coords_i.ndim != 2 or coords_i.shape[1] != 2:
                raise ValueError(f"coords must be (K,2), got {coords_i.shape} in {split_dir}")
            if values_i.ndim != 1:
                raise ValueError(f"values must be (K,), got {values_i.shape} in {split_dir}")
            if len(coords_i) != len(values_i):
                raise ValueError(
                    f"coords/value length mismatch: {len(coords_i)} vs {len(values_i)} in {split_dir}"
                )

            std_safe = max(float(std_i), 1e-6)

            gt_z = (gt_i - mean_i) / std_safe
            values_z = (values_i - mean_i) / std_safe

            self.samples.append({
                "name": f"{exp_name}_{i}",
                "gt": gt_z.astype(np.float32),
                "coords_idx": coords_i.astype(np.int64),
                "values": values_z.astype(np.float32),
                "gt_mean": mean_i,
                "gt_std": np.float32(std_safe),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "gt": torch.from_numpy(s["gt"]).unsqueeze(0),            # (1, H, W)
            "coords_idx": torch.from_numpy(s["coords_idx"]).long(), # (K, 2)
            "values": torch.from_numpy(s["values"]).float(),        # (K,)
            "gt_mean": torch.tensor(s["gt_mean"], dtype=torch.float32),
            "gt_std": torch.tensor(s["gt_std"], dtype=torch.float32),
            "name": s["name"],
            "id": idx,
        }