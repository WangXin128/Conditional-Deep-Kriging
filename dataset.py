import os
import torch
import numpy as np
from torch.utils.data import Dataset


def normalize_coords_index_to_unit(coords_idx: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Normalize integer index coordinates (ranging from 0 to grid_size − 1) to the interval [0, 1].
    coords_idx: (B, N, 2) or (N, 2)
    """
    return coords_idx.float() / (grid_size - 1)


def make_full_grid_coords(grid_size: int, device: torch.device) -> torch.Tensor:
    """
    Generate grid coordinates in [0, 1] range with shape (G, 2), where G = grid_size^2.
    """
    y = torch.linspace(0, 1, grid_size, device=device)
    x = torch.linspace(0, 1, grid_size, device=device)
    # meshgrid indexing='xy' means first dim is x, second is y
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    # stack -> (grid_size, grid_size, 2) -> (G, 2)
    # NOTE: The order here must match the coords format used in the dataset.
    # Usually image indexing is (row, col) -> (y, x).
    # If your coords are (x, y), you need to reverse the stacking order here.
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    return grid


class MagneticNpyDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "test", grid_size: int = 128, load_minmax: bool = True):
        super().__init__()
        self.split = split
        self.grid_size = grid_size

        # Build paths
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Data directory not found: {split_dir}")

        # 1. Load GT (Z-Score Normalized)
        # Shape: (N_samples, H, W)
        gt_path = os.path.join(split_dir, f"GT_{split}.npy")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        self.gts = np.load(gt_path).astype(np.float32)

        # 2. Load sampling coordinates
        # Shape: (N_samples,) object array -> each element is (K, 2)
        coords_path = os.path.join(split_dir, f"irregular_coords_{split}.npy")
        self.coords_arr = np.load(coords_path, allow_pickle=True)

        # 3. Load sampled values (Z-Score Normalized)
        # Shape: (N_samples,) object array -> each element is (K,)
        vals_path = os.path.join(split_dir, f"irregular_values_{split}.npy")
        self.vals_arr = np.load(vals_path, allow_pickle=True)

        self.has_stats = False
        self.means = None
        self.stds = None

        # 4. Load statistics (Mean/Std) for de-normalization
        # Even though the argument is called load_minmax, we now make it compatible with reading mean/std as well
        if load_minmax:
            mean_path = os.path.join(split_dir, f"GT_{split}_mean.npy")
            std_path = os.path.join(split_dir, f"GT_{split}_std.npy")

            # Also check legacy min/max naming to avoid errors
            min_path = os.path.join(split_dir, f"GT_{split}_min.npy")
            max_path = os.path.join(split_dir, f"GT_{split}_max.npy")

            if os.path.exists(mean_path) and os.path.exists(std_path):
                print(f"[Dataset] Loading Mean/Std from {split_dir}")
                self.means = np.load(mean_path).astype(np.float32)
                self.stds = np.load(std_path).astype(np.float32)
                self.has_stats = True
                self.stat_type = "zscore"
            elif os.path.exists(min_path) and os.path.exists(max_path):
                print(f"[Dataset] Loading Min/Max from {split_dir} (Legacy Mode)")
                # For interface compatibility, map min to mean (shift), and (max-min) to std (scale)
                # Then x * std + mean becomes x * (max-min) + min (assuming input is 0..1)
                # Or if input is -1..1: x = (val - min)/(max-min)*2 - 1
                # -> val = (x+1)/2 * (max-min) + min = x * (max-min)/2 + (max-min)/2 + min
                # This is a bit complex; here we keep a simplified handling:
                self.gt_mins = np.load(min_path).astype(np.float32)
                self.gt_maxs = np.load(max_path).astype(np.float32)
                self.has_stats = True
                self.stat_type = "minmax"
            else:
                print(f"[Dataset] Warning: No stat files (mean/std or min/max) found in {split_dir}")

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        # 1. GT Image (1, H, W)
        gt_img = torch.from_numpy(self.gts[idx]).unsqueeze(0)

        # 2. Coords (N, 2)
        coords = self.coords_arr[idx]  # (N, 2) int
        coords_t = torch.from_numpy(coords).long()

        # 3. Values (N,)
        vals = self.vals_arr[idx]  # (N,) float
        vals_t = torch.from_numpy(vals).float()

        item = {
            "gt": gt_img,
            "coords_idx": coords_t,
            "values": vals_t,
            "index": idx
        }

        # 4. Attach statistics
        if self.has_stats:
            if self.stat_type == "zscore":
                # Directly attach mean/std
                item["gt_mean"] = torch.tensor(self.means[idx], dtype=torch.float32)
                item["gt_std"] = torch.tensor(self.stds[idx], dtype=torch.float32)

            elif self.stat_type == "minmax":
                # Legacy compatibility: attach min/max
                item["gt_min"] = torch.tensor(self.gt_mins[idx], dtype=torch.float32)
                item["gt_max"] = torch.tensor(self.gt_maxs[idx], dtype=torch.float32)

                # To avoid errors in instance_optimize.py, also fabricate a mean/std
                # For MinMax in [-1, 1]:
                # phys = norm * scale_half + center
                # center = (max + min) / 2
                # scale_half = (max - min) / 2
                vmin = self.gt_mins[idx]
                vmax = self.gt_maxs[idx]
                item["gt_mean"] = torch.tensor((vmax + vmin) / 2.0, dtype=torch.float32)
                item["gt_std"] = torch.tensor((vmax - vmin) / 2.0, dtype=torch.float32)

        return item
