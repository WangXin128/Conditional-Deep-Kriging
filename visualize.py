import os
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(
        self,
        save_dir: str = "results_vis",
        sample_id: int = 34,
        cmap: str = "jet",
        levels: int = 20,
        dpi: int = 300,
    ):
        self.save_dir = save_dir
        self.sample_id = int(sample_id)
        self.cmap = cmap
        self.levels = int(levels)
        self.dpi = int(dpi)
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def _compute_vmin_vmax(gt: np.ndarray, pred: np.ndarray, vals: np.ndarray):
        vmin = min(float(np.nanmin(gt)), float(np.nanmin(pred)), float(np.nanmin(vals)))
        vmax = max(float(np.nanmax(gt)), float(np.nanmax(pred)), float(np.nanmax(vals)))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or abs(vmax - vmin) < 1e-12:
            vmin, vmax = 0.0, 1.0
        return vmin, vmax

    def _find_target_result(self, all_results: List[Dict]) -> Optional[Dict]:
        for r in all_results:
            if int(r.get("id", -1)) == self.sample_id:
                return r
        return None

    def _plot_one_row_three_cols(self, res: Dict, out_png: str):
        gt = np.asarray(res["gt"], dtype=np.float32)
        pred = np.asarray(res["pred"], dtype=np.float32)
        coords = np.asarray(res["coords"], dtype=np.float32)
        vals = np.asarray(res["input_vals"], dtype=np.float32)

        if gt.ndim != 2 or pred.ndim != 2:
            raise ValueError("res['gt'] and res['pred'] must be 2D arrays.")

        grid_h, grid_w = gt.shape
        X, Y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))

        vmin, vmax = self._compute_vmin_vmax(gt, pred, vals)
        levels = np.linspace(vmin, vmax, self.levels)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        plt.subplots_adjust(wspace=0.06, right=0.92)

        # 1) Ground Truth
        ax = axes[0]
        cf0 = ax.contourf(X, Y, gt, levels=levels, cmap=self.cmap, vmin=vmin, vmax=vmax)
        ax.contour(X, Y, gt, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
        ax.set_title("Ground Truth")
        ax.set_xlim(0, grid_w - 1)
        ax.set_ylim(grid_h - 1, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        # 2) Observations
        ax = axes[1]
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            s=14,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"Observed Points (n={len(coords)})")
        ax.set_xlim(0, grid_w - 1)
        ax.set_ylim(grid_h - 1, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        # 3) Prediction
        ax = axes[2]
        cf2 = ax.contourf(X, Y, pred, levels=levels, cmap=self.cmap, vmin=vmin, vmax=vmax)
        ax.contour(X, Y, pred, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
        ax.set_title("Prediction")
        ax.set_xlim(0, grid_w - 1)
        ax.set_ylim(grid_h - 1, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        # shared colorbar
        cbar = fig.colorbar(cf2, ax=axes, fraction=0.025, pad=0.02)
        cbar.ax.set_title("nT")

        fig.savefig(out_png, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight", format="pdf")
        plt.close(fig)

    def visualize_and_save(self, all_results: List[Dict], vis_count: int = 20):
        # vis_count 参数保留，只为兼容 train.py 的调用
        if not all_results:
            print("[Visualizer] all_results is empty, skip visualization.")
            return

        res = self._find_target_result(all_results)
        if res is None:
            print(
                f"[Visualizer] sample id {self.sample_id} not found in all_results. "
                f"Available id range: {[r.get('id', None) for r in all_results[:5]]} ..."
            )
            return

        out_png = os.path.join(self.save_dir, f"sample_{self.sample_id:03d}_gt_obs_pred.png")
        self._plot_one_row_three_cols(res, out_png)
        print(f"[Visualizer] Saved figure to {out_png}")