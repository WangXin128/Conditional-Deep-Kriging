# visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

SAVE_DIR = "results_vis"


# ===============================
# Formatter helper
# ===============================
def create_custom_formatter_and_label(vmin, vmax, unit=""):
    def formatter(x, pos):
        return f"{x:.0f}"
    label = unit
    return FuncFormatter(formatter), label


# ===============================
# Main visualization
# ===============================
def visualize_and_save(all_results, vis_count=20):
    os.makedirs(SAVE_DIR, exist_ok=True)
    colors_error = ["#2A9D8F", "#FFFAF0", "#E76F51"]
    cmap_premium_error = LinearSegmentedColormap.from_list(
        "custom_diverging", colors_error
    )
    cmap_main = 'jet'
    grid_size = 128

    print("📊 Generating Visualizations...")

    for idx, res in enumerate(all_results):
        if idx >= vis_count:
            break

        gt = res['gt'].reshape(grid_size, grid_size)
        pred = res['pred'].reshape(grid_size, grid_size)

        sample_x = res['coords'][:, 0]
        sample_y = res['coords'][:, 1]
        sample_val = res['input_vals']

        residual = pred - gt
        abs_error = np.abs(residual)

        vmin, vmax = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        levels = np.linspace(vmin, vmax, 21)
        fmt, lbl = create_custom_formatter_and_label(vmin, vmax, "nT")

        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.3)

        # 1. Input
        ax1 = plt.subplot(gs[0, 0])
        sc = ax1.scatter(
            sample_x, sample_y,
            c=sample_val, cmap=cmap_main,
            vmin=vmin, vmax=vmax,
            s=25, edgecolors='k', linewidth=0.2
        )
        ax1.set_title(f"Input (n={len(sample_x)})")
        ax1.set_xlim(0, 128)
        ax1.set_ylim(128, 0)
        ax1.set_aspect('equal')
        plt.colorbar(sc, ax=ax1, format=fmt).ax.set_title(lbl)

        # 2. GT
        ax2 = plt.subplot(gs[0, 1])
        cf2 = ax2.contourf(X, Y, gt, levels=levels, cmap=cmap_main)
        ax2.contour(X, Y, gt, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
        ax2.set_title("Ground Truth")
        ax2.set_xlim(0, 128)
        ax2.set_ylim(128, 0)
        ax2.set_aspect('equal')
        plt.colorbar(cf2, ax=ax2, format=fmt).ax.set_title(lbl)

        # 3. Pred
        ax3 = plt.subplot(gs[0, 2])
        cf3 = ax3.contourf(X, Y, pred, levels=levels, cmap=cmap_main)
        ax3.contour(X, Y, pred, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
        ax3.set_title("PC-NIERT Pred")
        ax3.set_xlim(0, 128)
        ax3.set_ylim(128, 0)
        ax3.set_aspect('equal')
        plt.colorbar(cf3, ax=ax3, format=fmt).ax.set_title(lbl)

        # 4. Error map
        limit = max(abs(residual.min()), abs(residual.max()), 1e-9)
        ax4 = plt.subplot(gs[1, 0])
        im4 = ax4.imshow(
            residual, cmap=cmap_premium_error,
            vmin=-limit, vmax=limit,
            extent=[0, 128, 128, 0]
        )
        ax4.set_title("Difference Map")
        efmt, elbl = create_custom_formatter_and_label(-limit, limit, "nT")
        plt.colorbar(im4, ax=ax4, format=efmt).ax.set_title(elbl)

        # 5. Histogram
        ax5 = plt.subplot(gs[1, 1:])
        valid_err = abs_error.flatten()[np.isfinite(abs_error.flatten())]
        if len(valid_err) > 0:
            ax5.hist(
                valid_err, bins=100, color='#E9967A', alpha=0.9,
                range=(0, np.percentile(valid_err, 99.5) * 1.1)
            )

        ax5.set_title("Error Histogram")
        ax5.text(
            0.95, 0.95,
            f"MAE: {res['mae']:.2f}\n"
            f"RMSE: {res['rmse']:.2f}\n"
            f"MAPE: {res['mape']:.2f}%\n"
            f"Time: {res['time']:.4f}s",
            transform=ax5.transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.9)
        )

        plt.savefig(
            os.path.join(SAVE_DIR, f"sample_{res['id']:03d}.png"),
            dpi=150
        )
        plt.close()

    # ===============================
    # Save TXT
    # ===============================
    with open(os.path.join(SAVE_DIR, "metrics_report.txt"), "w") as f:
        maes = [r['mae'] for r in all_results if np.isfinite(r['mae'])]
        avg_mae = np.mean(maes) if maes else -1
        f.write(f"Avg MAE: {avg_mae:.4f}\n")
        for r in all_results:
            f.write(
                f"ID {r['id']:03d} | "
                f"MAE {r['mae']:.4f} | "
                f"RMSE {r['rmse']:.4f} | "
                f"MAPE {r['mape']:.2f}% | "
                f"Time {r['time']:.4f}s\n"
            )
