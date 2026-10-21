"""Reproduce the existing publication figures using the full-size DOE data."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KMCS import save_topview_image, species_colors
from analyze_npz_3d import plot_voxels


def doe_matrix(x1, x2):
    return np.column_stack([np.ones_like(x1), x1, x2, x1*x2, x1*x1, x2*x2])


def main():
    data = json.loads((ROOT / "validation/full_size_doe_2026-10-21.json").read_text())
    rows = data["results"]
    target = ROOT / "figures"
    temperatures = np.array([r["T_C"] for r in rows], dtype=float)
    frequencies = np.array([r["f_Hz"] for r in rows], dtype=float)
    density = np.array([r["density_per_nm2"] for r in rows])
    x1 = 2*(frequencies-frequencies.min())/np.ptp(frequencies)-1
    x2 = 2*(temperatures-temperatures.min())/np.ptp(temperatures)-1
    matrix = doe_matrix(x1, x2)
    coefficients = np.linalg.lstsq(matrix, density, rcond=None)[0]
    fitted = matrix @ coefficients
    r2 = 1-np.sum((density-fitted)**2)/np.sum((density-density.mean())**2)
    f_grid, t_grid = np.meshgrid(
        np.linspace(frequencies.min(), frequencies.max(), 200),
        np.linspace(temperatures.min(), temperatures.max(), 200))
    grid_x1 = 2*(f_grid-frequencies.min())/np.ptp(frequencies)-1
    grid_x2 = 2*(t_grid-temperatures.min())/np.ptp(temperatures)-1
    z = (doe_matrix(grid_x1.ravel(), grid_x2.ravel()) @ coefficients).reshape(f_grid.shape)

    # Preserve the original DOE_plot.py presentation; only the data are new.
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[4.8, 1.45, .18],
                         height_ratios=[1.25, 4.2], wspace=.12, hspace=.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)
    cax = fig.add_subplot(gs[1, 2])
    fig.add_subplot(gs[0, 1:]).axis("off")
    levels = np.linspace(density.min(), z.max(), 10)
    lines = ax_main.contour(f_grid, t_grid, z, levels=levels, cmap="viridis", linewidths=1)
    ax_main.clabel(lines, inline=True, fontsize=8, fmt="%.3f")
    ax_main.scatter(frequencies, temperatures, s=38, facecolors="white",
                    edgecolors="black", linewidths=.8, zorder=3, clip_on=False)
    ax_main.axhline(800, color="tab:blue", linestyle="--", linewidth=1.2)
    ax_main.axvline(20, color="tab:orange", linestyle="--", linewidth=1.2)
    ax_main.set_xlabel("Frequency (Hz)")
    ax_main.set_ylabel("Temperature (Celsius)")
    ax_main.grid(True, color="0.9", linewidth=.6)
    freq_cut = sorted((r for r in rows if r["T_C"] == 800), key=lambda r:r["f_Hz"])
    temp_cut = sorted((r for r in rows if r["f_Hz"] == 20), key=lambda r:r["T_C"])
    ax_top.scatter([r["f_Hz"] for r in freq_cut], [r["density_per_nm2"] for r in freq_cut],
                   s=32, facecolors="white", edgecolors="tab:blue", linewidths=1, zorder=3, clip_on=False)
    ax_top.set_ylabel("Density")
    ax_top.set_title("OVAT: T = 800 C")
    ax_top.grid(True, color="0.9", linewidth=.6)
    ax_top.tick_params(labelbottom=False)
    ax_top.margins(y=.15)
    ax_side.scatter([r["density_per_nm2"] for r in temp_cut], [r["T_C"] for r in temp_cut],
                    s=32, facecolors="white", edgecolors="tab:orange", linewidths=1, zorder=3, clip_on=False)
    ax_side.set_xlabel("Density")
    ax_side.set_title("OVAT: f = 20 Hz")
    ax_side.grid(True, color="0.9", linewidth=.6)
    ax_side.tick_params(labelleft=False)
    ax_side.margins(x=.15)
    fig.suptitle("Number density per nm^2", y=.98)
    fig.colorbar(lines, cax=cax, label="Number density / nm^2")
    fig.subplots_adjust(top=.91)
    fig.savefig(target / "number_density.png", dpi=200)
    plt.close(fig)
    print(f"Quadratic fit R^2 = {r2:.6f}; fitted minimum = {z.min():.6g} (fit, not a simulation value)")

    with np.load(ROOT / "examples/snapshot_final.npz", allow_pickle=False) as saved:
        snapshot = {key:saved[key] for key in saved.files}
    save_topview_image(snapshot["Occ"], snapshot["H"], species_colors,
                       target / "snapshot_2mat.png", title="Snapshot final: pulse 300/300", scale=12)
    shutil.copyfile(target / "snapshot_2mat.png", ROOT / "examples/topview.png")
    fig = plot_voxels(snapshot, ROOT / "examples/snapshot_final.npz",
                      crop=None, subsample=2, zmin_filter=0, alpha=1)
    fig.savefig(target / "snapshot_2mat_3d.png", dpi=100)
    plt.close(fig)
    print("Updated the existing DOE, top-view and 3D figures in their original style.")


if __name__ == "__main__":
    main()
