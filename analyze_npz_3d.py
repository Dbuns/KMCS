"""Standalone viewer for saved KMCS ``.npz`` snapshots.

Run with no arguments to open a small Tkinter launcher.
CLI usage still works for scripting and batch export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if "--save" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors

MODES = ["info", "top_species", "height", "xz", "yz", "scatter3d", "voxels"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize a saved KMCS .npz snapshot."
    )
    parser.add_argument("--file", help="Path to the .npz snapshot")
    parser.add_argument(
        "--mode",
        default="scatter3d",
        choices=MODES,
        help="Visualization mode",
    )
    parser.add_argument("--x", type=int, help="x index for yz cross section")
    parser.add_argument("--y", type=int, help="y index for xz cross section")
    parser.add_argument(
        "--zmin",
        type=int,
        default=0,
        help="Only include occupied sites with z >= zmin for 3D modes",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Half-open crop bounds [xmin:xmax, ymin:ymax, zmin:zmax] for 3D modes",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Keep every Nth point in scatter mode, or stride by N in voxel mode",
    )
    parser.add_argument("--save", help="Optional output path for the figure")
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Marker size for scatter3d",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Transparency for 3D plotting",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter viewer even if CLI options are present",
    )
    return parser.parse_args(argv)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def describe_value(value: np.ndarray) -> str:
    if not isinstance(value, np.ndarray):
        return repr(value)
    if value.shape == ():
        return f"scalar dtype={value.dtype}, value={value.item()}"
    if value.size == 0:
        return f"shape={value.shape}, dtype={value.dtype}, empty"
    if np.issubdtype(value.dtype, np.number):
        return (
            f"shape={value.shape}, dtype={value.dtype}, "
            f"min={np.min(value)}, max={np.max(value)}"
        )
    return f"shape={value.shape}, dtype={value.dtype}"


def available_arrays_text(data: dict[str, np.ndarray]) -> str:
    lines = ["Available arrays in file:"]
    for name, value in data.items():
        lines.append(f"  - {name}: {describe_value(value)}")
    return "\n".join(lines)


def print_available_arrays(data: dict[str, np.ndarray]) -> None:
    print(available_arrays_text(data))


def require_array(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"Required array '{name}' is not present in the file.")
    return data[name]


def derive_occ_top(data: dict[str, np.ndarray]) -> np.ndarray:
    if "Occ_top" in data:
        return data["Occ_top"]
    occ = require_array(data, "Occ")
    height = require_array(data, "H")
    yy, xx = np.indices(height.shape)
    return occ[yy, xx, height]


def make_species_cmap(
    species_ids: np.ndarray,
) -> tuple[colors.ListedColormap, colors.BoundaryNorm]:
    ids = np.unique(species_ids.astype(int, copy=False))
    if ids.size == 0:
        ids = np.array([0], dtype=int)

    max_id = int(ids.max())
    color_table = np.ones((max_id + 1, 4), dtype=float)
    color_table[:, :3] = 0.96
    color_table[:, 3] = 1.0
    color_table[0, :3] = (0.08, 0.08, 0.08)

    base = plt.get_cmap("tab20")
    for species_id in ids:
        if species_id == 0:
            continue
        color_table[species_id] = base((species_id - 1) % base.N)

    cmap = colors.ListedColormap(color_table)
    bounds = np.arange(-0.5, max_id + 1.5, 1.0)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def add_species_colorbar(
    fig: plt.Figure, ax: plt.Axes, image, species_ids: np.ndarray
) -> None:
    ids = np.unique(species_ids.astype(int, copy=False))
    cbar = fig.colorbar(image, ax=ax, ticks=ids)
    cbar.set_label("Species id")


def validate_cross_section_index(index: int | None, upper_bound: int, name: str) -> int:
    if index is None:
        return upper_bound // 2
    if not 0 <= index < upper_bound:
        raise ValueError(f"{name} index must be in [0, {upper_bound - 1}], got {index}.")
    return index


def occupancy_stats_text(occ: np.ndarray) -> str:
    occupied = int(np.count_nonzero(occ))
    total = int(occ.size)
    fraction = occupied / total if total else 0.0
    return (
        f"Occ stats: shape={occ.shape}, occupied={occupied}/{total}, "
        f"occupancy_fraction={fraction:.6f}"
    )


def print_occ_stats(occ: np.ndarray) -> None:
    print(occupancy_stats_text(occ))


def maybe_warn_heavy_3d(mode: str, occ: np.ndarray, crop, subsample: int, zmin: int) -> None:
    if mode not in {"scatter3d", "voxels"}:
        return
    if crop is None and subsample <= 1 and zmin <= 0:
        print("Warning: plotting the full 3D volume without crop/subsample may be slow.")
        if mode == "voxels":
            print("Warning: voxel rendering is the heaviest option; prefer scatter3d first.")


def crop_occ(occ: np.ndarray, crop) -> tuple[np.ndarray, tuple[int, int, int]]:
    if crop is None:
        return occ, (0, 0, 0)

    xmin, xmax, ymin, ymax, zmin, zmax = crop
    ly, lx, lz = occ.shape
    if not (0 <= xmin < xmax <= lx and 0 <= ymin < ymax <= ly and 0 <= zmin < zmax <= lz):
        raise ValueError(
            "Crop must satisfy 0 <= xmin < xmax <= Lx, "
            "0 <= ymin < ymax <= Ly, 0 <= zmin < zmax <= Lz."
        )
    return occ[ymin:ymax, xmin:xmax, zmin:zmax], (xmin, ymin, zmin)


def scatter_inputs(
    occ: np.ndarray, crop, subsample: int, zmin_filter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if subsample < 1:
        raise ValueError("--subsample must be >= 1.")

    occ_view, offsets = crop_occ(occ, crop)
    local_zmin = max(0, zmin_filter - offsets[2])
    mask = occ_view != 0
    if local_zmin > 0:
        mask[:, :, :local_zmin] = False

    ys, xs, zs = np.nonzero(mask)
    species = occ_view[ys, xs, zs].astype(int, copy=False)

    if subsample > 1:
        keep = np.arange(xs.size) % subsample == 0
        xs, ys, zs, species = xs[keep], ys[keep], zs[keep], species[keep]

    xs = xs + offsets[0]
    ys = ys + offsets[1]
    zs = zs + offsets[2]
    return xs, ys, zs, species


def prepare_voxel_volume(
    occ: np.ndarray, crop, subsample: int, zmin_filter: int
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if subsample < 1:
        raise ValueError("--subsample must be >= 1.")

    occ_view, offsets = crop_occ(occ, crop)
    volume = occ_view.copy()

    local_zmin = max(0, zmin_filter - offsets[2])
    if local_zmin > 0:
        volume[:, :, :local_zmin] = 0

    if subsample > 1:
        volume = volume[::subsample, ::subsample, ::subsample]

    return volume, offsets


def save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    fig.tight_layout()
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")
        plt.close(fig)
        return
    plt.show()


def plot_top_species(data: dict[str, np.ndarray], source_path: Path) -> plt.Figure:
    top = derive_occ_top(data)
    cmap, norm = make_species_cmap(top)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(top, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(f"Top Species Map\n{source_path.name}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    add_species_colorbar(fig, ax, image, top)
    return fig


def plot_height(data: dict[str, np.ndarray], source_path: Path) -> plt.Figure:
    height = require_array(data, "H")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(height, origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_title(f"Height Map\n{source_path.name}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Top occupied z")
    return fig


def plot_xz(data: dict[str, np.ndarray], source_path: Path, y_index: int | None) -> plt.Figure:
    occ = require_array(data, "Occ")
    y_index = validate_cross_section_index(y_index, occ.shape[0], "y")
    section = occ[y_index, :, :]
    cmap, norm = make_species_cmap(section)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        section.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(f"x-z Cross Section at y={y_index}\n{source_path.name}")
    ax.set_xlabel("x index")
    ax.set_ylabel("z index")
    add_species_colorbar(fig, ax, image, section)
    return fig


def plot_yz(data: dict[str, np.ndarray], source_path: Path, x_index: int | None) -> plt.Figure:
    occ = require_array(data, "Occ")
    x_index = validate_cross_section_index(x_index, occ.shape[1], "x")
    section = occ[:, x_index, :]
    cmap, norm = make_species_cmap(section)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        section.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(f"y-z Cross Section at x={x_index}\n{source_path.name}")
    ax.set_xlabel("y index")
    ax.set_ylabel("z index")
    add_species_colorbar(fig, ax, image, section)
    return fig


def plot_scatter3d(
    data: dict[str, np.ndarray],
    source_path: Path,
    crop,
    subsample: int,
    zmin_filter: int,
    point_size: float,
    alpha: float,
) -> plt.Figure:
    occ = require_array(data, "Occ")
    xs, ys, zs, species = scatter_inputs(occ, crop, subsample, zmin_filter)
    cmap, norm = make_species_cmap(species)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if xs.size == 0:
        ax.text2D(0.2, 0.5, "No occupied voxels match the filter.", transform=ax.transAxes)
    else:
        ax.scatter(xs, ys, zs, c=species, cmap=cmap, norm=norm, s=point_size, alpha=alpha)

    ax.set_title(f"Occupied Sites Scatter3D\n{source_path.name}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_zlabel("z index")
    return fig


def plot_voxels(
    data: dict[str, np.ndarray],
    source_path: Path,
    crop,
    subsample: int,
    zmin_filter: int,
    alpha: float,
) -> plt.Figure:
    occ = require_array(data, "Occ")
    volume_yxz, offsets = prepare_voxel_volume(occ, crop, subsample, zmin_filter)
    volume_xyz = np.transpose(volume_yxz, (1, 0, 2))
    filled = volume_xyz != 0
    cmap, norm = make_species_cmap(volume_xyz)
    facecolors = cmap(norm(volume_xyz))
    facecolors[..., 3] = np.where(filled, alpha, 0.0)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if np.any(filled):
        ax.voxels(filled, facecolors=facecolors, edgecolor=(0, 0, 0, 0.05))
    else:
        ax.text2D(0.2, 0.5, "No occupied voxels match the filter.", transform=ax.transAxes)

    stride_note = f", stride={subsample}" if subsample > 1 else ""
    ax.set_title(
        f"Occupied Voxels{stride_note}\n"
        f"{source_path.name} (offset x={offsets[0]}, y={offsets[1]}, z={offsets[2]})"
    )
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_zlabel("z index")
    return fig


def build_figure(args: argparse.Namespace, data: dict[str, np.ndarray], source_path: Path) -> plt.Figure | None:
    if "Occ" in data:
        print_occ_stats(data["Occ"])
        maybe_warn_heavy_3d(args.mode, data["Occ"], args.crop, args.subsample, args.zmin)

    if args.mode == "info":
        return None
    if args.mode == "top_species":
        return plot_top_species(data, source_path)
    if args.mode == "height":
        return plot_height(data, source_path)
    if args.mode == "xz":
        return plot_xz(data, source_path, args.y)
    if args.mode == "yz":
        return plot_yz(data, source_path, args.x)
    if args.mode == "scatter3d":
        return plot_scatter3d(
            data, source_path, args.crop, args.subsample, args.zmin, args.point_size, args.alpha
        )
    if args.mode == "voxels":
        return plot_voxels(data, source_path, args.crop, args.subsample, args.zmin, args.alpha)
    raise ValueError(f"Unsupported mode: {args.mode}")


def run_analysis(args: argparse.Namespace) -> None:
    if not args.file:
        raise ValueError("Please choose a .npz file first.")

    source_path = Path(args.file)
    data = load_npz(source_path)

    print(f"Loaded file: {source_path}")
    print_available_arrays(data)

    fig = build_figure(args, data, source_path)
    if fig is not None:
        save_or_show(fig, args.save)


def parse_crop_text(text: str) -> list[int] | None:
    stripped = text.replace(",", " ").strip()
    if not stripped:
        return None
    values = [int(token) for token in stripped.split()]
    if len(values) != 6:
        raise ValueError("Crop must contain exactly 6 integers: xmin xmax ymin ymax zmin zmax.")
    return values


def guess_initial_file() -> str:
    runs_dir = Path("runs")
    candidates = sorted(runs_dir.rglob("snapshot_final.npz")) if runs_dir.exists() else []
    if candidates:
        return str(candidates[-1])
    return ""


def launch_gui(initial_file: str | None = None) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python environment.") from exc

    root = tk.Tk()
    root.title("KMCS NPZ Analyzer")
    root.geometry("760x440")

    file_var = tk.StringVar(value=initial_file or guess_initial_file())
    mode_var = tk.StringVar(value="scatter3d")
    x_var = tk.StringVar(value="")
    y_var = tk.StringVar(value="")
    zmin_var = tk.StringVar(value="0")
    subsample_var = tk.StringVar(value="2")
    crop_var = tk.StringVar(value="")
    point_size_var = tk.StringVar(value="8")
    alpha_var = tk.StringVar(value="0.85")
    status_var = tk.StringVar(
        value="Pick an .npz file, then click Plot. Default mode is the lightweight 3D scatter."
    )

    main = ttk.Frame(root, padding=12)
    main.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(6, weight=1)

    def set_info_text(text: str) -> None:
        info_text.configure(state="normal")
        info_text.delete("1.0", "end")
        info_text.insert("1.0", text)
        info_text.configure(state="disabled")

    def show_file_info(path_str: str) -> None:
        if not path_str:
            set_info_text("No file selected.")
            return
        data = load_npz(Path(path_str))
        text = available_arrays_text(data)
        if "Occ" in data:
            text = f"{text}\n{occupancy_stats_text(data['Occ'])}"
        set_info_text(text)

    def browse_file() -> None:
        initial_dir = str(Path(file_var.get()).parent) if file_var.get() else str(Path("runs"))
        chosen = filedialog.askopenfilename(
            title="Select KMCS snapshot",
            initialdir=initial_dir,
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        )
        if chosen:
            file_var.set(chosen)
            try:
                show_file_info(chosen)
                status_var.set("Loaded file info. Ready to plot.")
            except Exception as exc:  # pragma: no cover - GUI path
                set_info_text(f"Failed to read file:\n{exc}")
                status_var.set("Could not load selected file.")

    def build_args_from_gui() -> argparse.Namespace:
        x_value = x_var.get().strip()
        y_value = y_var.get().strip()
        return argparse.Namespace(
            file=file_var.get().strip(),
            mode=mode_var.get(),
            x=int(x_value) if x_value else None,
            y=int(y_value) if y_value else None,
            zmin=int(zmin_var.get().strip() or "0"),
            crop=parse_crop_text(crop_var.get()),
            subsample=int(subsample_var.get().strip() or "1"),
            save=None,
            point_size=float(point_size_var.get().strip() or "8"),
            alpha=float(alpha_var.get().strip() or "0.85"),
            gui=True,
        )

    def plot_from_gui() -> None:
        try:
            args = build_args_from_gui()
            if not args.file:
                raise ValueError("Please select a .npz file.")
            show_file_info(args.file)
            run_analysis(args)
            status_var.set(f"Opened {args.mode} plot for {Path(args.file).name}.")
        except Exception as exc:  # pragma: no cover - GUI path
            messagebox.showerror("KMCS NPZ Analyzer", str(exc))
            status_var.set("Plot failed. Check the message and try again.")

    ttk.Label(main, text="NPZ file").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Entry(main, textvariable=file_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
    ttk.Button(main, text="Browse...", command=browse_file).grid(
        row=0, column=2, sticky="ew", pady=(0, 8)
    )

    ttk.Label(main, text="Mode").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Combobox(main, textvariable=mode_var, values=MODES[1:], state="readonly").grid(
        row=1, column=1, sticky="w", pady=(0, 8)
    )

    options = ttk.Frame(main)
    options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    for idx in range(8):
        options.columnconfigure(idx, weight=1 if idx % 2 else 0)

    ttk.Label(options, text="x").grid(row=0, column=0, sticky="w")
    ttk.Entry(options, textvariable=x_var, width=8).grid(row=0, column=1, sticky="w", padx=(4, 12))
    ttk.Label(options, text="y").grid(row=0, column=2, sticky="w")
    ttk.Entry(options, textvariable=y_var, width=8).grid(row=0, column=3, sticky="w", padx=(4, 12))
    ttk.Label(options, text="zmin").grid(row=0, column=4, sticky="w")
    ttk.Entry(options, textvariable=zmin_var, width=8).grid(row=0, column=5, sticky="w", padx=(4, 12))
    ttk.Label(options, text="subsample").grid(row=0, column=6, sticky="w")
    ttk.Entry(options, textvariable=subsample_var, width=8).grid(
        row=0, column=7, sticky="w", padx=(4, 0)
    )

    ttk.Label(main, text="Crop").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Entry(main, textvariable=crop_var).grid(row=3, column=1, sticky="ew", pady=(0, 8))
    ttk.Label(main, text="xmin xmax ymin ymax zmin zmax").grid(
        row=3, column=2, sticky="w", pady=(0, 8)
    )

    extras = ttk.Frame(main)
    extras.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 8))
    ttk.Label(extras, text="Point size").grid(row=0, column=0, sticky="w")
    ttk.Entry(extras, textvariable=point_size_var, width=8).grid(
        row=0, column=1, sticky="w", padx=(4, 12)
    )
    ttk.Label(extras, text="Alpha").grid(row=0, column=2, sticky="w")
    ttk.Entry(extras, textvariable=alpha_var, width=8).grid(
        row=0, column=3, sticky="w", padx=(4, 12)
    )

    buttons = ttk.Frame(main)
    buttons.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))
    ttk.Button(buttons, text="Preview Info", command=lambda: show_file_info(file_var.get().strip())).grid(
        row=0, column=0, padx=(0, 8)
    )
    ttk.Button(buttons, text="Plot", command=plot_from_gui).grid(row=0, column=1)

    info_text = tk.Text(main, height=12, wrap="word")
    info_text.grid(row=6, column=0, columnspan=3, sticky="nsew")
    info_text.configure(state="disabled")

    ttk.Label(main, textvariable=status_var).grid(row=7, column=0, columnspan=3, sticky="w")

    if file_var.get():
        try:
            show_file_info(file_var.get())
        except Exception:
            set_info_text("Pick an .npz file to preview its arrays.")
    else:
        set_info_text("Pick an .npz file to preview its arrays.")

    root.mainloop()


def main() -> None:
    argv = sys.argv[1:]
    args = parse_args(argv)

    if args.gui or not argv:
        launch_gui(initial_file=args.file)
        return

    run_analysis(args)


if __name__ == "__main__":
    main()
