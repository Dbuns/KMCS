"""Measure pillar number density from saved KMCS ``.npz`` snapshots.

This is a standalone post-processing tool. It does not import or modify the
simulator.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

PIXEL_SIZE_NM = 0.39
DEFAULT_MIN_AREA_PIXELS = 3
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
Number = Union[int, float]
ScalarOrBlank = Union[int, float, str]
Metrics = Dict[str, Union[int, float]]
Metadata = Dict[str, ScalarOrBlank]
BatchRow = Dict[str, object]
BATCH_FIELDNAMES = [
    "run_folder",
    "snapshot_file",
    "snapshot_name",
    "progress_percent",
    "T",
    "f",
    "pulses",
    "Estatic",
    "E_ES",
    "seed",
    "species",
    "min_area_pixels",
    "connectivity",
    "raw_count",
    "filtered_count",
    "density_per_nm2",
    "density_per_um2",
    "mean_area_pixels",
    "median_area_pixels",
    "area_fraction",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pillar count and number density from a KMCS .npz snapshot, "
            "or batch-analyze multiple run folders."
        )
    )
    parser.add_argument("--file", help="Path to a single saved .npz snapshot")
    parser.add_argument(
        "--parent",
        help="Parent folder containing many run folders to batch-analyze",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        help="Explicit run folder paths to batch-analyze",
    )
    parser.add_argument(
        "--species",
        type=int,
        default=2,
        help="Top-surface species id to count as pillars",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA_PIXELS,
        help="Minimum connected-component area in pixels to keep",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="2D connected-component connectivity",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help=(
            "Single-file mode: save the overlay image. Overlays are still saved by default "
            "unless --no-overlay is used. Batch mode: also enables per-snapshot overlays."
        ),
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Batch mode only: save one overlay image per analyzed snapshot",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Single-file mode: skip writing the overlay image.",
    )
    parser.add_argument(
        "--csv",
        help="Single-file mode: CSV output path. Defaults to pillar_density.csv next to the .npz file.",
    )
    parser.add_argument(
        "--overlay",
        help=(
            "Single-file mode: overlay image path. Defaults to "
            "pillar_density_overlay_species_<id>.png next to the .npz file."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Batch mode: combined CSV path. Defaults to pillar_density_batch.csv. "
            "Single-file mode: also accepted as an alias for --csv."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter file picker even if CLI options are present.",
    )
    return parser.parse_args(argv)


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def print_available_arrays(data: Dict[str, np.ndarray]) -> None:
    print(available_arrays_text(data))


def available_arrays_text(data: Dict[str, np.ndarray]) -> str:
    lines = ["Available arrays in file:"]
    for name, value in data.items():
        if value.shape == ():
            lines.append(f"  - {name}: scalar dtype={value.dtype}, value={value.item()}")
        elif value.size and np.issubdtype(value.dtype, np.number):
            lines.append(
                f"  - {name}: shape={value.shape}, dtype={value.dtype}, "
                f"min={np.min(value)}, max={np.max(value)}"
            )
        else:
            lines.append(f"  - {name}: shape={value.shape}, dtype={value.dtype}")
    return "\n".join(lines)


def require_array(data: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"Required array '{name}' is not present in the file.")
    return data[name]


def derive_height_from_occ(occ: np.ndarray) -> np.ndarray:
    occupied = occ != 0
    reversed_any = occupied[:, :, ::-1]
    highest_from_top = np.argmax(reversed_any, axis=2)
    height = occ.shape[2] - 1 - highest_from_top
    height[~np.any(occupied, axis=2)] = 0
    return height.astype(np.int32, copy=False)


def top_species_from_snapshot(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return top species map as Top[y, x]."""
    if "Occ_top" in data:
        return data["Occ_top"]

    occ = require_array(data, "Occ")
    height = data["H"] if "H" in data else derive_height_from_occ(occ)
    yy, xx = np.indices(height.shape)
    return occ[yy, xx, height]


def structure_for_connectivity(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("Connectivity must be 4 or 8.")


def label_with_fallback(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask, structure=structure_for_connectivity(connectivity))
        return labels.astype(np.int32, copy=False), int(count)
    except ImportError:
        return label_components_numpy(mask, connectivity)


def neighbor_offsets(connectivity: int) -> List[Tuple[int, int]]:
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    return offsets


def label_components_numpy(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    current_label = 0
    offsets = neighbor_offsets(connectivity)
    ly, lx = mask.shape

    for y in range(ly):
        for x in range(lx):
            if not mask[y, x] or labels[y, x] != 0:
                continue

            current_label += 1
            labels[y, x] = current_label
            queue = deque([(y, x)])  # type: Deque[Tuple[int, int]]

            while queue:
                cy, cx = queue.popleft()
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < ly and 0 <= nx < lx and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current_label
                        queue.append((ny, nx))

    return labels, current_label


def component_areas(labels: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.array([], dtype=np.int64)
    return np.bincount(labels.ravel(), minlength=count + 1)[1:]


def filtered_labels(labels: np.ndarray, areas: np.ndarray, min_area: int) -> np.ndarray:
    keep = np.zeros(areas.size + 1, dtype=bool)
    keep[1:] = areas >= min_area
    return np.where(keep[labels], labels, 0)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    foreground = labels != 0
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return boundary & foreground


def make_species_cmap(top_species: np.ndarray) -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    ids = np.unique(top_species.astype(int, copy=False))
    max_id = max(int(ids.max()) if ids.size else 0, 3)
    color_table = np.ones((max_id + 1, 4), dtype=float)
    color_table[:, :3] = 0.94
    color_table[:, 3] = 1.0
    color_table[0, :3] = (0.08, 0.08, 0.08)

    base = plt.get_cmap("tab20")
    for species_id in ids:
        if species_id == 0:
            continue
        color_table[species_id] = base((species_id - 1) % base.N)

    cmap = colors.ListedColormap(color_table)
    norm = colors.BoundaryNorm(np.arange(-0.5, max_id + 1.5, 1.0), cmap.N)
    return cmap, norm


def compute_metrics(
    top_species: np.ndarray,
    target_species: int,
    raw_count: int,
    areas_pixels: np.ndarray,
    min_area: int,
    connectivity: int,
) -> Metrics:
    ly_pixels, lx_pixels = top_species.shape
    pixel_area_nm2 = PIXEL_SIZE_NM**2
    total_pixels = int(lx_pixels * ly_pixels)
    total_area_nm2 = total_pixels * pixel_area_nm2
    total_area_um2 = total_area_nm2 / 1_000_000.0

    kept_areas_pixels = areas_pixels[areas_pixels >= min_area]
    filtered_count = int(kept_areas_pixels.size)
    species_pixels = int(np.count_nonzero(top_species == target_species))

    mean_area_pixels = float(np.mean(kept_areas_pixels)) if filtered_count else 0.0
    median_area_pixels = float(np.median(kept_areas_pixels)) if filtered_count else 0.0
    mean_area_nm2 = mean_area_pixels * pixel_area_nm2
    median_area_nm2 = median_area_pixels * pixel_area_nm2

    return {
        "target_species": int(target_species),
        "connectivity": int(connectivity),
        "min_area_pixels": int(min_area),
        "Lx_pixels": int(lx_pixels),
        "Ly_pixels": int(ly_pixels),
        "pixel_size_nm": PIXEL_SIZE_NM,
        "area_nm2": float(total_area_nm2),
        "area_um2": float(total_area_um2),
        "pillar_count_raw": int(raw_count),
        "pillar_count_filtered": filtered_count,
        "density_per_nm2": float(filtered_count / total_area_nm2) if total_area_nm2 else 0.0,
        "density_per_um2": float(filtered_count / total_area_um2) if total_area_um2 else 0.0,
        "mean_pillar_area_pixels": mean_area_pixels,
        "median_pillar_area_pixels": median_area_pixels,
        "mean_pillar_area_nm2": mean_area_nm2,
        "median_pillar_area_nm2": median_area_nm2,
        "area_fraction_selected_species": float(species_pixels / total_pixels) if total_pixels else 0.0,
    }


def analyze_loaded_snapshot(
    data: Dict[str, np.ndarray],
    species: int,
    min_area: int,
    connectivity: int,
) -> Tuple[np.ndarray, np.ndarray, Metrics]:
    top_species = top_species_from_snapshot(data)
    mask = top_species == species
    labels, raw_count = label_with_fallback(mask, connectivity)
    areas_pixels = component_areas(labels, raw_count)
    labels_filtered = filtered_labels(labels, areas_pixels, min_area)
    metrics = compute_metrics(
        top_species=top_species,
        target_species=species,
        raw_count=raw_count,
        areas_pixels=areas_pixels,
        min_area=min_area,
        connectivity=connectivity,
    )
    return top_species, labels_filtered, metrics


def analyze_snapshot_file(
    source_path: Path,
    species: int,
    min_area: int,
    connectivity: int,
) -> Tuple[np.ndarray, np.ndarray, Metrics]:
    data = load_npz(source_path)
    return analyze_loaded_snapshot(data, species, min_area, connectivity)


def single_csv_row(
    metrics: Metrics, source_file: Path, overlay_path: Optional[Path]
) -> Dict[str, Union[int, float, str]]:
    return {
        "source_file": str(source_file),
        **metrics,
        "overlay_file": str(overlay_path) if overlay_path is not None else "",
    }


def write_single_csv(
    path: Path,
    metrics: Metrics,
    source_file: Path,
    overlay_path: Optional[Path],
) -> None:
    row = single_csv_row(metrics, source_file, overlay_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_batch_csv(path: Path, rows: List[BatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(
    path: Path,
    top_species: np.ndarray,
    labels_filtered: np.ndarray,
    target_species: int,
    metrics: Metrics,
) -> None:
    cmap, norm = make_species_cmap(top_species)
    boundaries = boundary_mask(labels_filtered)

    overlay_rgb = cmap(norm(top_species))[..., :3]
    overlay_rgb[boundaries] = (1.0, 0.05, 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].imshow(top_species, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
    axes[0].imshow(np.ma.masked_where(~boundaries, boundaries), origin="lower", cmap="Reds", alpha=0.9)
    axes[0].set_title(f"Top Species + Pillar Boundaries\nspecies {target_species}")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")
    fig.colorbar(image, ax=axes[0], label="Species id")

    labels_image = np.ma.masked_where(labels_filtered == 0, labels_filtered)
    axes[1].imshow(top_species == target_species, origin="lower", interpolation="nearest", cmap="gray_r")
    axes[1].imshow(labels_image, origin="lower", interpolation="nearest", cmap="nipy_spectral", alpha=0.8)
    axes[1].set_title(
        f"Filtered Components: {metrics['pillar_count_filtered']} pillars\n"
        f"{metrics['density_per_um2']:.3g} #/um^2"
    )
    axes[1].set_xlabel("x pixel")
    axes[1].set_ylabel("y pixel")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_results(metrics: Metrics, csv_path: Path, overlay_path: Optional[Path]) -> None:
    print(results_text(metrics, csv_path, overlay_path))


def results_text(metrics: Metrics, csv_path: Path, overlay_path: Optional[Path]) -> str:
    lines = [
        "",
        "Pillar density results",
        "----------------------",
        f"target_species: {metrics['target_species']}",
        f"pixel_size_nm: {metrics['pixel_size_nm']}",
        f"Lx_pixels: {metrics['Lx_pixels']}",
        f"Ly_pixels: {metrics['Ly_pixels']}",
        f"area_nm2: {metrics['area_nm2']:.6g}",
        f"area_um2: {metrics['area_um2']:.6g}",
        f"pillar_count_raw: {metrics['pillar_count_raw']}",
        f"pillar_count_filtered: {metrics['pillar_count_filtered']}",
        f"density_per_nm2: {metrics['density_per_nm2']:.6g}",
        f"density_per_um2: {metrics['density_per_um2']:.6g}",
        f"mean_pillar_area_pixels: {metrics['mean_pillar_area_pixels']:.6g}",
        f"median_pillar_area_pixels: {metrics['median_pillar_area_pixels']:.6g}",
        f"mean_pillar_area_nm2: {metrics['mean_pillar_area_nm2']:.6g}",
        f"median_pillar_area_nm2: {metrics['median_pillar_area_nm2']:.6g}",
        f"area_fraction_selected_species: {metrics['area_fraction_selected_species']:.6g}",
        "",
        f"CSV saved to: {csv_path}",
    ]
    if overlay_path is not None:
        lines.append(f"Overlay saved to: {overlay_path}")
    return "\n".join(lines)


def coerce_number(value: Optional[str]) -> ScalarOrBlank:
    if value is None:
        return ""

    stripped = value.strip().rstrip(",")
    if not stripped:
        return ""

    try:
        numeric = float(stripped)
    except ValueError:
        return ""

    if numeric.is_integer():
        return int(numeric)
    return numeric


def extract_first_number(text: str, patterns: List[str]) -> ScalarOrBlank:
    flags = re.MULTILINE | re.DOTALL
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = coerce_number(match.group(1))
            if value != "":
                return value
    return ""


def folder_name_number(folder: Path, label: str) -> ScalarOrBlank:
    match = re.search(rf"_{label}({NUMBER_PATTERN})(?:_|$)", folder.name)
    if not match:
        return ""
    return coerce_number(match.group(1))


def parse_dep_values(text: str) -> List[Number]:
    match = re.search(r"DEP:\s*\[\[(.*?)\]\]", text, re.DOTALL)
    if not match:
        return []

    values = []  # type: List[Number]
    for token in re.findall(NUMBER_PATTERN, match.group(1)):
        coerced = coerce_number(token)
        if coerced != "":
            values.append(coerced)
    return values


def metadata_value(*values: ScalarOrBlank) -> ScalarOrBlank:
    for value in values:
        if value != "":
            return value
    return ""


def parse_run_metadata(run_folder: Path) -> Metadata:
    metadata = {  # type: Metadata
        "T": "",
        "f": "",
        "pulses": "",
        "Estatic": "",
        "E_ES": "",
        "seed": "",
    }

    info_path = run_folder / "run_info.txt"
    text = ""
    if info_path.exists():
        try:
            text = info_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = info_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warn(f"Could not read metadata from {info_path}: {exc}")

    dep_values = parse_dep_values(text) if text else []

    dep_t = dep_values[0] if len(dep_values) > 0 else ""
    dep_f = dep_values[1] if len(dep_values) > 1 else ""
    dep_pulses = dep_values[2] if len(dep_values) > 2 else ""

    if text:
        metadata["T"] = metadata_value(
            extract_first_number(text, [rf"^\s*T\s*=\s*({NUMBER_PATTERN})"]),
            extract_first_number(text, [rf"^\s*doe_temperature_celsius\s*=\s*({NUMBER_PATTERN})"]),
            extract_first_number(text, [rf"DOE T\s*=\s*({NUMBER_PATTERN})\s*C"]),
            folder_name_number(run_folder, "T"),
            extract_first_number(text, [rf"^\s*doe_temperature_kelvin\s*=\s*({NUMBER_PATTERN})"]),
            dep_t,
        )
        metadata["f"] = metadata_value(
            extract_first_number(text, [rf"^\s*f\s*=\s*({NUMBER_PATTERN})"]),
            extract_first_number(text, [rf"^\s*doe_frequency_hz\s*=\s*({NUMBER_PATTERN})"]),
            extract_first_number(text, [rf"\bf\s*=\s*({NUMBER_PATTERN})\s*Hz"]),
            folder_name_number(run_folder, "f"),
            dep_f,
        )
        metadata["pulses"] = metadata_value(
            extract_first_number(text, [rf"^\s*pulses\s*=\s*({NUMBER_PATTERN})"]),
            extract_first_number(text, [rf"^\s*pulses_completed\s*=\s*({NUMBER_PATTERN})"]),
            dep_pulses,
        )
        metadata["Estatic"] = extract_first_number(text, [rf"^\s*Estatic\s*=\s*({NUMBER_PATTERN})"])
        metadata["E_ES"] = extract_first_number(text, [rf"^\s*E_ES\s*=\s*({NUMBER_PATTERN})"])
        metadata["seed"] = extract_first_number(text, [rf"^\s*Random seed\s*=\s*({NUMBER_PATTERN})"])
    else:
        metadata["T"] = folder_name_number(run_folder, "T")
        metadata["f"] = folder_name_number(run_folder, "f")
        metadata["pulses"] = dep_pulses

    return metadata


def parse_snapshot_progress(snapshot_path: Path) -> Tuple[str, Number]:
    snapshot_name = snapshot_path.stem
    if snapshot_name == "snapshot_final":
        return snapshot_name, 100

    match = re.fullmatch(rf"snapshot_({NUMBER_PATTERN})", snapshot_name)
    if not match:
        raise ValueError(
            f"Snapshot name '{snapshot_path.name}' does not match 'snapshot_<number>.npz' or 'snapshot_final.npz'."
        )

    numeric = float(match.group(1))
    if numeric.is_integer():
        return snapshot_name, int(numeric)
    return snapshot_name, numeric


def snapshot_sort_key(snapshot_path: Path) -> Tuple[float, str]:
    try:
        snapshot_name, progress = parse_snapshot_progress(snapshot_path)
        return float(progress), snapshot_name
    except ValueError:
        return float("inf"), snapshot_path.stem


def find_snapshot_files(run_folder: Path) -> List[Path]:
    snapshots = [path for path in run_folder.glob("snapshot_*.npz") if path.is_file()]
    return sorted(snapshots, key=snapshot_sort_key)


def list_child_run_folders(parent: Path) -> List[Path]:
    if not parent.exists():
        raise FileNotFoundError(f"Parent folder not found: {parent}")
    if not parent.is_dir():
        raise ValueError(f"Parent path is not a directory: {parent}")
    return sorted(path.resolve() for path in parent.iterdir() if path.is_dir())


def select_run_folder_range(
    available_folders: List[Path],
    start_name: str = "",
    end_name: str = "",
) -> List[Path]:
    if not available_folders:
        raise ValueError("No run folders are available in the selected parent folder.")

    names = [folder.name for folder in available_folders]
    start_value = start_name.strip() if start_name.strip() else names[0]
    end_value = end_name.strip() if end_name.strip() else names[-1]

    if start_value not in names:
        raise ValueError(f"Start folder '{start_value}' is not in the selected parent folder.")
    if end_value not in names:
        raise ValueError(f"End folder '{end_value}' is not in the selected parent folder.")

    start_index = names.index(start_value)
    end_index = names.index(end_value)
    if start_index > end_index:
        raise ValueError("Start folder must come before or match the end folder.")

    return available_folders[start_index : end_index + 1]


def resolve_run_folders(args: argparse.Namespace) -> List[Path]:
    folders = []  # type: List[Path]

    if args.parent:
        folders.extend(list_child_run_folders(Path(args.parent)))

    if args.folders:
        folders.extend(Path(folder) for folder in args.folders)

    unique_folders = []  # type: List[Path]
    seen = set()  # type: Set[Path]
    for folder in folders:
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_folders.append(resolved)

    for folder in unique_folders:
        if not folder.exists():
            raise FileNotFoundError(f"Run folder not found: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Run folder path is not a directory: {folder}")

    return unique_folders


def batch_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    if args.parent:
        return Path(args.parent) / "pillar_density_batch.csv"
    return Path("pillar_density_batch.csv")


def batch_overlay_path(snapshot_path: Path, species: int) -> Path:
    return snapshot_path.parent / f"pillar_density_overlay_{snapshot_path.stem}_species_{species}.png"


def batch_row(
    run_folder: Path,
    snapshot_path: Path,
    snapshot_name: str,
    progress_percent: Number,
    metadata: Metadata,
    metrics: Metrics,
) -> BatchRow:
    return {
        "run_folder": str(run_folder),
        "snapshot_file": str(snapshot_path),
        "snapshot_name": snapshot_name,
        "progress_percent": progress_percent,
        "T": metadata["T"],
        "f": metadata["f"],
        "pulses": metadata["pulses"],
        "Estatic": metadata["Estatic"],
        "E_ES": metadata["E_ES"],
        "seed": metadata["seed"],
        "species": metrics["target_species"],
        "min_area_pixels": metrics["min_area_pixels"],
        "connectivity": metrics["connectivity"],
        "raw_count": metrics["pillar_count_raw"],
        "filtered_count": metrics["pillar_count_filtered"],
        "density_per_nm2": metrics["density_per_nm2"],
        "density_per_um2": metrics["density_per_um2"],
        "mean_area_pixels": metrics["mean_pillar_area_pixels"],
        "median_area_pixels": metrics["median_pillar_area_pixels"],
        "area_fraction": metrics["area_fraction_selected_species"],
    }


def sort_value(value: object) -> Tuple[int, Union[float, str]]:
    if value == "":
        return (1, float("inf"))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (0, str(value))


def batch_row_sort_key(row: BatchRow) -> Tuple[Tuple[int, Union[float, str]], ...]:
    return (
        sort_value(row["T"]),
        sort_value(row["f"]),
        sort_value(row["run_folder"]),
        sort_value(row["progress_percent"]),
    )


def run_batch_analysis(args: argparse.Namespace) -> Path:
    run_folders = resolve_run_folders(args)
    if not run_folders:
        raise ValueError("No run folders were found. Use --parent or --folders.")

    output_path = batch_output_path(args)
    save_overlays = bool(args.save_overlays or args.save_overlay)

    rows = []  # type: List[BatchRow]
    successful_snapshots = 0

    print(f"Batch analysis started for {len(run_folders)} run folder(s).")
    for run_folder in run_folders:
        snapshot_files = find_snapshot_files(run_folder)
        if not snapshot_files:
            warn(f"No snapshot_*.npz files found in {run_folder}")
            continue

        metadata = parse_run_metadata(run_folder)
        print(f"Analyzing {run_folder.name}: {len(snapshot_files)} snapshot(s)")

        for snapshot_path in snapshot_files:
            try:
                snapshot_name, progress_percent = parse_snapshot_progress(snapshot_path)
                top_species, labels_filtered, metrics = analyze_snapshot_file(
                    snapshot_path,
                    species=args.species,
                    min_area=args.min_area,
                    connectivity=args.connectivity,
                )
                if save_overlays:
                    save_overlay(
                        batch_overlay_path(snapshot_path, args.species),
                        top_species,
                        labels_filtered,
                        args.species,
                        metrics,
                    )
                rows.append(
                    batch_row(
                        run_folder=run_folder,
                        snapshot_path=snapshot_path,
                        snapshot_name=snapshot_name,
                        progress_percent=progress_percent,
                        metadata=metadata,
                        metrics=metrics,
                    )
                )
                successful_snapshots += 1
            except Exception as exc:
                warn(f"Failed to analyze {snapshot_path}: {exc}")

    if not rows:
        raise ValueError("No snapshots were successfully analyzed.")

    rows.sort(key=batch_row_sort_key)
    write_batch_csv(output_path, rows)

    print("")
    print("Batch pillar density results")
    print("----------------------------")
    print(f"Snapshots analyzed: {successful_snapshots}")
    print(f"Combined CSV saved to: {output_path}")
    if save_overlays:
        print("Per-snapshot overlays were saved next to the snapshot files.")

    return output_path


def analyze_snapshot(args: argparse.Namespace) -> Tuple[Metrics, Path, Optional[Path]]:
    if not args.file:
        raise ValueError("Please select a .npz file.")

    source_path = Path(args.file)
    data = load_npz(source_path)

    print(f"Loaded file: {source_path}")
    print_available_arrays(data)

    top_species, labels_filtered, metrics = analyze_loaded_snapshot(
        data,
        species=args.species,
        min_area=args.min_area,
        connectivity=args.connectivity,
    )

    csv_target = args.csv or args.output
    csv_path = Path(csv_target) if csv_target else source_path.parent / "pillar_density.csv"
    overlay_path = (
        None
        if args.no_overlay
        else Path(args.overlay)
        if args.overlay
        else source_path.parent / f"pillar_density_overlay_species_{args.species}.png"
    )

    write_single_csv(csv_path, metrics, source_path, overlay_path)
    if overlay_path is not None:
        save_overlay(overlay_path, top_species, labels_filtered, args.species, metrics)

    return metrics, csv_path, overlay_path


def guess_initial_file() -> str:
    runs_dir = Path("runs")
    candidates = sorted(runs_dir.rglob("snapshot_final.npz")) if runs_dir.exists() else []
    if candidates:
        return str(candidates[-1])
    return ""


def launch_gui(initial_file: Optional[str] = None) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python environment.") from exc

    root = tk.Tk()
    root.title("KMCS Pillar Density Analyzer")
    root.geometry("920x760")

    file_var = tk.StringVar(value=initial_file or guess_initial_file())
    initial_parent = Path("runs")
    if not initial_parent.exists() and initial_file:
        initial_parent = Path(initial_file).resolve().parent.parent
    batch_parent_var = tk.StringVar(value=str(initial_parent) if initial_parent.exists() else "")
    batch_start_var = tk.StringVar(value="")
    batch_end_var = tk.StringVar(value="")
    batch_output_var = tk.StringVar(value="")
    species_var = tk.StringVar(value="2")
    min_area_var = tk.StringVar(value=str(DEFAULT_MIN_AREA_PIXELS))
    connectivity_var = tk.StringVar(value="8")
    save_overlay_var = tk.BooleanVar(value=True)
    batch_save_overlays_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="Use the Single Snapshot tab or the Batch Range tab.")
    available_run_folders = []  # type: List[Path]

    main = ttk.Frame(root, padding=12)
    main.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(3, weight=1)

    def set_output(text: str) -> None:
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("1.0", text)
        output.configure(state="disabled")

    def preview_file(path_text: str) -> None:
        if not path_text:
            set_output("No file selected.")
            return
        data = load_npz(Path(path_text))
        set_output(available_arrays_text(data))

    def browse_file() -> None:
        current = file_var.get().strip()
        initial_dir = str(Path(current).parent) if current else str(Path("runs"))
        chosen = filedialog.askopenfilename(
            title="Select KMCS snapshot",
            initialdir=initial_dir,
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        )
        if chosen:
            file_var.set(chosen)
            try:
                preview_file(chosen)
                status_var.set("File loaded. Ready to analyze.")
            except Exception as exc:  # pragma: no cover - GUI path
                set_output(f"Could not read selected file:\n{exc}")
                status_var.set("Could not read selected file.")

    def current_batch_output_default() -> str:
        parent_text = batch_parent_var.get().strip()
        if not parent_text:
            return str(Path("pillar_density_batch.csv"))
        return str(Path(parent_text) / "pillar_density_batch.csv")

    def update_batch_output_default(force: bool = False) -> None:
        current_value = batch_output_var.get().strip()
        if force or not current_value:
            batch_output_var.set(current_batch_output_default())

    def batch_selection_summary(folders: List[Path], output_path: Optional[Path] = None) -> str:
        lines = ["Selected run folders:"]
        total_snapshots = 0
        for folder in folders:
            snapshot_count = len(find_snapshot_files(folder))
            total_snapshots += snapshot_count
            lines.append(f"  - {folder.name}: {snapshot_count} snapshot(s)")

        lines.extend(
            [
                "",
                f"Folders in range: {len(folders)}",
                f"Snapshots discovered: {total_snapshots}",
                "Scan pattern: snapshot_*.npz",
                "Range is inclusive: start folder through end folder",
                f"Output CSV: {output_path or Path(batch_output_var.get().strip() or current_batch_output_default())}",
            ]
        )
        return "\n".join(lines)

    def refresh_batch_folder_choices(show_preview: bool = True) -> None:
        parent_text = batch_parent_var.get().strip()
        if not parent_text:
            available_run_folders[:] = []
            batch_start_var.set("")
            batch_end_var.set("")
            start_combo.configure(values=[])
            end_combo.configure(values=[])
            if show_preview:
                set_output("Choose a parent runs folder to load its run folders.")
            status_var.set("Choose a parent runs folder.")
            return

        parent_path = Path(parent_text)
        folders = list_child_run_folders(parent_path)
        available_run_folders[:] = folders
        names = [folder.name for folder in folders]

        start_combo.configure(values=names)
        end_combo.configure(values=names)

        if names:
            if batch_start_var.get() not in names:
                batch_start_var.set(names[0])
            if batch_end_var.get() not in names:
                batch_end_var.set(names[-1])
            update_batch_output_default()
            if show_preview:
                set_output(batch_selection_summary(select_run_folder_range(folders, batch_start_var.get(), batch_end_var.get())))
            status_var.set(f"Loaded {len(names)} run folders from {parent_path}.")
            return

        batch_start_var.set("")
        batch_end_var.set("")
        if show_preview:
            set_output("No child run folders were found in the selected parent folder.")
        status_var.set("No child run folders were found in the selected parent folder.")

    def browse_parent_folder() -> None:
        current = batch_parent_var.get().strip()
        initial_dir = current if current else str(Path("runs") if Path("runs").exists() else Path.cwd())
        chosen = filedialog.askdirectory(title="Select parent folder containing run folders", initialdir=initial_dir)
        if chosen:
            batch_parent_var.set(chosen)
            update_batch_output_default(force=True)
            try:
                refresh_batch_folder_choices(show_preview=True)
            except Exception as exc:  # pragma: no cover - GUI path
                set_output(f"Could not load run folders:\n{exc}")
                status_var.set("Could not load run folders from the selected parent folder.")

    def browse_batch_output() -> None:
        initial_path = batch_output_var.get().strip() or current_batch_output_default()
        chosen = filedialog.asksaveasfilename(
            title="Save combined batch CSV as",
            initialfile=Path(initial_path).name,
            initialdir=str(Path(initial_path).parent),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if chosen:
            batch_output_var.set(chosen)
            status_var.set("Batch output path updated.")

    def selected_batch_folders() -> List[Path]:
        refresh_batch_folder_choices(show_preview=False)
        return select_run_folder_range(available_run_folders, batch_start_var.get(), batch_end_var.get())

    def preview_batch_selection() -> None:
        folders = selected_batch_folders()
        set_output(batch_selection_summary(folders))
        status_var.set(f"Previewing {len(folders)} run folders in the selected range.")

    def args_from_gui() -> argparse.Namespace:
        return argparse.Namespace(
            file=file_var.get().strip(),
            parent=None,
            folders=None,
            species=int(species_var.get()),
            min_area=int(min_area_var.get()),
            connectivity=int(connectivity_var.get()),
            save_overlay=save_overlay_var.get(),
            save_overlays=False,
            no_overlay=not save_overlay_var.get(),
            csv=None,
            overlay=None,
            output=None,
            gui=True,
        )

    def batch_args_from_gui() -> argparse.Namespace:
        folders = selected_batch_folders()
        return argparse.Namespace(
            file=None,
            parent=None,
            folders=[str(folder) for folder in folders],
            species=int(species_var.get()),
            min_area=int(min_area_var.get()),
            connectivity=int(connectivity_var.get()),
            save_overlay=False,
            save_overlays=batch_save_overlays_var.get(),
            no_overlay=not batch_save_overlays_var.get(),
            csv=None,
            overlay=None,
            output=batch_output_var.get().strip() or current_batch_output_default(),
            gui=True,
        )

    def analyze_from_gui() -> None:
        try:
            args = args_from_gui()
            metrics, csv_path, overlay_path = analyze_snapshot(args)
            set_output(results_text(metrics, csv_path, overlay_path))
            status_var.set("Analysis complete. CSV and overlay saved next to the .npz file.")
        except Exception as exc:  # pragma: no cover - GUI path
            messagebox.showerror("KMCS Pillar Density Analyzer", str(exc))
            status_var.set("Analysis failed. Check the message and try again.")

    def analyze_batch_from_gui() -> None:
        try:
            args = batch_args_from_gui()
            selected_folders = [Path(folder) for folder in args.folders]
            output_path = run_batch_analysis(args)
            row_count = 0
            with output_path.open(newline="", encoding="utf-8") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            summary = batch_selection_summary(selected_folders, output_path=output_path)
            summary = f"{summary}\n\nCSV rows written: {row_count}"
            if batch_save_overlays_var.get():
                summary = f"{summary}\nPer-snapshot overlays: enabled"
            set_output(summary)
            status_var.set("Batch analysis complete. Combined CSV created successfully.")
        except Exception as exc:  # pragma: no cover - GUI path
            messagebox.showerror("KMCS Pillar Density Analyzer", str(exc))
            status_var.set("Batch analysis failed. Check the message and try again.")

    options = ttk.LabelFrame(main, text="Analysis Options", padding=10)
    options.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(options, text="Species").grid(row=0, column=0, sticky="w")
    ttk.Combobox(options, textvariable=species_var, values=["2", "3"], state="readonly", width=8).grid(
        row=0, column=1, sticky="w", padx=(4, 16)
    )
    ttk.Label(options, text="Min area pixels").grid(row=0, column=2, sticky="w")
    ttk.Entry(options, textvariable=min_area_var, width=8).grid(row=0, column=3, sticky="w", padx=(4, 16))
    ttk.Label(options, text="Connectivity").grid(row=0, column=4, sticky="w")
    ttk.Combobox(options, textvariable=connectivity_var, values=["4", "8"], state="readonly", width=8).grid(
        row=0, column=5, sticky="w", padx=(4, 16)
    )

    notebook = ttk.Notebook(main)
    notebook.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

    single_tab = ttk.Frame(notebook, padding=10)
    batch_tab = ttk.Frame(notebook, padding=10)
    notebook.add(single_tab, text="Single Snapshot")
    notebook.add(batch_tab, text="Batch Range")

    single_tab.columnconfigure(1, weight=1)
    batch_tab.columnconfigure(1, weight=1)

    ttk.Label(single_tab, text="NPZ file").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Entry(single_tab, textvariable=file_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
    ttk.Button(single_tab, text="Browse...", command=browse_file).grid(row=0, column=2, sticky="ew", pady=(0, 8))
    ttk.Checkbutton(single_tab, text="Save overlay", variable=save_overlay_var).grid(
        row=1, column=1, sticky="w", pady=(0, 8)
    )

    single_buttons = ttk.Frame(single_tab)
    single_buttons.grid(row=2, column=0, columnspan=3, sticky="w")
    ttk.Button(single_buttons, text="Preview File", command=lambda: preview_file(file_var.get().strip())).grid(
        row=0, column=0, padx=(0, 8)
    )
    ttk.Button(single_buttons, text="Analyze", command=analyze_from_gui).grid(row=0, column=1)

    single_info = (
        f"Single-file mode uses {PIXEL_SIZE_NM} nm per pixel and writes pillar_density.csv "
        "next to the selected snapshot."
    )
    ttk.Label(single_tab, text=single_info, wraplength=820).grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
    )

    ttk.Label(batch_tab, text="Parent folder").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Entry(batch_tab, textvariable=batch_parent_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
    batch_parent_buttons = ttk.Frame(batch_tab)
    batch_parent_buttons.grid(row=0, column=2, sticky="e", pady=(0, 8))
    ttk.Button(batch_parent_buttons, text="Browse...", command=browse_parent_folder).grid(row=0, column=0)
    ttk.Button(
        batch_parent_buttons,
        text="Refresh",
        command=lambda: refresh_batch_folder_choices(show_preview=True),
    ).grid(row=0, column=1, padx=(8, 0))

    ttk.Label(batch_tab, text="Start folder").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    start_combo = ttk.Combobox(batch_tab, textvariable=batch_start_var, state="readonly")
    start_combo.grid(row=1, column=1, sticky="ew", pady=(0, 8))

    ttk.Label(batch_tab, text="End folder").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    end_combo = ttk.Combobox(batch_tab, textvariable=batch_end_var, state="readonly")
    end_combo.grid(row=2, column=1, sticky="ew", pady=(0, 8))

    ttk.Label(batch_tab, text="Output CSV").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Entry(batch_tab, textvariable=batch_output_var).grid(row=3, column=1, sticky="ew", pady=(0, 8))
    ttk.Button(batch_tab, text="Save As...", command=browse_batch_output).grid(row=3, column=2, sticky="ew", pady=(0, 8))

    ttk.Checkbutton(batch_tab, text="Save per-snapshot overlays", variable=batch_save_overlays_var).grid(
        row=4, column=1, sticky="w", pady=(0, 8)
    )

    batch_buttons = ttk.Frame(batch_tab)
    batch_buttons.grid(row=5, column=0, columnspan=3, sticky="w")
    ttk.Button(batch_buttons, text="Preview Range", command=preview_batch_selection).grid(
        row=0, column=0, padx=(0, 8)
    )
    ttk.Button(batch_buttons, text="Analyze Range", command=analyze_batch_from_gui).grid(row=0, column=1)

    batch_info = (
        "Batch mode scans every folder from the selected start folder through the selected end folder, "
        "inclusive, and analyzes all snapshot_*.npz files it finds inside each run folder."
    )
    ttk.Label(batch_tab, text=batch_info, wraplength=820).grid(
        row=6, column=0, columnspan=3, sticky="w", pady=(8, 0)
    )

    output_info = ttk.Label(
        main,
        text="Results and previews appear below. Batch previews show which run folders and snapshots will be scanned.",
        wraplength=880,
    )
    output_info.grid(row=2, column=0, sticky="w", pady=(0, 8))

    output = tk.Text(main, height=20, wrap="word")
    output.grid(row=3, column=0, sticky="nsew")
    output.configure(state="disabled")

    ttk.Label(main, textvariable=status_var).grid(row=4, column=0, sticky="w", pady=(8, 0))

    start_combo.bind("<<ComboboxSelected>>", lambda _event: preview_batch_selection())
    end_combo.bind("<<ComboboxSelected>>", lambda _event: preview_batch_selection())

    if file_var.get():
        try:
            preview_file(file_var.get())
        except Exception:
            set_output("Choose a .npz file to preview its arrays.")
    else:
        set_output("Choose a .npz file to preview its arrays.")

    update_batch_output_default()
    try:
        refresh_batch_folder_choices(show_preview=False)
    except Exception:
        status_var.set("Choose a valid parent runs folder for batch mode.")

    root.mainloop()


def main() -> None:
    argv = sys.argv[1:]
    args = parse_args(argv)

    if args.gui or not argv:
        launch_gui(initial_file=args.file)
        return

    has_batch_inputs = bool(args.parent or args.folders)
    if args.file and has_batch_inputs:
        raise ValueError("Choose either --file for single-file analysis or --parent/--folders for batch analysis.")

    if has_batch_inputs:
        run_batch_analysis(args)
        return

    metrics, csv_path, overlay_path = analyze_snapshot(args)
    print_results(metrics, csv_path, overlay_path)


if __name__ == "__main__":
    main()
