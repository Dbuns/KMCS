"""
KMCS_new.py
===========

Kinetic Monte Carlo model for pulsed thin-film growth on a stepped substrate.
The code deposits atoms pulse-by-pulse, then relaxes only the atoms deposited
in the current pulse using a solid-on-solid lattice KMC model with local
hopping.

Model summary
-------------
- Solid-on-solid lattice with one species per lattice site.
- Pulsed deposition followed by KMC relaxation after each pulse.
- Active set limited to atoms deposited in the current pulse.
- Hop rate: k_hop = k0 * exp(-(Estatic + EN) / (kB * T))
- EN from six face-sharing neighbors only.
- Direction selection uses fixed move-family weights:
  cubic moves are unpenalized, diagonal moves carry the E_ES penalty.
- No elastic strain, no bulk diffusion, and no long-range interactions.

Quick start
-----------
1. Edit the simulation parameters in the CONFIG section near the bottom of the file.
   Start by choosing:
      RUN_TYPE = "single"   # one run using DEP exactly as written
      RUN_TYPE = "doe"      # batch over DOE_T_C and DOE_F_HZ
2. Run:
      python KMCS_new.py
3. Outputs are written to:
      runs/<timestamp>_<label>/
4. Useful files to inspect:
   - run_info.txt
   - roughness.csv
   - snapshot_final.npz
   - snapshot images (if enabled)
   - profile.txt / profile.pstats (if profiling is enabled)

Typical workflows
-----------------
- Single run (quick test):
  Set RUN_TYPE = "single". Edit DEP directly; DEP[0, 0] is temperature in K
  and DEP[0, 1] is frequency in Hz for the normal one-case run.
- DOE run:
  Set RUN_TYPE = "doe". DEP and E_vals are still required explicit inputs.
  DEP is used as the template experiment/model, and only temperature and
  frequency are overwritten from DOE_T_C and DOE_F_HZ. One folder is created
  for each T/f pair, plus a DOE summary CSV in runs/.
- Validation run (paper comparison):
  Use the validated parameter set, fixed seed, and documented analysis settings
  when comparing morphology, roughness, or pillar statistics to the study.
- Post-processing using .npz files:
  Use snapshot_*.npz files to reload H, Occ_top, and the final Occ array for
  later morphology plots, pillar analysis, or custom 2D/3D visualization.

Safe to edit
------------
- RUN_TYPE: "single" or "doe"
- DEP: deposition schedule and composition. DEP is always required.
- E_vals: species interaction-energy matrix. E_vals is always required.
- DOE_T_C and DOE_F_HZ when RUN_TYPE = "doe"
- Number of pulses
- Grid size (Lx, Ly, Lz)
- Step count W
- Estatic
- E_ES
- Output and profiling flags

Do not edit casually unless validating model internals
------------------------------------------------------
- Event selection logic
- Rate calculation functions
- Boundary mapping
- Move-family acceptance rules
- Source rejection / floater prevention
- Active-set bookkeeping
- KMC time update

Common mistakes
---------------
- Increasing max_events_per_pulse without thinking about pulse_interval = 1/f:
  this changes how far relaxation can proceed before the failsafe is reached.
- Changing Estatic, E_ES, or E_vals without tracking the consequence:
  these directly change event probabilities and therefore morphology.
- Treating intermediate snapshots as final morphology:
  25%, 50%, and 75% snapshots are progress diagnostics, not converged end states.
- Using too small a min_area in pillar analysis:
  this can count noise or fragmented clusters as real pillars.
- Editing KMC internals casually:
  changes to rates, event selection, move rules, or boundary logic can silently
  invalidate comparisons to validated runs.

Reproducibility
---------------
- The random seed in the CONFIG section controls the stochastic deposition and
  KMC event sequence.
- The same seed with the same parameters gives a reproducible run.
- Different seeds can change local morphology details, but robust physical
  trends should remain comparable across repeated runs.

Version / validation note
-------------------------
This version is intended to reproduce the validated study/paper workflow used
for the current project. If future versions change physics, event logic, or
analysis assumptions, document those changes clearly so comparisons across
versions remain meaningful.

Output files
------------
- run_info.txt: full record of parameters and counters
- roughness.csv: roughness and normalized proxy signal over the run
- snapshot_*.npz: saved state data for later analysis
- snapshot_*.png: top-view morphology images if enabled
- debug.gif / frames/: only when debug-frame saving is enabled
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image
import csv

# ============================================================
# Constants, state arrays, conventions
# ============================================================

# --- Physical constants (global, fixed) ---
kB = 8.617333262e-5      # Boltzmann constant [eV/K]
h  = 4.135667662e-15     # Planck constant [eV*s]
k0 = 1e12                # Hop attempt frequency [1/s]

# --- Global energy knobs (system-level, fixed) ---
Estatic = 1.000          # static barrier [eV]
E_ES    = 0.15           # Ehrlich–Schwoebel barrier [eV] applied to diagonal move families

# --- Conventions ---
# Species IDs:
#   0 = empty
#   1 = substrate (and/or the base material)
#   2..N = film species
#
# Arrays:
#   Occ[y, x, z] : uint8 species id at each voxel
#   H[y, x]      : int32 top occupied z index in each (y,x) column

species_colors = {
    1: (0.6, 0.6, 0.6),      # substrate = neutral gray
    2: (0.0, 0.45, 0.70),    # blue
    3: (0.90, 0.60, 0.0),    # orange
    4: (0.0, 0.62, 0.45),    # bluish green
    5: (0.80, 0.47, 0.65),   # reddish purple
    6: (0.95, 0.90, 0.25),   # yellow (use carefully on white bg)
}

TOPVIEW_STYLE = "species_only"        # choose from: "species_only", "species_plus_height"
TOPVIEW_HEIGHT_FLOOR = 0.85           # gentle, fixed shading floor for species_plus_height
TOPVIEW_HEIGHT_SPAN = 0.15            # keeps height contrast readable without washing out species

STEP_EDGE_W = 0
PROJECT_ROOT = Path(__file__).resolve().parent


def make_run_dir(label="debug", base="runs"):
    """Create one timestamped output folder for a simulation run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = str(label).strip().replace(" ", "_") if label else ""
    run_name = f"{timestamp}_{label}" if label else timestamp
    run_dir = PROJECT_ROOT / base / run_name
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, frames_dir


def save_run_info(run_dir, params_dict):
    """Save a plain-text summary of the parameters used for one run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run_info.txt"

    grid = params_dict.get("grid", {})
    constants = params_dict.get("constants", {})
    debug_params = params_dict.get("debug_params", {})
    validation_counters = params_dict.get("validation_counters", {})
    summary = params_dict.get("summary", {})
    notes = params_dict.get("notes", "")
    seed = params_dict.get("seed")

    lines = [
        f"Run timestamp: {params_dict.get('timestamp', '')}",
        "",
        "Grid:",
        f"Lx = {grid.get('Lx', '')}",
        f"Ly = {grid.get('Ly', '')}",
        f"Lz = {grid.get('Lz', '')}",
        "",
        "Energies (E_vals):",
        np.array2string(np.asarray(params_dict.get("E_vals", [])), separator=", "),
        "",
        "DEP:",
        np.array2string(np.asarray(params_dict.get("DEP", [])), separator=", "),
        "",
        "Constants:",
        f"Estatic = {constants.get('Estatic', '')}",
        f"E_ES = {constants.get('E_ES', '')}",
        f"TOPVIEW_STYLE = {constants.get('TOPVIEW_STYLE', '')}",
        f"kB = {constants.get('kB', '')}",
    ]

    if seed is not None:
        lines.extend(["", f"Random seed = {seed}"])

    if debug_params:
        lines.append("")
        lines.append("Debug parameters:")
        for key, value in debug_params.items():
            lines.append(f"{key} = {value}")

    if validation_counters:
        lines.append("")
        lines.append("Validation counters:")
        for key, value in validation_counters.items():
            lines.append(f"{key} = {value}")

    if notes:
        lines.extend(["", "Notes:", str(notes)])

    if summary:
        lines.extend(["", "Summary:"])
        for key, value in summary.items():
            lines.append(f"{key} = {value}")

    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved run info: {info_path}")
    return info_path

# ------------------------------------------------------------
# Utility: periodic neighbor indices in x/y
# ------------------------------------------------------------
def pbc_neighbors(x: int, y: int, Lx: int, Ly: int):
    """Return periodic neighbor indices (xm,xp, ym,yp)."""
    xm = (x - 1) % Lx
    xp = (x + 1) % Lx
    ym = (y - 1) % Ly
    yp = (y + 1) % Ly
    return xm, xp, ym, yp


def map_step_boundary(y, x, z, Ly, Lx, step_w=None):
    """
    Map coordinates back into the simulation cell.
    x wraps periodically; y wraps periodically with a z shift of +/-W
    to represent stepped boundaries.
    """
    if step_w is None:
        step_w = STEP_EDGE_W

    x = x % Lx
    while y < 0:
        y += Ly
        z += step_w
    while y >= Ly:
        y -= Ly
        z -= step_w
    return y, x, z


def mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz):
    """Return a boundary-mapped neighbor site, or None if the mapped z is outside the box."""
    y2, x2, z2 = map_step_boundary(y + dy, x + dx, z + dz, Ly, Lx)
    if z2 < 0 or z2 >= Lz:
        return None
    return y2, x2, z2

# ------------------------------------------------------------
# Substrate initialization
# ------------------------------------------------------------
def init_state_with_steps(Lx: int, Ly: int, Lz: int, W: int = 1, substrate_id: int = 1):
    """
    Initialize a stepped substrate with W terraces above the base layer.
    The resulting staircase profile is stored in Occ and H.
    """
    if Lz < 1:
        raise ValueError("Lz must be >= 1")
    if W < 0:
        raise ValueError("W must be >= 0")
    if W + 2 > Lz:
        raise ValueError(f"Lz={Lz} too small for W={W}. Need at least Lz >= W+2.")

    global STEP_EDGE_W
    STEP_EDGE_W = int(W)

    # Base substrate
    Occ = np.zeros((Ly, Lx, Lz), dtype=np.uint8)
    H   = np.zeros((Ly, Lx), dtype=np.int32)
    Occ[:, :, 0] = substrate_id
    H[:, :] = 0

    if W == 0:
        return Occ, H

    # Reproduce step positions
    arr = np.arange(-1, Ly + 2, 1, dtype=float)  # [-1, 0, ..., Ly+1]
    prof = np.ceil(arr * (W / Ly))               # staircase profile
    dprof = np.diff(prof)
    Steps = np.where(dprof == 1)[0] + 1          # +1 exactly like your code

    # Safety: Steps length should be W+1
    # (for W=1 => 2 entries, etc.)
    if Steps.size < (W + 1):
        # If rounding produced fewer steps, fall back to evenly spaced
        Steps = np.linspace(0, Ly - 1, W + 1).astype(int)

    # Apply terraces: i from 0..W => fill z=i+1
    for i in range(W + 1):
        z = i + 1
        y0 = int(Steps[i])
        if z >= Lz:
            break
        if y0 < 0:
            y0 = 0
        if y0 >= Ly:
            continue
        Occ[y0:, :, z] = substrate_id

    # Build H deterministically from Occ (simple + robust)
    # H[y,x] = highest z where Occ != 0
    for y in range(Ly):
        for x in range(Lx):
            zz = 0
            for z in range(Lz - 1, -1, -1):
                if Occ[y, x, z] != 0:
                    zz = z
                    break
            H[y, x] = zz

    return Occ, H

# ------------------------------------------------------------
# DEP parsing (layer schedule)
# ------------------------------------------------------------
def parse_dep_row(dep_row: np.ndarray):
    """
    Parse one DEP row of the form:
      [T[K], f, pulses, ML/pulse, M2, r2, M3, r3, ...]
    Returns:
      T, f, pulses, flux, species_ids (tuple), ratios (tuple normalized)
    """
    if dep_row.shape[0] < 4:
        raise ValueError("DEP row must have at least 4 entries: [T, f, pulses, ML/pulse, ...]")

    T, f, pulses, flux = dep_row[:4]

    pairs = dep_row[4:]
    species = []
    ratios = []
    for j in range(0, len(pairs), 2):
        if j + 1 >= len(pairs):
            break
        m = int(pairs[j])
        if m == 0:
            break
        species.append(m)
        ratios.append(float(pairs[j + 1]))

    if len(species) == 0:
        # Allow “no deposition” layer, but it’s usually a mistake
        species_ids = tuple()
        ratios_norm = tuple()
    else:
        ratios = np.array(ratios, dtype=float)
        s = ratios.sum()
        if s <= 0:
            raise ValueError("Sum of deposition ratios must be > 0")
        ratios_norm = tuple((ratios / s).tolist())
        species_ids = tuple(species)

    return float(T), float(f), int(pulses), float(flux), species_ids, ratios_norm

# ------------------------------------------------------------
# Deposition Logic
# ------------------------------------------------------------

def deposit_pulse(Occ, H, species_ids, ratios, flux_ML_per_pulse, rng):
    Ly, Lx, Lz = Occ.shape
    species_ids = np.asarray(species_ids, dtype=np.uint8)

    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()

    Natoms = int(round(flux_ML_per_pulse * Lx * Ly))
    if Natoms <= 0:
        return None, None, None, None  # nothing deposited

    H_before = H.copy()

    xs = rng.integers(0, Lx, size=Natoms, endpoint=False)
    ys = rng.integers(0, Ly, size=Natoms, endpoint=False)
    chosen = rng.choice(len(species_ids), size=Natoms, p=ratios)
    ss = species_ids[chosen]

    for x, y, s in zip(xs, ys, ss):
        z = H[y, x] + 1
        if z >= Lz:
            continue
        Occ[y, x, z] = s
        H[y, x] = z

    return xs, ys, ss, H_before


def save_debug_frame(Occ, H, species_colors, frames_dir, idx, title=None, scale=20):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    fname = frames_dir / f"frame_{idx:04d}.png"
    return save_topview_image(Occ, H, species_colors, fname, title=title, scale=scale)


def save_topview_image(Occ, H, species_colors, out_path, title=None, scale=20):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = topview_rgb(Occ, H, species_colors)
    Ly, Lx = H.shape
    fig_w = max(4.0, Lx * scale / 100.0)
    fig_h = max(4.0, Ly * scale / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100, facecolor="white")
    ax.imshow(img, origin="lower", interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=8)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out_path)


def save_snapshot_state(run_dir, snapshot_name, Occ, H, pulse_index, sim_time, event_count, save_full_occ=False):
    """Save lightweight snapshot state for later analysis without plotting."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / snapshot_name.replace(".png", ".npz")
    yy, xx = np.indices(H.shape)
    occ_top = Occ[yy, xx, H]
    payload = {
        "H": H.copy(),
        "Occ_top": occ_top,
        "pulse_index": np.int32(pulse_index),
        "simulation_time": np.float64(sim_time),
        "event_count": np.int32(event_count),
    }
    if save_full_occ:
        payload["Occ"] = Occ.copy()
    np.savez_compressed(out_path, **payload)
    return out_path


def compute_rms_roughness(H):
    """Return RMS roughness and mean height from the current height map."""
    h = H.astype(np.float64, copy=False)
    mean_height = float(h.mean())
    rq = float(np.sqrt(np.mean((h - mean_height) ** 2)))
    return rq, mean_height


def init_roughness_csv(run_dir):
    """Create the roughness CSV with a header and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "roughness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pulse", "progress_percent", "time", "event_count", "rq", "mean_height", "I_proxy"])
    return csv_path


def compute_roughness_proxy(rq, rq_min_so_far, rq_max_so_far):
    """Return a normalized 0..1 inverse-roughness proxy."""
    if rq_max_so_far <= rq_min_so_far:
        return 1.0
    i_proxy = 1.0 - (rq - rq_min_so_far) / (rq_max_so_far - rq_min_so_far)
    return float(np.clip(i_proxy, 0.0, 1.0))


def append_roughness_row(csv_path, H, pulse_index, progress_percent, sim_time, event_count, roughness_tracker):
    """Append one roughness measurement row using the already-available height map."""
    rq, mean_height = compute_rms_roughness(H)
    roughness_tracker["rq_min_so_far"] = min(roughness_tracker["rq_min_so_far"], rq)
    roughness_tracker["rq_max_so_far"] = max(roughness_tracker["rq_max_so_far"], rq)
    i_proxy = compute_roughness_proxy(
        rq,
        roughness_tracker["rq_min_so_far"],
        roughness_tracker["rq_max_so_far"],
    )
    with Path(csv_path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            int(pulse_index),
            float(progress_percent),
            float(sim_time),
            int(event_count),
            rq,
            mean_height,
            i_proxy,
        ])


def build_progress_save_plan(save_every_percent):
    """Return progress fractions for periodic within-pulse saving."""
    save_every_percent = float(save_every_percent)
    if save_every_percent <= 0 or save_every_percent > 100:
        raise ValueError("save_every_percent must satisfy 0 < value <= 100")
    n_steps = int(round(100.0 / save_every_percent))
    return [min(1.0, i * save_every_percent / 100.0) for i in range(1, n_steps + 1)]


def build_snapshot_plan(total_pulses, snapshot_fractions):
    if not snapshot_fractions or total_pulses <= 0:
        return {}

    plan = {}
    for frac in snapshot_fractions:
        frac = float(frac)
        target_pulse = total_pulses if frac >= 1.0 else max(1, int(np.ceil(total_pulses * frac)))
        filename = "snapshot_final.png" if frac >= 1.0 else f"snapshot_{int(round(frac * 100))}.png"
        plan.setdefault(target_pulse, []).append((filename, frac))
    return plan


def build_pulse_snapshot_plan(snapshot_fractions):
    """Return per-pulse snapshot targets keyed by progress fraction."""
    if not snapshot_fractions:
        return []
    plan = []
    for frac in snapshot_fractions:
        frac = float(frac)
        filename = "snapshot_final.png" if frac >= 1.0 else f"snapshot_{int(round(frac * 100))}.png"
        plan.append((min(1.0, frac), filename))
    return sorted(plan, key=lambda item: item[0])


def make_debug_gif(frames_dir, run_dir, output_name="debug.gif", fps=5):
    frames_dir = Path(frames_dir)
    run_dir = Path(run_dir)
    png_files = sorted(frames_dir.glob("frame_*.png"))
    if not png_files:
        print(f"No debug frames found in {frames_dir}.")
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / output_name
    duration_ms = max(1, int(round(1000 / fps)))
    imgs = [Image.open(str(p)).convert("P", palette=Image.ADAPTIVE) for p in png_files]
    imgs[0].save(
        str(out_path),
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )
    for img in imgs:
        img.close()
    print(f"Saved GIF: {out_path}")
    return out_path


def run_dep_plus_relax(Occ, H, DEP, E_vals, rng, plot_every=5, max_events_per_pulse=2000,
                      debug=False, run_dir=None, frames_dir=None, debug_fps=5,
                      validation_counters=None, snapshot_fractions=(),
                      roughness_save_every_percent=100):
    frame_count = 0
    total_pulses = sum(parse_dep_row(DEP[layer])[2] for layer in range(DEP.shape[0]))
    snapshot_plan = build_snapshot_plan(total_pulses, snapshot_fractions)
    pulse_snapshot_plan = build_pulse_snapshot_plan(snapshot_fractions) if total_pulses == 1 else []
    roughness_progress_plan = build_progress_save_plan(roughness_save_every_percent)
    if validation_counters is None:
        validation_counters = init_validation_counters()
    if debug and (run_dir is None or frames_dir is None):
        run_dir, frames_dir = make_run_dir(label="debug", base="runs")
        print(f"Saving debug output in {run_dir}")
    roughness_csv_path = init_roughness_csv(run_dir)
    rq0, _ = compute_rms_roughness(H)
    roughness_tracker = {
        "rq_min_so_far": rq0,
        "rq_max_so_far": rq0,
    }
    append_roughness_row(
        roughness_csv_path, H,
        pulse_index=0,
        progress_percent=0.0,
        sim_time=0.0,
        event_count=0,
        roughness_tracker=roughness_tracker,
    )

    for layer in range(DEP.shape[0]):
        T, f, pulses, flux, species_ids, ratios = parse_dep_row(DEP[layer])
        print(f"Layer {layer+1}: T={T}, pulses={pulses}, flux={flux}, species={species_ids}, ratios={ratios}")

        for p in range(pulses):
            pulse_time = 0.0
            pulse_event_count = 0
            pulse_index = validation_counters["pulses_completed"] + 1
            xs, ys, ss, H_before = deposit_pulse(Occ, H, species_ids, ratios, flux, rng)
            if xs is None:
                validation_counters["hops_per_pulse"].append(0)
                validation_counters["pulses_completed"] += 1
                append_roughness_row(
                    roughness_csv_path, H,
                    pulse_index=pulse_index,
                    progress_percent=100.0,
                    sim_time=0.0,
                    event_count=0,
                    roughness_tracker=roughness_tracker,
                )
            else:
                active = build_active_list_from_last_deposit(xs, ys, ss, H_before, H, Occ.shape[2])
                active_sites = build_active_site_set(active, Occ)

                if debug:
                    frame_count += 1
                    save_debug_frame(Occ, H, species_colors, frames_dir, frame_count,
                                     title=f"Layer {layer+1}, pulse {p+1}: deposit")

                pulse_interval = 1.0 / f
                frame_count, pulse_hops, end_reason, pulse_time, pulse_event_count = relax_after_pulse_step3a(
                    Occ, H, E_vals, active, T, rng,
                    pulse_interval=pulse_interval,
                    max_events=max_events_per_pulse,
                    debug=debug,
                    debug_dir=frames_dir,
                    frame_count=frame_count,
                    title_prefix=f"Layer {layer+1}, pulse {p+1}",
                    debug_fps=debug_fps,
                    validation_counters=validation_counters,
                    active_sites=active_sites,
                    run_dir=run_dir,
                    pulse_snapshot_plan=pulse_snapshot_plan,
                    pulse_index=pulse_index,
                    roughness_csv_path=roughness_csv_path,
                    roughness_progress_plan=roughness_progress_plan,
                    roughness_tracker=roughness_tracker,
                )
                validation_counters["total_accepted_hops"] += pulse_hops
                validation_counters["hops_per_pulse"].append(pulse_hops)
                validation_counters["pulses_completed"] += 1
                if end_reason == "time_limit":
                    validation_counters["pulses_ended_by_time"] += 1
                elif end_reason == "max_events":
                    validation_counters["pulses_ended_by_max_events"] += 1

            if total_pulses > 1:
                completed_pulses = validation_counters["pulses_completed"]
                for snapshot_name, frac in snapshot_plan.get(completed_pulses, []):
                    pct = int(round(frac * 100))
                    label = "final" if frac >= 1.0 else f"{pct}%"
                    title = f"Snapshot {label}: pulse {completed_pulses}/{total_pulses}"
                    save_snapshot_state(
                        run_dir, snapshot_name, Occ, H,
                        pulse_index=completed_pulses,
                        sim_time=pulse_time,
                        event_count=pulse_event_count,
                        save_full_occ=(frac >= 1.0),
                    )
                    save_topview_image(
                        Occ, H, species_colors,
                        Path(run_dir) / snapshot_name,
                        title=title,
                        scale=12,
                    )

            if plot_every and ((p+1) % plot_every == 0):
                show_topview(Occ, H, species_colors, title=f"Layer {layer+1}, pulse {p+1}")

    hops_per_pulse = validation_counters["hops_per_pulse"]
    if hops_per_pulse:
        validation_counters["average_hops_per_pulse"] = float(np.mean(hops_per_pulse))

    if debug:
        return make_debug_gif(frames_dir, run_dir, output_name="debug.gif", fps=debug_fps), validation_counters
    return None, validation_counters
                
def column_top_z(Occ, y, x):
    """Return highest occupied z in column (y,x)."""
    col = Occ[y, x, :]
    nz = np.nonzero(col)[0]
    return int(nz[-1]) if nz.size else 0

def fix_height_if_needed(Occ, H, y, x, z_removed):
    """
    If we removed the atom at (y,x,z_removed) and it was at the top,
    decrease H[y,x] until the next occupied voxel.
    """
    if H[y, x] != z_removed:
        return
    z = z_removed
    while z > 0 and Occ[y, x, z] == 0:
        z -= 1
    H[y, x] = z

def build_active_list_from_last_deposit(xs, ys, ss, H_before, H_after, Lz):
    """
    Build active list for atoms deposited in this pulse.
    We rely on the same order as deposition: each hit stacks.
    """
    active = []
    # reconstruct the z each atom landed on by replaying column heights
    # using a small local dict to track column increments
    local = {}
    for x, y, s in zip(xs, ys, ss):
        key = (y, x)
        z0 = local.get(key, H_before[y, x])
        z = z0 + 1
        local[key] = z
        if z < Lz:
            active.append([y, x, z, int(s)])
    return np.array(active, dtype=np.int32)  # columns: y,x,z,s

def neighbor_energy_6(Occ, E_vals, y, x, z, Ly, Lx, hop_geometry=None):
    """
    Compute EN from 6 face-neighbors at the same z +/- 1 in z.
    Uses E_vals with indexing [species-1, neighbor-1]; ignores neighbor=0.
    """
    s = int(Occ[y, x, z])
    if s == 0:
        return 0.0

    if hop_geometry is not None:
        return interaction_energy_from_neighbors(E_vals, s, hop_geometry.source_neighbor_species)

    nbrs = []
    for dy, dx, dz in ((0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)):
        site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Occ.shape[2])
        nbrs.append(int(Occ[site]) if site is not None else 0)
    return interaction_energy_from_neighbors(E_vals, s, nbrs)


def destination_energy_6(Occ, E_vals, s, y, x, z, Ly, Lx, hop_geometry=None, move_index=None):
    """
    Compute the local 6-neighbor interaction energy for species s
    if it were placed at destination site (y,x,z).
    """
    if hop_geometry is not None and move_index is not None:
        return interaction_energy_from_neighbors(E_vals, s, hop_geometry.destination_neighbor_species[move_index])

    nbrs = []
    for dy, dx, dz in ((0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)):
        site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Occ.shape[2])
        nbrs.append(int(Occ[site]) if site is not None else 0)
    return interaction_energy_from_neighbors(E_vals, s, nbrs)


def interaction_energy_from_neighbors(E_vals, s, neighbor_species):
    """Sum pair interactions for one species against a cached neighbor list."""
    EN = 0.0
    for n in neighbor_species:
        if n != 0:
            EN += float(E_vals[s - 1, n - 1])
    return EN


def rate_prefactor(T):
    """Fixed hop attempt frequency in 1/s."""
    return k0


# ------------------------------------------------------------
# Model-sensitive KMC internals
# These functions control event rates, allowed moves, event selection,
# and KMC time updates. Treat this section as model-validation code.
# ------------------------------------------------------------


def compute_rate_for_active(Occ, E_vals, active, idx, T,
                            active_sites=None, hop_geometry=None):
    """Compute the hop rate for a single active atom."""
    Ly, Lx, Lz = Occ.shape
    y, x, z, s = active[idx]
    if Occ[y, x, z] != s:
        return 0.0

    if hop_geometry is None:
        hop_geometry = build_hop_geometry(Occ, y, x, z)
    safe_source = source_removal_is_safe(
        Occ, active, y, x, z, None, active_sites=active_sites, hop_geometry=hop_geometry
    )
    if safe_source:
        Occ[y, x, z] = 0
        dests, _ = allowed_cubic_destinations(Occ, E_vals, y, x, z, s, T, hop_geometry=hop_geometry)
        Occ[y, x, z] = s
    else:
        dests = []
    if len(dests) == 0:
        return 0.0

    EN = neighbor_energy_6(Occ, E_vals, y, x, z, Ly, Lx, hop_geometry=hop_geometry)
    return rate_prefactor(T) * np.exp(-(Estatic + EN) / (kB * T))


def compute_rates_active(Occ, E_vals, active, T, active_sites=None,
                         return_hop_geometries=False):
    rates = np.zeros(active.shape[0], dtype=np.float64)
    hop_geometries = [None] * active.shape[0] if return_hop_geometries else None
    for i in range(active.shape[0]):
        hop_geometry = build_hop_geometry(Occ, active[i, 0], active[i, 1], active[i, 2])
        if return_hop_geometries:
            hop_geometries[i] = hop_geometry
        rates[i] = compute_rate_for_active(
            Occ, E_vals, active, i, T, active_sites=active_sites, hop_geometry=hop_geometry
        )
    if return_hop_geometries:
        return rates, hop_geometries
    return rates


def pick_index_weighted(rng, w):
    tot = w.sum()
    if tot <= 0:
        return None
    r = rng.random() * tot
    return int(np.searchsorted(np.cumsum(w), r))


def occupied_face_neighbors(Occ, y, x, z):
    """Count occupied 6-connected neighbors around a voxel."""
    Ly, Lx, Lz = Occ.shape

    count = 0
    for dy, dx, dz in ((0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)):
        site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz)
        if site is not None and Occ[site] != 0:
            count += 1
    return count


def face_neighbor_sites(y, x, z, Ly, Lx, Lz):
    """Return valid 6-connected neighbor sites with stepped boundary mapping."""
    sites = []
    for dy, dx, dz in ((0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)):
        site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz)
        if site is not None:
            sites.append(site)
    return sites


def destination_supported(Occ, y, x, z):
    """A hop destination is allowed only if it has at least one face-neighbor."""
    return occupied_face_neighbors(Occ, y, x, z) > 0


def site_occupied(Occ, y, x, z):
    """Return occupancy using the stepped boundary mapping."""
    Ly, Lx, Lz = Occ.shape
    site = map_step_boundary(y, x, z, Ly, Lx)
    y2, x2, z2 = site
    if z2 < 0 or z2 >= Lz:
        return False
    return Occ[y2, x2, z2] != 0


def all_sites_occupied(Occ, sites):
    return all(site_occupied(Occ, y, x, z) for y, x, z in sites)


def local_vic(Occ, y, x, z):
    """Return the local 3x3x3 neighborhood around (y,x,z) with stepped-boundary mapping."""
    Ly, Lx, Lz = Occ.shape
    Vic = np.zeros((3, 3, 3), dtype=np.uint8)
    for iy, dy in enumerate((-1, 0, 1)):
        for ix, dx in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz)
                if site is not None:
                    Vic[iy, ix, iz] = Occ[site]
    return Vic


def build_hop_geometry(Occ, y, x, z):
    """Cache source-centered local geometry once for one hop evaluation."""
    Ly, Lx, Lz = Occ.shape
    local_sites = [None] * HOP_LOCAL_COUNT
    local_species = [0] * HOP_LOCAL_COUNT
    for idx, (dy, dx, dz) in enumerate(HOP_LOCAL_OFFSETS):
        site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz)
        local_sites[idx] = site
        local_species[idx] = int(Occ[site]) if site is not None else 0

    source_vic = np.zeros((3, 3, 3), dtype=np.uint8)
    for iy, ix, iz, local_idx in LOCAL_CUBE_INDEX_TRIPLES:
        source_vic[iy, ix, iz] = local_species[local_idx]

    dest_occ = (source_vic != 0).astype(np.uint8)
    dest_occ[1, 1, 1] = 0

    cubic_neighbor_sites = tuple(local_sites[idx] for idx in FACE_OFFSET_INDICES)
    axis2_occupied = tuple(local_species[idx] != 0 for idx in AXIS2_OFFSET_INDICES)
    move_sites = tuple(local_sites[idx] for idx in MOVE_SITE_INDICES)
    source_neighbor_species = tuple(local_species[idx] for idx in FACE_OFFSET_INDICES)

    destination_neighbor_species = []
    for neighbor_indices in MOVE_FACE_INDICES:
        destination_neighbor_species.append(
            tuple(
                0 if idx == CENTER_LOCAL_INDEX else local_species[idx]
                for idx in neighbor_indices
            )
        )

    allowed_moves = tuple(
        destination_allowed_by_family(dest_occ, dy, dx, dz)
        for dy, dx, dz in MOVE_DISPLACEMENTS_18
    )

    return HopGeometry(
        source_vic=source_vic,
        dest_occ=dest_occ,
        cubic_neighbor_sites=cubic_neighbor_sites,
        axis2_occupied=axis2_occupied,
        move_sites=move_sites,
        source_neighbor_species=source_neighbor_species,
        destination_neighbor_species=tuple(destination_neighbor_species),
        allowed_moves=allowed_moves,
    )


def vic_count(Vic, coords):
    return sum(Vic[i, j, k] != 0 for i, j, k in coords)


def vic_all(Vic, coords):
    return all(Vic[i, j, k] != 0 for i, j, k in coords)


def local_cubic_neighbors(Vic):
    """Return cubic neighbor occupancies in the order left, right, up, down, below, above."""
    return np.array([
        Vic[1, 0, 1], Vic[1, 2, 1],
        Vic[0, 1, 1], Vic[2, 1, 1],
        Vic[1, 1, 0], Vic[1, 1, 2],
    ], dtype=np.uint8)


def active_neighbor_flags(active, Occ, y, x, z, active_sites=None, hop_geometry=None):
    """Return which of the 6 cubic neighbor sites currently contain active atoms."""
    Ly, Lx, Lz = Occ.shape
    if active_sites is None:
        active_sites = set()
        for ay, ax, az, s in active:
            if 0 <= az < Lz and Occ[ay, ax, az] == s:
                active_sites.add((int(ay), int(ax), int(az)))

    if hop_geometry is not None:
        sites = hop_geometry.cubic_neighbor_sites
    else:
        sites = [
            mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz)
            for dy, dx, dz in ((0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1))
        ]
    return [site in active_sites if site is not None else False for site in sites]


def prevents_void(Vic):
    return np.count_nonzero(local_cubic_neighbors(Vic)) == 6


def prevents_single_floater(Occ, active_flags, y, x, z, Vic, axis2_occupied=None):
    left, right, up, down, below, above = active_flags

    if axis2_occupied is None:
        axis2_occupied = {
            "left": site_occupied(Occ, y, x - 2, z),
            "right": site_occupied(Occ, y, x + 2, z),
            "up": site_occupied(Occ, y - 2, x, z),
            "down": site_occupied(Occ, y + 2, x, z),
            "below": site_occupied(Occ, y, x, z - 2),
            "above": site_occupied(Occ, y, x, z + 2),
        }
        above_blocked = axis2_occupied["above"]
        left_blocked = axis2_occupied["left"]
        right_blocked = axis2_occupied["right"]
        up_blocked = axis2_occupied["up"]
        down_blocked = axis2_occupied["down"]
        below_blocked = axis2_occupied["below"]
    else:
        left_blocked, right_blocked, up_blocked, down_blocked, below_blocked, above_blocked = axis2_occupied

    if above and vic_count(Vic, [(1, 0, 2), (1, 2, 2), (0, 1, 2), (2, 1, 2)]) == 0 and not above_blocked:
        return True
    if left and vic_count(Vic, [(1, 0, 0), (1, 0, 2), (0, 0, 1), (2, 0, 1)]) == 0 and not left_blocked:
        return True
    if right and vic_count(Vic, [(1, 2, 0), (1, 2, 2), (0, 2, 1), (2, 2, 1)]) == 0 and not right_blocked:
        return True
    if up and vic_count(Vic, [(0, 1, 0), (0, 1, 2), (0, 0, 1), (0, 2, 1)]) == 0 and not up_blocked:
        return True
    if down and vic_count(Vic, [(2, 1, 0), (2, 1, 2), (2, 0, 1), (2, 2, 1)]) == 0 and not down_blocked:
        return True
    if below and vic_count(Vic, [(1, 0, 0), (1, 2, 0), (0, 1, 0), (2, 1, 0)]) == 0 and not below_blocked:
        return True
    return False


def prevents_multiple_floater(Vic, active_flags):
    left, right, up, down, below, above = active_flags
    n_neighbors = np.count_nonzero(local_cubic_neighbors(Vic))

    if above and n_neighbors > 1 and sum([
        vic_all(Vic, [(1, 0, 1), (1, 0, 2)]),
        vic_all(Vic, [(1, 2, 1), (1, 2, 2)]),
        vic_all(Vic, [(0, 1, 1), (0, 1, 2)]),
        vic_all(Vic, [(2, 1, 1), (2, 1, 2)]),
    ]) == 0:
        return True

    if left and n_neighbors > 1 and sum([
        vic_all(Vic, [(1, 0, 0), (1, 1, 0)]),
        vic_all(Vic, [(1, 0, 2), (1, 1, 2)]),
        vic_all(Vic, [(0, 0, 1), (0, 1, 1)]),
        vic_all(Vic, [(2, 0, 1), (2, 1, 1)]),
    ]) == 0:
        return True

    if right and n_neighbors > 1 and sum([
        vic_all(Vic, [(1, 1, 0), (1, 2, 0)]),
        vic_all(Vic, [(1, 1, 2), (1, 2, 2)]),
        vic_all(Vic, [(0, 1, 1), (0, 2, 1)]),
        vic_all(Vic, [(2, 1, 1), (2, 2, 1)]),
    ]) == 0:
        return True

    if up and n_neighbors > 1 and sum([
        vic_all(Vic, [(0, 1, 0), (1, 1, 0)]),
        vic_all(Vic, [(0, 1, 2), (1, 1, 2)]),
        vic_all(Vic, [(0, 0, 1), (1, 0, 1)]),
        vic_all(Vic, [(0, 2, 1), (1, 2, 1)]),
    ]) == 0:
        return True

    if down and n_neighbors > 1 and sum([
        vic_all(Vic, [(1, 1, 0), (2, 1, 0)]),
        vic_all(Vic, [(1, 1, 2), (2, 1, 2)]),
        vic_all(Vic, [(1, 0, 1), (2, 0, 1)]),
        vic_all(Vic, [(1, 2, 1), (2, 2, 1)]),
    ]) == 0:
        return True

    if below and n_neighbors > 1 and sum([
        vic_all(Vic, [(1, 0, 0), (1, 0, 1)]),
        vic_all(Vic, [(1, 2, 0), (1, 2, 1)]),
        vic_all(Vic, [(0, 1, 0), (0, 1, 1)]),
        vic_all(Vic, [(2, 1, 0), (2, 1, 1)]),
    ]) == 0:
        return True

    return False


def source_rejection_reason(Occ, active, y, x, z, active_sites=None, hop_geometry=None):
    """Return the source-side rejection reason before destination evaluation."""
    if hop_geometry is None:
        hop_geometry = build_hop_geometry(Occ, y, x, z)
    Vic = hop_geometry.source_vic
    neighbors = local_cubic_neighbors(Vic)
    if np.count_nonzero(neighbors) == 0:
        return "isolated_source"
    if prevents_void(Vic):
        return "void"

    active_flags = active_neighbor_flags(active, Occ, y, x, z, active_sites=active_sites, hop_geometry=hop_geometry)
    if prevents_single_floater(Occ, active_flags, y, x, z, Vic, axis2_occupied=hop_geometry.axis2_occupied):
        return "single_floater"
    if prevents_multiple_floater(Vic, active_flags):
        return "multiple_floater"
    return None


def source_removal_is_safe(Occ, active, y, x, z, dest_site=None, active_sites=None, hop_geometry=None):
    """Convenience wrapper returning True when source removal is allowed."""
    return source_rejection_reason(
        Occ, active, y, x, z, active_sites=active_sites, hop_geometry=hop_geometry
    ) is None


def build_active_site_set(active, Occ):
    """Build a set of active atom coordinates for fast neighbor checks."""
    Ly, Lx, Lz = Occ.shape
    active_sites = set()
    for ay, ax, az, s in active:
        if 0 <= az < Lz and Occ[ay, ax, az] == s:
            active_sites.add((int(ay), int(ax), int(az)))
    return active_sites


def build_active_index_map(active, Occ):
    """Map active atom coordinates to their index for fast updates."""
    active_index = {}
    Ly, Lx, Lz = Occ.shape
    for idx, (ay, ax, az, s) in enumerate(active):
        if 0 <= az < Lz and Occ[ay, ax, az] == s:
            active_index[(int(ay), int(ax), int(az))] = idx
    return active_index


def affected_active_indices(active_index, y, x, z, Ly, Lx, Lz):
    """Return indices of active atoms within a 3x3x3 neighborhood."""
    indices = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                site = mapped_neighbor_site(y, x, z, dy, dx, dz, Ly, Lx, Lz)
                if site is None:
                    continue
                idx = active_index.get(site)
                if idx is not None:
                    indices.add(idx)
    return indices


def init_validation_counters():
    return {
        "total_accepted_hops": 0,
        "hops_per_pulse": [],
        "prevented_voids": 0,
        "prevented_single_floaters": 0,
        "prevented_multiple_floaters": 0,
        "pulses_completed": 0,
        "pulses_ended_by_time": 0,
        "pulses_ended_by_max_events": 0,
        "average_hops_per_pulse": 0.0,
    }


MOVE_DISPLACEMENTS_18 = [
    (0, -1, 0), (0, 1, 0),
    (-1, 0, 0), (1, 0, 0),
    (0, 0, -1), (0, 0, 1),
    (-1, -1, 0), (-1, 1, 0),
    (1, -1, 0), (1, 1, 0),
    (0, -1, -1), (0, 1, -1),
    (-1, 0, -1), (1, 0, -1),
    (0, -1, 1), (0, 1, 1),
    (-1, 0, 1), (1, 0, 1),
]

FACE_OFFSETS_6 = (
    (0, -1, 0), (0, 1, 0),
    (-1, 0, 0), (1, 0, 0),
    (0, 0, -1), (0, 0, 1),
)

LOCAL_CUBE_OFFSETS_27 = tuple(
    (dy, dx, dz)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    for dz in (-1, 0, 1)
)

AXIS2_OFFSETS = {
    "left": (0, -2, 0),
    "right": (0, 2, 0),
    "up": (-2, 0, 0),
    "down": (2, 0, 0),
    "below": (0, 0, -2),
    "above": (0, 0, 2),
}

MOVE_FACE_OFFSETS = {
    move: tuple((move[0] + dy, move[1] + dx, move[2] + dz) for dy, dx, dz in FACE_OFFSETS_6)
    for move in MOVE_DISPLACEMENTS_18
}

_HOP_LOCAL_OFFSETS = set(LOCAL_CUBE_OFFSETS_27)
_HOP_LOCAL_OFFSETS.update(AXIS2_OFFSETS.values())
for move in MOVE_DISPLACEMENTS_18:
    _HOP_LOCAL_OFFSETS.update(MOVE_FACE_OFFSETS[move])
HOP_LOCAL_OFFSETS = tuple(sorted(_HOP_LOCAL_OFFSETS))
HOP_LOCAL_INDEX = {offset: idx for idx, offset in enumerate(HOP_LOCAL_OFFSETS)}
HOP_LOCAL_COUNT = len(HOP_LOCAL_OFFSETS)
CENTER_LOCAL_INDEX = HOP_LOCAL_INDEX[(0, 0, 0)]
FACE_OFFSET_INDICES = tuple(HOP_LOCAL_INDEX[offset] for offset in FACE_OFFSETS_6)
AXIS2_ORDER = ("left", "right", "up", "down", "below", "above")
AXIS2_OFFSET_INDICES = tuple(HOP_LOCAL_INDEX[AXIS2_OFFSETS[name]] for name in AXIS2_ORDER)
MOVE_SITE_INDICES = tuple(HOP_LOCAL_INDEX[move] for move in MOVE_DISPLACEMENTS_18)
MOVE_FACE_INDICES = tuple(
    tuple(HOP_LOCAL_INDEX[offset] for offset in MOVE_FACE_OFFSETS[move])
    for move in MOVE_DISPLACEMENTS_18
)
LOCAL_CUBE_INDEX_TRIPLES = tuple(
    (dy + 1, dx + 1, dz + 1, HOP_LOCAL_INDEX[(dy, dx, dz)])
    for dy, dx, dz in LOCAL_CUBE_OFFSETS_27
)


class HopGeometry:
    """Lightweight per-hop cache for source-local mapped geometry."""
    __slots__ = (
        "source_vic",
        "dest_occ",
        "cubic_neighbor_sites",
        "axis2_occupied",
        "move_sites",
        "source_neighbor_species",
        "destination_neighbor_species",
        "allowed_moves",
    )

    def __init__(
        self,
        source_vic,
        dest_occ,
        cubic_neighbor_sites,
        axis2_occupied,
        move_sites,
        source_neighbor_species,
        destination_neighbor_species,
        allowed_moves,
    ):
        self.source_vic = source_vic
        self.dest_occ = dest_occ
        self.cubic_neighbor_sites = cubic_neighbor_sites
        self.axis2_occupied = axis2_occupied
        self.move_sites = move_sites
        self.source_neighbor_species = source_neighbor_species
        self.destination_neighbor_species = destination_neighbor_species
        self.allowed_moves = allowed_moves


def move_family(dy, dx, dz):
    if (dy == 0) and (dx == 0):
        return "cubic"
    if dz == 0:
        return "diag_inplane"
    if dz < 0:
        return "diag_down"
    return "diag_up"


def cubic_destination_allowed(occ, dy, dx, dz):
    """Cubic move acceptance rule for the local 3x3x3 neighborhood."""
    if (dy, dx, dz) == (0, -1, 0):  # left
        return (
            (occ[1, 0, 0] and occ[1, 1, 0]) or
            (occ[0, 0, 1] and occ[0, 1, 1]) or
            (occ[2, 0, 1] and occ[2, 1, 1]) or
            (occ[1, 0, 2] and occ[1, 1, 2])
        )
    if (dy, dx, dz) == (0, 1, 0):  # right
        return (
            (occ[1, 1, 0] and occ[1, 2, 0]) or
            (occ[0, 1, 1] and occ[0, 2, 1]) or
            (occ[2, 1, 1] and occ[2, 2, 1]) or
            (occ[1, 1, 2] and occ[1, 2, 2])
        )
    if (dy, dx, dz) == (-1, 0, 0):  # up
        return (
            (occ[0, 1, 0] and occ[1, 1, 0]) or
            (occ[0, 0, 1] and occ[1, 0, 1]) or
            (occ[0, 2, 1] and occ[1, 2, 1]) or
            (occ[0, 1, 2] and occ[1, 1, 2])
        )
    if (dy, dx, dz) == (1, 0, 0):  # down
        return (
            (occ[1, 1, 0] and occ[2, 1, 0]) or
            (occ[1, 0, 1] and occ[2, 0, 1]) or
            (occ[1, 2, 1] and occ[2, 2, 1]) or
            (occ[1, 1, 2] and occ[2, 1, 2])
        )
    if (dy, dx, dz) == (0, 0, -1):  # below
        return (
            (occ[1, 0, 0] and occ[1, 0, 1]) or
            (occ[1, 2, 0] and occ[1, 2, 1]) or
            (occ[0, 1, 0] and occ[0, 1, 1]) or
            (occ[2, 1, 0] and occ[2, 1, 1])
        )
    if (dy, dx, dz) == (0, 0, 1):  # above
        return (
            (occ[1, 0, 1] and occ[1, 0, 2]) or
            (occ[1, 2, 1] and occ[1, 2, 2]) or
            (occ[0, 1, 1] and occ[0, 1, 2]) or
            (occ[2, 1, 1] and occ[2, 1, 2])
        )
    return False


def inplane_diagonal_allowed(occ, dy, dx):
    """In-plane diagonal acceptance rule for the local 3x3x3 neighborhood."""
    if (dy, dx) == (-1, -1):  # left-up
        path_count = occ[1, 0, 1] + occ[0, 1, 1]
        if path_count == 1:
            return True
        if path_count == 0:
            return (
                (occ[0, 0, 0] + occ[1, 0, 0] + occ[0, 1, 0] + occ[1, 1, 0] == 4) or
                (occ[0, 0, 2] + occ[1, 0, 2] + occ[0, 1, 2] + occ[1, 1, 2] == 4)
            )
        return False

    if (dy, dx) == (1, -1):  # left-down
        path_count = occ[1, 0, 1] + occ[2, 1, 1]
        if path_count == 1:
            return True
        if path_count == 0:
            return (
                (occ[1, 0, 0] + occ[2, 0, 0] + occ[2, 1, 0] + occ[1, 1, 0] == 4) or
                (occ[1, 0, 2] + occ[2, 0, 2] + occ[2, 1, 2] + occ[1, 1, 2] == 4)
            )
        return False

    if (dy, dx) == (-1, 1):  # right-up
        path_count = occ[1, 2, 1] + occ[0, 1, 1]
        if path_count == 1:
            return True
        if path_count == 0:
            return (
                (occ[0, 2, 0] + occ[1, 2, 0] + occ[0, 1, 0] + occ[1, 1, 0] == 4) or
                (occ[0, 2, 2] + occ[1, 2, 2] + occ[0, 1, 2] + occ[1, 1, 2] == 4)
            )
        return False

    if (dy, dx) == (1, 1):  # right-down
        path_count = occ[1, 2, 1] + occ[2, 1, 1]
        if path_count == 1:
            return True
        if path_count == 0:
            return (
                (occ[2, 2, 0] + occ[1, 2, 0] + occ[2, 1, 0] + occ[1, 1, 0] == 4) or
                (occ[2, 2, 2] + occ[1, 2, 2] + occ[2, 1, 2] + occ[1, 1, 2] == 4)
            )
        return False

    return False


def vertical_diagonal_allowed(occ, dy, dx, dz):
    """Vertical diagonal acceptance rule for the local 3x3x3 neighborhood."""
    if dz < 0:
        if (dy, dx) == (0, -1):  # left-down-z
            path_count = occ[1, 0, 1] + occ[1, 1, 0]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[0, 0, 0] + occ[0, 0, 1] + occ[0, 1, 0] + occ[0, 1, 1] == 4) or
                    (occ[2, 0, 0] + occ[2, 0, 1] + occ[2, 1, 0] + occ[2, 1, 1] == 4)
                )
            return False

        if (dy, dx) == (0, 1):  # right-down-z
            path_count = occ[1, 2, 1] + occ[1, 1, 0]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[0, 2, 0] + occ[0, 2, 1] + occ[0, 1, 0] + occ[0, 1, 1] == 4) or
                    (occ[2, 2, 0] + occ[2, 2, 1] + occ[2, 1, 0] + occ[2, 1, 1] == 4)
                )
            return False

        if (dy, dx) == (-1, 0):  # up-down-z
            path_count = occ[0, 1, 1] + occ[1, 1, 0]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[0, 0, 0] + occ[0, 0, 1] + occ[1, 0, 0] + occ[1, 0, 1] == 4) or
                    (occ[0, 2, 0] + occ[0, 2, 1] + occ[1, 2, 0] + occ[1, 2, 1] == 4)
                )
            return False

        if (dy, dx) == (1, 0):  # down-down-z
            path_count = occ[2, 1, 1] + occ[1, 1, 0]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[2, 0, 0] + occ[2, 0, 1] + occ[1, 0, 0] + occ[1, 0, 1] == 4) or
                    (occ[2, 2, 0] + occ[2, 2, 1] + occ[1, 2, 0] + occ[1, 2, 1] == 4)
                )
            return False

    if dz > 0:
        if (dy, dx) == (0, -1):  # left-up-z
            path_count = occ[1, 0, 1] + occ[1, 1, 2]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[0, 0, 1] + occ[0, 0, 2] + occ[0, 1, 1] + occ[0, 1, 2] == 4) or
                    (occ[2, 0, 1] + occ[2, 0, 2] + occ[2, 1, 1] + occ[2, 1, 2] == 4)
                )
            return False

        if (dy, dx) == (0, 1):  # right-up-z
            path_count = occ[1, 2, 1] + occ[1, 1, 2]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[0, 2, 1] + occ[0, 2, 2] + occ[0, 1, 1] + occ[0, 1, 2] == 4) or
                    (occ[2, 2, 1] + occ[2, 2, 2] + occ[2, 1, 1] + occ[2, 1, 2] == 4)
                )
            return False

        if (dy, dx) == (-1, 0):  # up-up-z
            path_count = occ[0, 1, 1] + occ[1, 1, 2]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[0, 0, 1] + occ[0, 0, 2] + occ[1, 0, 1] + occ[1, 0, 2] == 4) or
                    (occ[0, 2, 1] + occ[0, 2, 2] + occ[1, 2, 1] + occ[1, 2, 2] == 4)
                )
            return False

        if (dy, dx) == (1, 0):  # down-up-z
            path_count = occ[2, 1, 1] + occ[1, 1, 2]
            if path_count == 1:
                return True
            if path_count == 0:
                return (
                    (occ[2, 0, 1] + occ[2, 0, 2] + occ[1, 0, 1] + occ[1, 0, 2] == 4) or
                    (occ[2, 2, 1] + occ[2, 2, 2] + occ[1, 2, 1] + occ[1, 2, 2] == 4)
                )
            return False

    return False


def destination_allowed_by_family(occ, dy, dx, dz):
    """Dispatch move acceptance to the appropriate 18-direction family rule."""
    family = move_family(dy, dx, dz)
    if family == "cubic":
        return cubic_destination_allowed(occ, dy, dx, dz)
    if family == "diag_inplane":
        return inplane_diagonal_allowed(occ, dy, dx)
    return vertical_diagonal_allowed(occ, dy, dx, dz)


def legacy_direction_weight(T, dy, dx, dz):
    """
    Direction-family weighting used by the current model:
    cubic moves carry weight 1 and diagonal moves carry exp(-E_ES / (kB*T)).
    """
    if move_family(dy, dx, dz) == "cubic":
        return 1.0
    return float(np.exp(-E_ES / (kB * T)))


def allowed_cubic_destinations(Occ, E_vals, y, x, z, s, T, hop_geometry=None):
    """Return allowed 18-direction destinations and their direction-family weights."""
    if hop_geometry is None:
        hop_geometry = build_hop_geometry(Occ, y, x, z)
    move_sites = hop_geometry.move_sites
    allowed_moves = hop_geometry.allowed_moves

    dests = []
    weights = []
    for move_index, (dy, dx, dz) in enumerate(MOVE_DISPLACEMENTS_18):
        site = move_sites[move_index]
        if site is None:
            continue
        y2, x2, z2 = site
        if Occ[y2, x2, z2] != 0:
            continue
        if not allowed_moves[move_index]:
            continue

        dests.append((y2, x2, z2))
        weights.append(legacy_direction_weight(T, dy, dx, dz))

    return dests, np.asarray(weights, dtype=np.float64)


def try_hop_cubic(Occ, H, E_vals, active, idx, T, rng, validation_counters=None, active_sites=None, hop_geometry=None):
    """
    Move an active atom through the current 18-direction move set.
    The atom is selected first, then a destination direction is drawn from the
    allowed move family weights.
    """
    Ly, Lx, Lz = Occ.shape
    y, x, z, s = active[idx]

    if Occ[y, x, z] != s:
        return False

    if hop_geometry is None:
        hop_geometry = build_hop_geometry(Occ, y, x, z)
    rejection_reason = source_rejection_reason(
        Occ, active, y, x, z, active_sites=active_sites, hop_geometry=hop_geometry
    )
    if rejection_reason is not None:
        if validation_counters is not None:
            if rejection_reason == "void":
                validation_counters["prevented_voids"] += 1
            elif rejection_reason == "single_floater":
                validation_counters["prevented_single_floaters"] += 1
            elif rejection_reason == "multiple_floater":
                validation_counters["prevented_multiple_floaters"] += 1
        return False

    Occ[y, x, z] = 0

    dests, weights = allowed_cubic_destinations(Occ, E_vals, y, x, z, s, T, hop_geometry=hop_geometry)
    if len(dests) == 0:
        Occ[y, x, z] = s
        return False

    dest_idx = pick_index_weighted(rng, weights)
    if dest_idx is None:
        Occ[y, x, z] = s
        return False

    y2, x2, z2 = dests[dest_idx]
    Occ[y2, x2, z2] = s

    if z2 > H[y2, x2]:
        H[y2, x2] = z2
    fix_height_if_needed(Occ, H, y, x, z_removed=z)

    active[idx, 0] = y2
    active[idx, 1] = x2
    active[idx, 2] = z2
    if active_sites is not None:
        active_sites.discard((int(y), int(x), int(z)))
        active_sites.add((int(y2), int(x2), int(z2)))

    return True


def try_hop_surface_only(Occ, H, active, idx, rng):
    """
    Move active atom idx to top of a random in-plane neighbor column.
    No ES, no diagonal, no floater checks yet.
    """
    Ly, Lx, Lz = Occ.shape
    y, x, z, s = active[idx]

    # still present?
    if Occ[y, x, z] != s:
        return False

    # choose among 4 in-plane neighbors uniformly
    # (later we’ll make direction selection energetic)
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    dy, dx = dirs[rng.integers(0, 4)]
    y2 = (y + dy) % Ly
    x2 = (x + dx) % Lx

    z2 = H[y2, x2] + 1
    if z2 >= Lz:
        return False

    # perform move
    Occ[y2, x2, z2] = s
    Occ[y, x, z] = 0

    # update heights
    if z2 > H[y2, x2]:
        H[y2, x2] = z2
    fix_height_if_needed(Occ, H, y, x, z_removed=z)

    # update active record
    active[idx, 0] = y2
    active[idx, 1] = x2
    active[idx, 2] = z2

    return True

def relax_after_pulse_step3a(Occ, H, E_vals, active, T, rng, pulse_interval,
                              max_events=2000,
                              debug=False, debug_dir=None, frame_count=0,
                              title_prefix=None, debug_fps=5,
                              validation_counters=None, active_sites=None,
                              run_dir=None, pulse_snapshot_plan=(),
                              pulse_index=1, roughness_csv_path=None,
                              roughness_progress_plan=(), roughness_tracker=None):
    """
    Perform time-based KMC relaxation until the next pulse or a failsafe limit.
    """
    t = 0.0
    event_count = 0
    accepted_hops = 0
    end_reason = None
    last_progress_step = -1
    next_event_report = 2000
    Ly, Lx, Lz = Occ.shape
    pulse_snapshot_plan = list(pulse_snapshot_plan)
    next_snapshot_idx = 0
    roughness_progress_plan = list(roughness_progress_plan)
    next_roughness_idx = 0

    if active_sites is None:
        active_sites = build_active_site_set(active, Occ)
    active_index = build_active_index_map(active, Occ)
    rates, hop_geometries = compute_rates_active(
        Occ, E_vals, active, T, active_sites=active_sites, return_hop_geometries=True
    )
    total_rate = rates.sum()

    while t < pulse_interval and event_count < max_events:
        if total_rate <= 0:
            end_reason = "no_events"
            break

        idx = pick_index_weighted(rng, rates)
        if idx is None:
            end_reason = "selection_failed"
            break

        u = max(rng.random(), np.finfo(np.float64).tiny)
        dt = -np.log(u) / total_rate
        t += dt
        event_count += 1

        y0, x0, z0, _ = active[idx]
        moved = try_hop_cubic(Occ, H, E_vals, active, idx, T, rng,
                              validation_counters=validation_counters,
                              active_sites=active_sites,
                              hop_geometry=hop_geometries[idx])
        if moved:
            y1, x1, z1, _ = active[idx]
            old_indices = affected_active_indices(active_index, y0, x0, z0, Ly, Lx, Lz)
            active_index.pop((int(y0), int(x0), int(z0)), None)
            active_index[(int(y1), int(x1), int(z1))] = idx
            new_indices = affected_active_indices(active_index, y1, x1, z1, Ly, Lx, Lz)
            affected = old_indices | new_indices
        else:
            affected = ()
        for j in affected:
            old_rate = rates[j]
            new_hop_geometry = build_hop_geometry(Occ, active[j, 0], active[j, 1], active[j, 2])
            hop_geometries[j] = new_hop_geometry
            new_rate = compute_rate_for_active(
                Occ, E_vals, active, j, T, active_sites=active_sites, hop_geometry=new_hop_geometry
            )
            rates[j] = new_rate
            total_rate += new_rate - old_rate

        progress = min(1.0, t / pulse_interval) if pulse_interval > 0 else 1.0
        event_progress = min(1.0, event_count / max_events) if max_events > 0 else 1.0
        snapshot_progress = max(progress, event_progress)
        while roughness_csv_path is not None and next_roughness_idx < len(roughness_progress_plan):
            target_frac = roughness_progress_plan[next_roughness_idx]
            if snapshot_progress + 1e-12 < target_frac:
                break
            append_roughness_row(
                roughness_csv_path, H,
                pulse_index=pulse_index,
                progress_percent=round(target_frac * 100.0, 10),
                sim_time=t,
                event_count=event_count,
                roughness_tracker=roughness_tracker,
            )
            next_roughness_idx += 1
        while run_dir is not None and next_snapshot_idx < len(pulse_snapshot_plan):
            target_frac, snapshot_name = pulse_snapshot_plan[next_snapshot_idx]
            if snapshot_progress + 1e-12 < target_frac:
                break
            pct = int(round(target_frac * 100))
            label = "final" if target_frac >= 1.0 else f"{pct}%"
            title = f"Snapshot {label}: {title_prefix}" if title_prefix else f"Snapshot {label}"
            save_snapshot_state(
                run_dir, snapshot_name, Occ, H,
                pulse_index=pulse_index,
                sim_time=t,
                event_count=event_count,
                save_full_occ=(target_frac >= 1.0),
            )
            save_topview_image(
                Occ, H, species_colors,
                Path(run_dir) / snapshot_name,
                title=title,
                scale=12,
            )
            next_snapshot_idx += 1
        progress_step = int(progress * 20)
        if event_count >= next_event_report or progress_step >= last_progress_step + 1:
            filled = int(progress * 10)
            bar = "#" * filled + "-" * (10 - filled)
            if title_prefix:
                print(
                    f"\r{title_prefix} [{bar}] t: {t:.4g} / {pulse_interval:.4g} s | events: {event_count}",
                    end="",
                    flush=True,
                )
            next_event_report = event_count + 2000
            last_progress_step = progress_step
        if debug and moved:
            frame_count += 1
            line1 = f"{title_prefix}, hop {event_count}" if title_prefix else f"hop {event_count}"
            line2 = f"t = {t:.4g} s / {pulse_interval:.4g} s"
            line3 = f"events = {event_count} / {max_events}"
            title = f"{line1}\n{line2}\n{line3}"
            save_debug_frame(Occ, H, species_colors, debug_dir, frame_count, title=title)
        if moved:
            accepted_hops += 1

    if end_reason is None:
        if t >= pulse_interval:
            end_reason = "time_limit"
        elif event_count >= max_events:
            end_reason = "max_events"
        else:
            end_reason = "completed"

    if title_prefix:
        print(
            f"\r{title_prefix} completed | t={t:.4g} / {pulse_interval:.4g} s | "
            f"events={event_count} | ended_by={end_reason}"
        )
    while run_dir is not None and next_snapshot_idx < len(pulse_snapshot_plan):
        target_frac, snapshot_name = pulse_snapshot_plan[next_snapshot_idx]
        pct = int(round(target_frac * 100))
        label = "final" if target_frac >= 1.0 else f"{pct}%"
        title = f"Snapshot {label}: {title_prefix}" if title_prefix else f"Snapshot {label}"
        save_snapshot_state(
            run_dir, snapshot_name, Occ, H,
            pulse_index=pulse_index,
            sim_time=t,
            event_count=event_count,
            save_full_occ=(target_frac >= 1.0),
        )
        save_topview_image(
            Occ, H, species_colors,
            Path(run_dir) / snapshot_name,
            title=title,
            scale=12,
        )
        next_snapshot_idx += 1
    while roughness_csv_path is not None and next_roughness_idx < len(roughness_progress_plan):
        target_frac = roughness_progress_plan[next_roughness_idx]
        append_roughness_row(
            roughness_csv_path, H,
            pulse_index=pulse_index,
            progress_percent=round(target_frac * 100.0, 10),
            sim_time=t,
            event_count=event_count,
            roughness_tracker=roughness_tracker,
        )
        next_roughness_idx += 1
    return frame_count, accepted_hops, end_reason, t, event_count

# ------------------------------------------------------------
# Visualization functions
# ------------------------------------------------------------

def top_surface_species(Occ: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Return Top[y,x] = species id at the top occupied voxel."""
    Ly, Lx = H.shape
    Top = np.zeros((Ly, Lx), dtype=np.uint8)
    for y in range(Ly):
        for x in range(Lx):
            Top[y, x] = Occ[y, x, H[y, x]]
    return Top

def topview_rgb(Occ: np.ndarray,
                H: np.ndarray,
                species_colors,
                bg=(0.0, 0.0, 0.0),
                style=TOPVIEW_STYLE,
                height_floor=TOPVIEW_HEIGHT_FLOOR,
                height_span=TOPVIEW_HEIGHT_SPAN) -> np.ndarray:
    """
    Build an RGB image where:
      - base color depends on species id
      - optional brightness depends gently on height
    species_colors: {species_id: (r,g,b)} in 0..1
    """
    Top = top_surface_species(Occ, H)

    Ly, Lx = H.shape
    rgb = np.zeros((Ly, Lx, 3), dtype=np.float32)
    rgb[:, :, :] = bg

    for s, col in species_colors.items():
        mask = (Top == s)
        if not np.any(mask):
            continue
        rgb[mask, 0] = col[0]
        rgb[mask, 1] = col[1]
        rgb[mask, 2] = col[2]

    if style == "species_plus_height":
        h = H.astype(np.float32, copy=False)
        h_norm = (h - h.min()) / (h.max() - h.min() + 1e-12)
        brightness = height_floor + height_span * h_norm
        rgb *= brightness[:, :, None]
    elif style != "species_only":
        raise ValueError(f"Unknown TOPVIEW_STYLE: {style}")

    return np.clip(rgb, 0.0, 1.0)

def show_topview(Occ, H, species_colors, title=None):
    img = topview_rgb(Occ, H, species_colors)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# CONFIG SECTION
# This is the main place students should edit routine run settings.
# ------------------------------------------------------------

def default_run_config():
    """Return the current working simulation configuration."""
    return {
        "RUN_TYPE": "single",  # options: "single", "doe"
        "seed": 42,
        "Lx": 128,
        "Ly": 128,
        "Lz": 35,
        "W": 1,
        "plot_every": 0,
        "max_events_per_pulse": 100_000_000,
        "debug_fps": 5,
        "enable_profiling": False,
        "profile_pulses": 10,
        "save_snapshots": True,
        "save_final_state": True,
        "save_debug_frames": False,
        "enable_plotting": False,
        "roughness_save_every_percent": 100,
        # DEP columns: T_K, f_Hz, pulses, ML_per_pulse, then species_id, ratio pairs.
        "DEP": np.array([[600.0 + 273.15, 2, 50, 1 / 25, 2, 1/3, 3, 2/3]], dtype=float),
        # E_vals[i, j] is the interaction energy between species i+1 and neighbor species j+1.
        "E_vals": np.array([[0.00, 0.25, 0.49, 0.25],
                            [0.25, 0.51, 0.25, 0.25],
                            [0.49, 0.25, 0.51, 0.25],
                            [0.25, 0.25, 0.25, 0.51]], dtype=np.float32),
        # DOE overwrites only DEP[0, 0] temperature and DEP[0, 1] frequency.
        "DOE_T_C": [700, 750, 800, 850, 900],
        "DOE_F_HZ": [0.5, 2, 10, 20, 50],
        "notes": "main KMCS run",
    }


def frequency_label(frequency_hz):
    """Return a filesystem-friendly frequency label."""
    return f"{float(frequency_hz):g}".replace(".", "p")


def run_kmcs_case(config, DEP_run, run_label=None, run_note=None, case_metadata=None):
    """Run one KMCS case using the supplied DEP matrix."""
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz = int(config["Lx"]), int(config["Ly"]), int(config["Lz"])
    W = int(config["W"])
    plot_every = int(config["plot_every"])
    max_events_per_pulse = int(config["max_events_per_pulse"])
    debug_fps = int(config["debug_fps"])
    enable_profiling = bool(config["enable_profiling"])
    profile_pulses = int(config["profile_pulses"])
    save_snapshots = bool(config["save_snapshots"])
    save_final_state = bool(config["save_final_state"])
    save_debug_frames = bool(config["save_debug_frames"])
    enable_plotting = bool(config["enable_plotting"])
    roughness_save_every_percent = float(config["roughness_save_every_percent"])

    DEP_run = np.array(DEP_run, dtype=float, copy=True)
    T_kelvin = float(DEP_run[0, 0])
    T_celsius = T_kelvin - 273.15
    frequency_hz = float(DEP_run[0, 1])

    snapshot_fractions = (0.25, 0.50, 0.75) if save_snapshots else ()
    if save_final_state and 1.0 not in snapshot_fractions:
        snapshot_fractions = snapshot_fractions + (1.0,)

    label = run_label or f"T{int(round(T_celsius))}C_f{frequency_label(frequency_hz)}Hz"
    run_label_parts = ["kmcs", label]
    if enable_profiling:
        run_label_parts.append("profile")
    if save_debug_frames:
        run_label_parts.append("debug")
    full_run_label = "_".join(run_label_parts)
    run_dir, frames_dir = make_run_dir(label=full_run_label, base="runs")
    run_timestamp = run_dir.name.replace(f"_{full_run_label}", "")
    print(f"Saving run output in {run_dir}")

    Occ, H = init_state_with_steps(Lx, Ly, Lz, W=W, substrate_id=1)
    validation_counters = init_validation_counters()
    debug = save_debug_frames
    notes = run_note or f"{config['notes']} | single run from DEP"
    case_metadata = case_metadata or {}

    run_params = {
        "timestamp": run_timestamp if run_timestamp is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "grid": {"Lx": Lx, "Ly": Ly, "Lz": Lz},
        "DEP": DEP_run,
        "E_vals": config["E_vals"],
        "constants": {
            "Estatic": Estatic,
            "E_ES": E_ES,
            "TOPVIEW_STYLE": TOPVIEW_STYLE,
            "kB": kB,
        },
        "debug_params": {
            "relaxation_mode": "time-based KMC, t_end = 1/f",
            "plot_every": plot_every,
            "max_events_per_pulse": max_events_per_pulse,
            "debug_fps": debug_fps,
            "step_count_W": W,
            "debug_frames_enabled": save_debug_frames,
            "snapshots_enabled": save_snapshots,
            "snapshot_fractions": list(snapshot_fractions),
            "profile_enabled": enable_profiling,
            "profile_pulses": profile_pulses if enable_profiling else None,
            "plotting_enabled": enable_plotting,
            "save_final_state": save_final_state,
            "roughness_save_every_percent": roughness_save_every_percent,
            **case_metadata,
        },
        "seed": seed,
        "notes": notes,
    }
    save_run_info(run_dir, run_params)

    if enable_profiling:
        import cProfile
        import io
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        gif_path, validation_counters = run_dep_plus_relax(
            Occ, H, DEP_run, config["E_vals"], rng,
            plot_every=plot_every,
            max_events_per_pulse=max_events_per_pulse,
            debug=debug,
            run_dir=run_dir,
            frames_dir=frames_dir,
            debug_fps=debug_fps,
            validation_counters=validation_counters,
            snapshot_fractions=snapshot_fractions,
            roughness_save_every_percent=roughness_save_every_percent,
        )
        profiler.disable()
        stats_path = Path(run_dir) / "profile.pstats"
        profiler.dump_stats(str(stats_path))
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(30)
        text_path = Path(run_dir) / "profile.txt"
        text_path.write_text(s.getvalue(), encoding="utf-8")
        print(f"Saved profile stats: {stats_path}")
        print(f"Saved profile report: {text_path}")
    else:
        gif_path, validation_counters = run_dep_plus_relax(
            Occ, H, DEP_run, config["E_vals"], rng,
            plot_every=plot_every,
            max_events_per_pulse=max_events_per_pulse,
            debug=debug,
            run_dir=run_dir,
            frames_dir=frames_dir,
            debug_fps=debug_fps,
            validation_counters=validation_counters,
            snapshot_fractions=snapshot_fractions,
            roughness_save_every_percent=roughness_save_every_percent,
        )

    run_params["validation_counters"] = validation_counters
    final_roughness, final_mean_height = compute_rms_roughness(H)
    run_params["summary"] = {
        "average_hops_per_pulse": validation_counters.get("average_hops_per_pulse", 0.0),
        "final_mean_height": float(final_mean_height),
        "final_roughness": float(final_roughness),
    }
    save_run_info(run_dir, run_params)
    if gif_path is not None:
        print(f"Saved debug GIF to {gif_path}")

    return {
        "T_C": float(T_celsius),
        "T_K": float(T_kelvin),
        "f_Hz": float(frequency_hz),
        "run_dir": str(run_dir),
        "validation_counters": validation_counters,
    }


def run_single_kmcs(config, run_label=None):
    """Run one KMCS case using DEP exactly as defined in the config."""
    DEP_run = np.array(config["DEP"], dtype=float, copy=True)
    T_kelvin = float(DEP_run[0, 0])
    frequency_hz = float(DEP_run[0, 1])
    T_celsius = T_kelvin - 273.15
    label = run_label or f"single_T{int(round(T_celsius))}C_f{frequency_label(frequency_hz)}Hz"
    note = (
        f"{config['notes']} | single run from DEP | "
        f"T={T_celsius:.0f}C ({T_kelvin:.2f} K), f={frequency_hz:g} Hz"
    )
    return run_kmcs_case(config, DEP_run, run_label=label, run_note=note)


def write_doe_summary(summary_rows, out_path):
    """Write one DOE summary row per completed run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["T_C", "T_K", "f_Hz", "run_folder"])
        for row in summary_rows:
            writer.writerow([row["T_C"], row["T_K"], row["f_Hz"], row["run_dir"]])
    print(f"Saved DOE summary: {out_path}")
    return out_path


def run_doe(config, temperatures_celsius, frequencies_hz, summary_path=None):
    """Run a simple nested DOE over temperature and frequency."""
    summary_rows = []
    for T_celsius in temperatures_celsius:
        for frequency_hz in frequencies_hz:
            DEP_run = np.array(config["DEP"], dtype=float, copy=True)
            T_kelvin = float(T_celsius) + 273.15
            DEP_run[0, 0] = T_kelvin
            DEP_run[0, 1] = float(frequency_hz)
            label = f"doe_T{int(round(T_celsius))}C_f{frequency_label(frequency_hz)}Hz"
            note = (
                f"{config['notes']} | DOE run | "
                f"T={T_celsius:.0f}C ({T_kelvin:.2f} K), f={frequency_hz:g} Hz"
            )
            result = run_kmcs_case(
                config,
                DEP_run,
                run_label=label,
                run_note=note,
                case_metadata={
                    "doe_temperature_celsius": float(T_celsius),
                    "doe_temperature_kelvin": T_kelvin,
                    "doe_frequency_hz": float(frequency_hz),
                },
            )
            summary_rows.append(result)

    if summary_path is None:
        summary_path = PROJECT_ROOT / "runs" / "doe_summary.csv"
    write_doe_summary(summary_rows, summary_path)
    return summary_rows


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    config = default_run_config()
    run_type = str(config["RUN_TYPE"]).strip().lower()

    if run_type == "single":
        run_single_kmcs(config)
    elif run_type == "doe":
        run_doe(config, config["DOE_T_C"], config["DOE_F_HZ"])
    else:
        raise ValueError('RUN_TYPE must be "single" or "doe"')
