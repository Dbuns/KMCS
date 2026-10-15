"""Summarize saved full DOE conditions without rerunning simulations."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def analyse(output):
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        print(f"No recorded runs yet in {output}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = manifest["settings"]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output.glob("*/case.json"))]
    valid = [r for r in rows if r["status"] == "complete"]
    temperatures = sorted(settings["temperatures_C"])
    frequencies = sorted(settings["frequencies_Hz"])
    expected = len(temperatures) * len(frequencies) * len(settings["seeds"])
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    table_tree = ast.parse((ROOT / "DOE_plot.py").read_text(encoding="utf-8"))
    historical = next(ast.literal_eval(node.value) for node in table_tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "data" for t in node.targets))
    lookup = {(float(t), float(f)): float(d) for f, t, d in historical}
    refs = json.loads((ROOT / "validation/reference_data.json").read_text(encoding="utf-8"))
    summary = []
    for t in temperatures:
        for f in frequencies:
            values = [r["density_per_nm2"] for r in valid if r["T_C"] == t and r["f_Hz"] == f]
            summary.append(dict(T_C=t, f_Hz=f, completed_seeds=len(values), expected_seeds=len(settings["seeds"]),
                mean_density_per_nm2=float(np.mean(values)) if values else "",
                sd_density_per_nm2=float(np.std(values, ddof=1)) if len(values)>1 else (0. if values else ""),
                historical_verified_python_density=lookup.get((float(t), float(f)), "")))
    with (analysis / "density_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    status = dict(engine=manifest["engine"], complete=len(valid), expected=expected,
                  failed_or_invalid=sum(r["status"] in ("failed", "invalid") for r in rows),
                  all_conditions_complete=len(valid)==expected,
                  smoke_test=manifest["smoke_test"], trend_checks={})
    (analysis / "summary.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not valid:
        print(f"No valid completed conditions; {len(rows)} saved records. See case.json/error.txt files.")
        return
    grid = np.full((len(temperatures), len(frequencies)), np.nan)
    for row in summary:
        if row["mean_density_per_nm2"] != "":
            grid[temperatures.index(row["T_C"]), frequencies.index(row["f_Hz"])] = row["mean_density_per_nm2"]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(frequencies)), labels=[f"{v:g}" for v in frequencies])
    ax.set_yticks(range(len(temperatures)), labels=[f"{v:g}" for v in temperatures])
    for i, j in np.ndindex(grid.shape):
        ax.text(j, i, f"{grid[i,j]:.3g}" if np.isfinite(grid[i,j]) else "pending", ha="center", va="center",
                color="white" if np.isfinite(grid[i,j]) and grid[i,j] < np.nanmean(grid) else "black", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"{manifest['engine']} DOE: {len(valid)}/{expected} valid completed runs")
    fig.colorbar(im, ax=ax, label="Species-2 feature density (nm⁻²)")
    fig.savefig(analysis / "density_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, key, fixed_key, fixed, anchor, refkey, expected_sign in [
        (axes[0], "T_C", "f_Hz", 20., 800., "temperature_sweep", -1),
        (axes[1], "f_Hz", "T_C", 800., 20., "frequency_sweep", 1)]:
        data = [r for r in summary if r[fixed_key] == fixed and r["mean_density_per_nm2"] != ""]
        data.sort(key=lambda r: r[key])
        xs = np.array([r[key] for r in data])
        ys = np.array([r["mean_density_per_nm2"] for r in data])
        anchors = np.flatnonzero(xs == anchor)
        if len(anchors) == 1 and ys[anchors[0]] > 0:
            ax.plot(xs, ys / ys[anchors[0]], "o-", color="#D55E00", label=f"Fresh {manifest['engine']}")
        else:
            ax.text(.03, .03, "Fresh sweep anchor not complete yet", transform=ax.transAxes,
                    va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        ref = refs[refkey]
        refx = np.array(ref["temperature_C" if key == "T_C" else "frequency_Hz"])
        anchor_index = int(np.flatnonzero(refx == anchor)[0])
        for name, field, color in [("Experiment", "experiment", "#222222"),
                                   ("Original paper KMC", "original_paper_kmcs", "#8B5AA6")]:
            values = np.array(ref[field])
            ax.plot(refx, values / values[anchor_index], "o:", label=name, color=color, markersize=4)
        historical_cut = sorted((t if key == "T_C" else f, d) for f,t,d in historical
            if (f if key == "T_C" else t) == fixed and (key != "T_C" or 700<=t<=900))
        hx, hy = np.array(historical_cut).T
        ax.plot(hx, hy / hy[int(np.flatnonzero(hx == anchor)[0])], "s--", color="#009E73", label="Verified Python DOE")
        if len(xs) >= 2:
            slope = float(np.polyfit(xs, ys, 1)[0])
            status["trend_checks"][key] = dict(expected_direction=bool(expected_sign*slope>0),
                slope=slope, all_sweep_seeds_complete=all(r["completed_seeds"] == r["expected_seeds"] for r in data)
                and len(data)==(len(temperatures) if key=="T_C" else len(frequencies)))
        ax.set_xlabel("Temperature (°C)" if key=="T_C" else "Frequency (Hz)")
        ax.set_ylabel("Relative number density (central point = 1)")
        ax.set_title("At 20 Hz" if key=="T_C" else "At 800 °C")
        if key=="f_Hz":
            ax.set_xscale("log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Independent normalization compares trends, not absolute density")
    fig.savefig(analysis / "reference_trends.png", dpi=180)
    plt.close(fig)

    # Fix the seed in advance. Do not choose the visually closest morphology.
    seed = settings["seeds"][0]
    fig, axes = plt.subplots(len(temperatures), len(frequencies),
        figsize=(max(4, 2.5*len(frequencies)), max(4, 2.6*len(temperatures))), squeeze=False)
    for i,t in enumerate(temperatures):
        for j,f in enumerate(frequencies):
            ax = axes[i,j]
            match = next((r for r in valid if r["T_C"]==t and r["f_Hz"]==f and r["seed"]==seed), None)
            if match:
                with np.load(output / match["snapshot"], allow_pickle=False) as saved:
                    top = saved["Occ_top"]
                rgb = np.zeros((*top.shape,3))
                for species,color in {1:(.6,.6,.6),2:(0,.45,.70),3:(.9,.6,0)}.items():
                    rgb[top==species] = color
                ax.imshow(rgb, origin="lower", interpolation="nearest")
            else:
                ax.text(.5,.5,"pending / invalid",ha="center",va="center",transform=ax.transAxes)
            ax.set_title(f"{t:g} °C, {f:g} Hz",fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{manifest['engine']} top views, seed {seed} | grid {settings['grid']} | {settings['pulses']} pulses\n"
                 "Blue: LMO | Orange: LLTO | Gray: substrate",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(analysis / "topviews.png",dpi=160)
    plt.close(fig)
    (analysis / "summary.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    lines = ["# Work-PC DOE results", "",
        f"Engine: {manifest['engine']}. Valid completed runs: **{len(valid)}/{expected}**.", "",
        "**SMOKE TEST ONLY — not scientific results.**" if manifest["smoke_test"] else
        f"Grid {settings['grid']}, {settings['pulses']} pulses, seeds {settings['seeds']}. "
        "No model parameters were fitted to the reference curves.", "",
        "Failed, event-capped and integrity-failing cases are excluded from the plots. "
        "Partial plots are marked with their completion count; do not treat a partial grid as validated.", "",
        "![DOE density](density_heatmap.png)", "", "![Reference trends](reference_trends.png)", "",
        "![DOE top views](topviews.png)", "", "## Recorded checks", "",
        "```json", json.dumps(status,indent=2), "```", "",
        "## Notes", "",
        "- The built-in frequency list contains 10 Hz, while the original-paper table has 8 Hz. "
        "These points are kept distinct; no value is substituted or relabelled.",
        "- The verified historical DOE table has no 900 °C / 0.5 Hz value; its CSV reference cell is blank.",
        "- Reference normalization is independent for each dataset. Absolute densities may differ.",
        "- One default seed gives no estimate of seed variability. Trend flags are a first check, not statistical proof.",
        "- Inspect the final NPZ states and run_info.txt as well as the curves before a new scientific release.", ""]
    (analysis / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Analysis saved: {analysis}. Valid completed runs: {len(valid)}/{expected}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"workpc_results/corrected")
    analyse(parser.parse_args().results)
