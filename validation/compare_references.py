"""Compare completed paired runs with user-supplied density trend references."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def plot_topview(ax, snapshot, title):
    """Plot numeric saved species, using the simulator's standard palette."""
    with np.load(snapshot, allow_pickle=False) as data:
        top = data["Occ_top"]
    rgb = np.zeros((*top.shape, 3))
    for species, color in {1: (.6, .6, .6), 2: (0, .45, .70), 3: (.9, .6, 0)}.items():
        rgb[top == species] = color
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def make_topview_comparisons(rows, manifest, out):
    seed = manifest["seeds"][0]
    conditions = sorted({(float(r["T_C"]), 20.) for r in rows if float(r["f_Hz"]) == 20.})
    conditions += sorted({(800., float(r["f_Hz"])) for r in rows
                          if float(r["T_C"]) == 800. and float(r["f_Hz"]) != 20.})
    fig, axes = plt.subplots(len(conditions), 2, figsize=(7.6, 3.5 * len(conditions)), squeeze=False)
    for i, (temp, freq) in enumerate(conditions):
        for j, version in enumerate(["previous", "corrected"]):
            row = next(r for r in rows if r["version"] == version and int(r["seed"]) == seed
                       and float(r["T_C"]) == temp and float(r["f_Hz"]) == freq)
            label = "Published Python" if version == "previous" else "Corrected Python"
            plot_topview(axes[i, j], row["snapshot"], f"{label} — {temp:g} °C, {freq:g} Hz")
    fig.suptitle(f"Matched top views — seed {seed}, {manifest['grid']}×{manifest['grid']}, {manifest['pulses']} pulses\n"
                 "Blue: LMO | Orange: LLTO | Gray: substrate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / "topviews_side_by_side.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # The bundled example has different growth conditions. Keep the distinction
    # explicit rather than implying that the small screen reproduces this image.
    example = REPO / "examples/snapshot_final.npz"
    pair = [next(r for r in rows if r["version"] == version and int(r["seed"]) == 42
                 and float(r["T_C"]) == 900. and float(r["f_Hz"]) == 20.)
            for version in ["previous", "corrected"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    plot_topview(axes[0], example, "Archived repository example\n900 °C, 2 Hz | 128×128 | 300 pulses")
    for ax, row, label in zip(axes[1:], pair, ["Published Python screen", "Corrected Python screen"]):
        plot_topview(ax, row["snapshot"],
                     f"{label}\n900 °C, 20 Hz | {row['grid']}×{row['grid']} | {row['pulses']} pulses")
    fig.suptitle("Repository reference and fresh top views — seed 42 throughout", fontsize=12)
    fig.text(.5, .025, "Left has different frequency, film thickness and field of view. "
             "Only the middle/right pair is a matched comparison.\n"
             "Blue: LMO | Orange: LLTO | Gray: substrate. Panels are fitted to equal display sizes, not equal physical areas.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(out / "repository_example_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Output directory containing results.csv and manifest.json")
    args = parser.parse_args()
    out = args.results.resolve()
    rows = list(csv.DictReader((out / "results.csv").open(encoding="utf-8")))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    refs = json.loads((REPO / "validation/reference_data.json").read_text(encoding="utf-8"))
    # Extract the fixed data table without executing its plotting script.
    tree = ast.parse((REPO / "DOE_plot.py").read_text(encoding="utf-8"))
    table = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "data" for t in node.targets))
    valid = [r for r in rows if r["complete"] == "True"]
    make_topview_comparisons(valid, manifest, out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    checks = []
    screen_stats = []
    for ax, key, fixed_key, fixed, anchor, ref_key, expected in [
        (axes[0], "T_C", "f_Hz", 20., 800., "temperature_sweep", -1),
        (axes[1], "f_Hz", "T_C", 800., 20., "frequency_sweep", 1),
    ]:
        reference = refs[ref_key]
        xs = np.array(reference["temperature_C" if key == "T_C" else "frequency_Hz"])
        datasets = [
            ("Experiment", xs, reference["experiment"], "#222222", "o:", "reference"),
            ("Original paper KMC", xs, reference["original_paper_kmcs"], "#8B5AA6", "D-.", "reference"),
        ]
        cut = sorted((t if key == "T_C" else f, d) for f, t, d in table
                     if (f if key == "T_C" else t) == fixed
                     and (key != "T_C" or 700 <= t <= 900))
        datasets.append(("Verified Python DOE table", np.array([x for x, _ in cut]),
                         [d for _, d in cut], "#009E73", "s--", "reference"))
        for version, color, style in [("previous", "#0072B2", "^-"),
                                       ("corrected", "#D55E00", "v-")]:
            subset = [r for r in valid if r["version"] == version and float(r[fixed_key]) == fixed]
            x = sorted({float(r[key]) for r in subset})
            y = []
            for value in x:
                points = [float(r["density_per_nm2"]) for r in subset if float(r[key]) == value]
                y.append(float(np.mean(points)))
                screen_stats.append(dict(version=version, sweep=key, x=value,
                                         mean=y[-1], sd=float(np.std(points, ddof=1)), n=len(points)))
            datasets.append(("Published Python (screen)" if version == "previous" else "Corrected Python (screen)",
                             np.array(x), y, color, style, "screen"))
        for label, x, y, color, style, kind in datasets:
            y = np.asarray(y, dtype=float)
            matches = np.flatnonzero(x == anchor)
            if len(matches) != 1 or y[matches[0]] <= 0:
                raise ValueError(f"Missing or nonpositive normalization anchor for {label}")
            relative = y / y[matches[0]]
            slope = float(np.polyfit(x, relative, 1)[0])
            check = dict(sweep=key, dataset=label, kind=kind, expected_sign=bool(expected * slope > 0),
                         monotonic=bool(np.all(expected * np.diff(y) >= 0)), slope=slope,
                         x=x.tolist(), density=y.tolist(), normalized=relative.tolist())
            checks.append(check)
            ax.plot(x, relative, style, color=color, label=label, markersize=5,
                    linewidth=1.8 if kind == "screen" else 1.3)
        ax.set_title("Temperature dependence at 20 Hz" if key == "T_C" else "Frequency dependence at 800 °C")
        ax.set_xlabel("Temperature (°C)" if key == "T_C" else "Frequency (Hz)")
        ax.set_ylabel("Relative number density (800 °C = 1)" if key == "T_C" else "Relative number density (20 Hz = 1)")
        if key == "f_Hz":
            ax.set_xscale("log")
            ax.set_xticks([.5, 2, 8, 10, 20, 50], labels=["0.5", "2", "8", "10", "20", "50"])
            ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7.8, frameon=False, loc="upper right" if key == "T_C" else "upper left")
    fig.suptitle("Number-density trends: experiment, original model, verified DOE, and paired Python runs", fontsize=12)
    fig.text(.5, .015, f"Screen: {manifest['grid']}×{manifest['grid']}, {manifest['pulses']} pulses, {len(manifest['seeds'])} seeds. "
             "Each dataset normalized independently; screen seed spread is shown in trends.png.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out / "reference_trends.png", dpi=200)
    plt.close(fig)
    integrity = []
    for row in rows:
        with np.load(row["snapshot"], allow_pickle=False) as data:
            occ, height = data["Occ"], data["H"]
            expected_film = int(row["pulses"]) * round(int(row["grid"])**2 / 25)
            retained = int(np.count_nonzero((occ == 2) | (occ == 3)))
            expected_height = np.max(np.where(occ != 0, np.arange(occ.shape[2]), 0), axis=2)
            integrity.append(dict(version=row["version"], T_C=float(row["T_C"]), f_Hz=float(row["f_Hz"]),
                                  seed=int(row["seed"]), mass_ok=retained == expected_film,
                                  heights_ok=bool(np.array_equal(height, expected_height))))
    paired_summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    raw_checks = []
    for key, fixed_key, fixed, sign in [("T_C", "f_Hz", 20., -1), ("f_Hz", "T_C", 800., 1)]:
        for version in ["previous", "corrected"]:
            subset = [r for r in valid if r["version"] == version and float(r[fixed_key]) == fixed]
            xs = sorted({float(r[key]) for r in subset})
            means = [np.mean([float(r["raw_density_per_nm2"]) for r in subset if float(r[key]) == x]) for x in xs]
            slope = float(np.polyfit(xs, means, 1)[0])
            raw_checks.append(dict(version=version, sweep=key, expected_sign=bool(sign*slope > 0), slope=slope))
    all_pass = (len(valid) == len(rows) and all(c["expected_sign"] for c in checks)
                and all(r["mass_ok"] and r["heights_ok"] for r in integrity))
    result = dict(screen_direction_check_passed=all_pass, completed_cases=len(valid), total_cases=len(rows),
                  reference_checks=checks, raw_component_checks=raw_checks, integrity=integrity)
    (out / "reference_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# KMCS paired trend comparison", "",
             "**Reduced-size mean-trend screening: " + ("PASS" if all_pass else "NOT ESTABLISHED") + ".**", "",
             "The previous engine is KMCS.py from public commit c1b0590. The corrected engine is the "
             "local 1.1.0.dev0 candidate. Neither engine was tuned to match these references.", "",
             f"Completed cases: {len(valid)}/{len(rows)}. All input and source identities are in manifest.json.", "",
             f"Setup: {manifest['grid']}×{manifest['grid']}×64, W=1, {manifest['pulses']} pulses, "
             f"seeds {manifest['seeds']}, default energies, 0.04 ML/pulse requested, LMO:LLTO=1:2. "
             "At this small grid, rounding deposits 10 blocks/pulse: 250 blocks total, or 0.9765625 ML.", "",
             "These are early-growth surface features, not a test of mature vertical pillar connectivity. "
             "Both engines use the same species-2, 8-connected finite-image analysis with minimum area 3 pixels "
             "and 0.39 nm/pixel. Absolute densities are not required to match across datasets.", "",
             "## Measured screen densities (nm^-2; mean ± sample SD)", "",
             "| Sweep | Value | Published Python | Corrected Python |", "|---|---:|---:|---:|"]
    for key in ["T_C", "f_Hz"]:
        values = sorted({r['x'] for r in screen_stats if r['sweep'] == key})
        for value in values:
            cells = []
            for version in ["previous", "corrected"]:
                stat = next(r for r in screen_stats if r['sweep'] == key and r['x'] == value and r['version'] == version)
                cells.append(f"{stat['mean']:.5f} ± {stat['sd']:.5f}")
            lines.append(f"| {'Temperature (°C), 20 Hz' if key == 'T_C' else 'Frequency (Hz), 800 °C'} | {value:g} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Comparison with supplied references", "",
              "The two original-paper tables were transcribed from Daniel's screenshots into "
              "validation/reference_data.json. The verified Python DOE table was read at full precision "
              "from DOE_plot.py; it agrees with the supplied rounded Excel screenshot. It remains historical "
              "reference data, not fresh output from either engine in this screen.", "",
              "Each curve in reference_trends.png is divided by its own central point: 800 °C for "
              "temperature or 20 Hz for frequency. This compares shape and direction without fitting "
              "densities. The two experimental central values (4.948e-5 and 4.828e-5) are preserved separately.", ""]
    for c in checks:
        lines.append(f"- {c['dataset']}, {c['sweep']}: expected mean-trend direction = {c['expected_sign']}; monotonic points/means = {c['monotonic']}.")
    lines += ["", "## Seed variability and size-filter sensitivity", ""]
    for trend in paired_summary['trends']:
        expected = -1 if trend['expected'] == 'decreasing' else 1
        slopes = trend['per_seed_slopes']
        flat = sum(abs(s) <= 1e-12 for s in slopes)
        hits = sum(expected*s > 1e-12 for s in slopes)
        opposite = len(slopes) - flat - hits
        lines.append(f"- {trend['version']}, {trend['variable']}: {hits}/{len(slopes)} seed slopes follow the expected direction; {flat} flat; {opposite} opposite.")
    for raw in raw_checks:
        lines.append(f"- Counting all components (minimum area 1), {raw['version']}, {raw['sweep']}: expected mean-trend direction = {raw['expected_sign']}.")
    lines += ["", "Seed SD is not a confidence interval. Three seeds and a 16×16 grid give limited precision; "
              "a single feature changes density by about 0.02568 nm^-2. Disagreement in individual seeds "
              "must not be hidden by the mean curves.", "", "## Integrity and scope", "",
              f"- Event-capped pulses: {sum(int(r['capped_pulses']) for r in rows)}.",
              f"- Every saved state retains its expected film-block count: {all(r['mass_ok'] for r in integrity)}.",
              f"- Every height map agrees with occupied sites: {all(r['heights_ok'] for r in integrity)}.",
              "- No energy or analysis-parameter tuning was performed.",
              "- The screen covers temperature 700–900 °C at 20 Hz and frequency 10–50 Hz at 800 °C. "
              "It does not test the whole DOE surface or fresh low-frequency simulations below 10 Hz.",
              "- The original 32×32, 50-pulse pilot remains in ../paired_trends_32_50. Completed cases are "
              "preserved separately; the incomplete pilot is not combined with these mean curves.",
              "- This is a directional early-growth screen, not a 128×128, 300-pulse VAN validation, "
              "quantitative experimental fit, or verification of every possible input condition.", "",
              "## Top-view comparisons", "",
              "### Matched published-code and corrected-code runs", "",
              "![Matched top views, published Python on the left and corrected Python on the right](topviews_side_by_side.png)", "",
              f"All pairs above use seed {manifest['seeds'][0]}, the same {manifest['grid']}×{manifest['grid']} "
              f"lattice and {manifest['pulses']} pulses. The temperatures and frequencies are labelled in each row. "
              "The first seed was selected in advance, not chosen for the closest visual match. "
              "Identical starting seeds do not imply identical later random paths after hopping logic changes.", "",
              "### Existing repository example alongside the new runs", "",
              "![Archived repository top view beside the published and corrected screening runs](repository_example_comparison.png)", "",
              "The left panel is re-rendered from the repository's examples/snapshot_final.npz, corresponding "
              "to the existing two-material top-view figure. Its recorded settings are 900 °C, 2 Hz, "
              "128×128 cells, 300 pulses, seed 42. The middle/right pair uses 900 °C, 20 Hz, 16×16 cells, "
              "25 pulses, seed 42. This provides visual context; comparing the left panel directly with "
              "either screen would confound frequency, thickness and field of view. The archived image is "
              "historical example data, not a fresh rerun of the current public engine.", "",
              "## Density comparison figures", "",
              "![Normalized number-density curves against the supplied references](reference_trends.png)", "",
              "![Raw mean number-density curves with seed standard deviation](trends.png)", "",
              "## Outputs", "", "- topviews_side_by_side.png: matched published/corrected top views.",
              "- repository_example_comparison.png: archived repository example and the matched screen pair, with conditions labelled.",
              "- reference_trends.png: normalized curves against all three reference datasets.",
              "- trends.png: raw paired-screen means and sample SD.",
              "- morphologies.png: predefined seed 7, old/new top views at matched conditions.",
              "- results.csv: all individual measured results.", "- reference_comparison.json: machine-readable checks.", ""]
    (out / "reference_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ['screen_direction_check_passed', 'completed_cases', 'total_cases']}, indent=2))


if __name__ == "__main__":
    main()
