"""Paired old/new Python trend comparison; leaves both engines unmodified."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analyze_pillar_density import analyze_loaded_snapshot


def run_case(job):
    model = types.ModuleType("comparison_engine")
    model.__file__ = str(REPO / "KMCS.py")
    exec(compile(job["source"], model.__file__, "exec"), model.__dict__)
    root = Path(job["output"]) / f"{job['version']}_T{job['T_C']:g}_f{job['f_Hz']:g}_seed{job['seed']}"
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["source_sha256"] != job["source_sha256"]:
            raise ValueError(f"Source changed for existing result: {root}")
        return saved
    model.PROJECT_ROOT = root
    config = model.default_run_config()
    config.update(Lx=job["grid"], Ly=job["grid"], Lz=64, W=1,
                  seed=job["seed"], save_snapshots=False, save_final_state=True,
                  save_debug_frames=False, enable_profiling=False, plot_every=0,
                  max_events_per_pulse=100_000_000)
    config["DEP"] = np.array([[job["T_C"]+273.15, job["f_Hz"],
                               job["pulses"], 1/25, 2, 1/3, 3, 2/3]], dtype=float)
    start = time.monotonic()
    with (root / "run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        result = model.run_single_kmcs(config)
    elapsed = time.monotonic()-start
    snapshot = Path(result["run_dir"]) / "snapshot_final.npz"
    with np.load(snapshot, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    _, _, metrics = analyze_loaded_snapshot(arrays, species=2, min_area=3, connectivity=8)
    _, _, raw = analyze_loaded_snapshot(arrays, species=2, min_area=1, connectivity=8)
    counters = result["validation_counters"]
    valid = counters["pulses_ended_by_max_events"] == 0 and counters["pulses_completed"] == job["pulses"]
    row = {key: job[key] for key in ["version", "T_C", "f_Hz", "seed", "grid", "pulses", "source_sha256"]}
    row.update(complete=valid, elapsed_s=elapsed,
               capped_pulses=counters["pulses_ended_by_max_events"],
               accepted_hops=counters["total_accepted_hops"],
               density_per_nm2=metrics["density_per_nm2"],
               raw_density_per_nm2=raw["density_per_nm2"],
               pillar_count=metrics["pillar_count_filtered"],
               mean_area_pixels=metrics["mean_pillar_area_pixels"],
               area_fraction=metrics["area_fraction_selected_species"],
               snapshot=str(snapshot))
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def summarize(rows, args):
    out = args.output
    rows.sort(key=lambda r: (r["version"], r["T_C"], r["f_Hz"], r["seed"]))
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["complete"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    trends = []
    for ax, variable, fixed_key, fixed, expected in [
        (axes[0], "T_C", "f_Hz", 20., "decreasing"),
        (axes[1], "f_Hz", "T_C", 800., "increasing"),
    ]:
        for version, color in [("previous", "#0072B2"), ("corrected", "#D55E00")]:
            subset = [r for r in valid if r["version"] == version and r[fixed_key] == fixed]
            xs = sorted({r[variable] for r in subset})
            means, spreads = [], []
            for x in xs:
                values = [r["density_per_nm2"] for r in subset if r[variable] == x]
                means.append(float(np.mean(values)))
                spreads.append(float(np.std(values, ddof=1)) if len(values)>1 else 0.)
            ax.errorbar(xs, means, yerr=spreads, marker="o", capsize=3, label=version, color=color)
            if len(xs)>1:
                slope = float(np.polyfit(xs, means, 1)[0])
                seed_slopes=[]
                for seed in args.seeds:
                    seeded=sorted([r for r in subset if r['seed']==seed], key=lambda r:r[variable])
                    if len(seeded)==len(xs):
                        seed_slopes.append(float(np.polyfit([r[variable] for r in seeded],
                                                          [r['density_per_nm2'] for r in seeded],1)[0]))
                trends.append(dict(version=version, variable=variable, expected=expected,
                    mean_slope=slope, per_seed_slopes=seed_slopes,
                    expected_sign=slope<0 if expected=="decreasing" else slope>0,
                    monotonic_means=bool(np.all(np.diff(means)<=0)) if expected=="decreasing"
                                    else bool(np.all(np.diff(means)>=0))))
        ax.set_xlabel("Temperature (°C)" if variable=="T_C" else "Frequency (Hz)")
        ax.set_ylabel("Species-2 feature density (nm⁻²)")
        ax.set_title("Temperature sweep at 20 Hz" if variable=="T_C" else "Frequency sweep at 800 °C")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f"Paired Python comparison: {args.grid}×{args.grid}, {args.pulses} pulses, {len(args.seeds)} seeds\n"
                 "Error bars: sample SD; incomplete event-capped runs excluded", fontsize=10)
    fig.savefig(out/"trends.png", dpi=180)
    plt.close(fig)
    conditions=sorted({(r['T_C'],r['f_Hz']) for r in valid})
    if conditions:
        fig, axes=plt.subplots(2,len(conditions),figsize=(max(4,2.4*len(conditions)),5),squeeze=False)
        for i,version in enumerate(['previous','corrected']):
            for j,(temp,freq) in enumerate(conditions):
                ax=axes[i,j]
                candidates=[r for r in valid if r['version']==version and r['seed']==args.seeds[0]
                            and r['T_C']==temp and r['f_Hz']==freq]
                if candidates:
                    with np.load(candidates[0]['snapshot']) as data:
                        top=data['Occ_top']
                    rgb=np.zeros((*top.shape,3))
                    for species,color in {1:(.6,.6,.6),2:(0,.45,.70),3:(.9,.6,0)}.items():
                        rgb[top==species]=color
                    ax.imshow(rgb,origin='lower',interpolation='nearest')
                ax.set_title(f'{version}\n{temp:g} °C, {freq:g} Hz',fontsize=9)
                ax.axis('off')
        fig.suptitle(f'Example top views: seed {args.seeds[0]} (blue = LMO, orange = LLTO, gray = substrate)')
        fig.tight_layout()
        fig.savefig(out/'morphologies.png',dpi=160)
        plt.close(fig)
    summary=dict(grid=args.grid,pulses=args.pulses,seeds=args.seeds,
                 completed_cases=len(valid),total_cases=len(rows),trends=trends,
                 analysis=dict(species=2,min_area=3,connectivity=8,pixel_size_nm=.39,boundaries='finite image'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='c1b0590')
    parser.add_argument('--grid',type=int,default=16)
    parser.add_argument('--pulses',type=int,default=25)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,42,159])
    parser.add_argument('--temperatures',type=float,nargs='+',default=[700,800,900])
    parser.add_argument('--frequencies',type=float,nargs='+',default=[10,20,50])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    sources={
        'previous':subprocess.check_output(['git','show',args.baseline_ref+':KMCS.py'],cwd=REPO).decode('utf-8'),
        'corrected':(REPO/'KMCS.py').read_text(encoding='utf-8')}
    conditions=sorted({(t,20.) for t in args.temperatures}|{(800.,f) for f in args.frequencies})
    jobs=[]
    for version,source in sources.items():
        for temp,freq in conditions:
            for seed in args.seeds:
                jobs.append(dict(version=version,source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    T_C=temp,f_Hz=freq,seed=seed,grid=args.grid,pulses=args.pulses,output=str(args.output)))
    manifest={key:value for key,value in vars(args).items() if key!='output'}
    manifest['sources']={key:hashlib.sha256(value.encode()).hexdigest() for key,value in sources.items()}
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Output directory already contains a different comparison; use another directory')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Starting {len(jobs)} paired jobs with {args.workers} workers',flush=True)
    rows=[]
    # Interleave versions so early progress contains paired results.
    jobs.sort(key=lambda j:(j['T_C'],j['f_Hz'],j['seed'],j['version']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,job) for job in jobs]
        for future in as_completed(futures):
            row=future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['version']} T={row['T_C']:g} f={row['f_Hz']:g} seed={row['seed']} "
                  f"density={row['density_per_nm2']:.6g}, {row['elapsed_s']:.1f}s, complete={row['complete']}",flush=True)
    summarize(rows,args)


if __name__=='__main__':
    main()
