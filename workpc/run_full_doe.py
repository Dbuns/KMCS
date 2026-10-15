"""Resumable full-size wrapper around KMCS.run_doe; never changes KMC physics."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
# Avoid each independent simulation also starting a BLAS thread pool.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_pillar_density import analyze_loaded_snapshot


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_engine_source(engine):
    if engine == "corrected":
        return (ROOT / "KMCS.py").read_text(encoding="utf-8")
    frozen = ROOT / "published_reference/KMCS.py"
    if frozen.exists():
        return frozen.read_text(encoding="utf-8")
    # Development checkout only. The portable archive includes this file.
    return subprocess.check_output(["git", "show", "c1b0590:KMCS.py"], cwd=ROOT).decode("utf-8")


def build_config(model, settings, temperature, frequency, seed):
    config = model.default_run_config()
    lx, ly, lz = settings["grid"]
    config.update(RUN_TYPE="doe", Lx=lx, Ly=ly, Lz=lz, W=settings["W"], seed=seed,
                  max_events_per_pulse=settings["max_events_per_pulse"], plot_every=0,
                  save_snapshots=True, save_final_state=True, save_debug_frames=False,
                  enable_profiling=False, notes=settings["notes"])
    pairs = [value for pair in zip(settings["species"], settings["ratios"]) for value in pair]
    config["DEP"] = np.array([[temperature + 273.15, frequency,
                               settings["pulses"], settings["flux_ML_per_pulse"], *pairs]], dtype=float)
    config["E_vals"] = np.array(settings["E_vals"], dtype=np.float32)
    return config


def import_engine(source):
    model = types.ModuleType("kmcs_workpc_engine")
    model.__file__ = str(ROOT / "KMCS.py")
    exec(compile(source, model.__file__, "exec"), model.__dict__)
    return model


def execute_case(job):
    case_dir = Path(job["output"]) / job["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    attempt = case_dir / ("attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    attempt.mkdir()
    record = {key: job[key] for key in ("case_id", "engine", "T_C", "f_Hz", "seed")}
    record.update(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  case_fingerprint=job["fingerprint"], attempt=attempt.relative_to(job["output"]).as_posix())
    status_path = case_dir / "case.json"
    save_json(status_path, record)
    started = time.monotonic()
    try:
        model = import_engine(job["source"])
        model.PROJECT_ROOT = attempt
        config = build_config(model, job["settings"], job["T_C"], job["f_Hz"], job["seed"])
        with (attempt / "run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            # Use the actual built-in DOE function for each checkpointed condition.
            result = model.run_doe(config, [job["T_C"]], [job["f_Hz"]],
                                   summary_path=attempt / "builtin_doe_summary.csv")[0]
        run_dir = Path(result["run_dir"])
        snapshot = run_dir / "snapshot_final.npz"
        with np.load(snapshot, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        analysis = job["settings"]["analysis"]
        _, _, metrics = analyze_loaded_snapshot(arrays, species=analysis["species"],
            min_area=analysis["min_area_pixels"], connectivity=analysis["connectivity"])
        occ, height = arrays["Occ"], arrays["H"]
        lx, ly, _ = job["settings"]["grid"]
        expected_film = job["settings"]["pulses"] * round(job["settings"]["flux_ML_per_pulse"] * lx * ly)
        film_count = int(np.count_nonzero(np.isin(occ, job["settings"]["species"])))
        expected_height = np.max(np.where(occ != 0, np.arange(occ.shape[2]), 0), axis=2)
        counters = result["validation_counters"]
        integrity_ok = film_count == expected_film and np.array_equal(height, expected_height)
        complete = (integrity_ok and counters["pulses_ended_by_max_events"] == 0
                    and counters["pulses_completed"] == job["settings"]["pulses"])
        record.update(status="complete" if complete else "invalid", integrity_ok=bool(integrity_ok),
            pulses_completed=counters["pulses_completed"], capped_pulses=counters["pulses_ended_by_max_events"],
            accepted_hops=counters["total_accepted_hops"], film_count=film_count, expected_film_count=expected_film,
            density_per_nm2=metrics["density_per_nm2"], feature_count=metrics["pillar_count_filtered"],
            mean_area_pixels=metrics["mean_pillar_area_pixels"], area_fraction=metrics["area_fraction_selected_species"],
            snapshot=snapshot.relative_to(job["output"]).as_posix(),
            run_dir=run_dir.relative_to(job["output"]).as_posix())
    except Exception as error:
        record.update(status="failed", error=str(error))
        (attempt / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    record.update(elapsed_s=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(status_path, record)
    return record


def collect_records(output):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output.glob("*/case.json"))]


def write_summary(output):
    records = collect_records(output)
    if not records:
        return
    fields = ["case_id", "engine", "T_C", "f_Hz", "seed", "status", "density_per_nm2", "feature_count",
              "mean_area_pixels", "area_fraction", "capped_pulses", "integrity_ok", "elapsed_s", "run_dir", "snapshot", "error"]
    temporary = output / "doe_results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "doe_results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "workpc/full_doe_config.json")
    parser.add_argument("--engine", choices=["corrected", "published"], default="corrected")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--check", action="store_true", help="Validate the full plan without simulating")
    parser.add_argument("--smoke-test", action="store_true", help="One tiny independent plumbing run; not scientific data")
    parser.add_argument("--analyse-only", action="store_true", help="Rebuild CSV/plots from saved case records")
    args = parser.parse_args()
    if args.analyse_only:
        output = (args.output or ROOT / "workpc_results" / args.engine).resolve()
        write_summary(output)
        from analyse_full_doe import analyse
        analyse(output)
        return
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    source = load_engine_source(args.engine)
    model = import_engine(source)
    if args.smoke_test:
        settings.update(grid=[8, 8, 12], pulses=2, temperatures_C=[800], frequencies_Hz=[1e8],
                        seeds=[42], max_events_per_pulse=10000, workers=1)
    for field in ("temperatures_C", "frequencies_Hz", "seeds"):
        if not settings[field]:
            raise ValueError(f"{field} must not be empty")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in settings["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    workers = args.workers if args.workers is not None else settings["workers"]
    if workers < 1:
        raise ValueError("workers must be at least one")
    workers = min(workers, max(1, (os.cpu_count() or 1) - 1))
    output = (args.output or ROOT / ("workpc_smoke_results" if args.smoke_test else "workpc_results") / args.engine).resolve()
    manifest = dict(engine=args.engine, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    settings=settings, smoke_test=args.smoke_test,
                    software=dict(python=platform.python_version(), numpy=np.__version__, kmcs=model.__version__))
    # Runtime versions are part of the resume identity: never mix environments silently.
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    jobs = []
    if settings["species"] != [2, 3] or len(settings["ratios"]) != 2:
        raise ValueError("This validation profile requires film species 2 and 3")
    if any(not np.isfinite(t) or t <= -273.15 for t in settings["temperatures_C"]):
        raise ValueError("Invalid DOE temperature")
    if any(not np.isfinite(f) or f <= 0 for f in settings["frequencies_Hz"]):
        raise ValueError("Invalid DOE frequency")
    for t in sorted(settings["temperatures_C"]):
        for f in sorted(settings["frequencies_Hz"], reverse=True):
            for seed in settings["seeds"]:
                config = build_config(model, settings, t, f, seed)
                if hasattr(model, "validate_run_inputs"):
                    model.validate_run_inputs(config, config["DEP"])
                else:
                    model.parse_dep_row(config["DEP"][0])
                case_id = f"T{t:g}_f{f:g}_seed{seed}".replace(".", "p")
                jobs.append(dict(case_id=case_id, engine=args.engine, T_C=t, f_Hz=f, seed=seed,
                                 settings=settings, source=source, fingerprint=fingerprint, output=str(output)))
    if len({job["case_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate temperatures, frequencies or seeds in the plan")
    print(f"{'SMOKE TEST - NOT SCIENTIFIC DATA' if args.smoke_test else 'FULL-SIZE DOE'}: {args.engine}", flush=True)
    print(f"Grid {settings['grid']}, {settings['pulses']} pulses, seeds {settings['seeds']}", flush=True)
    print(f"{len(jobs)} conditions, up to {workers} independent processes; output: {output}", flush=True)
    if args.check:
        print("Plan validated. No simulations started.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output contains a different code/config/environment. Use a new --output folder.")
    save_json(manifest_path, manifest)
    pending = []
    for job in jobs:
        path = output / job["case_id"] / "case.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if (record.get("status") == "complete" and record.get("case_fingerprint") == fingerprint
                and (output / record.get("snapshot", "MISSING")).is_file()):
            print(f"SKIP completed {job['case_id']}", flush=True)
        else:
            pending.append(job)
    print(f"{len(jobs)-len(pending)} cached; {len(pending)} to run. Completed conditions resume safely.", flush=True)
    # Keep only 'workers' cases active, so PAUSE_AFTER_CURRENT can stop scheduling.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        queue = iter(pending)
        def submit_one():
            if (output / "PAUSE_AFTER_CURRENT").exists():
                return
            job = next(queue, None)
            if job is not None:
                futures[pool.submit(execute_case, job)] = job
        for _ in range(workers):
            submit_one()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                job = futures.pop(future)
                record = future.result()
                write_summary(output)
                print(f"{record['status'].upper()} {job['case_id']} ({record['elapsed_s']/60:.1f} min)", flush=True)
                if record["status"] != "complete":
                    print(f"Inspect {output / job['case_id']}; incomplete runs are excluded from trends.", flush=True)
                submit_one()
    write_summary(output)
    from analyse_full_doe import analyse
    analyse(output)
    if (output / "PAUSE_AFTER_CURRENT").exists():
        print("Paused after active cases. Remove PAUSE_AFTER_CURRENT before resuming.", flush=True)
    records = collect_records(output)
    if any(record["status"] in ("failed", "invalid") for record in records):
        return 1
    if args.smoke_test and sum(record["status"] == "complete" for record in records) != len(jobs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
