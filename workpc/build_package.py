"""Build the work-PC archive from tracked files plus a frozen public baseline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COMMIT = "c1b05906d06d762b451ee8349b1f2ef377a46b7e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Archive already exists; choose a new filename")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    contents = {name: (ROOT / name).read_bytes() for name in names if name}
    contents["published_reference/KMCS.py"] = subprocess.check_output(
        ["git", "show", f"{PUBLIC_COMMIT}:KMCS.py"], cwd=ROOT)
    contents["START_HERE.md"] = contents["workpc/README.md"]
    contents["published_reference/README.md"] = (
        "# Published Python baseline\n\nFrozen KMCS.py from Dbuns/KMCS commit " + PUBLIC_COMMIT
        + ".\nIncluded only for optional matched comparisons. See the root LICENSE and CONTRIBUTORS.md.\n"
    ).encode()
    for name in contents:
        parts = Path(name).parts
        if any(part in (".git", ".venv", "__pycache__", "runs", "workpc_results", "workpc_smoke_results") for part in parts):
            raise ValueError(f"Refusing unintended archive entry: {name}")
        if name.endswith(".bat"):
            contents[name] = contents[name].replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        candidate_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        public_baseline_commit=PUBLIC_COMMIT,
        sha256={name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())})
    contents["PACKAGE_MANIFEST.json"] = json.dumps(metadata, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr("KMCS_full_DOE/" + name, data)
    print(f"Created {args.output.resolve()} ({args.output.stat().st_size:,} bytes, {len(contents)} files)")
    print("SHA256 " + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
