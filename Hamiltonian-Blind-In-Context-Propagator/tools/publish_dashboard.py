"""Export the dashboard data and publish it on GitHub Pages with one command.

Run from the code root (the folder that contains `tools/`):

    python -m tools.publish_dashboard --label "n8 no S" --notes "use_S=false" \
        --runs runs --results results --data datasets/n8/test.h5 \
        --ckpt runs/local_A/last.pt --T 32 --horizon 64 --sample 0

Steps: build the export (tools.export_dashboard) -> <git_root>/docs/data/runs/<YYYYmmdd-HHMMSS>_<label>.json
-> update docs/data/index.json (new export first and `latest`, keep the last --keep) ->
git add docs/data, git commit -m "dashboard: <label>", git push.
GitHub Pages must serve main:/docs (see README).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.export_dashboard import add_export_args, build_export

WARN_BYTES = 5 * 1024 * 1024


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], check=check, capture_output=True, text=True)


def git_root(start: Path) -> Path:
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"], check=True,
                             capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as ex:
        raise SystemExit(f"not inside a git repository ({ex})")
    return Path(out)


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "export"


def read_index(path: Path) -> dict:
    if path.exists():
        try:
            idx = json.loads(path.read_text(encoding="utf-8"))
            idx.setdefault("exports", [])
            return idx
        except json.JSONDecodeError:
            print(f"warning: {path} is not valid JSON, starting a new index", file=sys.stderr)
    return {"latest": None, "exports": []}


def write_index(path: Path, idx: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_index(idx: dict, entry: dict, keep: int, data_dir: Path) -> list[str]:
    """Put entry first and make it `latest`; drop the exports beyond `keep`. Returns removed files."""
    exports = [e for e in idx["exports"] if e.get("file") != entry["file"]]
    exports.insert(0, entry)
    removed = [e["file"] for e in exports[keep:]]
    idx["exports"], idx["latest"] = exports[:keep], entry["file"]
    for f in removed:
        (data_dir / f).unlink(missing_ok=True)
    return removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_export_args(ap)
    ap.add_argument("--label", default="export", help="short name shown in the Export menu")
    ap.add_argument("--notes", default="", help="free text shown in the dashboard banner")
    ap.add_argument("--keep", type=int, default=20, help="number of exports to keep (default 20)")
    ap.add_argument("--no-push", action="store_true", help="commit but do not push")
    ap.add_argument("--dry-run", action="store_true", help="build the export and report, write/commit nothing")
    ap.add_argument("--docs-dir", default=None, help="override <git_root>/docs (testing; skips git)")
    a = ap.parse_args(argv)
    if a.keep < 1:
        raise SystemExit("--keep must be >= 1")

    root = git_root(Path.cwd())
    docs = Path(a.docs_dir) if a.docs_dir else root / "docs"
    data_dir = docs / "data"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rel = f"runs/{stamp}_{slug(a.label)}.json"
    target = data_dir / rel

    export = build_export(a, a.notes)
    payload = json.dumps(export, separators=(",", ":"), ensure_ascii=False)
    size = len(payload.encode("utf-8"))
    print(f"export: {size / 1e6:.2f} MB, {len(export['runs'])} runs, {len(export['results'])} result sets, "
          f"{len(export['datasets'])} datasets, sample={'yes' if 'sample' in export else 'no'}")
    if size > WARN_BYTES:
        print(f"warning: export is larger than 5 MB ({size / 1e6:.1f} MB): the repository and the page get "
              "heavy; lower --horizon, drop --data/--baselines or export fewer runs", file=sys.stderr)
    entry = dict(file=rel, label=a.label, generated_at=export["generated_at"], git_commit=export["git_commit"],
                 notes=a.notes)
    if a.dry_run:
        print(f"[dry-run] would write {target} and set latest={rel} (keep {a.keep}); nothing written")
        return 0

    (data_dir / "runs").mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    idx_path = data_dir / "index.json"
    idx = read_index(idx_path)
    removed = update_index(idx, entry, a.keep, data_dir)
    write_index(idx_path, idx)
    print(f"wrote {target}\nupdated {idx_path}: latest={rel}, {len(idx['exports'])} exports"
          + (f", removed {len(removed)} old" if removed else ""))
    if a.docs_dir:
        return 0

    paths = [str(data_dir.relative_to(root))]
    git(root, "add", "-A", "--", *paths)
    if git(root, "diff", "--cached", "--quiet", "--", *paths, check=False).returncode == 0:
        print("nothing to commit")
        return 0
    git(root, "commit", "-m", f"dashboard: {a.label}", "--", *paths)
    print(f"committed: dashboard: {a.label}")
    if a.no_push:
        print("--no-push: skipping git push")
        return 0
    res = git(root, "push", check=False)
    if res.returncode != 0:
        print(res.stderr.strip(), file=sys.stderr)
        raise SystemExit("git push failed: the export is committed locally, fix the remote and push manually")
    print("pushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
