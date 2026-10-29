"""Publish the finished HPC runs to the dashboard. Run on the LOGIN node (needs internet), from the code folder:
    GH_TOKEN=... python scripts/hpc/publish.py
Mirrors the Colab publish cell: n8_resid_wide and n8_resid_long (n8_ref/n8_resid come along for comparison if present)."""
import os, shutil, subprocess, sys

tok = os.environ.get("GH_TOKEN")
if not tok:
    sys.exit("set GH_TOKEN (GitHub token with push rights)")
run = lambda a, **k: subprocess.run(a, check=True, **k)
run(["git", "remote", "set-url", "origin", f"https://{tok}@github.com/Alessio-Pecilli/Hamiltonian-Blind-In-Context-Propagator.git"])
run(["git", "config", "user.name", "Alessio Pecilli"]); run(["git", "config", "user.email", "alessiopecilli@gmail.com"])
jobs = [("n8_resid_wide", ["n8_ref", "n8_resid", "n8_resid_wide", "hl_fair"], "n8 residual, modello grande (d=256, L=6, m=64), 20k step (HPC)",
         "Residual anchor, modello di dimensione hpc, n=8, 20000 step. Verifica se il limite e la capacita. HL equo ricalcolato dopo la correzione di ehrenfest_system."),
        ("n8_resid_long", ["n8_ref", "n8_resid", "n8_resid_long"], "n8 residual, 60k step (HPC)",
         "Residual anchor, modello locale, 60000 step invece di 20000. Verifica se il limite e la durata del training.")]
for name, keep, label, notes in jobs:
    marker = f"results/{name}/.published"
    if os.path.exists(marker): print(name, "gia pubblicato"); continue
    if not os.path.exists(f"results/{name}/results.json"): print(name, "skip: manca results.json"); continue
    shutil.rmtree("stage_publish", ignore_errors=True)
    for k in keep:
        if not os.path.exists(f"results/{k}/results.json"): continue
        os.makedirs(f"stage_publish/runs/{k}", exist_ok=True); os.makedirs(f"stage_publish/results/{k}")
        if k != "hl_fair":
            for f in ["config.json", "log.jsonl", "last.pt"]:
                if os.path.exists(f"runs/{k}/{f}"): shutil.copy(f"runs/{k}/{f}", f"stage_publish/runs/{k}")
        shutil.copy(f"results/{k}/results.json", f"stage_publish/results/{k}")
    run(["git", "pull", "--rebase", "-q"])
    r = subprocess.run([sys.executable, "-m", "tools.publish_dashboard", "--label", label, "--notes", notes, "--runs", "stage_publish/runs",
        "--results", "stage_publish/results", "--data", "datasets/n8/test.h5", "--ckpt", f"runs/{name}/last.pt",
        "--T", "32", "--horizon", "64", "--sample", "0", "--keep", "100"])
    if r.returncode == 0: open(marker, "w").close()
    shutil.rmtree("stage_publish", ignore_errors=True)
