"""Publish the finished HPC runs to the dashboard. Run on the LOGIN node (needs internet), from the code folder:
    python scripts/hpc/publish.py      # token from ~/.hbicp_gh_token or $GH_TOKEN
Mirrors the Colab publish cell: n8_resid_wide and n8_resid_long (n8_ref/n8_resid come along for comparison if present),
then n10_small and n12_small (no oracle baselines there, so the sample rollout uses persistence/dmd/dmd4 only)."""
import json, os, shutil, subprocess, sys

tok = os.environ.get("GH_TOKEN")
tf = os.path.expanduser("~/.hbicp_gh_token")  # chmod 600, created once by the user; never committed
if not tok and os.path.exists(tf):
    tok = open(tf).read().strip()
if not tok:
    sys.exit("no token: put it in ~/.hbicp_gh_token (see README_HPC.md) or set GH_TOKEN")
run = lambda a, **k: subprocess.run(a, check=True, **k)
run(["git", "remote", "set-url", "origin", f"https://{tok}@github.com/Alessio-Pecilli/Hamiltonian-Blind-In-Context-Propagator.git"])
run(["git", "config", "user.name", "Alessio Pecilli"]); run(["git", "config", "user.email", "alessiopecilli@gmail.com"])
N8 = dict(data="datasets/n8/test.h5", base=["persistence", "dmd", "dmd4", "oracle", "hl_oracle"])
BIG = dict(base=["persistence", "dmd", "dmd4"])
jobs = [("n8_resid_wide", ["n8_ref", "n8_resid", "n8_resid_wide", "hl_fair"], "n8 residual, modello grande (d=256, L=6, m=64), 20k step (HPC)",
         "Residual anchor, modello di dimensione hpc, n=8, 20000 step. Verifica se il limite e la capacita. HL equo ricalcolato dopo la correzione di ehrenfest_system."),
        ("n8_resid_long", ["n8_ref", "n8_resid", "n8_resid_long"], "n8 residual, 60k step (HPC)",
         "Residual anchor, modello locale, 60000 step invece di 20000. Verifica se il limite e la durata del training."),
        ("n10_small", ["n10_resid", "n10_small", "hl_fair_n10"], "n10: modello piccolo vs grande vs DMD (HPC)",
         "n=10 qubit, 8000 hamiltoniane, 40000 step, residual anchor. Modello piccolo (d=128, L=3, m=32, 1.45M param) a confronto con il grande (d=256, L=6, m=64) e con DMD. HL equo (5 traiettorie, T=64) per riferimento."),
        ("n12_small", ["n10_small", "n12_small"], "n12: modello piccolo vs DMD (HPC)",
         "n=12 qubit, 3000 traiettorie di training, 40000 step, residual anchor, modello piccolo (d=128, L=3, m=32). Confronto con DMD e persistence. n10_small incluso come riferimento."),
        # (src, shown name, model only): the two result sets get readable names, baselines appear once
        ("n10_small_norm", [("n10_small", "n10_small_rel", "perdita standard", False), ("n10_small_norm", "n10_small_norm", "perdita normalizzata", True)],
         "n10: perdita normalizzata vs standard vs DMD, metriche relative (HPC)",
         "n=10 qubit, modello piccolo (d=128, L=3, m=32), 40000 step. Perdita standard contro perdita normalizzata sull'ampiezza del segnale vero (soglia 0.03). Metriche relative: errore diviso per l'ampiezza (k*_rel, soglia 25%) e correlazione previsto-vero, su nodi e correlatori di coppia. La traiettoria di esempio usa il modello con perdita normalizzata.")]
extra = {"n8_resid_wide": N8, "n8_resid_long": N8, "n10_small": dict(BIG, data="datasets/n10/test.h5"), "n12_small": dict(BIG, data="datasets/n12/test.h5"),
         "n10_small_norm": dict(BIG, data="datasets/n10/test.h5")}
for name, keep, label, notes in jobs:
    marker = f"results/{name}/.published"
    if os.path.exists(marker): print(name, "gia pubblicato"); continue
    if not os.path.exists(f"results/{name}/results.json"): print(name, "skip: manca results.json"); continue
    shutil.rmtree("stage_publish", ignore_errors=True)
    for k in keep:
        run_k, res_k, show, model_only = k if isinstance(k, tuple) else (k, k, k, False)
        if not os.path.exists(f"results/{res_k}/results.json"): continue
        os.makedirs(f"stage_publish/runs/{show}", exist_ok=True); os.makedirs(f"stage_publish/results/{show}")
        if not run_k.startswith("hl_fair"):
            for f in ["config.json", "log.jsonl", "last.pt"]:
                if os.path.exists(f"runs/{run_k}/{f}"): shutil.copy(f"runs/{run_k}/{f}", f"stage_publish/runs/{show}")
        res = json.load(open(f"results/{res_k}/results.json"))
        if model_only: res = {m: v for m, v in res.items() if m.startswith("model")}
        json.dump(res, open(f"stage_publish/results/{show}/results.json", "w"))
    run(["git", "pull", "--rebase", "-q"])
    r = subprocess.run([sys.executable, "-m", "tools.publish_dashboard", "--label", label, "--notes", notes, "--runs", "stage_publish/runs",
        "--results", "stage_publish/results", "--data", extra[name]["data"], "--ckpt", f"runs/{name}/last.pt", "--baselines", *extra[name]["base"],
        "--T", "32", "--horizon", "64", "--sample", "0", "--keep", "100"])
    if r.returncode == 0: open(marker, "w").close()
    shutil.rmtree("stage_publish", ignore_errors=True)

# do not leave the token in .git/config
run(["git", "remote", "set-url", "origin", "https://github.com/Alessio-Pecilli/Hamiltonian-Blind-In-Context-Propagator.git"])
