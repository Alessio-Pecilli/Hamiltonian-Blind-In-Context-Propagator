"""Publish the finished HPC runs to the dashboard. Run on the LOGIN node (needs internet), from the code folder:
    python scripts/hpc/publish.py      # token from ~/.hbicp_gh_token or $GH_TOKEN
Mirrors the Colab publish cell: n8_resid_wide and n8_resid_long (n8_ref/n8_resid come along for comparison if present),
then n10_small and n12_small (no oracle baselines there, so the sample rollout uses persistence/dmd/dmd4 only).
OFFLINE=1 (inside a job, scripts/hpc/publish.sbatch: compute nodes have no internet and the login node a 600 s CPU limit):
build and commit the exports only; then run this script again on the login node to push them."""
import json, os, shutil, subprocess, sys

OFFLINE = os.environ.get("OFFLINE") == "1"

tok = os.environ.get("GH_TOKEN")
tf = os.path.expanduser("~/.hbicp_gh_token")  # chmod 600, created once by the user; never committed
if not tok and os.path.exists(tf):
    tok = open(tf).read().strip()
if not tok and not OFFLINE:
    sys.exit("no token: put it in ~/.hbicp_gh_token (see README_HPC.md) or set GH_TOKEN")
run = lambda a, **k: subprocess.run(a, check=True, **k)
if not OFFLINE: run(["git", "remote", "set-url", "origin", f"https://{tok}@github.com/Alessio-Pecilli/Hamiltonian-Blind-In-Context-Propagator.git"])
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
         "n=10 qubit, modello piccolo (d=128, L=3, m=32), 40000 step. Perdita standard contro perdita normalizzata sull'ampiezza del segnale vero (soglia 0.03). Metriche relative: errore diviso per l'ampiezza (k*_rel, soglia 25%) e correlazione previsto-vero, su nodi e correlatori di coppia. La traiettoria di esempio usa il modello con perdita normalizzata."),
        ("n10_small_roll", [("n10_small", "n10_small_rel", "previsione in un colpo solo", False),
                            ("n10_small", "n10_small_blk32", "a blocchi da 32, senza nuovo training", True),
                            ("n10_small_varh", "n10_small_varh", "orizzonte variabile fino a 64", True),
                            ("n10_small_roll", "n10_small_roll", "roll out alternato, blocchi da 16", True)],
         "n10: roll out a blocchi e orizzonte variabile (HPC)",
         "n=10 qubit, modello piccolo (d=128, L=3, m=32), 40000 step, perdita standard. (1) modello di riferimento, previsione dei 64 passi in un colpo solo; (2) lo stesso modello valutato a blocchi da 32 passi rimessi nel contesto; (3) training con orizzonte estratto a caso tra 8 e 64 passi; (4) training a blocchi da 16 passi (4 blocchi) con scheduled sampling: il blocco rimesso nel contesto e' la previsione del modello con probabilita' crescente da 0 a 1 nella prima meta' del training; valutato a blocchi. (3) e (4) con contesto di training fino a 96 invece di 128 (le traiettorie di training sono lunghe 161). La traiettoria di esempio usa il modello (4), a blocchi."),
        ("n10_sets_roll", [("n10_sets_roll", "n10_sets_roll", "64 hamiltoniane, test su hamiltoniane nuove", False),
                           ("n10_small_roll", "n10_small_roll_on_sets", "8000 hamiltoniane, stesso test", True)],
         "n10: training su 64 hamiltoniane fisse, test su hamiltoniane nuove (HPC)",
         "n=10 qubit, modello piccolo, roll out a blocchi da 16 (come la voce precedente), 40000 step. Training su 64 hamiltoniane fisse (grafo, J, h), 125 stati iniziali ciascuna; test su 20 hamiltoniane mai viste, 25 stati ciascuna. Confronto con lo stesso modello addestrato su 8000 hamiltoniane diverse (una traiettoria ciascuna). Baseline: DMD stimato sul contesto (dmd) e libreria di 64 operatori DMD, uno per hamiltoniana di training, combinati con pesi stimati sul contesto (dmd_lib)."),
        ("n10_sets_roll_seen", [("n10_sets_roll", "n10_sets_roll_seen", "64 hamiltoniane, test su hamiltoniane viste", False)],
         "n10: training su 64 hamiltoniane fisse, test sulle stesse con stati nuovi (HPC)",
         "Come la voce precedente, ma il test usa le 64 hamiltoniane del training con stati iniziali nuovi (8 per hamiltoniana): misura quanto si perde passando a parametri mai visti.")]
extra = {"n8_resid_wide": N8, "n8_resid_long": N8, "n10_small": dict(BIG, data="datasets/n10/test.h5"), "n12_small": dict(BIG, data="datasets/n12/test.h5"),
         "n10_small_norm": dict(BIG, data="datasets/n10/test.h5"), "n10_small_roll": dict(BIG, data="datasets/n10/test.h5", block=16),
         "n10_sets_roll": dict(base=["persistence", "dmd", "dmd_lib"], data="datasets/n10sets/test.h5", block=16),
         "n10_sets_roll_seen": dict(base=["persistence", "dmd", "dmd_lib"], data="datasets/n10sets/test_seen.h5", block=16, ckpt="runs/n10_sets_roll/last.pt")}
FAMS = {"tfim": "TFIM su grafo casuale", "xxz": "XXZ su grafo casuale", "heisenberg": "Heisenberg disordinato", "lr_ising": "Ising a lungo raggio",
        "chain": "catena TFIM uniforme (vecchio progetto)", "chain_dis": "catena TFIM con J disordinati (vecchio progetto)"}
MIXB = dict(base=["persistence", "dmd", "dmd4"], block=16, ckpt="runs/n10_mix_roll/last.pt")
jobs.append(("n10_mix_roll", [("n10_mix_roll", "n10_mix_roll", "training su 6 famiglie, test misto", False),
                              ("n10_small_roll", "n10_small_roll_on_mix", "training solo TFIM, stesso test", True)],
             "n10: training su 6 famiglie di hamiltoniane, test misto (HPC)",
             "n=10 qubit, modello piccolo, roll out a blocchi da 16. Training su 12000 hamiltoniane di sei famiglie (2000 ciascuna): TFIM, XXZ e Heisenberg disordinato su grafo casuale, Ising a lungo raggio, catene TFIM aperte del vecchio progetto (uniforme e con J disordinati). Test su 600 hamiltoniane nuove delle stesse famiglie. Confronto con il modello addestrato solo su TFIM e con DMD."))
extra["n10_mix_roll"] = dict(MIXB, data="datasets/n10mix/test.h5")
jobs.append(("n10_mix_roll_on_n10", [("n10_small_roll", "n10_small_roll", "training solo TFIM", False), ("n10_mix_roll", "n10_mix_roll_on_n10", "training su 6 famiglie", True)],
             "n10: training su 6 famiglie, test sul TFIM originale (HPC)", "Il modello addestrato sulle sei famiglie valutato sul test TFIM delle voci precedenti: quanto costa la varieta sul caso di partenza."))
extra["n10_mix_roll_on_n10"] = dict(MIXB, data="datasets/n10/test.h5")
for fam, desc in FAMS.items():
    jobs.append((f"n10_mix_roll_on_{fam}", [(f"n10_mix_roll", f"n10_mix_roll_on_{fam}", "training su 6 famiglie", False),
                                            ("n10_small_roll", f"n10_small_roll_on_{fam}", "training solo TFIM", True)],
                 f"n10: training su 6 famiglie, test su {desc} (HPC)", f"Come la voce del test misto, ma il test contiene solo la famiglia: {desc} (300 hamiltoniane nuove)."))
    extra[f"n10_mix_roll_on_{fam}"] = dict(MIXB, data=f"datasets/n10mix/test_{fam}.h5")
os.environ.setdefault("HBICP_DMD_LIB", os.path.abspath("datasets/n10sets/dmd_lib.npy"))
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
    if not OFFLINE: run(["git", "pull", "--rebase", "-q"])
    r = subprocess.run([sys.executable, "-m", "tools.publish_dashboard", "--label", label, "--notes", notes, "--runs", "stage_publish/runs",
        "--results", "stage_publish/results", "--data", extra[name]["data"], "--ckpt", extra[name].get("ckpt", f"runs/{name}/last.pt"), "--baselines", *extra[name]["base"],
        "--T", "32", "--horizon", "64", "--sample", "0", "--keep", "100", *(["--no-push"] if OFFLINE else []), *(["--block", str(extra[name]["block"])] if "block" in extra[name] else [])])
    if r.returncode == 0: open(marker, "w").close()
    shutil.rmtree("stage_publish", ignore_errors=True)

if OFFLINE: sys.exit(0)
try:
    run(["git", "pull", "--rebase", "-q"]); run(["git", "push", "-q"])  # exports committed by an OFFLINE job
finally:  # do not leave the token in .git/config
    run(["git", "remote", "set-url", "origin", "https://github.com/Alessio-Pecilli/Hamiltonian-Blind-In-Context-Propagator.git"])
