"""Training (Sec. 5).

    python train.py --preset local                      # Stage 1 (n = 8)
    python train.py --preset hpc                        # Stage 2 (n = 10, one GPU)
    python train.py --preset local --overfit            # sanity check: single-batch overfit
    python train.py --preset local --set model.head=B   # Head B ablation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from config import git_commit, load_config
from data.dataset import gather_targets, make_loader
from eval.metrics import evaluate_model
from losses import total_loss
from model.propagator import Propagator, count_parameters


def lr_at(step: int, oc: dict, total: int) -> float:
    if step < oc["warmup"]:
        return oc["lr"] * (step + 1) / oc["warmup"]
    p = min(1.0, (step - oc["warmup"]) / max(1, total - oc["warmup"]))
    return oc["lr_min"] + 0.5 * (oc["lr"] - oc["lr_min"]) * (1 + math.cos(math.pi * p))


def make_optimizer(model, oc):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)  # biases / LayerNorm: no decay
    return torch.optim.AdamW([{"params": decay, "weight_decay": oc["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=oc["lr"], betas=tuple(oc["betas"]))


class Logger:
    def __init__(self, out_dir):
        self.f = open(os.path.join(out_dir, "log.jsonl"), "a")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(out_dir)
        except Exception:
            self.tb = None

    def log(self, step, d, prefix="train"):
        self.f.write(json.dumps({"step": step, "split": prefix, **d}) + "\n")
        self.f.flush()
        if self.tb:
            for k, v in d.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self.tb.add_scalar(f"{prefix}/{k}", v, step)


def save_ckpt(path, model, opt, step, cfg):
    # write-then-rename: a disconnect mid-write on Drive must not corrupt last.pt
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, config=cfg,
                    git_commit=git_commit()), path + ".tmp")
    os.replace(path + ".tmp", path)


def summarize_eval(res: dict) -> dict:
    out = {}
    for T, r in res.items():
        out[f"T{T}/rmse_n_k1"] = r["rmse_n"][0]
        out[f"T{T}/rmse_n_kH"] = r["rmse_n"][-1]
        out[f"T{T}/rmse_e_kH"] = r["rmse_e"][-1]
        out[f"T{T}/k_star"] = r["k_star"]
        out[f"T{T}/fid_mean16"] = float(np.mean(r["fidelity2"][:16]))
        for k in ("auroc_gate", "auroc_attn", "gen_auroc_adj", "gen_corr_J", "gen_corr_h",
                  "probe_r2_J", "probe_r2_h", "S_F"):
            if k in r:
                out[f"T{T}/{k}"] = r[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--preset", default="local", choices=["local", "hpc"])
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.steps=1000")
    ap.add_argument("--overfit", action="store_true", help="single-batch overfit check")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.preset, args.set)
    tc, oc, dc = cfg["train"], cfg["optim"], cfg["data"]
    torch.manual_seed(tc["seed"]); np.random.seed(tc["seed"]); random.seed(tc["seed"])
    device = torch.device(args.device)
    out_dir = tc["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cfg["git_commit"] = git_commit()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    model = Propagator(cfg).to(device)
    opt = make_optimizer(model, oc)
    print(f"params: {count_parameters(model):,}  head={model.head_type}  device={device}")
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]

    loader = make_loader(dc["train"], (tc["T_min"], tc["T_max"]), tc["H"], tc["anchors"],
                         tc["batch_size"], shuffle=True, num_workers=dc.get("num_workers", 0),
                         preload=dc.get("preload", True), min_anchor=tc.get("min_anchor", 8))
    logger = Logger(out_dir)
    total = tc["steps"]
    fixed_batch = None
    t_last = time.time()
    while step < total:
        for batch in loader:
            if args.overfit:
                fixed_batch = fixed_batch or batch
                batch = fixed_batch
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, oc, total)
            H = random.randint(tc.get("H_min", tc["H"]), tc["H"]) if tc.get("var_horizon") else tc["H"]
            tn, te = gather_targets(batch, H)
            pred = model(batch["node"], batch["edge"], batch["dt"], batch["anchors"], batch["T"], H)
            loss, logs = total_loss(pred, batch, tn, te, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
            opt.step()
            step += 1
            if step % tc["log_every"] == 0 or step == 1:
                logs.update(lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                            sec_per_step=(time.time() - t_last) / (1 if step == 1 else tc["log_every"]))
                t_last = time.time()
                logger.log(step, logs)
                print(f"step {step:7d}  loss {logs['loss']:.3e}  pred {logs['pred']:.3e}  "
                      f"phys {logs['phys']:.2e}  probe {logs['probe']:.3f}  "
                      f"S_F {logs.get('S_F', 0):.3f}  lr {logs['lr']:.2e}", flush=True)
            if args.overfit and logs["pred"] < 1e-4:
                print(f"overfit check PASSED at step {step}: L_pred = {logs['pred']:.2e}")
                return
            if not args.overfit and dc.get("val") and step % tc["eval_every"] == 0:
                res = evaluate_model(model, dc["val"], tc["eval_T"], tc["H"], device,
                                     batch_size=tc["batch_size"], max_batches=tc["eval_batches"])
                s = summarize_eval(res)
                logger.log(step, s, "val")
                print("  val: " + "  ".join(f"{k}={v:.3g}" for k, v in s.items()), flush=True)
            if step % tc["ckpt_every"] == 0 or step == total:
                save_ckpt(os.path.join(out_dir, "last.pt"), model, opt, step, cfg)
            if step >= total:
                break
    if args.overfit:
        print(f"overfit check did NOT reach L_pred < 1e-4 in {total} steps")


if __name__ == "__main__":
    main()
