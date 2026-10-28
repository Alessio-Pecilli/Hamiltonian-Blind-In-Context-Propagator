"""Loss of Sec. 4: L = L_pred + lambda_phys L_phys + lambda_probe L_probe + lambda_theta ||theta||^2."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from data.physics import PAULIS, SI

# P16[a, b] = sigma^a (x) sigma^b, a, b in {0, x, y, z}
_P16 = None


def pauli16(device) -> torch.Tensor:
    global _P16
    if _P16 is None or _P16.device != torch.device(device):
        import numpy as np
        ps = (SI,) + PAULIS
        P = np.stack([np.stack([np.kron(a, b) for b in ps]) for a in ps])  # (4, 4, 4, 4)
        _P16 = torch.tensor(P, dtype=torch.complex64, device=device)
    return _P16


def offdiag(n, device):
    return ~torch.eye(n, dtype=torch.bool, device=device)


def k_weights(H: int, mode: str, device) -> torch.Tensor:
    if mode == "linear":
        w = torch.arange(1, H + 1, device=device, dtype=torch.float32)
        return w / w.mean()
    return torch.ones(H, device=device)


def _sq_err(pn, pe, tn, te):
    """(..., H): mean square over 3n node entries plus mean square over 9n(n-1) off-diagonal edge entries."""
    n = pn.shape[-2]
    ln = ((pn - tn) ** 2).mean((-1, -2))
    mask = offdiag(n, pn.device)[..., None]
    le = (((pe - te) ** 2) * mask).sum((-1, -2, -3)) / (9 * n * (n - 1))
    return ln + le


def pred_loss(pn, pe, tn, te, w_k=None, anchor=None):
    """pn/tn (..., H, n, 3), pe/te (..., H, n, n, 9). Averaged over anchors and k (Eq. 27).
    anchor = (node0 (B, A, n, 3), edge0 (B, A, n, n, 9)): per-k normalisation, each k divided by the
    batch persistence error at that k (detached), so no horizon dominates."""
    l = _sq_err(pn, pe, tn, te)                                            # (B, A, H)
    if anchor is not None:
        p = _sq_err(anchor[0][..., None, :, :], anchor[1][..., None, :, :, :], tn, te)
        l = l / (p.detach().mean((0, 1)) + 1e-6)
    elif w_k is not None:
        l = l * w_k
    return l.mean()


def rho_pairs(pn: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Two-qubit density matrices for all i<j from predicted node (..., n, 3) / edge (..., n, n, 9).
    Returns (..., P, 4, 4) complex64, P = n(n-1)/2 (Eq. 28)."""
    n = pn.shape[-2]
    iu, ju = torch.triu_indices(n, n, 1, device=pn.device)
    lead = pn.shape[:-2]
    P = iu.numel()
    c = torch.zeros(*lead, P, 4, 4, device=pn.device, dtype=torch.float32)
    c[..., 0, 0] = 1.0
    c[..., 1:, 0] = pn[..., iu, :]
    c[..., 0, 1:] = pn[..., ju, :]
    c[..., 1:, 1:] = pe[..., iu, ju, :].reshape(*lead, P, 3, 3)
    return 0.25 * torch.einsum("...ab,abxy->...xy", c.to(torch.complex64), pauli16(pn.device))


def phys_loss(pn, pe):
    """mean relu(-lambda_min(rho_ij)) over pairs, anchors, horizons."""
    rho = rho_pairs(pn.float(), pe.float()).reshape(-1, 4, 4)
    # chunked: cuSOLVER batched eigvalsh allocates a huge workspace for ~3e4 matrices (OOM at H = 64)
    lam = torch.cat([torch.linalg.eigvalsh(c)[:, 0] for c in rho.split(2048)])
    return F.relu(-lam).mean()


def probe_loss(probe: dict, batch: dict):
    n = batch["h"].shape[-1]
    mask = offdiag(n, batch["h"].device).expand_as(batch["J"])
    l_h = F.mse_loss(probe["h"], batch["h"])
    l_J = F.mse_loss(probe["J"][mask], batch["J"][mask])
    l_a = F.binary_cross_entropy_with_logits(probe["adj_logit"][mask], batch["adj"][mask].float())
    return l_h + l_J + l_a, dict(probe_h=l_h.item(), probe_J=l_J.item(), probe_adj=l_a.item())


def total_loss(pred: dict, batch: dict, tn, te, cfg: dict):
    lc = cfg["loss"]
    H = tn.shape[2]
    w = k_weights(H, lc.get("w_k", "uniform"), tn.device)  # (H,)
    anchor = (pred["node0"], pred["edge0"]) if lc.get("per_k_norm", False) else None
    l_pred = pred_loss(pred["node"], pred["edge"], tn, te, w, anchor)
    l_phys = phys_loss(pred["node"], pred["edge"]) if lc["lambda_phys"] > 0 else tn.new_zeros(())
    if "probe" in pred and lc["lambda_probe"] > 0:  # never a target of the body: only for the old runs
        l_probe, probe_logs = probe_loss(pred["probe"], batch)
    else:
        l_probe, probe_logs = tn.new_zeros(()), {}
    l_theta = (pred["theta"] ** 2).sum(-1).mean() if "theta" in pred else tn.new_zeros(())
    loss = l_pred + lc["lambda_phys"] * l_phys + lc["lambda_probe"] * l_probe \
        + lc["lambda_theta"] * l_theta
    logs = dict(loss=loss.item(), pred=l_pred.item(), phys=l_phys.item(), probe=l_probe.item(),
                theta=l_theta.item(), **probe_logs)
    if "S_F" in pred:
        logs["S_F"] = pred["S_F"].mean().item()
    return loss, logs
