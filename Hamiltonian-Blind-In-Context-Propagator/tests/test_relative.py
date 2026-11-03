"""Relative metrics and the amplitude-normalised loss are scale-free."""
import numpy as np
import torch

from eval.metrics import RelMetrics
from losses import pred_loss


def _obs(rng, N=3, H=5, n=4, scale=1.0):
    return scale * rng.normal(size=(N, H, n, 3)), scale * rng.normal(size=(N, H, n, n, 9))


def test_rel_metrics_perfect_zero_and_scaled():
    rng = np.random.default_rng(0)
    tn, te = _obs(rng)
    for pn, pe, rel, corr in ((tn, te, 0.0, 1.0), (0 * tn, 0 * te, 1.0, None), (2 * tn, 2 * te, 1.0, 1.0)):
        m = RelMetrics(); m.add(pn, tn, pe, te)
        rn = ((pn - tn) ** 2).mean((-1, -2)) ** 0.5
        mask = ~np.eye(tn.shape[-2], dtype=bool)
        re = ((pe - te) ** 2)[..., mask, :].mean((-1, -2)) ** 0.5
        r = m.result(rn.mean(0), re.mean(0))
        if rel == 0.0:
            assert np.allclose(r["rel_rmse_n"], 0) and r["k_star_rel"] == 5
        else:
            assert np.allclose(r["rel_rmse_n"], 1, rtol=1e-9) and r["k_star_rel"] == 0
        if corr is not None:
            assert np.allclose(r["corr_n"], corr) and np.allclose(r["corr_e"], corr)


def test_amp_norm_loss_is_scale_free():
    rng = np.random.default_rng(1)
    tn, te = (torch.tensor(x, dtype=torch.float32) for x in _obs(rng))
    pn, pe = 0.9 * tn, 0.9 * te
    l1 = pred_loss(pn, pe, tn, te, amp_floor=1e-4)
    l2 = pred_loss(0.1 * pn, 0.1 * pe, 0.1 * tn, 0.1 * te, amp_floor=1e-4)
    assert torch.allclose(l1, l2, rtol=1e-4) and torch.allclose(l1, torch.tensor(0.01), rtol=1e-3)
