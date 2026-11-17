"""Pair batches (same Hamiltonian) and cross-prediction with the partner's generator."""
import h5py
import numpy as np
import torch

from data.dataset import PairBatchSampler, permute_sites
from data.generate import generate_file
from test_model import make_model, random_inputs


def test_pair_sampler(tmp_path):
    p = str(tmp_path / "sets.h5")
    generate_file(p, n=4, ntraj=24, L=12, dt=0.1, family="tfim", seed=1, verbose=False, n_sets=6, set_seed=2)
    with h5py.File(p) as f:
        sid, J = f["set_id"][()], f["J"][()]
    batches = list(PairBatchSampler(sid, 8, seed=0))
    assert len(batches) == 3
    for b in batches:
        assert len(b) == 8 and len(set(b)) == 8
        for i in range(0, 8, 2):
            assert sid[b[i]] == sid[b[i + 1]] and np.array_equal(J[b[i]], J[b[i + 1]])
        assert len({sid[i] for i in b}) == 4


def test_cross_predict_permutation(tiny_cfg):
    """Partner = the same window with its sites permuted: once moved to this window's site order, its generator
    is this window's generator (the model is site-equivariant), so the cross prediction is the own prediction."""
    tiny_cfg["model"]["residual"] = True
    model = make_model(tiny_cfg).double().eval()
    node, edge, dt = (x.double() for x in random_inputs(1, 16, 5))
    perm = np.array([3, 0, 4, 1, 2])
    item = permute_sites(dict(node=node[0].numpy(), edge=edge[0].numpy(), adj=np.zeros((5, 5)),
                              J=np.zeros((5, 5)), h=np.zeros(5)), perm)
    node2 = torch.stack([node[0], torch.from_numpy(item["node"])])
    edge2 = torch.stack([edge[0], torch.from_numpy(item["edge"])])
    T = torch.tensor([12, 12])
    with torch.no_grad():
        pred = model(node2, edge2, dt.repeat(2), T[:, None], T, 6, keep_feats=True)
        P = torch.tensor(np.stack([np.arange(5), perm]))
        pn, pe, mis = model.cross_predict(pred, torch.tensor([1, 0]), P, 6)
    assert mis.max() < 1e-8
    assert torch.allclose(pn, pred["node"][:, 0], atol=1e-10) and torch.allclose(pe, pred["edge"][:, 0], atol=1e-10)
    # a partner from another window does change the prediction
    with torch.no_grad():
        other = model(node2.flip(1).contiguous(), edge2.flip(1).contiguous(), dt.repeat(2), T[:, None], T, 6, keep_feats=True)
        both = {k: torch.cat([pred[k], other[k]]) if torch.is_tensor(pred[k]) and k != "dt" else pred[k] for k in pred}
        both["dt"] = dt.repeat(4)
        both["G"] = torch.cat([pred["G"], other["G"]])
        pn2, _, mis2 = model.cross_predict(both, torch.tensor([2, 3, 0, 1]), torch.cat([P, P]), 6)
    assert mis2.min() > 1e-3 and not torch.allclose(pn2[0], pred["node"][0, 0])
