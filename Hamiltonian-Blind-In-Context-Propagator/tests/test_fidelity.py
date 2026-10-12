import numpy as np
import torch

from eval.metrics import fidelity_pairs, rho_pairs_np
from losses import rho_pairs


def _product_state(n=3, H=2):
    """All qubits in |0>: <Z_i> = 1, <Z_i Z_j> = 1, everything else 0."""
    node = np.zeros((1, H, n, 3)); node[..., 2] = 1.0
    edge = np.zeros((1, H, n, n, 9)); edge[..., 8] = 1.0
    edge[..., np.arange(n), np.arange(n), :] = 0.0
    return node, edge


def test_rho_matches_torch_version():
    rng = np.random.default_rng(0)
    node, edge = rng.uniform(-1, 1, (2, 3, 4, 3)), rng.uniform(-1, 1, (2, 3, 4, 4, 9))
    a = rho_pairs_np(node, edge)
    b = rho_pairs(torch.tensor(node), torch.tensor(edge)).numpy()
    assert np.allclose(a, b, atol=1e-6)


def test_fidelity_identical_is_one():
    node, edge = _product_state()
    assert np.allclose(fidelity_pairs(node, node, edge, edge), 1.0, atol=1e-9)


def test_fidelity_maximally_mixed_prediction():
    """rho_hat = I/4 against the pure |00><00|: F = <00|I/4|00> = 1/4."""
    node, edge = _product_state()
    f = fidelity_pairs(np.zeros_like(node), node, np.zeros_like(edge), edge)
    assert np.allclose(f, 0.25, atol=1e-9)


def test_fidelity_orthogonal_states_is_zero():
    node, edge = _product_state()
    flip_n = -node
    flip_e = edge.copy()          # |11>: <Z> = -1, <ZZ> = +1 -> same edge, flipped node
    assert np.allclose(fidelity_pairs(flip_n, node, flip_e, edge), 0.0, atol=1e-9)


def test_fidelity_unphysical_prediction_stays_in_unit_interval():
    node, edge = _product_state()
    rng = np.random.default_rng(1)
    pn, pe = rng.uniform(-1, 1, node.shape), rng.uniform(-1, 1, edge.shape)
    f = fidelity_pairs(pn, node, pe, edge)
    assert np.all((f >= 0) & (f <= 1))
