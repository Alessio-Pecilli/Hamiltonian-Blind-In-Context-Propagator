"""Step 1-2 of the implementation plan: generator and loader."""
import h5py
import numpy as np
import torch

from data import physics
from data.dataset import TrajectoryDataset, collate, permute_sites
from data.generate import FAMILIES, simulate


def test_norm_and_energy_conservation():
    tr = simulate(6, 80, 0.1, "tfim", 11, keep_state=True, check=False)
    norm = np.einsum("td,td->t", tr.Psi.conj(), tr.Psi).real
    assert np.abs(norm - 1).max() < 1e-10
    E = physics.energy(tr.H, tr.Psi)
    assert np.abs(E - E[0]).max() < 1e-8


def test_krylov_path_matches_dense():
    tr = simulate(5, 20, 0.1, "tfim", 3, keep_state=True)
    old = physics.DENSE_MAX_N
    physics.DENSE_MAX_N = 0
    try:
        tr2 = simulate(5, 20, 0.1, "tfim", 3, keep_state=True)
    finally:
        physics.DENSE_MAX_N = old
    assert np.abs(tr.Psi - tr2.Psi).max() < 1e-8


def test_observables_symmetry_and_range(tiny_h5):
    with h5py.File(tiny_h5) as f:
        node, edge = f["node"][()], f["edge"][()]
    n = node.shape[2]
    swap = [3 * b + a for a in range(3) for b in range(3)]
    assert np.allclose(edge, edge.transpose(0, 1, 3, 2, 4)[..., swap], atol=1e-6)
    assert np.all(edge[:, :, np.arange(n), np.arange(n)] == 0)
    assert np.abs(node).max() <= 1 + 1e-6 and np.abs(edge).max() <= 1 + 1e-6


def test_product_state_observables():
    """For a product state E[i,j,a,b] = N[i,a] N[j,b]."""
    rng = np.random.default_rng(0)
    psi = physics.random_product_state(4, rng)
    N, E = physics.observables_single(psi, 4)
    assert np.allclose(E[0, 1].reshape(3, 3), np.outer(N[0], N[1]), atol=1e-12)
    assert np.allclose((N ** 2).sum(-1), 1.0)  # pure single-qubit states


def test_regenerate_from_seed_bit_for_bit(tiny_h5):
    with h5py.File(tiny_h5) as f:
        node, edge, seed, J = f["node"][3], f["edge"][3], f["seed"][3], f["J"][3]
        a = dict(f.attrs)
    tr = simulate(int(a["n"]), int(a["L_traj"]), float(a["dt"]), str(a["family"]), int(seed))
    assert np.array_equal(tr.node, node) and np.array_equal(tr.edge, edge)
    assert np.array_equal(tr.J, J)


def test_heisenberg_equations_tfim():
    """The HL baseline relies on Eq. (29): check it against finite differences at small dt."""
    dt = 0.002
    tr = simulate(5, 30, dt, "tfim", 5)
    N, E, J, h = tr.node.astype(float), tr.edge.astype(float), tr.J, tr.h
    d = (N[2:] - N[:-2]) / (2 * dt)
    Nm, Em = N[1:-1], E[1:-1]
    assert np.abs(d[..., 0] - 2 * np.einsum("ij,tij->ti", J, Em[..., 5])).max() < 1e-3
    assert np.abs(d[..., 1] - (-2 * np.einsum("ij,tij->ti", J, Em[..., 2]) + 2 * h * Nm[..., 2])).max() < 1e-3
    assert np.abs(d[..., 2] + 2 * h * Nm[..., 1]).max() < 1e-3


def test_other_families_run():
    for fam in FAMILIES:
        simulate(5, 10, 0.1, fam, 1)


def test_chain_families():
    """The open TFIM chains of the older project: nearest-neighbour bonds only."""
    for fam in ("chain", "chain_dis"):
        tr = simulate(5, 4, 0.1, fam, 2)
        assert tr.adj.sum() == 8 and all(tr.adj[i, i + 1] for i in range(4))
        assert np.array_equal(tr.J != 0, tr.adj)
    tr = simulate(5, 4, 0.1, "chain", 2)
    assert np.ptp(tr.h) == 0 and np.ptp(tr.J[tr.adj]) == 0 and 0.2 <= tr.h[0] <= 2.0


def test_mixed_file(tmp_path):
    """family = mix: families in turn, stored per trajectory, regenerated bit for bit; the oracle follows them."""
    from data.generate import generate_file
    from eval.baselines import run_baselines
    p = str(tmp_path / "mix.h5")
    generate_file(p, n=4, ntraj=2 * len(FAMILIES), L=20, dt=0.1, family="mix", seed=3, verbose=False)
    with h5py.File(p) as f:
        fams = [x.decode() for x in f["fam"][()]]
        assert fams == list(FAMILIES) * 2
        for k in (1, 4, 9):
            tr = simulate(4, 20, 0.1, fams[k], int(f["seed"][k]))
            assert np.array_equal(tr.node, f["node"][k])
    assert len(TrajectoryDataset(p, (8, 8), 4, preload=True)[0]["node"]) == 13
    r = run_baselines(p, ["oracle"], [8], 8, verbose=False)["oracle"][8]
    assert max(r["rmse_n"]) < 1e-5
    # with parameter sets the family goes with the set
    generate_file(p, n=4, ntraj=12, L=12, dt=0.1, family="mix", seed=3, verbose=False, n_sets=3, set_seed=1,
                  mix=("tfim", "xxz"))
    with h5py.File(p) as f:
        assert [x.decode() for x in f["fam"][:6]] == ["tfim", "xxz", "tfim"] * 2
        assert np.array_equal(f["J"][0], f["J"][3]) and not np.array_equal(f["J"][0], f["J"][1])


def test_loader_permutation(tiny_h5):
    torch.manual_seed(0)
    ds = TrajectoryDataset(tiny_h5, (8, 16), 4, anchors=2, permute=False)
    item = ds[2]
    perm = np.random.default_rng(1).permutation(ds.n)
    p = permute_sites(item, perm)
    assert np.array_equal(p["node"], item["node"][:, perm])
    assert np.array_equal(p["edge"][:, 1, 4], item["edge"][:, perm[1], perm[4]])
    assert np.array_equal(p["J"][0, 3], item["J"][perm[0], perm[3]])
    batch = collate([ds[0], ds[1], ds[2]])
    T = batch["T"]
    assert batch["node"].shape[1] == int(T.max()) + 4 + 1
    assert (batch["anchors"][:, 0] == T).all() and (batch["anchors"] <= T[:, None]).all()


def test_anchor_range(tiny_h5):
    """Sec. 5.1: anchors = last context position + positions in [t0+8, t0+T) (relative: [8, T))."""
    torch.manual_seed(0)
    ds = TrajectoryDataset(tiny_h5, (8, 24), 4, anchors=4, permute=False)
    for k in range(300):
        it = ds[k % len(ds)]
        T, a = it["T"], it["anchors"]
        assert a[0] == T
        assert all((8 <= x < T) or (T <= 8 and x == T) for x in a[1:])
        assert it["t0"] + T + 4 < ds.L
