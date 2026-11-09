"""Step 7 of the implementation plan: metrics and baselines."""
import numpy as np

from data.generate import simulate
from eval.baselines import estimate_tfim, flatten_obs, run_baselines, unflatten_obs
from eval.metrics import auroc, k_star


def test_oracle_is_exact(tiny_h5):
    res = run_baselines(tiny_h5, ["oracle"], [10, 20], 16, max_traj=3, verbose=False)
    for r in res["oracle"].values():
        assert max(r["rmse_n"]) < 1e-6 and max(r["rmse_e"]) < 1e-6


def test_flatten_roundtrip(tiny_h5):
    tr = simulate(5, 4, 0.1, "tfim", 0)
    n2, e2 = unflatten_obs(flatten_obs(tr.node, tr.edge), 5)
    assert np.array_equal(n2, tr.node) and np.array_equal(e2, tr.edge)
    assert flatten_obs(tr.node, tr.edge).shape[-1] == 3 * 5 + 9 * 10


def test_hl_recovers_parameters_small_dt():
    tr = simulate(5, 40, 0.01, "tfim", 2)
    J, h = estimate_tfim(tr.node, tr.edge, 0.01, l1=0.0)
    assert np.abs(J - tr.J).max() < 0.05 and np.abs(h - tr.h).max() < 0.05


def test_metric_helpers():
    assert k_star(np.array([0.01, 0.02, 0.06, 0.01])) == 2
    assert k_star(np.array([0.1])) == 0
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75


def test_parameter_sets_and_dmd_lib(tmp_path, monkeypatch):
    import h5py
    from data.generate import generate_file
    from eval.baselines import dmd_lib, fit_dmd_lib
    tr, te = str(tmp_path / "tr.h5"), str(tmp_path / "te.h5")
    generate_file(tr, n=4, ntraj=60, L=160, dt=0.1, family="tfim", seed=1, verbose=False, n_sets=3, set_seed=5)
    generate_file(te, n=4, ntraj=3, L=40, dt=0.1, family="tfim", seed=2, verbose=False, n_sets=3, set_seed=5)
    with h5py.File(tr) as f, h5py.File(te) as g:
        assert np.array_equal(f["h"][0], f["h"][3]) and not np.array_equal(f["h"][0], f["h"][1])
        assert np.array_equal(f["h"][:3], g["h"][:3])                        # same set_seed: same Hamiltonians
        assert not np.allclose(f["node"][0, 0], g["node"][0, 0])             # ...but new initial states
        node, edge = g["node"][0], g["edge"][0]
    lib = str(tmp_path / "lib.npy")
    fit_dmd_lib(tr, lib)
    monkeypatch.setenv("HBICP_DMD_LIB", lib)
    pn, _ = dmd_lib(node[:9], edge[:9], 30)
    # the 1- and 2-body observables are not closed across initial states, so even the right operator
    # drifts; only the first step is tight
    assert np.isfinite(pn).all() and np.abs(pn[0] - node[9]).max() < 0.05
