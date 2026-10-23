import numpy as np

from data import physics
from eval import hl_fair
from eval.baselines import flatten_obs


def _setup(n=3, L=6, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    fr = hl_fair.Fitter(n, dt)
    theta = rng.normal(size=fr.K)
    psi = physics.random_product_state(n, rng)
    Psi, *_ = fr.states(theta, psi, L)
    node, edge = physics.observables(Psi, n)
    return fr, theta, psi, flatten_obs(node, edge), node, edge


def test_observables_match_physics():
    fr, theta, psi, obs, *_ = _setup()
    Psi, *_ = fr.states(theta, psi, 6)
    assert np.allclose(fr.observables(Psi), obs)


def test_gradient_matches_finite_differences():
    fr, theta, psi, obs, *_ = _setup()
    rng = np.random.default_rng(1)
    th = theta + 0.3 * rng.normal(size=fr.K)
    ps = psi + 0.2 * (rng.normal(size=psi.shape) + 1j * rng.normal(size=psi.shape))
    ps /= np.linalg.norm(ps)
    L0, gth, gr, gi = fr.loss_grad(th, ps, obs)
    e = 1e-6
    for k in rng.choice(fr.K, 6, replace=False):
        d = np.zeros(fr.K); d[k] = e
        fd = (fr.loss_grad(th + d, ps, obs, False)[0] - fr.loss_grad(th - d, ps, obs, False)[0]) / (2 * e)
        assert abs(fd - gth[k]) < 1e-6 * (1 + abs(fd))
    for s in rng.choice(len(ps), 4, replace=False):
        for g, v in ((gr, 1.0), (gi, 1j)):
            d = np.zeros(len(ps), complex); d[s] = e * v
            fd = (fr.loss_grad(th, ps + d, obs, False)[0] - fr.loss_grad(th, ps - d, obs, False)[0]) / (2 * e)
            assert abs(fd - g[s]) < 1e-6 * (1 + abs(fd))


def test_trajectory_fit_recovers_forecast():
    fr, theta, psi, obs, node, edge = _setup(n=3, L=12)
    th, ps, loss = hl_fair.fit(node, edge, 0.1, fit_psi=True, iters=300)
    assert loss < 1e-4
