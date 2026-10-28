"""Fair Hamiltonian-learning baseline: a generic 2-local Hamiltonian fitted to the observed context only.

H(theta) = sum_k theta_k P_k over every one- and two-qubit Pauli (K = 3n + 9n(n-1)/2). The P_k are in
the same order as flatten_obs(), so the observables o_t[k] = <psi_t|P_k|psi_t> and the parameters
share one index. Stage 1: Ehrenfest least squares (group lasso per pair). Stage 2: trajectory matching
with an exact analytic gradient (eigenbasis + Daleckii-Krein), L-BFGS-B. psi_0 is either rebuilt from
o_0 (`hl_fair_o0`, valid only for pure product states and t0 = 0) or fitted jointly (`hl_fair_fit`).
"""
from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.sparse as sp

from data import physics

EPS = np.zeros((3, 3, 3))
for _a, _b, _c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    EPS[_a, _b, _c], EPS[_a, _c, _b] = 1.0, -1.0
LETTERS = "XYZ"


def pauli_terms(n: int):
    """[(site, letter), ...] per parameter, ordered like flatten_obs: nodes (i, a), then pairs i<j x (a, b)."""
    terms = [[(i, LETTERS[a])] for i in range(n) for a in range(3)]
    for i in range(n):
        for j in range(i + 1, n):
            terms += [[(i, LETTERS[a]), (j, LETTERS[b])] for a in range(3) for b in range(3)]
    return terms


class Fitter:
    """Dense exact simulation, trajectory loss and gradient for one n."""

    def __init__(self, n: int, dt: float):
        self.n, self.dt, self.D = n, dt, 1 << n
        D = self.D
        mats = [physics.pauli_string_sparse(n, ops) for ops in pauli_terms(n)]
        self.K = len(mats)
        self.new = np.empty((self.K, D), np.int64)    # P_k |s> = phase[k, s] |new[k, s]>
        self.phase = np.empty((self.K, D), np.complex128)
        for k, M in enumerate(mats):
            M = M.tocsc()
            self.new[k] = M.indices                   # one nonzero per column
            self.phase[k] = M.data
        # (P_k psi)[r] = phase2[k, r] psi[new[k, r]]  (new is an involution)
        self.phase2 = np.take_along_axis(self.phase, self.new, 1)
        self.flat = (self.new * D + np.arange(D)[None]).ravel()

    # ---- model --------------------------------------------------------------------------
    def hamiltonian(self, theta):
        w = (theta[:, None] * self.phase).ravel()
        H = np.bincount(self.flat, w.real, self.D ** 2) + 1j * np.bincount(self.flat, w.imag, self.D ** 2)
        return H.reshape(self.D, self.D)

    def states(self, theta, psi0, L):
        """Psi (L, D) = exp(-i H dt t) psi0, plus eigendecomposition."""
        E, V = np.linalg.eigh(self.hamiltonian(theta))
        c = V.conj().T @ psi0
        tau = self.dt * np.arange(L)
        return (np.exp(-1j * np.outer(tau, E)) * c) @ V.T, E, V, c, tau

    def observables(self, Psi):
        G = Psi[:, self.new]                                         # (L, K, D)
        return np.einsum("tr,kr,tkr->tk", Psi.conj(), self.phase2, G).real

    def loss_grad(self, theta, psi0, obs, grad_psi=True):
        """L = mean_{t,k} (o_hat - o)^2; returns L, dL/dtheta, dL/d(Re psi0), dL/d(Im psi0)."""
        L = obs.shape[0]
        Psi, E, V, c, tau = self.states(theta, psi0, L)
        G = Psi[:, self.new]
        res = np.einsum("tr,kr,tkr->tk", Psi.conj(), self.phase2, G).real - obs
        a = 2 * res / res.size                                      # dL = sum a_tk d o_hat_tk
        g = np.einsum("tk,kr,tkr->tr", a, self.phase2, G)           # dL = 2 Re sum_t g_t^H dpsi_t
        gam = g @ V.conj()                                          # gamma_t = V^H g_t
        Eb = 0.5 * (E[:, None] + E[None])
        dE = E[:, None] - E[None]
        M = np.zeros((self.D, self.D), np.complex128)
        for t in range(1, L):                                       # t = 0 does not depend on H
            F = -1j * tau[t] * np.exp(-1j * tau[t] * Eb) * np.sinc(tau[t] * dE / (2 * np.pi))
            M += gam[t].conj()[:, None] * F
        M *= c[None]
        N = V @ M.T @ V.conj().T
        gth = 2 * np.real(np.einsum("ks,ks->k", N[np.arange(self.D)[None], self.new], self.phase))
        if not grad_psi:
            return float((res ** 2).mean()), gth
        h = V @ (np.exp(1j * np.outer(tau, E)) * gam).sum(0)  # sum_t U_t^H g_t
        return float((res ** 2).mean()), gth, 2 * h.real, 2 * h.imag


# ---- stage 1: Ehrenfest least squares -----------------------------------------------------
def ehrenfest_system(node, edge, dt):
    """Linear system  d<sigma_i^c>/dt = A theta  from central differences; returns A, b."""
    n = node.shape[1]
    iu, ju = np.triu_indices(n, 1)
    pair = -np.ones((n, n), int)
    pair[iu, ju] = np.arange(len(iu))
    pair[ju, iu] = pair[iu, ju]  # symmetric table: the i > j view uses the same column
    K = 3 * n + 9 * len(iu)
    mid_n, mid_e = node[1:-1], edge[1:-1]
    d = (node[2:] - node[:-2]) / (2 * dt)
    T = len(mid_n)
    A = np.zeros((T, n, 3, K))
    for i in range(n):
        for c in range(3):
            for a in range(3):
                for dd in range(3):
                    e = EPS[c, a, dd]
                    if e == 0:
                        continue
                    A[:, i, c, 3 * i + a] += 2 * e * mid_n[:, i, dd]
                    for j in range(n):
                        if j == i:
                            continue
                        for b in range(3):
                            col = 3 * n + 9 * pair[i, j] + (3 * a + b if i < j else 3 * b + a)
                            A[:, i, c, col] += 2 * e * mid_e[:, i, j, 3 * dd + b]
    return A.reshape(-1, K), d.reshape(-1)


def estimate_2local(node, edge, dt, lam=2e-3, iters=2000):
    """Group lasso (one group = the 9 couplings of a pair, fields unpenalised) by proximal gradient."""
    n = node.shape[1]
    A, b = ehrenfest_system(node.astype(np.float64), edge.astype(np.float64), dt)
    N = len(b)
    step = N / np.linalg.norm(A, 2) ** 2
    x = np.zeros(A.shape[1]); y = x.copy(); tk = 1.0
    for _ in range(iters):
        z = y - step * A.T @ (A @ y - b) / N
        zp = z[3 * n:].reshape(-1, 9)
        nr = np.linalg.norm(zp, axis=1, keepdims=True)
        z[3 * n:] = (zp * np.maximum(1 - step * lam / np.maximum(nr, 1e-12), 0)).ravel()
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        x_new = z
        y = x_new + (tk - 1) / t_new * (x_new - x)
        x, tk = x_new, t_new
    return x


def product_state_from_o0(node0):
    """Pure product state whose Bloch vectors are the (renormalised) rows of node0 (n, 3)."""
    psi = np.ones(1, np.complex128)
    for x, y, z in node0:
        th, ph = np.arccos(np.clip(z / max(np.sqrt(x * x + y * y + z * z), 1e-12), -1, 1)), np.arctan2(y, x)
        psi = np.kron(psi, np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)]))
    return psi


# ---- stage 2: trajectory matching ---------------------------------------------------------
def fit(node, edge, dt, fit_psi: bool, iters=None, lam=2e-3):
    from eval.baselines import flatten_obs
    n = node.shape[1]
    obs = flatten_obs(node.astype(np.float64), edge.astype(np.float64))
    fr = Fitter(n, dt)
    theta = estimate_2local(node, edge, dt, lam)
    psi = product_state_from_o0(node[0].astype(np.float64))
    K, D = fr.K, fr.D
    if fit_psi:
        x0 = np.concatenate([theta, psi.real, psi.imag])

        def f(x):
            xs = x[K:K + D] + 1j * x[K + D:]
            nx = np.linalg.norm(xs)
            Lv, gth, gr, gi = fr.loss_grad(x[:K], xs / nx, obs)
            h = (gr + 1j * gi) / 2                                   # = h of the derivation
            h = (h - np.real(np.vdot(xs / nx, h)) * xs / nx) / nx
            return Lv, np.concatenate([gth, 2 * h.real, 2 * h.imag])
        r = scipy.optimize.minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=iters or 300))
        xs = r.x[K:K + D] + 1j * r.x[K + D:]
        return r.x[:K], xs / np.linalg.norm(xs), r.fun
    r = scipy.optimize.minimize(lambda th: fr.loss_grad(th, psi, obs, grad_psi=False), theta, jac=True,
                                method="L-BFGS-B", options=dict(maxiter=iters or 80))
    return r.x, psi, r.fun


def forecast(theta, psi0, n, dt, T, H):
    Hm = sp.csr_matrix(Fitter(n, dt).hamiltonian(theta))
    Psi = physics.evolve(Hm, psi0, dt, T + H + 1)[T + 1:]
    return physics.observables(Psi, n)


def _hl_fair(node, edge, H, attrs, t_abs, fit_psi):
    t0 = t_abs - (len(node) - 1)
    if not fit_psi and (str(attrs.get("init", "product")) != "product" or t0 != 0):
        raise ValueError("hl_fair_o0 needs pure product initial states and a window starting at t=0; use hl_fair_fit")
    dt = float(attrs["dt"])
    theta, psi, _ = fit(node, edge, dt, fit_psi)
    pn, pe = forecast(theta, psi, node.shape[1], dt, len(node) - 1, H)
    return pn.astype(np.float32), pe.astype(np.float32)


def hl_fair_o0(node, edge, H, attrs=None, t_abs=None, **_):
    return _hl_fair(node, edge, H, attrs, t_abs, False)


def hl_fair_fit(node, edge, H, attrs=None, t_abs=None, **_):
    return _hl_fair(node, edge, H, attrs, t_abs, True)
