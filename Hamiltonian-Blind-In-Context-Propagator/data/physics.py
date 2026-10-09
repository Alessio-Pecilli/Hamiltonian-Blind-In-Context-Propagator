"""Exact quantum simulation utilities (spec Sec. 2.1-2.4).

Conventions
-----------
* Site i corresponds to tensor axis i of psi reshaped to (2,)*n, i.e. site 0 is the
  most significant bit of the computational-basis index (C order).
* Pauli index a in {0,1,2} = {x,y,z}. Edge channel c = 3*a + b holds <sigma_i^a sigma_j^b>.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SI = np.eye(2, dtype=np.complex128)
PAULIS = (SX, SY, SZ)
# P9[3a+b] = sigma^a (x) sigma^b, first factor = first site
P9 = np.stack([np.kron(PAULIS[a], PAULIS[b]) for a in range(3) for b in range(3)])

DENSE_MAX_N = 12  # n <= 12: dense propagator; above: Krylov (expm_multiply)


# ----------------------------------------------------------------------------------------
# Hamiltonians as sums of Pauli strings
# ----------------------------------------------------------------------------------------
def pauli_string_sparse(n: int, ops: list[tuple[int, str]], coef: complex = 1.0) -> sp.csr_matrix:
    """Sparse matrix of coef * prod_k sigma^{p_k}_{i_k}. ops = [(site, 'X'|'Y'|'Z'), ...]."""
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    new = idx.copy()
    phase = np.full(dim, coef, dtype=np.complex128)
    for site, p in ops:
        bit = (idx >> (n - 1 - site)) & 1
        sign = 1 - 2 * bit  # (-1)^b
        if p == "X":
            new ^= 1 << (n - 1 - site)
        elif p == "Y":
            new ^= 1 << (n - 1 - site)
            phase *= 1j * sign  # Y|0> = i|1>, Y|1> = -i|0>
        elif p == "Z":
            phase *= sign
        else:
            raise ValueError(p)
    # P|s> = phase(s) |new(s)>  ->  matrix element [new(s), s]
    return sp.csr_matrix((phase, (new, idx)), shape=(dim, dim))


def build_hamiltonian(n: int, terms: list[tuple[float, list[tuple[int, str]]]]) -> sp.csr_matrix:
    H = sp.csr_matrix((1 << n, 1 << n), dtype=np.complex128)
    for c, ops in terms:
        if c != 0.0:
            H = H + pauli_string_sparse(n, ops, c)
    return H.tocsr()


def tfim_terms(J: np.ndarray, h: np.ndarray) -> list:
    """H = - sum_{i<j} J_ij Z_i Z_j - sum_i h_i X_i  (Eq. 3)."""
    n = len(h)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                terms.append((-float(J[i, j]), [(i, "Z"), (j, "Z")]))
    for i in range(n):
        terms.append((-float(h[i]), [(i, "X")]))
    return terms


# ----------------------------------------------------------------------------------------
# States and evolution
# ----------------------------------------------------------------------------------------
def random_product_state(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random single-qubit states, tensored (Sec. 2.2)."""
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)
    theta = np.arccos(cos_t)
    psi = np.ones(1, dtype=np.complex128)
    for i in range(n):
        q = np.array([np.cos(theta[i] / 2), np.exp(1j * phi[i]) * np.sin(theta[i] / 2)])
        psi = np.kron(psi, q)
    return psi


def evolve(H: sp.csr_matrix, psi0: np.ndarray, dt: float, L: int) -> np.ndarray:
    """Return Psi of shape (L, 2^n) with Psi[t] = exp(-i H dt)^t psi0."""
    dim = psi0.shape[0]
    n = int(np.log2(dim))
    out = np.empty((L, dim), dtype=np.complex128)
    if n <= DENSE_MAX_N:
        U = scipy.linalg.expm(-1j * dt * H.toarray())
        psi = psi0.copy()
        out[0] = psi
        for t in range(1, L):
            psi = U @ psi
            out[t] = psi
    else:
        out[:] = spla.expm_multiply(
            -1j * H, psi0, start=0.0, stop=(L - 1) * dt, num=L, endpoint=True
        )
    return out


def energy(H: sp.csr_matrix, Psi: np.ndarray) -> np.ndarray:
    """<H>_t for every row of Psi (L, dim)."""
    HPsi = (H @ Psi.T).T
    return np.real(np.einsum("td,td->t", Psi.conj(), HPsi))


# ----------------------------------------------------------------------------------------
# Observables (Sec. 2.4)
# ----------------------------------------------------------------------------------------
def observables(Psi: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Node (L, n, 3) and edge (L, n, n, 9) expectation values, float64.

    E[t, i, j, 3a+b] = <sigma_i^a sigma_j^b>, E[t,i,i] = 0, E[t,j,i,3b+a] = E[t,i,j,3a+b].
    """
    L = Psi.shape[0]
    T = Psi.reshape((L,) + (2,) * n)
    node = np.zeros((L, n, 3))
    edge = np.zeros((L, n, n, 9))
    for i in range(n):
        A = np.moveaxis(T, i + 1, 1).reshape(L, 2, -1)
        rho = np.einsum("tar,tbr->tab", A, A.conj())
        for a in range(3):
            node[:, i, a] = np.real(np.einsum("tpq,qp->t", rho, PAULIS[a]))
    swap = np.array([3 * b + a for a in range(3) for b in range(3)])  # c=(a,b) -> (b,a)
    for i in range(n):
        for j in range(i + 1, n):
            A = np.moveaxis(T, (i + 1, j + 1), (1, 2)).reshape(L, 4, -1)
            rho = np.einsum("tar,tbr->tab", A, A.conj())
            e = np.real(np.einsum("tpq,cqp->tc", rho, P9))
            edge[:, i, j] = e
            edge[:, j, i] = e[:, swap]
    return node, edge


def observables_single(psi: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    node, edge = observables(psi[None], n)
    return node[0], edge[0]
