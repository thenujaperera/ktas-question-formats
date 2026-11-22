"""Turn pairwise judgements into a priority vector. Pure functions, no labels."""
from itertools import combinations

import numpy as np

RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}   # Saaty random index


def ahp_eigen(A):
    """Principal right eigenvector of a reciprocal matrix, normalised to sum 1."""
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def consistency_ratio(A):
    n = len(A)
    _, lam = ahp_eigen(A)
    return ((lam - n) / (n - 1)) / RI[n] if n in RI else np.nan


def ahp_from_logodds(L):
    """Log least squares priorities: row means of a skew-symmetric log-ratio matrix (= geometric mean method)."""
    return np.asarray(L).mean(axis=1)


def bradley_terry(W, iters=500, prior=0.1):
    """MM algorithm (Hunter 2004). W[i][j] = (possibly fractional) wins of i over j.

    A small symmetric pseudo-count `prior` per pair keeps strengths finite when a patient wins or loses
    every comparison. Returns log-strengths centred at 0.
    """
    W = np.asarray(W, float).copy()
    n = len(W)
    W = W + prior * (1 - np.eye(n))
    N = W + W.T
    p = np.ones(n)
    for _ in range(iters):
        new = W.sum(1) / np.array([sum(N[i, j] / (p[i] + p[j]) for j in range(n) if j != i) for i in range(n)])
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(new - p)) < 1e-10:
            p = new
            break
        p = new
    return np.log(p)


def flip_rate(signs_fwd, signs_rev):
    """Share of pairs whose verdict (A wins / tie / B wins) changes when the A/B order is swapped."""
    pairs = list(signs_fwd)
    return float(np.mean([signs_fwd[k] != signs_rev[k] for k in pairs])) if pairs else np.nan


__all__ = ["ahp_eigen", "consistency_ratio", "ahp_from_logodds", "bradley_terry", "flip_rate", "combinations"]
