"""
Consistency diagnostics for pairwise comparison matrices.

Two measures, both computed from the matrix alone (no ground truth needed):

  CR         - Saaty's consistency ratio. CARDINAL. Checks whether the
               MAGNITUDES cohere. CR <= 0.10 is the conventional threshold.
  cycle_rate - fraction of triples containing a preference loop. ORDINAL.
               Checks only whether the ORDERING contradicts itself.

The project's core claim is that CR catches incoherence that cycle_rate
structurally cannot see.
"""

import itertools
import numpy as np

# Saaty's Random Index: the average CI of randomly generated reciprocal
# matrices of size n. Used to normalise CI so CR is comparable across sizes.
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}


def priority_vector(A):
    """Priority weights = principal eigenvector, normalised to sum to 1."""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = int(np.argmax(eigenvalues.real))
    w = np.abs(eigenvectors[:, idx].real)
    return w / w.sum()


def consistency_ratio(A):
    """
    Saaty's CR.

      lambda_max = principal eigenvalue
      CI         = (lambda_max - n) / (n - 1)
      CR         = CI / RI(n)

    Returns (CR, lambda_max, CI). CR is 0.0 for n <= 2 (always consistent).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n <= 2:
        return 0.0, float(n), 0.0

    eigenvalues = np.linalg.eigvals(A)
    lambda_max = float(np.max(eigenvalues.real))

    ci = (lambda_max - n) / (n - 1)
    ri = RANDOM_INDEX.get(n)
    if ri is None or ri == 0:
        return 0.0, lambda_max, ci
    return ci / ri, lambda_max, ci


def cycle_rate(A):
    """
    Fraction of 3-element subsets whose ordinal preferences form a loop
    (i beats j beats k beats i).

    Only the DIRECTION of each comparison is used: i beats j iff A[i][j] > 1.
    This is the ordinal check the existing literature uses.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n < 3:
        return 0.0

    beats = A > 1.0
    triples = list(itertools.combinations(range(n), 3))
    cycles = 0
    for i, j, k in triples:
        forward = beats[i, j] and beats[j, k] and beats[k, i]
        backward = beats[j, i] and beats[k, j] and beats[i, k]
        if forward or backward:
            cycles += 1
    return cycles / len(triples)


def diagnose(A):
    """Both measures plus the gate verdict, as a dict."""
    cr, lam, ci = consistency_ratio(A)
    return {
        "n": int(np.asarray(A).shape[0]),
        "lambda_max": round(lam, 4),
        "CI": round(ci, 4),
        "CR": round(cr, 4),
        "cycle_rate": round(cycle_rate(A), 4),
        "cardinally_coherent": bool(cr <= 0.10),
        "ordinally_coherent": bool(cycle_rate(A) == 0.0),
    }


# ---------------------------------------------------------------- matrix build

SAATY_SCALE = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2,
               1, 2, 3, 4, 5, 6, 7, 8, 9]


def build_matrix(n, comparisons):
    """
    Assemble a reciprocal matrix from elicited pairwise values.

    comparisons: dict {(i, j): value} for i < j, where value is
    "how many times more important is i than j" on Saaty's 1-9 scale
    (or a reciprocal like 1/3 if j is more important).

    Diagonal is 1; A[j][i] is forced to 1/A[i][j].
    """
    A = np.ones((n, n), dtype=float)
    for (i, j), v in comparisons.items():
        A[i, j] = v
        A[j, i] = 1.0 / v
    return A


def pairs(n):
    """Every (i, j) pair with i < j - the comparisons you need to elicit."""
    return list(itertools.combinations(range(n), 2))
