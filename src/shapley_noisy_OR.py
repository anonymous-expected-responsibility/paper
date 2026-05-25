"""
Shapley Value Algorithms for the Noisy-OR Cooperative Game
===========================================================

This module provides three implementations of Shapley values for the
Noisy-OR characteristic function:

    v(S) = 1 - product_{j in S} (1 - w_j)

where each player i has weight w_i in [0, 1].

The implementations are:

1. brute_force_shapley_all_mpq
   --------------------------------
   Exact exponential-time reference implementation.

   - Enumerates all coalitions.
   - Uses exact rational arithmetic (gmpy2.mpq).
   - Complexity: O(n * 2^(n-1)).
   - Intended for validation and correctness testing only.

2. noisy_or_shapley_mpq
   --------------------------------
   Exact O(n^2) implementation using elementary symmetric sums.

   - Uses exact rational arithmetic (gmpy2.mpq).
   - Closed-form derivation avoids coalition enumeration.
   - Complexity: O(n^2).
   - Suitable as a mathematically exact reference for benchmarking.

3. noisy_or_shapley_float64
   --------------------------------
   High-performance float64 implementation.

   - Uses NumPy and Numba.
   - O(n^2) time.
   - Optionally enforces conservation (sum(phi) == v(N)).
   - Intended for large-scale experiments and benchmarking.

Design Philosophy
-----------------
- The mpq implementations provide ground-truth exact results.
- The float64 implementation provides scalable performance.
- All algorithms compute the same Shapley values under the
  Noisy-OR model; they differ only in arithmetic precision and speed.

This file contains no benchmarking, plotting, or I/O.
It is a pure algorithmic library module.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Set

from gmpy2 import mpq
import numpy as np
from numba import njit


def brute_force_shapley_all_mpq(weights: List[mpq]) -> List[mpq]:
    """
    Compute exact Shapley values for a Noisy-OR cooperative game using the
    brute-force (exponential-time) definition.

    The game is defined over n players with per-player Noisy-OR parameters
    weights[i] in [0, 1] (exact rationals, gmpy2.mpq).

    For any coalition S subset of {1, ..., n}, the characteristic function is:

        v(S) = 1 - product_{j in S} (1 - w_j)

    where w_j is the Noisy-OR activation probability for player j,
    and v(empty set) = 0.

    The Shapley value for player i is defined as:

        phi_i = sum_{S subset N \\ {i}}
                    [ |S|! (n-|S|-1)! / n! ] *
                    ( v(S union {i}) - v(S) )

    This implementation explicitly enumerates all coalitions S for each player i.
    It is exponential in n and intended only as a mathematically exact
    reference implementation for testing faster algorithms.

    Args:
        weights:
            A non-empty list of gmpy2.mpq values, one per player.
            Each weight must lie in the closed interval [0, 1].
            The list is 0-indexed: weights[i] corresponds to player (i+1)
            in the mathematical definition.

    Returns:
        List[mpq]:
            A list of length n where the i-th entry is the exact
            Shapley value phi_i (as gmpy2.mpq).

    Raises:
        AssertionError:
            If weights is not a non-empty list of mpq,
            or if any weight lies outside [0, 1].

    Complexity:
        Time:  O(n * 2^(n-1)) coalition evaluations
        Space: O(n)

    Notes:
        - Symmetry shortcut:
            If all weights are equal, then all Shapley values are equal and
            can be computed in O(n) time as:

                phi_i = v(N) / n

        - Player indexing:
            Internally coalitions use 1-based labels {1, ..., n}
            to match the Shapley definition, while weights remain 0-based.

    Example:
        >>> from gmpy2 import mpq
        >>> brute_force_shapley_all_mpq([mpq(1, 2), mpq(1, 3)])
        [mpq(5, 12), mpq(1, 4)]
    """

    # ---- Input validation (keep it strict: this is a reference implementation) ----
    if not isinstance(weights, list) or not weights:
        raise AssertionError("weights must be a non-empty list")
    if not all(isinstance(w, mpq) for w in weights):
        raise AssertionError("all weights must be gmpy2.mpq")
    if not all(mpq(0) <= w <= mpq(1) for w in weights):
        raise AssertionError("weights must lie in [0,1]")

    n: int = len(weights)

    # ---- Symmetry shortcut: equal players get equal Shapley values ----
    w0: mpq = weights[0]
    if all(w == w0 for w in weights):
        v_all: mpq = mpq(1) - (mpq(1) - w0) ** n
        share: mpq = v_all / mpq(n)
        return [share] * n

    one: mpq = mpq(1)

    def coalition_value(S: Set[int]) -> mpq:
        """
        v(S) = 1 - ∏_{j∈S} (1 - w_j), with j using 1-based player indices.
        """
        prod: mpq = one
        for j in S:
            prod *= (one - weights[j - 1])  # convert 1-based player id -> 0-based index
        return one - prod

    # Precompute factorials for Shapley coefficients:
    #   coeff(S) = |S|! (n-|S|-1)! / n!
    factorials: List[int] = [math.factorial(k) for k in range(n + 1)]
    denom: int = factorials[n]  # n!

    # Set of players using 1-based labels to mirror the Shapley definition.
    N: Set[int] = set(range(1, n + 1))

    phis: List[mpq] = [mpq(0) for _ in range(n)]

    # ---- Brute-force Shapley computation ----
    for i in range(1, n + 1):
        # Accumulate φ_i over all coalitions S ⊆ N\{i}.
        acc: mpq = mpq(0)

        others = N - {i}
        # Enumerate S by size to compute the factorial-based coefficient cheaply.
        for s_size in range(n):  # |S| = 0..n-1
            for combo in combinations(others, s_size):
                S: Set[int] = set(combo)

                # Shapley weight: |S|!(n-|S|-1)! / n!
                coeff_num: int = factorials[s_size] * factorials[n - s_size - 1]
                coeff: mpq = mpq(coeff_num, denom)

                # Marginal contribution of i to coalition S.
                marginal: mpq = coalition_value(S | {i}) - coalition_value(S)

                acc += coeff * marginal

        phis[i - 1] = acc  # back to 0-based output indexing

    return phis



def noisy_or_shapley_mpq(weights: List[mpq]) -> List[mpq]:
    """
    Compute exact Shapley values for a Noisy-OR cooperative game in O(n^2) time
    using elementary symmetric sums.

    This function implements the closed-form O(n^2) algorithm for the Shapley
    value in a Noisy-OR game:

        v(S) = 1 - product_{j in S} (1 - w_j)

    where each weight w_i lies in [0, 1] and is represented exactly as a
    gmpy2.mpq rational number.

    The algorithm avoids enumerating all coalitions. Instead, it:

    1. Computes the full set of elementary symmetric sums of a_i = (1 - w_i).
    2. For each player i, reconstructs the symmetric sums excluding i.
    3. Applies the closed-form Shapley formula derived from the
       elementary symmetric expansion.

    This implementation is mathematically exact and runs in O(n^2) time,
    making it suitable as a high-precision reference implementation.

    Args:
        weights:
            A non-empty list of gmpy2.mpq values in [0, 1].
            weights[i] corresponds to player i (0-based indexing).

    Returns:
        List[mpq]:
            A list of exact Shapley values (mpq), one per player.

    Raises:
        AssertionError:
            If weights is not a non-empty list of mpq values
            or if any weight lies outside [0, 1].

    Complexity:
        Time:  O(n^2)
        Space: O(n)

    Notes:
        - Symmetry shortcut:
            If all weights are equal, all Shapley values are equal:
                phi_i = v(N) / n
        - All arithmetic is exact rational arithmetic via gmpy2.mpq.
    """

    # ---- Input validation ----
    assert isinstance(weights, list) and len(weights) > 0, \
        "weights must be a non-empty list"
    assert all(isinstance(w, mpq) for w in weights), \
        "all weights must be gmpy2.mpq"
    assert all(mpq(0) <= w <= mpq(1) for w in weights), \
        "weights must lie in [0,1]"

    n: int = len(weights)

    # ---- Symmetry shortcut: identical players ----
    w0: mpq = weights[0]
    if all(w == w0 for w in weights):
        v_all: mpq = mpq(1) - (mpq(1) - w0) ** n
        share: mpq = v_all / mpq(n)
        return [share] * n

    # a_i = 1 - w_i
    a: List[mpq] = [mpq(1) - wi for wi in weights]

    # ------------------------------------------------------------
    # Step 1: Compute full elementary symmetric sums
    #
    # e[k] = sum over all subsets T of size k of product_{j in T} a_j
    #
    # This is computed via the standard dynamic programming recurrence.
    # ------------------------------------------------------------
    e: List[mpq] = [mpq(0)] * (n + 1)
    e[0] = mpq(1)

    for m in range(1, n + 1):
        am: mpq = a[m - 1]
        for k in range(m, 0, -1):
            e[k] = e[k] + am * e[k - 1]

    # Precompute inverse binomial coefficients:
    # inv_combs[k] = 1 / C(n-1, k)
    inv_combs: List[mpq] = [mpq(1, math.comb(n - 1, k)) for k in range(n)]

    inv_n: mpq = mpq(1, n)

    phi: List[mpq] = [mpq(0)] * n

    # ------------------------------------------------------------
    # Step 2+3: For each player i
    #
    # Reconstruct elementary symmetric sums excluding i
    # and compute the closed-form Shapley expression.
    # ------------------------------------------------------------
    for i in range(n):
        ai: mpq = a[i]

        # e_minus_i[k] = elementary symmetric sum of size k
        # excluding element i
        e_minus_i: List[mpq] = [mpq(0)] * n
        e_minus_i[0] = mpq(1)

        # Recurrence:
        # e_minus_i[k] = e[k] - a_i * e_minus_i[k-1]
        for k in range(1, n):
            e_minus_i[k] = e[k] - ai * e_minus_i[k - 1]

        # Compute sum_k e_minus_i[k] / C(n-1, k)
        s: mpq = mpq(0)
        for k in range(n):
            s += e_minus_i[k] * inv_combs[k]

        # Final Shapley value for player i
        phi[i] = weights[i] * inv_n * s

    return phi


@njit(cache=True, fastmath=False)
def _noisy_or_shapley_float64(
    weights: np.ndarray,
    inv_combs: np.ndarray,
) -> np.ndarray:
    """
    Compute Shapley values for a Noisy-OR game in O(n^2) time using float64
    arithmetic (Numba-compiled kernel).

    This is the high-performance numerical analogue of the exact mpq
    implementation. All arithmetic is performed in float64.

    The Noisy-OR game is defined as:

        v(S) = 1 - product_{j in S} (1 - w_j)

    The algorithm:

    1. Computes a_i = 1 - w_i.
    2. Builds the full elementary symmetric sums of a.
    3. Reconstructs symmetric sums excluding each player i.
    4. Applies the closed-form Shapley expression.

    Args:
        weights:
            1D float64 array of shape (n,) with values in [0, 1].
        inv_combs:
            1D float64 array of shape (n,), where:
                inv_combs[k] = 1 / C(n-1, k)

    Returns:
        np.ndarray:
            1D float64 array of Shapley values of length n.

    Raises:
        ValueError:
            If weights is empty,
            if inv_combs has incorrect length,
            or if any weight lies outside [0, 1].

    Complexity:
        Time:  O(n^2)
        Space: O(n)

    Notes:
        - No conservation correction is applied here.
        - Designed to be called from Python wrapper.
        - Numba JIT-compiled for performance.
    """

    n: int = weights.size

    if n <= 0:
        raise ValueError("weights must be non-empty")
    if inv_combs.size != n:
        raise ValueError("inv_combs must have length n")

    # ------------------------------------------------------------
    # a_i = 1 - w_i
    # ------------------------------------------------------------
    a = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = weights[i]
        if w < 0.0 or w > 1.0:
            raise ValueError("weights must lie in [0,1]")
        a[i] = 1.0 - w

    # ------------------------------------------------------------
    # Step 1: Full elementary symmetric sums e[0..n]
    # ------------------------------------------------------------
    e = np.zeros(n + 1, dtype=np.float64)
    e[0] = 1.0

    for m in range(1, n + 1):
        am = a[m - 1]
        for k in range(m, 0, -1):
            e[k] = e[k] + am * e[k - 1]

    inv_n: float = 1.0 / float(n)
    phi = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------
    # Step 2+3: Reconstruct e^{-i} and compute Shapley values
    # ------------------------------------------------------------
    e_minus_i = np.empty(n, dtype=np.float64)

    for i in range(n):
        ai = a[i]

        e_minus_i[0] = 1.0
        for k in range(1, n):
            e_minus_i[k] = e[k] - ai * e_minus_i[k - 1]

        s: float = 0.0
        for k in range(n):
            s += e_minus_i[k] * inv_combs[k]

        phi[i] = weights[i] * inv_n * s

    return phi


@njit(cache=True, fastmath=False)
def _enforce_conservation_last_inplace(
    phi: np.ndarray,
    weights: np.ndarray,
) -> None:
    """
    Enforce efficiency (conservation) in-place for float64 Shapley values.

    The Shapley values must satisfy:

        sum(phi) = v(N)

    where for the Noisy-OR game:

        v(N) = 1 - product_i (1 - w_i)

    Due to floating-point rounding, small deviations may occur.
    This function adjusts the final entry phi[-1] so that the
    equality holds exactly in float arithmetic.

    Args:
        phi:
            1D float64 array of Shapley values (modified in-place).
        weights:
            1D float64 array of Noisy-OR weights.

    Returns:
        None

    Complexity:
        Time:  O(n)
        Space: O(1)

    Notes:
        - Only the last component is adjusted.
        - Intended for post-processing after the O(n^2) kernel.
    """

    n: int = phi.size
    if n == 0:
        return

    # Compute v(N) = 1 - product_i (1 - w_i)
    prod: float = 1.0
    for i in range(weights.size):
        prod *= (1.0 - weights[i])
    v: float = 1.0 - prod

    # Compute current sum of Shapley values
    s: float = 0.0
    for i in range(n):
        s += phi[i]

    # Adjust final component to enforce conservation
    phi[n - 1] = phi[n - 1] + (v - s)


def noisy_or_shapley_float64(
    weights: np.ndarray,
    conserve: bool = True,
) -> np.ndarray:
    """
    Public float64 API for computing Noisy-OR Shapley values.

    This function:
      1. Converts inputs to float64.
      2. Constructs inverse binomial coefficients.
      3. Calls the Numba-compiled O(n^2) kernel.
      4. Optionally enforces conservation.

    Args:
        weights:
            1D array-like of values in [0, 1].
        conserve:
            If True (default), enforce sum(phi) == v(N)
            via in-place correction of the last entry.

    Returns:
        np.ndarray:
            1D float64 array of Shapley values.

    Raises:
        ValueError:
            If weights is not a non-empty 1D array.

    Complexity:
        Time:  O(n^2)
        Space: O(n)

    Notes:
        - This is the high-performance approximation of the exact mpq version.
        - Suitable for benchmarking and large-scale experiments.
    """

    w = np.asarray(weights, dtype=np.float64)

    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1D array")

    n: int = int(w.size)

    # inv_combs[k] = 1 / C(n-1, k)
    inv_combs = np.empty(n, dtype=np.float64)
    for k in range(n):
        inv_combs[k] = 1.0 / float(math.comb(n - 1, k))

    phi = _noisy_or_shapley_float64(w, inv_combs)

    if conserve:
        _enforce_conservation_last_inplace(phi, w)

    return phi
