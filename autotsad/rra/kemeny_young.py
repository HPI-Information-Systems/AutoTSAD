# Copied and modified from:
# https://github.com/mononitogoswami/tsad-model-selection/blob/master/src/tsadams/model_selection/rank_aggregation.py
# Modifications:
# - Removed all functions not related to 'kemeny'
# - Added kemeny_young_iterative function from the SELECT paper

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import combinations, permutations
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from .mallows_kendall import distance


def kemeny_young_iterative(rank_list: np.ndarray, approx: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Kemeny-Young rank aggregation implementation based on Matlab code from RayanaEtAl2015 [1].

    Parameters
    ----------
    rank_list: np.ndarray
        Permutations/Ranks (candidates x voters)
    approx: bool
        Whether to use the approximation from [1] or the exact algorithm. The exact algorithm is very slow
        O(n!).

    References
    ----------
    [1] Rayana, S. and Akoglu, L. 2016. Less is More: Building Selective Anomaly Ensembles.
        In: ACM Trans. Knowl. Discov. Data 10, 4. DOI: 10.1145/2890508-
    """
    candidates, voters = rank_list.shape
    # determine maximum number of votes for matrix type
    dtype = np.uint8 if voters < 255 else np.uint16
    pref_matrix = np.zeros((candidates, candidates), dtype=dtype)

    for pair in combinations(range(candidates), 2):
        for voter in range(voters):
            idx1 = rank_list[pair[0], voter]
            idx2 = rank_list[pair[1], voter]

            if idx1 < idx2:
                pref_matrix[pair[0], pair[1]] += 1
            else:
                pref_matrix[pair[1], pair[0]] += 1

    if approx:
        # This is just an approximation made in [1]:
        scores = np.sum(pref_matrix, axis=1)
        score_index_df = pd.DataFrame({"score": scores, "index": np.arange(candidates) + 1})
        sorted_df = score_index_df.sort_values(by="score", ascending=False)
        aggregated_rank_list = sorted_df["index"].values

    else:
        # This is the correct way to compute the final ranking:
        best_rank_candidate = ()
        best_rank_score = -1
        for rank_candidate in permutations(range(candidates), candidates):
            score = 0
            for i in range(len(rank_candidate)-1):
                score += np.sum(pref_matrix[rank_candidate[i], rank_candidate[i+1:]])
            if score > best_rank_score:
                best_rank_score = score
                best_rank_candidate = rank_candidate
        aggregated_rank_list = np.argsort(best_rank_candidate) + 1
        scores = aggregated_rank_list.max() - aggregated_rank_list

    return aggregated_rank_list, scores


def kemeny(ranks: np.ndarray,
           weights: Optional[np.ndarray] = None,
           verbose: bool = True) -> Tuple[float, np.ndarray]:
    """Kemeny-Young optimal rank aggregation [1]

    We include the ability to incorporate weights of metrics/permutations.

    Parameters
    ----------
    ranks:
        Permutations/Ranks
    weights:
        Weight of each rank/permutation.
    verbose:
        Controls verbosity

    References
    ----------
    [1] Conitzer, V., Davenport, A., & Kalagnanam, J. (2006, July). Improved bounds for computing Kemeny rankings.
        In AAAI (Vol. 6, pp. 620-626).
        https://www.aaai.org/Papers/AAAI/2006/AAAI06-099.pdf
    [2] http://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html#note4
    """

    _, n_models = ranks.shape

    # Minimize C.T * X
    edge_weights = _build_graph(ranks, weights)
    c = -1 * edge_weights.ravel().reshape((-1, 1))

    # Defining variables
    x = cp.Variable((n_models**2, 1), boolean=True)

    # Defining the objective function
    objective = cp.Maximize(c.T @ x)

    # Defining the constraints
    def idx(i: int, j: int) -> int:
        return n_models * i + j

    # Constraints for every pair
    pairwise_constraints = np.zeros(((n_models * (n_models - 1)) // 2, n_models**2))
    for row, (i, j) in zip(pairwise_constraints, combinations(range(n_models), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros(((n_models * (n_models - 1) * (n_models - 2)), n_models**2))
    for row, (i, j, k) in zip(triangle_constraints,
                              permutations(range(n_models), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

    constraints = [
        pairwise_constraints @ x == np.ones((pairwise_constraints.shape[0], 1)),
        triangle_constraints @ x >= np.ones((triangle_constraints.shape[0], 1))
    ]

    # Solving the problem
    problem = cp.Problem(objective, constraints)

    if verbose:
        print("Is DCP:", problem.is_dcp())
    problem.solve(verbose=verbose, warm_start=True)

    aggregated_rank = x.value.reshape((n_models, n_models)).sum(axis=1)
    objective = np.mean([distance(r, aggregated_rank) for r in ranks])

    return objective, aggregated_rank


def _build_graph(ranks: np.ndarray, metric_weights: Optional[np.ndarray] = None) -> np.ndarray:
    n_metrics, n_models = ranks.shape
    if metric_weights is None:
        metric_weights = np.ones((n_metrics, 1))
    else:
        metric_weights = metric_weights.reshape((-1, 1))
    edge_weights = np.zeros((n_models, n_models))

    for i, j in combinations(range(n_models), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = (metric_weights.T @ (preference < 0).astype(np.int_).reshape((-1, 1))).squeeze()  # prefers i to j
        h_ji = (metric_weights.T @ (preference > 0).astype(np.int_).reshape((-1, 1))).squeeze()  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights
