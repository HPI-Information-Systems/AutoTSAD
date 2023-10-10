# Copied and modified from:
# https://github.com/mononitogoswami/tsad-model-selection/blob/master/src/tsadams/model_selection/rank_aggregation.py
# Modifications:
# - Removed 'kemeny' and all related functions
# - Removed 'pagerank' as supported metric in influence-calculation
# - Removed 'weights' parameter from all functions
# - Fixed and added type hints

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##########################################
# Functions for rank aggregation
##########################################

from itertools import combinations
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from .mallows_kendall import borda_partial, distance, median


##########################################
# Trimmed Rank Aggregators
##########################################
def trimmed_partial_borda(ranks: np.ndarray,
                          top_k: Optional[int] = None,
                          top_kr: Optional[int] = None,
                          aggregation_type: str = "borda",
                          metric: str = "influence",
                          n_neighbors: int = 6) -> np.ndarray:
    """Computes the trimmed borda rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks

    top_k: int
        How many items to consider for partial rank aggregation.
        By default top_k=None.

    top_kr: int
        How many permutations to use for rank aggregation.
        By default, top_kr=None. If top is None, then use
        agglomerative clustering.

    aggregation_type: int
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems.

    metric: str
        Metric of rank reliability. By default, metric='influence'.

    n_neighbors: int
        Number of neighbours to use for proximity based reliability
    """
    reliability = _get_reliability(ranks=ranks,
                                   metric=metric,
                                   aggregation_type=aggregation_type,
                                   top_k=top_k,
                                   n_neighbors=n_neighbors)

    if top_kr is None:
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1 * reliability)[:top_kr], :]

    if top_k is not None:
        return partial_borda(ranks=trimmed_ranks, top_k=top_k)
    else:
        return partial_borda(ranks=trimmed_ranks)


def trimmed_borda(ranks: np.ndarray,
                  top_k: Optional[int] = None,
                  top_kr: Optional[int] = None,
                  aggregation_type: str = "borda",
                  metric: str = "influence",
                  n_neighbors: int = 6) -> np.ndarray:
    """Computes the trimmed borda rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks

    top_k: int
        How many items to consider for partial rank aggregation.
        By default top_k=None.

    top_kr: int
        How many permutations to use for rank aggregation.
        By default top_kr=None. If top is None, then use
        agglomerative clustering.

    aggregation_type: int
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems.

    metric: str
        Metric of rank reliablity. By default metric='influence'.

    n_neighbors: int
        Number of neighbours to use for proximity based reliability
    """
    reliability = _get_reliability(ranks=ranks,
                                   metric=metric,
                                   aggregation_type=aggregation_type,
                                   top_k=top_k,
                                   n_neighbors=n_neighbors)

    if top_kr is None:
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1 * reliability)[:top_kr], :]
    return borda(ranks=trimmed_ranks)


##########################################
# Rank Aggregators
##########################################
def partial_borda(ranks: np.ndarray, top_k: int = 5) -> np.ndarray:
    # Top-k Borda Rank Aggregation

    ranks = ranks.astype(float)
    ranks = np.nan_to_num(x=ranks,
                          nan=ranks.shape[1] + 1)  # If ranks already have NaNs
    # Mask higher ranks
    x, y = np.where((ranks > (top_k - 1)))
    for x_i, y_i in zip(x, y):
        ranks[x_i, y_i] = np.NaN
    aggregated_rank = np.nan_to_num(x=borda_partial(ranks, w=1, k=top_k),
                                    nan=ranks.shape[1] - 1).astype(int)

    return aggregated_rank


def borda(ranks: np.ndarray) -> np.ndarray:
    aggregated_rank = median(ranks)

    return aggregated_rank


def minimum_influence(ranks: np.ndarray,
                      aggregation_type: str = "borda",
                      top_k: Optional[int] = None) -> np.ndarray:
    reliability = influence(ranks, aggregation_type=aggregation_type, top_k=top_k)
    return ranks[np.argmax(reliability), :]


##########################################
# Helper functions
##########################################
def _get_reliability(ranks,
                     metric="influence",
                     aggregation_type="borda",
                     top_k=None,
                     n_neighbors=6):
    if metric == "influence":
        reliability = influence(ranks,
                                aggregation_type=aggregation_type,
                                top_k=top_k)
    elif metric == "proximity":
        reliability = proximity(ranks, n_neighbors=n_neighbors, top_k=top_k)
    elif metric == "averagedistance":
        reliability = averagedistance(ranks, top_k=top_k)
    return reliability


def _get_trimmed_ranks_clustering(ranks, reliability):
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(reliability.reshape((-1, 1)))

    most_reliable_cluster_idx = np.argmax([
        np.sum(reliability[np.where(clustering == 0)[0]]),
        np.sum(reliability[np.where(clustering == 1)[0]])
    ])
    trimmed_ranks = ranks[np.where(clustering == most_reliable_cluster_idx)[0], :]  # <--- NOTE: We used this

    return trimmed_ranks


##########################################
# Functions to compute influence
##########################################
def objective(ranks: np.ndarray, aggregation_type: str = "borda", top_k: Optional[int] = None):
    if aggregation_type == "borda":
        sigma_star = borda(ranks=ranks)
    elif aggregation_type == "partial_borda":
        sigma_star = partial_borda(ranks=ranks, top_k=top_k)
    else:
        raise ValueError(f"Invalid aggregation_type, {aggregation_type} is not supported!")

    return np.mean([distance(r, sigma_star) for r in ranks])


def influence(ranks: np.ndarray, aggregation_type: str = "borda", top_k: Optional[int] = None) -> np.array:
    """Computes the reciprocal influence of each permutation/rank on the objective. Ranks with
    higher influence (and lower reciprocal influence) are more outlying.
    """
    N, n = ranks.shape
    objective_values = []

    if (aggregation_type == "partial_borda") and (top_k is None):
        raise ValueError("top_k must be specified!")

    objective_all = objective(ranks, aggregation_type=aggregation_type, top_k=top_k)  # Objective when using all the permutations

    for i in combinations(np.arange(N), N - 1):
        objective_values.append(
            objective(ranks[i, :], aggregation_type=aggregation_type, top_k=top_k)
        )

    # If removing a permutation results in a higher decrease in the objective
    # then it is more likely to be outlying
    influence = objective_all - np.array(objective_values[::-1])  # Reverse the list
    reliability = -influence

    # influence --
    # +ve -- metric good
    # -ve influence is bad
    # low positive influence or high positive influence?

    return reliability


def proximity(ranks: np.ndarray, n_neighbors: int = 6, top_k: Optional[int] = None) -> np.array:
    """Computes the proximity of each rank to its nearest neighbours. Ranks with higher proximity are more central.
    """
    if top_k is not None:
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k - 1)))
        for x_i, y_i in zip(x, y):
            ranks[x_i, y_i] = np.NaN

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", metric=distance)
    neigh.fit(ranks)

    proximity = 1 / neigh.kneighbors(ranks)[0].mean(axis=1)

    return proximity


def averagedistance(ranks: np.ndarray, top_k: Optional[int] = None) -> np.array:
    """Computes the average distance of each rank to all other ranks.
    Lower average implies that a rank is more reliable.
    """
    if top_k is not None:
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k - 1)))
        for x_i, y_i in zip(x, y):
            ranks[x_i, y_i] = np.NaN

    tol = 1e-6
    averagedist = squareform(pdist(ranks, metric=distance)).mean(axis=1)
    return 1 / (tol + averagedist)
