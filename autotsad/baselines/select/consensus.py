from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from autotsad.rra.kemeny_young import kemeny_young_iterative
from autotsad.rra.rra import robust_rank_aggregation_impl
from autotsad.system.execution.gaussian_scaler import GaussianScaler
from .base_algorithms import scores_to_ranks, ranks_to_scores
from .jings_method import unify_em_jings


def inverse_ranking(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i_ranks = 1 / ranks
    i_ranks = np.mean(i_ranks, axis=0)
    scores = i_ranks.copy()
    ranks = rankdata(1 - i_ranks, method="max")
    return ranks, scores


def kemeny_young(ranks: np.ndarray, approx: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    ranks, scores = kemeny_young_iterative(ranks.T, approx=approx)
    return ranks, scores


def robust_rank_aggregation(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return robust_rank_aggregation_impl(ranks)


def mixture_modeling(scores: np.ndarray, aggregation, cut_off: int = 20, iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    scores = unify_em_jings(scores, cut_off, iterations, n_jobs=1)
    scores = aggregation(scores, axis=0)
    ranks = scores_to_ranks(1 - scores)
    return ranks, scores


def unification(scores: np.ndarray, aggregation) -> Tuple[np.ndarray, np.ndarray]:
    scores = GaussianScaler().fit_transform(scores)
    scores = aggregation(scores, axis=0)
    ranks = scores_to_ranks(1 - scores)
    return ranks, scores


if __name__ == '__main__':
    # ranks = np.array([[3, 1, 4, 2, 2], [1, 2, 3, 4, 3], [1, 2, 4, 3, 5]])
    debug_names = np.array(["Memphis", "Nashville", "Knoxville", "Chattanooga"])
    ranks = np.vstack([
        np.tile([1, 2, 4, 3], (42, 1)),  # Memphis
        np.tile([4, 1, 3, 2], (26, 1)),  # Nashville
        np.tile([4, 3, 2, 1], (15, 1)),  # Chattanooga
        np.tile([4, 3, 1, 2], (17, 1)),  # Knoxville
    ])
    scores = ranks_to_scores(ranks)
    print("Ranks:", ranks.shape)
    print("Ranks:", ranks)

    def _get_ranking(ranks: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({"rank": ranks, "name": debug_names})
        return df.sort_values(by="rank")["name"].values

    # final_ranks, final_scores = inverse_ranking(ranks)
    final_ranks, final_scores = kemeny_young(ranks, approx=False)
    # final_ranks, final_scores, _, _ = robust_rank_aggregation(ranks)
    # final_ranks, final_scores = mixture_modeling(scores, aggregation=np.max)
    # final_ranks, final_scores = unification(scores, aggregation=np.mean)
    print("FinalRank:", final_ranks)
    print("Scores:", final_scores)

    print("                   ", debug_names)
    print("First ranking:     ", ranks[0, :])
    print("Aggregated:        ", final_ranks)
    print("First ranking:", _get_ranking(ranks[0, :]))
    print("Aggregated ranking:", _get_ranking(final_ranks))
