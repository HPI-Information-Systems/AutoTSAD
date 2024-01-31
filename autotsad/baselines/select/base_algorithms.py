from typing import Tuple

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from autotsad.config import ALGORITHMS
from autotsad.dataset import TestDataset
from autotsad.tsad_algorithms.interop import exec_algo, params_default


def execute_base_algorithms(dataset: TestDataset, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    algorithms = ALGORITHMS
    jobs = [(algorithm, params_default(algorithm)) for algorithm in algorithms]
    with tqdm_joblib(tqdm(desc="Running base algorithms on dataset", total=len(jobs))):
        scores = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(exec_algo)(dataset, algorithm, algo_params)
            for algorithm, algo_params in jobs
        )
    scores = np.vstack(scores)
    ranks = scores_to_ranks(scores, invert_order=True)
    return scores, ranks


def scores_to_ranks(scores: np.ndarray, invert_order: bool = False) -> np.ndarray:
    """Converts scores to ranks.

    If the input consists of anomaly scores, the ranks should be inverted so that higher scores get lower ranks. Use
    ``invert_order=True`` for that. E.g.:

    >>> scores = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.1]]).T
    >>> scores_to_ranks(scores, invert_order=True)
    array([[3, 2, 1],
           [2, 1, 3]])

    Parameters
    ----------
    scores : np.ndarray
        The scores to convert to ranks. Each row represents the scores for one instance. If a 1-dim. array is passed,
        the ranks are computed for that array.
    invert_order : bool
        Whether to invert the order of the ranks so that highest score gets rank 1. Defaults to False so that lowest
        score gets rank 1.

    Returns
    -------
    np.ndarray
        The ranks for each score. Each row represents the ranks for one instance.
    """
    ranks = rankdata(scores, method="max", axis=1 if scores.ndim > 1 else None)
    ranks = ranks.astype(np.int_)
    if invert_order:
        ranks = ranks.max() - ranks + 1
    return ranks


def ranks_to_scores(ranks: np.ndarray) -> np.ndarray:
    """Converts ranks to scores.

    Parameters
    ----------
    ranks : np.ndarray
        The ranks to convert to scores. Each row represents the ranks for one instance. If a 1-dim. array is passed,
        the scores are computed for that array.

    Returns
    -------
    np.ndarray
        The scores for each rank. Each row represents the scores for one instance.
    """
    scores = 1. / ranks
    # scores = ranks.max() - ranks
    # scores = scores.astype(np.float_)
    # if scores.ndim == 1:
    #     scores = scores.reshape(-1, 1)
    # scores = MinMaxScaler().fit_transform(scores)
    # if scores.ndim == 1:
    #     scores = scores.ravel()
    return scores
