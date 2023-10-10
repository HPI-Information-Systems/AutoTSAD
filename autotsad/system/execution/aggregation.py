from pathlib import Path
from typing import Sequence, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from timeeval.metrics.thresholding import ThresholdingStrategy, SigmaThresholding
from timeeval.utils.hash_dict import hash_dict

from autotsad.config import config
from autotsad.system.execution.gaussian_scaler import GaussianScaler


def load_scores(algorithm_instances: Sequence[str],
                dataset_id: str = config.general.cache_key,
                base_scores_path: Path = config.general.tmp_path / "scores",
                normalization_method: Optional[str] = config.general.score_normalization_method,
                ensure_finite: bool = True) -> pd.DataFrame:
    scores = []
    instances = list(algorithm_instances)
    for algo_instance in instances:
        path = base_scores_path / f"{dataset_id}-{algo_instance}.csv"
        scores.append(np.genfromtxt(path, delimiter=","))
    scores = np.asarray(scores).T

    if ensure_finite:  # before normalization (if algorithm did something strange)
        scores, instances = ensure_finite_scores(scores, instances)

    if normalization_method is not None:
        scores = normalize_scores(scores, normalization_method)

    if ensure_finite:  # after normalization (if normalization did something strange)
        scores, instances = ensure_finite_scores(scores, instances)

    return pd.DataFrame(scores, columns=instances, dtype=np.float_)


def ensure_finite_scores(scores: np.ndarray, instances: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    # filter out instances that have non-finite results
    filter_mask = np.all(np.isfinite(scores), axis=0)
    scores = scores[:, filter_mask]
    instances = np.asarray(instances, dtype=str)[filter_mask]
    return scores, instances


def normalize_scores(scores: np.ndarray,
                     normalization_method: str = config.general.score_normalization_method) -> np.ndarray:
    if normalization_method == "minmax":
        return MinMaxScaler().fit_transform(scores)
    elif normalization_method == "gaussian":
        return GaussianScaler().fit_transform(scores)
    else:
        raise ValueError(f"Unknown score normalization method '{normalization_method}'!")


def aggregate_scores(scores: np.ndarray, agg_method: str = config.general.score_aggregation_method) -> np.ndarray:
    if agg_method == "max":
        return np.max(scores, axis=1)
    elif agg_method == "mean":
        return np.mean(scores, axis=1)
    elif agg_method == "custom":
        return _custom_score_aggregation(scores)
    else:
        raise ValueError(f"Unknown score aggregation method '{agg_method}'!")


def algorithm_instances(df_results: pd.DataFrame) -> pd.Series:
    return df_results["algorithm"] + "-" + df_results["params"].apply(hash_dict)


def _custom_score_aggregation(scores: np.ndarray,
                              thresholding: ThresholdingStrategy = SigmaThresholding(2)) -> np.ndarray:
    combined_score = np.empty_like(scores, dtype=np.float_)
    for i in range(scores.shape[1]):
        s = scores[:, i]
        s = MinMaxScaler().fit_transform(s.reshape(-1, 1)).ravel()
        predictions = thresholding.fit_transform(np.zeros_like(s, dtype=np.int_), s).astype(np.bool_)
        combined_score[:, i] = s.copy() - thresholding.threshold
        combined_score[~predictions, i] = 0

    combined_score = np.nanmax(combined_score, axis=1)
    return combined_score
