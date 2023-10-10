#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from optuna import Trial
from pyod.models.lof import LOF

from ..util import random_seed, default_reverse_windowing

supports_sliding_window_view = True
supports_tumbling_window_view = False
supports_nan_view = False


@dataclass
class CustomParameters:
    window_size: int = 10
    n_neighbors: int = 1
    leaf_size: int = 10
    distance_metric_order: int = 1
    n_jobs: int = 1
    algorithm: str = "auto"  # using default is fine
    distance_metric: str = "minkowski"  # using default is fine
    random_state: int = 42
    use_column_index: int = 0

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            window_size=trial.suggest_int("window_size", 10, 1000),
            n_neighbors=trial.suggest_int("n_neighbors", 1, 100),
            leaf_size=trial.suggest_int("leaf_size", 10, 100),
            distance_metric_order=trial.suggest_int("distance_metric_order", 1, 4),
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            window_size=period_size if period_size > 1 else 100,
            n_neighbors=50,
            leaf_size=20,
            distance_metric_order=2
        )


def main(data: np.ndarray,
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True) -> np.ndarray:
    # Use smallest positive float as contamination
    contamination = np.finfo(np.float_).eps

    # preprocess data
    if not cut_mode:
        data = sliding_window_view(data, window_shape=params.window_size)

    with random_seed(params.random_state):
        clf = LOF(
            contamination=contamination,
            n_neighbors=params.n_neighbors,
            leaf_size=params.leaf_size,
            n_jobs=params.n_jobs,
            algorithm=params.algorithm,
            metric=params.distance_metric,
            metric_params=None,
            p=params.distance_metric_order,
        )
        clf.fit(data)
        scores = clf.decision_scores_

    if postprocess:
        scores = default_reverse_windowing(scores, params)
    return scores
