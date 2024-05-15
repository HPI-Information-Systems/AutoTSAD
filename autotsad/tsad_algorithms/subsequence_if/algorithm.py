#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from optuna import Trial
from pyod.models.iforest import IForest

from ..util import random_seed, default_reverse_windowing


supports_sliding_window_view = True
supports_tumbling_window_view = False
supports_nan_view = False


@dataclass
class CustomParameters:
    window_size: int = 10
    n_trees: float = 10
    max_samples: float = 0.01
    max_features: float = 0.01
    bootstrap: bool = False
    random_state: int = 42
    verbose: int = 0
    n_jobs: int = 1

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            window_size=trial.suggest_int("window_size", 10, 1000),
            n_trees=trial.suggest_int("n_trees", 10, 1000),
            max_features=trial.suggest_float("max_features", 0.01, 1.0),
            max_samples=trial.suggest_float("max_samples", 0.01, 1.0),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False])
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            window_size=period_size if period_size > 1 else 100,
            n_trees=500,
            max_features=1.0,
            max_samples=1.0,
            bootstrap=True
        )

    @staticmethod
    def default() -> CustomParameters:
        return CustomParameters(
            window_size=10,
            n_trees=10,
            max_features=0.01,
            max_samples=0.01,
            bootstrap=False
        )


def main(data: np.ndarray,
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True) -> np.ndarray:
    # Use smallest positive float as contamination
    contamination = np.nextafter(0, 1)

    # preprocess data
    if not cut_mode:
        data = sliding_window_view(data, window_shape=params.window_size)

    with random_seed(params.random_state):
        clf = IForest(
            contamination=contamination,
            n_estimators=params.n_trees,
            max_samples=params.max_samples or "auto",
            max_features=params.max_features,
            bootstrap=params.bootstrap,
            random_state=params.random_state,
            verbose=params.verbose,
            n_jobs=params.n_jobs,
        )
        clf.fit(data)
        scores = clf.decision_scores_

    if postprocess:
        scores = default_reverse_windowing(scores, params)

    return scores
