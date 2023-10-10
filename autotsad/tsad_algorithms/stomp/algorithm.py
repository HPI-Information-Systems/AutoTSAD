#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from optuna import Trial
from stumpy import stump
from timeeval.utils.window import ReverseWindowing

from ..util import random_seed


supports_sliding_window_view = False
supports_tumbling_window_view = False
supports_nan_view = True


@dataclass
class CustomParameters:
    anomaly_window_size: int = 10
    n_jobs: int = 1
    random_state: int = 42

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            anomaly_window_size=trial.suggest_int("anomaly_window_size", 10, 1000),
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            anomaly_window_size=period_size if period_size > 1 else 100,
        )


def main(data: np.ndarray,
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True) -> np.ndarray:
    with random_seed(params.random_state):
        scores = stump(data, m=params.anomaly_window_size)[:, 0].astype(np.float_)

    if postprocess:
        window_size = params.anomaly_window_size
        if window_size < 4:
            print("WARN: window_size must be at least 4. Dynamically fixing it by setting window_size to 4")
            window_size = 4
        scores = ReverseWindowing(window_size).fit_transform(scores)
    return scores
