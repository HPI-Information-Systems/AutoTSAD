from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from optuna import Trial

from .model import KMeansAD

supports_sliding_window_view = True
supports_tumbling_window_view = False
supports_nan_view = False


@dataclass
class CustomParameters:
    n_clusters: int = 2
    window_size: int = 10
    stride: int = 1
    random_state: int = 42

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            n_clusters=trial.suggest_int("n_clusters", 2, 100),
            window_size=trial.suggest_int("window_size", 10, 1000),
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            window_size=100,  # guesstimate, actually: AnomalyLengthHeuristic(agg_type='max')
            n_clusters=50,
        )


def main(data: np.ndarray,
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True) -> np.ndarray:

    # prepare parameters
    params_dict = asdict(params)
    params_dict["k"] = params_dict["n_clusters"]
    del params_dict["n_clusters"]

    detector = KMeansAD(**params_dict)
    detector.fit(data, preprocess=not cut_mode)
    anomaly_scores = detector.predict(data, preprocess=not cut_mode, postprocess=postprocess)

    return anomaly_scores
