#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from optuna import Trial

from .dwt_mlead import DWT_MLEAD
from ..util import random_seed, supress_output

supports_sliding_window_view = False
supports_tumbling_window_view = False
supports_nan_view = True


@dataclass
class CustomParameters:
    start_level: int = 1
    quantile_epsilon: float = 0.1
    random_state: int = 42
    use_column_index: int = 0

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            quantile_epsilon=trial.suggest_float("quantile_epsilon", 0.001, 0.5),
            start_level=trial.suggest_int("start_level", 1, 10),
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            quantile_epsilon=0.1,
            start_level=3,
        )

    @staticmethod
    def default() -> CustomParameters:
        return CustomParameters(
            quantile_epsilon=0.01,
            start_level=1,
        )


def main(data: np.ndarray,
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True) -> np.ndarray:

    def call_dwt_mlead(series: np.ndarray) -> np.ndarray:
        with random_seed(params.random_state), supress_output(stderr=True):
            detector = DWT_MLEAD(series,
                                 start_level=params.start_level,
                                 quantile_boundary_type="percentile",
                                 quantile_epsilon=params.quantile_epsilon,
                                 track_coefs=False,  # just used for plotting
                                 )
            scores = detector.detect()

            # print("\n=== Cluster anomalies ===")
            # clusters = detector.find_cluster_anomalies(point_scores, d_max=2.5, anomaly_counter_threshold=2)
            # for c in clusters:
            #     print(f"  Anomaly at {c.center} with score {c.score}")

            # detector.plot(coefs=False, point_anomaly_scores=point_scores)
            return scores

    if cut_mode:
        # split data array on NaNs
        nan_indices = np.nonzero(np.isnan(data))[0]
        # print(f"{data.shape=}")
        # print(f"{nan_indices=}")
        segments = np.split(data, nan_indices)
        segment_scores = []
        for data in segments:
            mask = np.isfinite(data)
            # print(f"{mask.shape=}, {np.nonzero(~mask)[0]=}")
            data = data[mask]
            # print(f"{data.shape=}")
            if data.shape[0] == 0:
                segment_scores.append(np.array([np.nan], dtype=np.float_))
                continue

            point_scores = np.full(len(mask), fill_value=np.nan, dtype=np.float_)
            point_scores[mask] = call_dwt_mlead(data)
            # print(f"\n{point_scores.shape=}\n")
            segment_scores.append(point_scores)

        # concatenate the scores
        scores = np.concatenate(segment_scores)
        # print(f"final {scores.shape=}")
    else:
        scores = call_dwt_mlead(data)

    return scores
