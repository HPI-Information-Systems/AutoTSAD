from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optuna import Study
    from optuna.trial import FrozenTrial


class EarlyStoppingCallback:
    """Early stopping callback for Optuna.

    This callback stops the optimization when the number of trials that achieve a near perfect score (above or below
    the ``threshold`` depending on the ``direction``) exceed a certain amount (``min_rounds``).

    Parameters
    ----------
    min_rounds : int
        The minimum number of trials that must exceed the threshold before the optimization is stopped.
    threshold : float
        The threshold for the score.
    direction : str, optional
        The direction of the optimization. Either "minimize" or "maximize". If "minimize", values below the threshold
        are considered successful. If "maximize", values above the threshold are considered successful. Default:
        "minimize".
    """

    def __init__(self, min_rounds: int, threshold: float, direction: str = "minimize") -> None:
        self.min_rounds = min_rounds
        self.threshold = threshold
        if direction == "minimize":
            self._operator = operator.lt
        elif direction == "maximize":
            self._operator = operator.gt
        else:
            raise ValueError(f"Direction '{direction}' is invalid!")

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        scores = np.array([t.value for t in study.trials], dtype=np.float_)
        if self.should_stop(scores):
            study.stop()

    def should_stop(self, scores: np.array) -> bool:
        n_successful_trials = np.count_nonzero(scores >= self.threshold)
        return n_successful_trials >= self.min_rounds

    @staticmethod
    def from_config() -> EarlyStoppingCallback:
        from ...config import config
        return EarlyStoppingCallback(
            min_rounds=config.optimization.stop_heuristic_n,
            threshold=config.optimization.stop_heuristic_quality_threshold,
            direction="maximize"
        )
