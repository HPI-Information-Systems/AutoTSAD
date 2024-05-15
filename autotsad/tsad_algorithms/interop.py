from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Union, Set, TYPE_CHECKING

import numpy as np

from ..dataset import Dataset, TrainDataset

if TYPE_CHECKING:
    from optuna.trial import BaseTrial

ALGORITHM_MODULES: Set[str] = {"subsequence_lof", "subsequence_knn", "subsequence_if", "stomp", "kmeans", "torsk",
                               "dwt_mlead", "grammarviz"}


def _load_algorithm_module(algo: str) -> Any:
    if algo not in ALGORITHM_MODULES:
        raise KeyError(f"Unknown algorithm {algo}")

    return importlib.import_module(f".{algo}.algorithm", __package__)


def _get_window_size(dataset: TrainDataset, params: Optional[Any] = None) -> int:
    try:
        window_size = params.window_size
    except AttributeError:
        try:
            window_size = params.anomaly_window_size
        except AttributeError:
            try:
                window_size = params.context_window_size
            except AttributeError:
                try:
                    window_size = dataset.period_size
                except AttributeError:
                    window_size = 100
    return window_size


def _sliding_algo(dataset: TrainDataset, module: Any, params: Optional[Any] = None) -> np.ndarray:
    if params is None:
        params = module.CustomParameters.timeeval(period_size=dataset.period_size)

    # pre-compute sliding windows
    window_size = _get_window_size(dataset, params)
    dataset_window_view = dataset.sliding_window_view(window_size=window_size)
    data = dataset_window_view.data

    # call algo
    # print("INPUT", data.shape)
    scores = module.main(data, params=params, cut_mode=True, postprocess=False)
    # print("OUTPUT", scores.shape)

    # reverse windowing based on pre-computed windows
    # print("before custom reverse windowing", scores.shape)
    scores = dataset_window_view.reverse_windowing(scores)
    # print("after custom reverse windowing", scores.shape)
    return scores


def _tumbling_algo(dataset: TrainDataset, module: Any, params: Optional[Any] = None) -> np.ndarray:
    if params is None:
        params = module.CustomParameters.timeeval(period_size=dataset.period_size)

    # pre-compute tumbling windows
    window_size = _get_window_size(dataset, params)
    dataset_window_view = dataset.tumbling_window_view(
        window_size=window_size,
        train_window_size=params.train_window_size,
        prediction_window_size=params.prediction_window_size,
    )
    data = dataset_window_view.data

    # call algo
    # print("INPUT", data.shape)
    scores = module.main(data, params=params, cut_mode=True, postprocess=False)
    # print("OUTPUT", scores.shape)

    # reverse windowing based on pre-computed windows
    # print("before custom reverse windowing", scores.shape)
    scores = dataset_window_view.reverse_windowing(scores)
    # print("after custom reverse windowing", scores.shape)
    return scores


def _nan_separated_algo(dataset: TrainDataset, module: Any, params: Optional[Any] = None) -> np.ndarray:
    if params is None:
        params = module.CustomParameters.timeeval(period_size=dataset.period_size)

    # stomp input: NaN-separated regions
    dataset_view = dataset.nan_separated_view()
    data = dataset_view.data

    # call algo
    # print("INPUT", data.shape, data)
    scores = module.main(data, params=params, cut_mode=True, postprocess=True)
    # print("OUTPUT", scores.shape, scores)

    # reverse windowing based on NaN-separated regions
    # print("before custom reverse windowing", scores.shape)
    scores = dataset_view.reverse_windowing(scores)
    # print("after custom reverse windowing", scores.shape)
    return scores


def exec_algo(dataset: Union[Dataset, np.ndarray],
              algorithm: str,
              params: Optional[Any] = None,
              ignore_cuts: bool = False) -> np.ndarray:
    module = _load_algorithm_module(algorithm)

    if not ignore_cuts:
        if isinstance(dataset, Dataset):
            dataset = dataset.data
        if params:
            return module.main(dataset, params=params, cut_mode=False, postprocess=True)
        else:
            return module.main(dataset, cut_mode=False, postprocess=True)

    if not isinstance(dataset, TrainDataset):
        raise ValueError("ignore_cuts=True is only supported for TrainDatasets.")

    if hasattr(module, "supports_custom_cut_handling") and module.supports_custom_cut_handling:
        if params is None:
            params = module.CustomParameters.timeeval(period_size=dataset.period_size)
        return module.main(dataset, params=params, cut_mode=True, postprocess=True)
    if module.supports_sliding_window_view:
        return _sliding_algo(dataset, module, params)
    elif module.supports_tumbling_window_view:
        return _tumbling_algo(dataset, module, params)
    elif module.supports_nan_view:
        return _nan_separated_algo(dataset, module, params)
    else:
        raise NotImplementedError(f"{algorithm} in cut_mode is not supported in AutoTSAD.")


def params_from_dict(params: Dict[str, Any], algorithm: str) -> Any:
    module = _load_algorithm_module(algorithm)
    return module.CustomParameters(**params)


def params_from_trial(trial: BaseTrial, algorithm: str) -> Any:
    module = _load_algorithm_module(algorithm)
    return module.CustomParameters.from_trial(trial)


def params_default(algorithm: str) -> Any:
    module = _load_algorithm_module(algorithm)
    return module.CustomParameters()


def params_timeeval(algorithm: str, period_size: int) -> Any:
    module = _load_algorithm_module(algorithm)
    return module.CustomParameters.timeeval(int(period_size))


def params_bad_default(algorithm: str) -> Any:
    module = _load_algorithm_module(algorithm)
    return module.CustomParameters.default()
