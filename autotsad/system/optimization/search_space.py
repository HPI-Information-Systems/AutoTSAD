from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Iterator, Any, Union, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from optuna.storages import BaseStorage

from ...config import config
from ...dataset import Dataset


@dataclass(init=True, frozen=False, repr=True)
class SearchSpaceState:
    algorithm: str
    dataset: str
    status: str = "active"
    n_trials_completed: int = 0
    n_trials_desired: int = 0
    best_trial_id: int = 0
    best_trial_runtime: int = 0  # FIXME: record runtime in seconds
    best_score: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    seed_params: List[Dict[str, Any]] = field(default_factory=list)
    proxy: Optional[str] = None

    @property
    def study_name(self) -> str:
        return f"{self.algorithm}-{self.dataset}"

    @property
    def active(self) -> bool:
        return self.status == "active"

    def max_trials_reached(self) -> None:
        self._stop("max_trials")

    def max_quality_reached(self) -> None:
        self._stop("max_quality")

    def algo_pruned(self) -> None:
        self._stop("algo_pruned")

    def dataset_pruned(self, proxy_dataset: str) -> None:
        self._stop("dataset_pruned")
        self.proxy = proxy_dataset

    def terminate(self) -> None:
        self._stop("terminated")

    def _stop(self, reason: str) -> None:
        if reason not in ("max_trials", "max_quality", "terminated", "algo_pruned", "dataset_pruned", "error"):
            raise ValueError(f"Reason '{reason}' is invalid!")
        self.status = reason

    @staticmethod
    def default_study_tags(algorithm: str, dataset: str) -> Dict[str, Any]:
        return {"algorithm": algorithm, "dataset": dataset, "status": "active"}


@dataclass
class ProxiedState:
    algorithm: str
    dataset: str
    proxy: SearchSpaceState


class SearchSpace(Mapping):

    def __init__(self, algorithms: Sequence[str], datasets: Sequence[str]):
        algorithms = np.array(sorted(algorithms), dtype=np.str_)
        datasets = np.array(sorted(datasets), dtype=np.str_)

        self._log = logging.getLogger(f"autotsad.optimization.{self.__class__.__name__}")
        self._data = pd.DataFrame(index=algorithms, columns=datasets)

        for algo in algorithms:
            for dataset in datasets:
                self._data.loc[algo, dataset] = SearchSpaceState(
                    algorithm=algo,
                    dataset=dataset
                )

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def algorithms(self) -> Sequence[str]:
        return self._data.index.values

    @property
    def datasets(self) -> Sequence[str]:
        return self._data.columns.values

    def __getitem__(self, indexer: Any) -> Union[SearchSpaceState, Sequence[SearchSpaceState]]:
        # allow passing a dataset-instance instead of just the dataset name
        if isinstance(indexer, Dataset):
            indexer = indexer.name

        if isinstance(indexer, str):
            if indexer in self.algorithms:
                return self._data.loc[indexer, :].values
            elif indexer in self.datasets:
                return self._data[indexer].values
            else:
                raise KeyError(f"{indexer} is not a valid key!")

        elif isinstance(indexer, tuple) and len(indexer) == 2:
            result = self._data.loc[indexer]
            if isinstance(result, (pd.DataFrame, pd.Series)):
                result = result.values
            return result

        else:
            raise KeyError(f"{indexer} is not a supported index!")

    def __len__(self) -> int:
        return sum(self.shape)

    def __iter__(self) -> Iterator[SearchSpaceState]:
        for i, s in self._data.iterrows():
            for c in s:
                yield c

    def __repr__(self) -> str:
        return f"SearchSpace\n----------\n{self._data}"

    def __str__(self) -> str:
        return self.__repr__()

    def get_proxies(self) -> List[ProxiedState]:
        states = []
        for s in self:
            if s.proxy is not None:
                proxy = self[s.algorithm, s.proxy]
                states.append(ProxiedState(
                    algorithm=s.algorithm,
                    dataset=s.dataset,
                    proxy=proxy,
                ))
        return states

    def sync_from_optuna(self, storage: BaseStorage) -> None:
        import optuna

        self._log.info("Reading existing state from Optuna storage")
        for study in optuna.get_all_study_summaries(storage=storage):
            algo = study.user_attrs.get("algorithm", "")
            dataset = study.user_attrs.get("dataset", "")
            if study.n_trials < 1 or study.best_trial is None:
                self._log.debug(f"Study {study.study_name} was not run, skipping sync.")
                continue

            try:
                state = self[algo, dataset]
            except KeyError:
                self._log.warning(f"Study {study.study_name} is not part of the current search space, skipping sync.")
                continue

            state.n_trials_completed = study.n_trials
            state.best_trial_id = study.best_trial._trial_id
            state.best_score = study.best_trial.value
            state.best_params = study.best_trial.params

            if "status" in study.user_attrs:
                state.status = str(study.user_attrs["status"])

            if state.n_trials_completed >= config.optimization.max_trails_per_study:
                state.max_trials_reached()

    def sync_to_optuna(self, storage: BaseStorage) -> None:
        self._log.info("Writing progress to Optuna storage")
        for state in self:
            try:
                study_id = storage.get_study_id_from_name(f"{state.study_name}_study")
                storage.set_study_user_attr(study_id, "algorithm", state.algorithm)
                storage.set_study_user_attr(study_id, "dataset", state.dataset)
                storage.set_study_user_attr(study_id, "status", state.status)
            except KeyError:
                self._log.debug(f"Study {state.study_name} does not exist in storage, skipping.")
                pass

    def save(self, filename: Path) -> None:
        data = self._data
        data = data.applymap(lambda x: json.dumps(x.__dict__))
        data.to_csv(filename, index=True, header=True)

    @staticmethod
    def load(filename: Path) -> SearchSpace:
        ss = SearchSpace([], [])
        data = pd.read_csv(filename, index_col=0, header=0)
        data = data.applymap(lambda x: SearchSpaceState(**json.loads(x)))
        ss._data = data
        return ss


if __name__ == '__main__':
    """Test cases for the SearchSpace class"""
    ss = SearchSpace(["sub-lof", "kmeans"], ["d1", "d2", "d3", "d4"])

    print(ss.algorithms)
    print(ss.datasets)

    print(ss.shape, ss._data.shape)
    assert ss.shape == ss._data.shape == (2, 4)
    print("data")
    print(ss._data)

    algorithm = "kmeans"
    print(f"State for algo {algorithm}")
    print(ss[algorithm])
    assert len(ss[algorithm]) == len(ss.datasets) == ss.shape[1]

    dataset = "d2"
    print(f"State for dataset {dataset}")
    print(ss[dataset])
    assert len(ss[dataset]) == len(ss.algorithms) == ss.shape[0]

    print(f"State for {algorithm} & {dataset}")
    t = algorithm, dataset
    print(ss[t])
    assert ss[algorithm, dataset].study_name == f"{algorithm}-{dataset}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "search_space.csv"
        ss.save(tmp_file)
        ss2 = SearchSpace.load(tmp_file)
    print(ss2._data)
    pd.testing.assert_frame_equal(ss._data, ss2._data)

    print("\n")
    print("duplicated test")
    algorithm: str = "kmeans"
    name: str
    ss._data.columns = ["d1", "d2", "d4", "d4"]
    print(ss._data.loc[algorithm, "d4"])
    tmp = ss[algorithm, "d4"]
    print(tmp)
    print(type(tmp))

    try:
        tmp = ss["unknown", "d2"]
        assert False, "Expected KeyError"
    except KeyError:
        assert True
    try:
        tmp = ss["kmeans", "unknown"]
        assert False, "Expected KeyError"
    except KeyError:
        assert True
    try:
        tmp = ss["kmeans",]
        assert False, "Expected KeyError"
    except KeyError:
        assert True
