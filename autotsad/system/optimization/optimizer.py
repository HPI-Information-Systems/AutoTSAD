from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Dict, Union, Callable, List, ContextManager, Type, Optional, Sequence, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .optuna_auto_storage import OptunaStorageManager
from .remote_funcs import study_job, StudyJobResult
from .search_space import SearchSpace, SearchSpaceState
from ..hyperparameters import ParamSetting
from ...config import config
from ...dataset import TrainingDatasetCollection
from ...util import show_full_df


class Optimizer(ContextManager):
    def __init__(self,
                 state_file_path: Path,
                 dataset_collection: TrainingDatasetCollection,
                 continue_from_existing: bool = False):
        self._log = logging.getLogger(f"autotsad.optimization.{self.__class__.__name__}")
        self.state_file_path = state_file_path
        if self.state_file_path.exists():
            self._log.info("Loading optimization state")
            self.state = SearchSpace.load(self.state_file_path)
        else:
            self._log.info("Using empty optimization state")
            self.state = SearchSpace(
                algorithms=list(config.optimization.algorithms),
                datasets=[d.name for d in dataset_collection.training_datasets],
            )
        self.datasets = dataset_collection
        self.storage_mgr = OptunaStorageManager.from_config(config)

        if continue_from_existing:
            self.sync_from_optuna()

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        self.close()
        return False

    def run_optimization_iteration(self, tags: Dict[str, Union[str, int, float]] = {}) -> None:
        # collect desired studies
        pending_studies = []
        for s in self.state:
            if s.active and s.n_trials_desired > s.n_trials_completed:
                pending_studies.append(s)

        # run studies
        n_studies = len(pending_studies)
        self._log.info(f"Running optimization iteration with {n_studies} studies")
        if 0 < self._log.level <= logging.DEBUG:
            for d in set([s.dataset for s in pending_studies]):
                limits = config.general.adjusted_training_limits(self.datasets[d].length)
                if limits['enabled']:
                    self._log.debug(f"{d}: Effective resource limits: "
                                    f"{limits['time_limit']}s, {limits['memory_limit']}mb")
                else:
                    self._log.debug(f"{d}: Effective resource limits: unlimited")
        with tqdm_joblib(tqdm(desc="Optimizing algorithms", total=n_studies, disable=not config.general.progress)):
            results: List[StudyJobResult] = joblib.Parallel(min(config.general.n_jobs, n_studies))(
                joblib.delayed(study_job)(
                    s.algorithm,
                    self.datasets[s.dataset],
                    config.optimization.metric(),
                    n_trials=s.n_trials_desired - s.n_trials_completed,
                    general_config=config.general,
                    optimization_config=config.optimization,
                    tags=tags,
                    seed_params=s.seed_params,
                )
                for s in pending_studies
            )

            # update state
            for r in results:
                s = self.state[r.algorithm, r.dataset]
                s.n_trials_completed = r.n_trials
                s.best_score = r.best_score
                s.best_params = r.best_params
                s.best_trial_id = r.best_trial_id

                self._apply_stop_conditions(s, r)

            self.state.sync_to_optuna(self.storage_mgr.get())

    def update_all_states(self, fn: Callable[[SearchSpace], SearchSpace]) -> None:
        self.state = fn(self.state)

    def update_state(self, fn: Callable[[SearchSpaceState], None]) -> None:
        for s in self.state:
            fn(s)

    def set_seed_params(self, algorithm: str, dataset: str, seed_params: Dict[str, Any]) -> None:
        self.state[algorithm, dataset].seed_params.append(seed_params)

    def prune_worst_algorithm(self) -> None:
        self._log.debug("Pruning worst algorithm for each dataset")
        for d in self.datasets:
            studies = self.state[d.name]
            studies = [s for s in studies if s.active]
            if len(studies) > 1:
                worst_study = min(studies, key=lambda s: s.best_score)
                diffs = [s.best_score - worst_study.best_score for s in studies]
                self._log.info(f"{d.name}: Pruning algo {worst_study.algorithm}; diffs={diffs}")
                worst_study.algo_pruned()
            else:
                self._log.debug(f"{d.name}: Only a single algorithm left; skipping algorithm pruning!")

    def prune_consolidated_datasets(self, studies_to_deactivate: Sequence[Tuple[str, str, str]]) -> None:
        self._log.info(f"Pruning {len(studies_to_deactivate)} consolidated datasets")
        for a, d, proxy in studies_to_deactivate:
            self.state[a, d].dataset_pruned(proxy)

    def active_studies(self) -> int:
        return sum([1 for s in self.state if s.active])

    def is_fresh(self) -> bool:
        return sum([s.n_trials_completed for s in self.state]) == 0

    def display_state(self, fmt: Callable[[SearchSpaceState], str], max_rows: Optional[int] = None) -> None:
        print("\n\nSTATE")
        with show_full_df(max_rows):
            print(self.state._data.applymap(fmt).T)

    def save_state(self) -> None:
        # rotate state file
        i = 0
        old_name = self.state_file_path
        while old_name.exists():
            old_name = self.state_file_path.with_suffix(f".{i}{self.state_file_path.suffix}")
            i += 1
        if self.state_file_path.exists():
            self.state_file_path.rename(old_name)

        # save state
        self.state.save(self.state_file_path)
        self.sync_to_optuna()

    def sync_from_optuna(self) -> None:
        self.state.sync_from_optuna(storage=self.storage_mgr.get())

    def sync_to_optuna(self) -> None:
        self.state.sync_to_optuna(storage=self.storage_mgr.get())

    def close(self) -> None:
        self.save_state()
        self.storage_mgr.stop()

    def _apply_stop_conditions(self, s: SearchSpaceState, result: StudyJobResult) -> None:
        if s.n_trials_completed >= config.optimization.max_trails_per_study:
            # stop if max trials reached
            s.max_trials_reached()

        if config.optimization.use_stop_heuristic():
            # stop studies that repeatedly achieve perfect results
            from .optuna_early_stopping_callback import EarlyStoppingCallback

            scores = np.array(result.all_scores, dtype=np.float_)
            if EarlyStoppingCallback.from_config().should_stop(scores):
                s.max_quality_reached()

        if config.optimization.use_optuna_terminator():
            # stop studies that cannot improve further (judged by optuna)
            import optuna
            from .optuna_terminator import optuna_terminator

            study = optuna.load_study(study_name=result.study_id, storage=self.storage_mgr.get())
            if optuna_terminator.should_terminate(study):
                s.terminate()

    def load_results_for_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        import optuna
        from optuna.trial import TrialState

        results = pd.DataFrame(
            index=pd.MultiIndex.from_frame(candidates[["algorithm", "params_id"]]),
            columns=pd.MultiIndex.from_product([
                [d for d in self.state.datasets],
                ["quality", "duration"]
            ]),
            dtype=np.float_
        )

        storage = self.storage_mgr.get()
        for i, row in candidates.iterrows():
            algorithm = row["algorithm"]
            params = row["params"]
            params_id = row["params_id"]
            self._log.info(f"Loading all optimization results for {algorithm} {params}")
            for d in self.state.datasets:
                study_name = self.state[algorithm, d].study_name
                study = optuna.load_study(study_name=f"{study_name}_study", storage=storage)
                trials = study.get_trials(states=(TrialState.COMPLETE,))
                trials = [t for t in trials if ParamSetting(params) == ParamSetting(t.params)]
                if len(trials) == 0:
                    self._log.debug(f"  {d}: no matching trial found!")
                    continue

                t = trials[np.argmax([t.value for t in trials])]
                quality = t.value
                duration = t.duration.total_seconds()
                self._log.debug(f"  {d}: found best trial {t.number} with value {quality=:.4f}, {duration=:.2f}s")
                results.loc[(algorithm, params_id), (d, "quality")] = quality
                results.loc[(algorithm, params_id), (d, "duration")] = duration
        return results
