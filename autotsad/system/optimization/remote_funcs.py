import warnings
from dataclasses import asdict
from typing import Tuple, Optional, Callable, NamedTuple, Any, Dict, List, Union

import numpy as np
import optuna
from optuna import Trial
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import CmaEsSampler, BaseSampler, TPESampler
from optuna.trial import TrialState
from timeeval import Metric

from autotsad.config import GeneralSection, OptimizationSection
from autotsad.dataset import Dataset, TrainDataset
from autotsad.system.optimization.optuna_auto_storage import OptunaStorageReference, OptunaStorageType
from autotsad.system.optimization.optuna_early_stopping_callback import EarlyStoppingCallback
from autotsad.system.optimization.search_space import SearchSpaceState
from autotsad.system.pynisher import pynish_func, PynisherException
from autotsad.tsad_algorithms.interop import params_timeeval, params_from_trial, params_default, exec_algo


def timeeval_params_job(algorithm: str,
                        dataset: TrainDataset,
                        metric: Metric,
                        config: GeneralSection) -> Tuple[str, str, float]:
    def objective() -> float:
        params = params_timeeval(algorithm, period_size=dataset.period_size)
        scores = exec_algo(dataset, algorithm, params, ignore_cuts=True)
        quality = metric(dataset.label, scores)
        return quality

    result: float = np.nan
    try:
        result = pynish_func(
            objective,
            **config.adjusted_training_limits(dataset.length),
            name=f"OPT_TIMEEVAL: {dataset.name}-{algorithm}"
        )()
    except PynisherException as e:
        print(f"ERROR: {algorithm} failed on {dataset.name} with TimeEval params because {repr(e)}")

    return algorithm, dataset.name, result


def _create_objective(
        algorithm: str,
        dataset: Dataset,
        metric: Metric,
        config: GeneralSection,
) -> Callable[[Trial], float]:
    def internal_obj(params: Any) -> float:
        scores = exec_algo(dataset, algorithm, params, ignore_cuts=True)
        return metric(dataset.label, scores)

    def objective(trial: Trial) -> float:
        # IMPORTANT: retrieve parameter from optuna within the same process and just limit the algo execution and metric
        # calculation
        params = params_from_trial(trial, algorithm)

        # if we already checked this parameter set, return the cached result
        complete_trials = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        for t in complete_trials[::-1]:
            if trial.params == t.params:
                return t.value

        limited_objective = pynish_func(
            internal_obj,
            **config.adjusted_training_limits(dataset.length),
            name=f"OPT: {algorithm}-{dataset.name}"
        )
        return limited_objective(params)

    return objective


def _create_sampler(n_startup_trials: int = 50, seed: Optional[int] = None) -> BaseSampler:
    return CmaEsSampler(
        n_startup_trials=n_startup_trials,
        independent_sampler=TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials // 2,
            multivariate=True,
            warn_independent_sampling=False
        ),
        warn_independent_sampling=False,
        seed=seed,
        with_margin=False,  # True sometimes stalls optimization!
    )


class StudyJobResult(NamedTuple):
    algorithm: str
    dataset: str
    study_id: str
    best_trial_id: int
    best_score: float
    best_params: Dict[str, Any]
    n_trials: int
    all_scores: List[float]


def study_job(algorithm: str,
              dataset: TrainDataset,
              metric: Metric,
              n_trials: int,
              general_config: GeneralSection,
              optimization_config: OptimizationSection,
              tags: Dict[str, Union[str, int, float]],
              seed_params: List[Dict[str, Any]]) -> StudyJobResult:
    print(f"Processing study for {algorithm} on dataset {dataset.name} ({n_trials} trials)...")
    objective = _create_objective(algorithm, dataset, metric, general_config)

    optuna.logging.set_verbosity(optimization_config.optuna_logging_level)
    with warnings.catch_warnings(), OptunaStorageReference(
            tmp_path=general_config.cache_dir(),
            storage_type=OptunaStorageType.from_string(optimization_config.optuna_storage_type)
    ) as storage:
        warnings.filterwarnings(action="ignore", category=ExperimentalWarning)
        warnings.filterwarnings(action="ignore", category=UserWarning,
                                message="Cannot compute metric for a constant value")

        sampler = _create_sampler(
            n_startup_trials=max(50, optimization_config.n_trials_startup),
            seed=general_config.seed
        )
        study = optuna.create_study(
            study_name=f"{algorithm}-{dataset.name}_study",
            storage=storage,
            sampler=sampler,
            direction="maximize",
            load_if_exists=True,
        )
        # add default tags and update user attributes
        tags.update(SearchSpaceState.default_study_tags(algorithm, dataset.name))
        for k, v in tags.items():
            study.set_user_attr(k, v)

        if len(study.trials) == 0:
            # aid search with two good guesses for parameters
            # study.enqueue_trial(asdict(params_default(algorithm)),
            #                     user_attrs={"tag": "default"})
            study.enqueue_trial(asdict(params_timeeval(algorithm, period_size=dataset.period_size)),
                                user_attrs={"tag": "timeeval"})
            # add additional seed parameters
            for params in seed_params:
                study.enqueue_trial(params,
                                    user_attrs={"tag": "seed"},
                                    skip_if_exists=True)

        callbacks: List[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
        if optimization_config.use_stop_heuristic():
            callbacks.append(EarlyStoppingCallback(
                min_rounds=optimization_config.stop_heuristic_n,
                threshold=optimization_config.stop_heuristic_quality_threshold
            ))
        if optimization_config.use_optuna_terminator():
            from .optuna_terminator import optuna_terminator_callback
            callbacks.append(optuna_terminator_callback)
        study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=Exception, callbacks=callbacks,
                       show_progress_bar=False)  # progress bar clashes with our study progress bar

    try:
        best_trial = study.best_trial
    except ValueError:  # catch failing single trial if optimization is disabled (otherwise, this is very unlikely)
        print(f"... failed! No trial completed successfully for {algorithm} on dataset {dataset.name}!")
        return StudyJobResult(
            algorithm=algorithm,
            dataset=dataset.name,
            study_id=study.study_name,
            best_trial_id=1,
            best_score=.0,
            best_params={},
            n_trials=len(study.trials),
            all_scores=[.0]
        )

    print(f"... done! Best params for {algorithm} on dataset {dataset.name} with "
          f"RangePrAUC={best_trial.value} so far: {best_trial.params}")
    return StudyJobResult(
        algorithm=algorithm,
        dataset=dataset.name,
        study_id=study.study_name,
        best_trial_id=best_trial.number,
        best_score=best_trial.value,
        best_params=best_trial.params,
        n_trials=len(study.trials),
        all_scores=[t.value for t in study.trials]
    )
