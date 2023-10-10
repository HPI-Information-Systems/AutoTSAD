from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from timeeval import Metric
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .consolidation import main as consolidate
from .optimizer import Optimizer
from .remote_funcs import timeeval_params_job
from .search_space import SearchSpace, SearchSpaceState
from ..execution.main import execute_algorithms
from ..hyperparameters import ParamSetting
from ..timer import Timers
from ...config import config
from ...dataset import TrainingDatasetCollection, TrainingTSDataset
from ...tsad_algorithms.interop import params_default
from ...util import load_serialized_results, save_serialized_results


def _get_logger() -> logging.Logger:
    return logging.getLogger("autotsad.optimization")


def test_timeeval_params_on(datasets: List[TrainingTSDataset], metric: Metric, series_name: str) -> pd.Series:
    log = _get_logger()
    log.debug(f"  {series_name} series: {len(datasets)} dataset")
    df_results = pd.DataFrame(index=list(config.optimization.algorithms))
    df_results.index.name = "algorithm"

    n_tasks = len(config.optimization.algorithms) * len(datasets)
    with tqdm_joblib(tqdm(desc=f"Testing TimeEval parameters ({series_name})",
                          total=n_tasks,
                          disable=not config.general.progress)):
        results = joblib.Parallel(n_jobs=min(config.general.n_jobs, n_tasks))(
            joblib.delayed(timeeval_params_job)(algorithm, dataset, metric, config.general)
            for algorithm in config.optimization.algorithms
            for dataset in datasets
        )
    for a, d, q in results:
        df_results.loc[a, d] = q
    return df_results.T.std()


def validate_param_sensitivity(dataset_collection: TrainingDatasetCollection) -> Dict[str, str]:
    log = _get_logger()
    std_path = config.general.cache_dir() / "algorithm-score-stddevs.csv"

    if std_path.exists():
        log.info("Loading algorithm stddevs")
        df_std = pd.read_csv(std_path, index_col=0)

    else:
        log.info("Executing algorithms on dataset optimization dimension series with default parameters")
        log.debug(f"Dataset candidates {len(dataset_collection)}")
        df_std = pd.DataFrame(index=list(config.optimization.algorithms))

        # run algos with TimeEval parameters for dataset optimization dimensions
        for opt_series in ["base", "anomaly_length", "anomaly_type"]:
            datasets = dataset_collection.get_base_optimization_series(opt_series)
            df_std[opt_series] = test_timeeval_params_on(datasets, config.optimization.metric(), series_name=opt_series)

        df_std = df_std.sort_index()
        df_std.to_csv(std_path, index=True)
        log.info("saved algorithm stddevs.")

    log.debug(f"Algorithm stddevs:\n{df_std}")

    algo_data_dim_mapping = {}
    for c in df_std.index:
        if c not in config.optimization.algorithms:
            log.warning(f"Algorithm {c} was excluded!")
            continue
        argmax = df_std.loc[c, :].argmax()
        algo_data_dim_mapping[c] = df_std.columns[argmax]

    log.debug("Sensitive dataset dimensions for each algorithm:")
    for algo in algo_data_dim_mapping:
        log.debug(f"{algo}: {algo_data_dim_mapping[algo]}")
    path = config.general.cache_dir() / "sensitive-data-dims.json"
    with path.open("w") as fh:
        json.dump(algo_data_dim_mapping, fh)
    return algo_data_dim_mapping


def find_seed_params_for_sensitive_dims(optimizer: Optimizer,
                                        dataset_collection: TrainingDatasetCollection) -> Dict[str, Any]:
    log = _get_logger()
    Timers.start(["Sensitivity analysis", "Detect sensitive dim."])
    log.info("Validating algorithm parameter sensitivity to find their sensitive data dimension...")
    sensitive_data_dims = validate_param_sensitivity(dataset_collection)
    Timers.stop("Detect sensitive dim.")

    # set desired state
    def update(state: SearchSpace) -> SearchSpace:
        for algorithm in sensitive_data_dims:
            for dataset in dataset_collection.get_base_optimization_series(sensitive_data_dims[algorithm]):
                state[algorithm, dataset.name].n_trials_desired = config.optimization.n_trials_sensitivity
        state.save(config.general.cache_dir() / "initial-state.csv")
        return state
    Timers.start("Prepare state")
    optimizer.update_all_states(update)
    Timers.stop("Prepare state")

    log.info("Optimizing algorithms for each dataset in their respective sensitive data dimension...")
    Timers.start("Valid. optimization")
    optimizer.run_optimization_iteration(tags={"study_type": "sensitivity"})

    # map best hyperparameters to respective sensitive data dimension value (considered as new default/seed params)
    sensitivity_results = {}
    for algorithm in sensitive_data_dims:
        seed_params = {}
        sensitive_data_dim = sensitive_data_dims[algorithm]
        for dataset in dataset_collection.get_base_optimization_series(sensitive_data_dim):
            data_dim_value = dataset.opt_dims[sensitive_data_dim]
            seed_params[str(data_dim_value)] = optimizer.state[algorithm, dataset.name].best_params
        sensitivity_results[algorithm] = {
            "sensitive_data_dim": sensitive_data_dim,
            "seed_params": seed_params
        }
    Timers.stop("Valid. optimization")
    Timers._log = logging.getLogger("autotsad.Timer")
    Timers.stop("Sensitivity analysis")
    return sensitivity_results


def _resolve_proxy_relations(dataset_collection: TrainingDatasetCollection, state: SearchSpace) -> None:
    log = _get_logger()
    log.info("Resolving proxy relations")
    Timers.start("Resolving proxies")
    tasks = [(s.algorithm, s.dataset, s.proxy.best_params) for s in state.get_proxies()]
    if len(tasks) > 0:
        log.info(f"Executing {len(tasks)} tasks to determine proxy performance")
        results = execute_algorithms(dataset_collection, config.optimization.metric(), config.general.n_jobs, tasks)

        for i, row in results.iterrows():
            a, p, d, q, _ = row
            if state[a, d].best_score - q > config.optimization.proxy_allowed_quality_deviation:
                log.warning(f"New score for {a} on {d} is worse than before: {q} < {state[a, d].best_score}, "
                            f"using original best trial and parameters!")
            else:
                state[a, d].best_score = q
                state[a, d].best_params = p
    Timers.stop("Resolving proxies")


def _select_best_instances(dataset_collection: TrainingDatasetCollection, state: SearchSpace) -> pd.DataFrame:
    log = _get_logger()
    log.info("  selecting best algorithm instance for each training dataset")
    best_algos = []
    for d in dataset_collection.training_datasets:
        candidates = [s for s in state[d.name] if s.status not in ["algo_pruned"]]
        if len(candidates) == 1:
            idx = 0
        else:
            # find best performing algo
            idx = np.argmax([s.best_score for s in candidates])
        algo = candidates[idx].algorithm
        params = candidates[idx].best_params
        best_algos.append((algo, hash(ParamSetting(params)), params))
    best_algos = pd.DataFrame(best_algos, columns=["algorithm", "params_id", "params"])
    best_algos = best_algos.groupby(["algorithm", "params_id"], sort=False).agg(
        params=("params", "first"),
        no_datasets=("params", "count"),
    )
    log.info("  ...done.")
    return best_algos.reset_index()


def _reintroduce_default_params(best_algos: pd.DataFrame) -> pd.DataFrame:
    log = _get_logger()
    additional_instances = []
    log.info("  reintroducing default parameters...")
    for algo in best_algos["algorithm"].unique():
        default_params = asdict(params_default(algo))
        tmp = best_algos[best_algos["algorithm"] == algo]
        # filter out non-optimized parameters
        if len(tmp) > 0:
            param_names = tmp["params"].iloc[0].keys()
            default_params = {k: default_params[k] for k in param_names}

        # default params are not already included --> add them
        params_id = hash(ParamSetting(default_params))
        if len(tmp[tmp["params_id"] == params_id]) == 0:
            log.info(f"    reintroducing {algo} with default parameters ({default_params})")
            additional_instances.append({
                "algorithm": algo,
                "params": default_params,
                "params_id": params_id,
                "no_datasets": 0,
            })
    best_algos = pd.concat([best_algos, pd.DataFrame(additional_instances)], ignore_index=True, axis=0)
    log.info("  ...done reintroducing default parameters.")
    return best_algos


def _compute_proxy_metrics(dataset_collection: TrainingDatasetCollection, optimizer: Optimizer, best_algos: pd.DataFrame) -> pd.DataFrame:
    log = _get_logger()
    Timers.start(["Computing proxy metric", "Loading existing results"])
    log.info("  computing proxy metrics ...")

    log.info("    loading optimization results from Optuna storage")
    best_algo_results = optimizer.load_results_for_candidates(best_algos)
    best_algo_results = best_algo_results.sort_index()

    # build task list for missing executions
    nans = np.isnan(best_algo_results.values)
    missing_results = best_algo_results.loc[np.any(nans, axis=1), np.any(nans, axis=0)]
    best_algos = best_algos.set_index(["algorithm", "params_id"])
    tasks = []
    for dataset in missing_results.columns.remove_unused_levels().levels[0]:
        for algorithm, params_id in missing_results[dataset].index:
            if np.isnan(missing_results.loc[(algorithm, params_id), dataset]).any():
                params = best_algos.loc[(algorithm, params_id), "params"]
                tasks.append((algorithm, dataset, params))
    Timers.stop("Loading existing results")

    if len(tasks) > 0:
        log.info(f"    executing {len(tasks)} tasks to determine missing proxy metrics")
        Timers.start("Executing proxy metric tasks")
        results = execute_algorithms(dataset_collection, config.optimization.metric(), config.general.n_jobs, tasks)

        for i, row in results.iterrows():
            a, p, d, q, t = row
            params_id = hash(ParamSetting(p))
            best_algo_results.loc[(a, params_id), (d, "quality")] = q
            best_algo_results.loc[(a, params_id), (d, "duration")] = t

        Timers.stop("Executing proxy metric tasks")

    df_tmp = pd.DataFrame({
        "mean_train_quality": best_algo_results.loc[:, pd.IndexSlice[:, "quality"]].mean(axis=1).values,
        "mean_train_runtime": best_algo_results.loc[:, pd.IndexSlice[:, "duration"]].mean(axis=1).values,
    }, index=best_algo_results.index)
    best_algos = pd.merge(best_algos, df_tmp, how="inner", left_index=True, right_index=True)
    best_algos = best_algos.reset_index()
    log.info("  ... done computing proxy metrics")
    Timers.stop("Computing proxy metric")
    return best_algos


def run_optimization_steps(dataset_collection: TrainingDatasetCollection) -> pd.DataFrame:
    log = _get_logger()
    Timers.start("Optimization")
    state_file_path = config.general.cache_dir() / "optimization-state.csv"
    with Optimizer(state_file_path, dataset_collection, continue_from_existing=True) as optimizer:
        if optimizer.is_fresh() and config.optimization.enabled():
            print("\n############################################################")
            print("#         STEP 2a: Validating algorithm sensitivity        #")
            print("############################################################")
            sensitivity_result_file_path = config.general.cache_dir() / "sensitivity-optimization-results.json"

            if sensitivity_result_file_path.exists():
                log.info("Loading sensitivity optimization results")
                with sensitivity_result_file_path.open("r") as fh:
                    sensitivity_results = json.load(fh)
            else:
                log.info("Optimizing each algorithm with datasets of its sensitive dimension")
                sensitivity_results = find_seed_params_for_sensitive_dims(optimizer, dataset_collection)
                with sensitivity_result_file_path.open("w") as fh:
                    json.dump(sensitivity_results, fh)
                optimizer.save_state()

            print("\nNew default parameters:")
            for algorithm in sensitivity_results:
                print("Algorithm", algorithm)
                dim = sensitivity_results[algorithm]["sensitive_data_dim"]
                for k in sensitivity_results[algorithm]["seed_params"]:
                    print(f"  {dim}={k}:")
                    print(f"    {sensitivity_results[algorithm]['seed_params'][k]}")

            log.info("Setting new seed parameters")
            for algorithm in sensitivity_results:
                sensitive_data_dim = sensitivity_results[algorithm]["sensitive_data_dim"]
                seed_params = sensitivity_results[algorithm]["seed_params"]
                # set as new default params for all datasets with same sensitive data dim value
                for d in dataset_collection.training_datasets:
                    for data_dim_value in seed_params:
                        # data_dim_value is serialized as string --> compare to string repr
                        if str(d.opt_dims[sensitive_data_dim]) == data_dim_value:
                            optimizer.set_seed_params(algorithm, d.name, seed_params[data_dim_value])

        print("\n############################################################")
        print("#       STEP 2b: Optimizing algorithm hyperparameters      #")
        print("############################################################")
        Timers.start("Hyperparams opt.")
        iterations = 0
        while optimizer.active_studies() > 0:
            optimizer.save_state()
            optimizer.display_state(lambda x: f"{x.status} ({x.n_trials_completed})")

            iterations += 1
            if iterations == 1:
                # first iteration should do double the number of trials to benefit from exploration
                n_trials = config.optimization.n_trials_step * 2
            else:
                n_trials = config.optimization.n_trials_step
            if config.optimization.disabled:
                # only perform one trial (containing the default parameters)
                n_trials = 1
            log.info(f"Optimizing all algorithms on all datasets (iteration {iterations}: {n_trials} trials)")

            def update(s: SearchSpaceState) -> None:
                new_n_trials = min(
                    # only perform one trial (containing the default parameters) if disabled
                    config.optimization.max_trails_per_study if config.optimization.enabled() else 1,
                    s.n_trials_desired + n_trials
                )
                s.n_trials_desired = new_n_trials
            optimizer.update_state(update)
            optimizer.run_optimization_iteration()

            if config.optimization.disabled:
                # optimization is disabled, only perform one iteration (containing the default parameters)
                # thus, we can also safely skip the algo pruning and the consolidation step
                optimizer.update_state(lambda x: x.max_trials_reached())
                break

            # disable the worst performing algorithm per dataset
            optimizer.prune_worst_algorithm()

            # consolidate datasets with similar hyperparameter selection and dataset characteristics
            log.info("Consolidating datasets with similar hyperparameter selection and dataset characteristics")
            _, studies_to_deactivate = consolidate(
                dataset_collection, consider_dataset_characteristics=True, key=f"{iterations}"
            )
            optimizer.prune_consolidated_datasets(studies_to_deactivate.values)

        optimizer.save_state()
        optimizer.display_state(lambda x: f"{x.status} ({x.n_trials_completed})")
        Timers.stop("Hyperparams opt.")
        print("All studies processed")

        print("\n############################################################")
        print("#        STEP 2c: Ranking algorithms on training data      #")
        print("############################################################")

        print("Resolving proxy relations")
        if config.optimization.enabled():
            _resolve_proxy_relations(dataset_collection, optimizer.state)

        print("Selecting best algorithm instances")
        Timers.start("Selecting best performers")
        best_algos = _select_best_instances(dataset_collection, optimizer.state)
        print(f"  selected {best_algos.shape[0]} candidate instances")

        if config.optimization.reintroduce_default_params and config.optimization.enabled():
            n_before = best_algos.shape[0]
            best_algos = _reintroduce_default_params(best_algos)
            print(f"  reintroduced {best_algos.shape[0] - n_before} default parameter instances")

        print("  computing proxy metrics...")
        best_algos = _compute_proxy_metrics(dataset_collection, optimizer, best_algos)
        print("  ...done.")

    best_algos = best_algos.sort_values(["no_datasets", "mean_train_quality"], ascending=False)
    best_algos = best_algos.reset_index(drop=True).drop(columns="params_id")
    Timers.stop("Selecting best performers")

    print("  best algorithms:")
    print(best_algos.iloc[:config.general.max_algorithm_instances])
    Timers.stop("Optimization")
    return best_algos


def optimize_algorithms(dataset_collection: TrainingDatasetCollection) -> pd.DataFrame:
    print("\n############################################################")
    print("#                  STEP 2: Optimization                    #")
    print("############################################################")
    instances_cache_path = config.general.cache_dir() / "ranked-instances.csv"

    if instances_cache_path.exists():
        selected_instances = load_serialized_results(instances_cache_path)
    else:
        selected_instances = run_optimization_steps(dataset_collection)
        save_serialized_results(selected_instances, instances_cache_path)

    return selected_instances
