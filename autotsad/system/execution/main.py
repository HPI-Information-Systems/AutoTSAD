from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from timeeval import Metric
from timeeval.utils.hash_dict import hash_dict
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .aggregation import aggregate_scores
from .remote_func_execute import execution_job
from .remote_func_test_combination import test_single_combination
from ..hyperparameters import ParamSetting
from ..logging import LOGGING_Q
from ..timer import Timers
from ...config import config, ALGORITHM_SELECTION_METHODS, SCORE_AGGREGATION_METHODS, SCORE_NORMALIZATION_METHODS
from ...dataset import TrainingDatasetCollection, TestDataset
from ...util import save_serialized_results, load_serialized_results, show_full_df


def execute_algorithms(dataset_collection: TrainingDatasetCollection,
                       metric: Metric,
                       parallelism: int,
                       tasks: List[Tuple[str, str, Dict[str, Any]]],
                       score_dirpath: Optional[Path] = None) -> pd.DataFrame:
    if score_dirpath is not None:
        score_dirpath = score_dirpath.resolve()
        score_dirpath.mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Executing", total=len(tasks), disable=not config.general.progress)):
        results = joblib.Parallel(n_jobs=min(parallelism, len(tasks)))(
            joblib.delayed(execution_job)(
                algorithm, dataset_collection[dataset_name], params, metric, score_dirpath, config.general
            )
            for algorithm, dataset_name, params in tasks
        )
    return pd.DataFrame(results, columns=["algorithm", "params", "dataset", "quality", "duration"])


def create_result_plot(test_data: TestDataset, results: pd.DataFrame, scores_path: Path,
                       combined_score: Optional[np.ndarray] = None,
                       selection_method: str = config.general.algorithm_selection_method,
                       save_plot: bool = False) -> None:
    import matplotlib.pyplot as plt
    from timeeval.metrics.thresholding import SigmaThresholding
    from sklearn.preprocessing import MinMaxScaler

    dataset_id = test_data.hexhash
    dataset_name = results["dataset"].iloc[0] if "dataset" in results.columns else test_data.name
    fig, axs = plt.subplots(2 + len(results), 1, sharex="col", figsize=(10, 1.5 * (1 + len(results))))
    axs[0].set_title(f"{selection_method} ranking for {dataset_name}")
    test_data.plot(ax=axs[0])

    # reset index to allow loc-indexing
    results = results.reset_index(drop=True)
    scores = np.empty((len(test_data), len(results)), dtype=np.float_)
    for i in range(2, len(results) + 2):
        algo, params, quality = results.loc[i-2, ["algorithm", "params", "quality"]]
        label = f"{algo} {params} ({quality:.0%})"
        s = np.genfromtxt(scores_path / f"{dataset_id}-{algo}-{hash_dict(params)}.csv", delimiter=",")
        s = MinMaxScaler().fit_transform(s.reshape(-1, 1)).ravel()
        scores[:, i-2] = s

        thresholding = SigmaThresholding(factor=2)
        predictions = thresholding.fit_transform(test_data.label, s)

        axs[i].plot(s, label=label, color="black")
        axs[i].hlines([thresholding.threshold], 0, s.shape[0], label=f"{thresholding}", color="red")
        axs[i].plot(predictions, label="predictions", color="orange")

        axs[i].legend()

    if combined_score is None:
        combined_score = aggregate_scores(scores)
    axs[1].set_title("Combined score")
    axs[1].plot(combined_score, label="Combined score", color="green")

    if save_plot:
        plt.savefig(config.general.result_dir() / "selected-instances.pdf")


def execute_on_test(dataset_collection: TrainingDatasetCollection,
                    selected_instances: pd.DataFrame,
                    score_dirpath: Path) -> pd.DataFrame:
    log = logging.getLogger("autotsad.execution")
    dataset_name = dataset_collection.test_data.name
    result_cache_path = config.general.cache_dir() / "execution-results.csv"

    if result_cache_path.exists():
        print("Loading algorithm execution results from cache")
        results = load_serialized_results(result_cache_path)
        log.info("Loaded algorithm results from cache")

    else:
        tasks = []
        for i, instance in selected_instances.iterrows():
            tasks.append((instance["algorithm"], dataset_name, instance["params"]))

        print(f"Running {len(tasks)} best algorithm instances on test data")
        log.info(f"Executing {len(tasks)}/{joblib.effective_n_jobs(config.general.n_jobs)} algorithm instances on test dataset {dataset_name}")
        Timers.start("Algorithm Execution")
        results = execute_algorithms(dataset_collection, config.optimization.metric(), config.general.n_jobs, tasks,
                                     score_dirpath=score_dirpath)

        # merge test results with training results
        selected_instances["params_id"] = selected_instances["params"].apply(lambda x: hash(ParamSetting(x)))
        results["params_id"] = results["params"].apply(lambda x: hash(ParamSetting(x)))
        results = results.drop(columns=["params"])
        results = pd.merge(selected_instances, results, how="left", on=["algorithm", "params_id"], sort=False)
        results = results.drop(columns=["params_id"])
        results["dataset_id"] = dataset_collection.test_data.hexhash

        save_serialized_results(results, result_cache_path)
        Timers.stop("Algorithm Execution")

    return results


def compute_single_result(dataset_collection: TrainingDatasetCollection, results: pd.DataFrame) -> pd.DataFrame:
    log = logging.getLogger("autotsad.execution")
    ranking_result_dir = config.general.result_dir() / "rankings"
    max_instances = config.general.max_algorithm_instances
    scores_path = config.general.tmp_path / "scores"
    selection = config.general.algorithm_selection_method
    normalization = config.general.score_normalization_method
    aggregation = config.general.score_aggregation_method

    print("Selecting diverse algorithm instances")
    log.info(f"Selecting diverse algorithm instances with {selection} strategy, {normalization} normalization and "
             f"{aggregation} aggregation.")
    Timers.start("Selecting and combining")
    metrics = test_single_combination(
                dataset=dataset_collection.test_data,
                results=results,
                result_dir=ranking_result_dir,
                selection=selection,
                normalization=normalization,
                aggregation=aggregation,
                max_instances=max_instances,
                scores_path=scores_path,
            )
    df_metrics = pd.DataFrame([metrics], index=[f"{selection}-{normalization}-{aggregation}"])
    df_metrics.to_csv(config.general.result_dir() / "metrics.csv")
    Timers.stop("Selecting and combining")

    return df_metrics


def compute_all_combinations(dataset_collection: TrainingDatasetCollection, results: pd.DataFrame) -> pd.DataFrame:
    log = logging.getLogger("autotsad.execution")
    jobs = [(selection, normalization, aggregation)
            for selection in ALGORITHM_SELECTION_METHODS
            for normalization in SCORE_NORMALIZATION_METHODS
            for aggregation in SCORE_AGGREGATION_METHODS]

    # capture some variables in closure:
    ranking_result_dir = config.general.result_dir() / "rankings"
    max_instances = config.general.max_algorithm_instances
    scores_path = config.general.tmp_path / "scores"
    q = LOGGING_Q

    def _test_single_combination_job(selection: str, normalization: str, aggregation: str) -> Dict[str, Any]:
        from autotsad.system.logging import setup_process_logging

        with setup_process_logging(q):
            return test_single_combination(
                dataset=dataset_collection.test_data,
                results=results,
                result_dir=ranking_result_dir,
                selection=selection,
                normalization=normalization,
                aggregation=aggregation,
                max_instances=max_instances,
                scores_path=scores_path,
            )

    log.info("Computing all combinations of algorithm selection, normalization and aggregation methods in parallel "
             f"({len(jobs)} / {joblib.effective_n_jobs(config.general.n_jobs)})")
    Timers.start("Computing all combinations")
    with tqdm_joblib(tqdm(desc="Algorithm Selections", total=len(jobs), disable=not config.general.progress)):
        all_metrics = joblib.Parallel(n_jobs=min(config.general.n_jobs, len(jobs)))(
            joblib.delayed(_test_single_combination_job)(selection, normalization, aggregation)
            for selection, normalization, aggregation in jobs
        )

    all_metrics = pd.DataFrame(all_metrics, index=[f"{s}-{n}-{a}" for s, n, a in jobs])
    all_metrics.to_csv(config.general.result_dir() / "metrics.csv")
    Timers.stop("Computing all combinations")

    return all_metrics


def execute_and_rank(dataset_collection: TrainingDatasetCollection, selected_instances: pd.DataFrame) -> np.ndarray:
    log = logging.getLogger("autotsad.execution")
    print("\n############################################################")
    print("#                   STEP 2: Execution                      #")
    print("############################################################")
    score_dirpath = config.general.tmp_path / "scores"
    Timers.start("Execution")

    print("\n############################################################")
    print("#             STEP 3a: Algorithm Execution                 #")
    print("############################################################")
    results = execute_on_test(dataset_collection, selected_instances, score_dirpath=score_dirpath)
    results = results[~results["quality"].isna()]

    print("\n############################################################")
    print("#   STEP 3b: Algorithm Instance Selection and Ensembling   #")
    print("############################################################")
    if config.general.compute_all_combinations:
        print("Computing all combinations of algorithm selection and ensembling techniques")
        metrics = compute_all_combinations(dataset_collection, results)

    else:
        print(f"Computing single result using\n"
              f"  - algorithm selection method: {config.general.algorithm_selection_method}\n"
              f"  - score normalization method: {config.general.score_normalization_method}\n"
              f"  - score aggregation method:   {config.general.score_aggregation_method}")
        metrics = compute_single_result(dataset_collection, results)

    with show_full_df():
        if "RangePrAUC" in metrics.columns:
            metrics = metrics.sort_values("RangePrAUC", ascending=False)
        print("\n\nAutoTSAD Performance:")
        print(metrics)
        print("\n\n")

    # determine best combination for returning score
    metric_name = config.optimization.metric_name
    selection_method = metrics.sort_values(by=metric_name, ascending=False).index[0]
    log.info(f"Determined best combination for returning score: {selection_method}")
    print(f"Best combination: {selection_method} with {metric_name}={metrics.loc[selection_method, metric_name]:2.2%}")
    path = config.general.result_dir() / "rankings" / selection_method / "combined-score.csv"
    log.info(f"Using combined score from {path}")
    scoring = np.genfromtxt(path, delimiter=",")

    Timers.stop("Execution")

    # save results
    target_result_path = config.general.result_dir() / "execution-results.csv"
    log.info(f"Storing algorithm execution results in {target_result_path}")
    save_serialized_results(results, target_result_path)

    scores_source_path = score_dirpath
    scores_target_path = config.general.result_dir() / "scores"
    log.info(f"Storing algorithm scores in {scores_target_path}")
    scores_target_path.mkdir(parents=True, exist_ok=True)
    instances = results["dataset_id"] + "-" + results["algorithm"] + "-" + results["params"].apply(hash_dict)
    for instance in instances:
        shutil.copy2(scores_source_path / f"{instance}.csv", scores_target_path / f"{instance}.csv")

    # plot results
    if config.general.plot_final_scores:
        import matplotlib.pyplot as plt

        selected_results = load_serialized_results(
            config.general.result_dir() / "rankings" / selection_method / "selected-instances.csv"
        )
        create_result_plot(dataset_collection.test_data, selected_results, score_dirpath,
                           combined_score=scoring,
                           selection_method=selection_method,
                           save_plot=True)
        plt.show()

    print("\n############################################################")
    print("#                        FINISHED!                         #")
    print("############################################################")
    return scoring
