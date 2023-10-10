import argparse
import sys
import time
from pathlib import Path
from typing import Sequence, Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from durations import Duration
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import Engine, create_engine as create_pg_engine, MetaData, Table, insert, select
from timeeval import Metric, Algorithm, ResourceConstraints
from timeeval.adapters.docker import DockerTimeoutError, DockerMemoryError, SCORES_FILE_NAME
from timeeval.metrics.thresholding import SigmaThresholding
from timeeval.utils.hash_dict import hash_dict
from timeeval_experiments.algorithms import sand, kmeans

sys.path.append(".")

from autotsad.config import ALGORITHMS, METRIC_MAPPING
from autotsad.dataset import TrainingDatasetCollection, TestDataset, Dataset
from autotsad.evaluation import evaluate_result, evaluate_individual_results
from autotsad.system.execution.main import execute_algorithms

DOCKER_ALGORITHMS = {
    "SAND": sand(),
    "k-Means": kmeans(),
}


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the baseline algorithms on the supplied dataset and store "
                                                 "the results in the DB.")
    add_database_arguments(parser)
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset file!")
    parser.add_argument("--metric", type=str, choices=list(METRIC_MAPPING.keys()), default="RangePrAUC",
                        help="The metric used for computing the quality of individual results (e.g. for ranking).")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp"), help="Directory for the baseline scorings "
                                                                           "(can be deleted afterwards).")
    return parser.parse_args(args)


def add_database_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-host", type=str, default="172.17.17.32:5432", help="Database hostname (and port)")


def create_engine(args: argparse.Namespace) -> Engine:
    db_user = "autotsad"
    db_pw = "holistic-tsad2023"
    db_database_name = "akita"

    engine = _create_engine_from_url(f"postgresql+psycopg2://{db_user}:{db_pw}@{args.db_host}/{db_database_name}")
    return engine


def _create_engine_from_url(url: str) -> Engine:
    return create_pg_engine(
        url,
        isolation_level="SERIALIZABLE",
        # echo=True,
        future=True,
    )


def upload_results(method: str, dataset_id: str, baseline_results: pd.DataFrame, baseline_metrics: Dict[str, float],
                   engine: Engine, tmp_path: Path) -> None:
    metadata_obj = MetaData()
    baseline_execution_table = Table("baseline_execution", metadata_obj, autoload_with=engine, schema="autotsad")
    algorithm_scoring_table = Table("algorithm_scoring", metadata_obj, autoload_with=engine, schema="autotsad")
    ranking_table = Table("algorithm_ranking", metadata_obj, autoload_with=engine, schema="autotsad")
    ranking_entry_table_metadata = {"name": "algorithm_ranking_entry", "schema": "autotsad"}
    scoring_table_metadata = {"name": "scoring", "schema": "autotsad"}

    def _create_scoring(algorithm: str, params: Dict[str, Any], dataset: str, dataset_id: str, quality: float) -> int:
        hyper_params_id = hash_dict(params)
        res = conn.execute(insert(algorithm_scoring_table).values({
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "hyper_params_id": hyper_params_id,
            "hyper_params": params,
            "range_pr_auc": quality,
            # "range_roc_auc": ...,
            # "precision_at_k": ...,
            # "runtime": ...,
        }).returning(algorithm_scoring_table.c.id)).first()
        scoring_id = res[0]

        scoring_path = tmp_path / "baseline-scores" / f"{dataset_id}-{algorithm}-{hyper_params_id}.csv"
        if not scoring_path.exists():
            print("    ... missing scoring file, skipping!")
            # raise ValueError(f"Could not find scoring file {path}!")
            return scoring_id

        t0 = time.time_ns()
        df_scoring = pd.DataFrame()
        df_scoring["score"] = np.genfromtxt(scoring_path, delimiter=",")
        df_scoring["time"] = df_scoring.index
        df_scoring["algorithm_scoring_id"] = scoring_id
        t1 = time.time_ns()

        df_scoring.to_sql(con=conn, **scoring_table_metadata, if_exists="append", index=False)
        t2 = time.time_ns()
        print(f"    ... {dataset}-{algorithm}-{hyper_params_id} done (local {(t1 - t0) / 1e9:.2f}s, DB {(t2 - t1) / 1e9:.2f}s)")
        return scoring_id

    def _create_ranking(scoring_ids: List[int]) -> int:
        res = conn.execute(insert(ranking_table).values({"experiment_id": None}))
        ranking_id = res.inserted_primary_key[0]
        print(f"  created ranking with ID {ranking_id}")

        # create ranking entries
        df_ranking = pd.DataFrame({"algorithm_scoring_id": scoring_ids})
        df_ranking["rank"] = df_ranking.index
        df_ranking["ranking_id"] = ranking_id
        df_ranking.to_sql(con=conn, **ranking_entry_table_metadata, if_exists="append", index=False)
        print(f"  added {len(df_ranking)} ranking entries to ranking {ranking_id}")
        return ranking_id

    def _create_execution(ranking_id: Optional[int] = None, scoring_id: Optional[int] = None) -> None:
        if ranking_id is None and scoring_id is None:
            raise ValueError("Either ranking_id or scoring_id must be set!")
        data = {
            "dataset_id": dataset_id,
            "name": method,
            "runtime": None,  # TODO: measure runtime!
            "range_pr_auc": baseline_metrics["RangePrAUC"],
            "range_roc_auc": baseline_metrics["RangeRocAUC"],
            "precision_at_k": baseline_metrics["PrecisionAtK"],
            "precision": baseline_metrics["RangePrecision"],
            "recall": baseline_metrics["RangeRecall"],
        }
        if scoring_id is not None:
            data["algorithm_scoring_id"] = scoring_id
        if ranking_id is not None:
            data["algorithm_ranking_id"] = ranking_id
        conn.execute(insert(baseline_execution_table).values(data))

    with engine.begin() as conn:
        # check if already executed
        res = conn.execute(select(baseline_execution_table.c.id)
                           .where(baseline_execution_table.c.dataset_id == dataset_id)
                           .where(baseline_execution_table.c.name == method)).first()
        if res is not None:
            print(f"Baseline {method} was already executed for dataset {dataset_id} and has ID {res[0]}.")
            return

        # upload scorings
        scoring_ids = []
        for _, (algorithm, params, dataset, dataset_id, quality) in baseline_results.iterrows():
            scoring_id = _create_scoring(algorithm, params, dataset, dataset_id, quality)
            scoring_ids.append(scoring_id)

        if len(scoring_ids) < 2:
            scoring_id = scoring_ids[0]
            _create_execution(scoring_id=scoring_id)

        else:
            ranking_id = _create_ranking(scoring_ids)
            _create_execution(ranking_id=ranking_id)
        print(f"  added execution for {method} (dataset_id={dataset_id})")


def main(sys_args: List[str]) -> None:
    args = parse_args(sys_args)
    engine = create_engine(args)

    tmp_path: Path = args.tmp_dir
    dataset_path: Path = args.dataset_path
    metric: str = args.metric
    test_dataset = TestDataset.from_file(dataset_path)

    method = "default-baseline"
    print(f"Processing {method} on dataset {test_dataset.name} ({test_dataset.hexhash})...")
    baseline_results = execute_baselines(test_dataset, test_dataset_path=dataset_path, tmp_path=tmp_path, metric=metric)
    baseline_metrics = evaluate_result(test_dataset, baseline_results, tmp_path / "baseline-scores")
    # create_result_plot(test_dataset, baseline1_results, config.general.tmp_path / "baseline-scores")
    upload_results(method, test_dataset.hexhash, baseline_results, baseline_metrics, engine, tmp_path)

    baseline_algorithms = ["SAND", "k-Means"]
    print(f"Processing {baseline_algorithms} on dataset {test_dataset.name} ({test_dataset.hexhash})...")
    baseline2_results = execute_baselines(test_dataset, test_dataset_path=dataset_path, algorithms=baseline_algorithms, tmp_path=tmp_path, metric=metric)
    baseline2_metrics = evaluate_individual_results(test_dataset, baseline2_results, tmp_path / "baseline-scores")

    for method in baseline_algorithms:
        results = baseline2_results[baseline2_results["algorithm"] == method]
        metrics = baseline2_metrics[method]
        upload_results(method, test_dataset.hexhash, results, metrics, engine, tmp_path)


def evaluate_existing(dataset: Dataset, scores_path: Path, metric: Metric) -> float:
    scores = np.loadtxt(scores_path, delimiter=",")
    if np.any(dataset.label):
        return metric(dataset.label, scores)
    else:
        return -1.


def _run_docker_algorithm(
        algo: Algorithm,
        dataset: Dataset,
        dataset_path: Path,
        scores_path: Path,
        metric: Metric,
        params: Dict[str, Any] = {},
        parallelism: int = 1) -> Dict[str, Any]:
    params_id = hash_dict(params)
    dataset_id = getattr(dataset, "hexhash", dataset.name)
    results_path = scores_path
    scores_path = scores_path / f"{dataset_id}-{algo.name}-{params_id}.csv"
    args = {
        "results_path": results_path,
        "hyper_params": params,
        "resource_constraints": ResourceConstraints(
            tasks_per_host=parallelism,
            execute_timeout=Duration("4 hours"),
        )
    }
    try:
        print(f"Executing Docker algorithm {algo.name} with params {params}")
        scores = algo.execute(dataset_path, args)
        if algo.postprocess is not None:
            scores = algo.postprocess(scores, args)
        (results_path / SCORES_FILE_NAME).unlink(missing_ok=True)
        np.savetxt(scores_path, scores, delimiter=",")
    except (DockerTimeoutError, DockerMemoryError):
        print(f"Execution of {algo.name} with params {params} timed out, returning empty scores")
        length = pd.read_csv(dataset_path).shape[0]
        scores = np.full(length, 0, dtype=np.float_)
        np.savetxt(scores_path, scores, delimiter=",")  # type: ignore

    quality = evaluate_existing(dataset, scores_path, metric)

    return {"algorithm": algo.name, "params": params, "dataset": dataset.name, "dataset_id": dataset_id,
            "quality": quality}


def _execute_docker_algorithms(dataset: Dataset,
                               test_dataset_path: Path,
                               base_scores_path: Path,
                               metric: Metric,
                               parallelism: int,
                               tasks: List[Tuple[str, Dict[str, Any]]], ) -> pd.DataFrame:
    if base_scores_path is not None:
        base_scores_path = base_scores_path.resolve()
        base_scores_path.mkdir(parents=True, exist_ok=True)

    results = joblib.Parallel(n_jobs=min(parallelism, len(tasks)))(
        joblib.delayed(_run_docker_algorithm)(
            DOCKER_ALGORITHMS[algorithm],
            dataset,
            dataset_path=test_dataset_path,
            scores_path=base_scores_path,
            metric=metric,
            params=params,
            parallelism=parallelism,
        )
        for algorithm, params in tasks
    )
    return pd.DataFrame(results, columns=["algorithm", "params", "dataset", "dataset_id", "quality"])


def execute_baselines(test_dataset: TestDataset,
                      test_dataset_path: Path,
                      tmp_path: Path,
                      algorithms: Sequence[str] = ALGORITHMS,
                      metric: str = "RangePrAUC",
                      parallelism: int = -1) -> pd.DataFrame:
    print("Running default algorithm instances on test data")

    dataset_name = test_dataset.name
    dataset_id = test_dataset.hexhash
    dataset_collection = TrainingDatasetCollection.from_base_timeseries(test_dataset)
    base_scores_path = tmp_path / "baseline-scores"
    metric = METRIC_MAPPING[metric]

    tasks = []
    docker_tasks = []
    params = {}
    params_hash = hash_dict(params)
    results = []
    for a in algorithms:
        scores_path = base_scores_path / f"{dataset_id}-{a}-{params_hash}.csv"
        if scores_path.exists():
            print(f"{a} was already executed on {dataset_name}, using existing scores.")
            quality = evaluate_existing(test_dataset, scores_path, metric)
            results.append({"algorithm": a, "params": params, "dataset": dataset_name,
                            "dataset_id": dataset_id, "quality": quality, "duration": np.nan})
        elif a in DOCKER_ALGORITHMS:
            docker_tasks.append((a, params))
        else:
            tasks.append((a, dataset_name, params))
    results = pd.DataFrame(results)

    if len(tasks) > 0:
        print(f"Executing {len(tasks)} algorithms on test dataset")
        parallelism = min(joblib.effective_n_jobs(parallelism), len(tasks))
        new_results = execute_algorithms(dataset_collection, metric, parallelism, tasks, score_dirpath=base_scores_path)
        new_results.insert(3, "dataset_id", dataset_id)
        results = pd.concat([results, new_results], axis=0, ignore_index=True)

    if len(docker_tasks) > 0:
        print(f"Executing {len(docker_tasks)} Docker algorithms on test dataset")
        parallelism = min(joblib.effective_n_jobs(parallelism), len(docker_tasks))
        new_results = _execute_docker_algorithms(test_dataset, test_dataset_path, base_scores_path, metric,
                                                 parallelism, docker_tasks)
        results = pd.concat([results, new_results], axis=0, ignore_index=True)

    results["quality"] = results["quality"].astype(np.float_)
    results = results.sort_values("quality", ascending=False)
    print(results)
    return results


def _plot_baseline_results(test_data: TestDataset, results: pd.DataFrame, scores_path: Path) -> None:
        dataset_id = test_data.hexhash
        dataset_name = results["dataset"].iloc[0] if "dataset" in results.columns else test_data.name

        # reset index to allow loc-indexing
        results = results.reset_index(drop=True)
        for i in range(results.shape[0]):
            algo, params, quality = results.loc[i, ["algorithm", "params", "quality"]]

            fig, axs = plt.subplots(2, 1, sharex="col", figsize=(10, 3))
            axs[0].set_title(f"Baseline {algo} for {dataset_name}")
            test_data.plot(ax=axs[0])

            label = f"{algo} {params} ({quality:.0%})"
            s = np.genfromtxt(scores_path / f"{dataset_id}-{algo}-{hash_dict(params)}.csv", delimiter=",")
            s = MinMaxScaler().fit_transform(s.reshape(-1, 1)).ravel()

            thresholding = SigmaThresholding(factor=2)
            # thresholding = PercentileThresholding()
            predictions = thresholding.fit_transform(test_data.label, s)

            axs[1].plot(s, label=label, color="black")
            axs[1].hlines([thresholding.threshold], 0, s.shape[0], label=f"{thresholding}", color="red")
            axs[1].plot(predictions, label="predictions", color="orange")

            axs[1].legend()


if __name__ == '__main__':
    main(sys.argv[1:])
