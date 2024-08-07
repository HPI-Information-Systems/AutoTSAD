# Execute with the following result backups:
# - 2021-12-03_runtime-gutentag-2-merged
# - 2022-02-21_runtime-benchmark-2-merged
# - 2023-08-17-timeeval-tsb-uad
# - 2023-08-18-timeeval-iops-and-norma
# - 2023-08-25-timeeval-SAND
# - 2024-04-01-tods-synthetic

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Connection, MetaData, Table, insert, select
from timeeval.constants import ANOMALY_SCORES_TS
from timeeval.utils.results_path import generate_experiment_path

from autotsad.config import (
    BASELINE_KMEANS_NAME,
    BASELINE_MAX_NAME,
    BASELINE_MEAN_NAME,
    BASELINE_SAND_NAME,
    METRIC_MAPPING,
)
from autotsad.database.database import Database


def _get_convert(s: pd.Series, column: str) -> Optional[str]:
    value = s.get(column, None)
    if value is None or (isinstance(value, np.number) and ~np.isfinite(value)):
        return None
    return str(value)


def upload_single_result(
    method: str,
    execution: pd.Series,
    results_path: Path,
    conn: Connection,
    omit_scores: bool = False,
) -> None:
    metadata_obj = MetaData()
    baseline_execution_table = Table(
        "baseline_execution", metadata_obj, autoload_with=conn, schema="autotsad"
    )
    algorithm_scoring_table = Table(
        "algorithm_scoring", metadata_obj, autoload_with=conn, schema="autotsad"
    )
    scoring_table_metadata = {"name": "scoring", "schema": "autotsad"}

    dataset_id = execution["hexhash"]
    range_pr_auc = _get_convert(execution, "RANGE_PR_AUC")
    range_roc_auc = _get_convert(execution, "RANGE_ROC_AUC")
    precision_at_k = _get_convert(execution, "PRECISION@K(None)")
    precision = _get_convert(execution, "RangePrecision")
    recall = _get_convert(execution, "RangeRecall")

    # check if already executed
    res = conn.execute(
        select(baseline_execution_table.c.id)
        .where(baseline_execution_table.c.dataset_id == dataset_id)
        .where(baseline_execution_table.c.name == method)
    ).first()
    if res is not None:
        print(
            f"  baseline {method} was already executed for dataset {dataset_id} "
            f"({execution['collection']} {execution['dataset']}) and has ID {res[0]}."
        )
        return

    print(f"  uploading execution for dataset {dataset_id} and baseline {method}...")
    if method == BASELINE_MEAN_NAME:
        print(f"    no scoring for baseline method {method}.")
        scoring_id = None

    else:
        # upload scorings
        scoring_path = (
            generate_experiment_path(
                base_results_dir=results_path,
                algorithm_name=execution["algorithm"],
                hyper_params_id=execution["hyper_params_id"],
                collection_name=execution["orig_collection"],
                dataset_name=execution["dataset"],
                repetition_number=1,
            )
            / ANOMALY_SCORES_TS
        )

        if omit_scores:
            scoring_id = None
        elif not scoring_path.exists():
            print(f"    ... missing scoring file, skipping scoring upload ({scoring_path})")
            scoring_id = None
            # raise ValueError(f"Could not find scoring file {path}!")

        else:
            res = conn.execute(
                insert(algorithm_scoring_table)
                .values(
                    {
                        "dataset_id": dataset_id,
                        "algorithm": execution["algorithm"],
                        "hyper_params_id": _get_convert(execution, "hyper_params_id"),
                        "hyper_params": _get_convert(execution, "hyper_params"),
                        "range_pr_auc": range_pr_auc,
                        "range_roc_auc": range_roc_auc,
                        "precision_at_k": precision_at_k,
                        "runtime": execution["overall_runtime"],
                    }
                )
                .returning(algorithm_scoring_table.c.id)
            ).first()
            scoring_id = res[0]

            df_scoring = pd.DataFrame()
            df_scoring["score"] = np.genfromtxt(scoring_path, delimiter=",")
            df_scoring["time"] = df_scoring.index
            df_scoring["algorithm_scoring_id"] = scoring_id

            df_scoring.to_sql(
                con=conn, **scoring_table_metadata, if_exists="append", index=False
            )
            print(f"    ... {dataset_id}-{method} done")

    data = {
        "dataset_id": dataset_id,
        "name": method,
        "runtime": execution["overall_runtime"],
        "range_pr_auc": range_pr_auc,
        "range_roc_auc": range_roc_auc,
        "precision_at_k": precision_at_k,
        "precision": precision,
        "recall": recall,
        "algorithm_scoring_id": scoring_id,
    }
    conn.execute(insert(baseline_execution_table).values(data))
    print(f"  added execution for {method} (dataset_id={dataset_id})")


def load_timeeval_baseline_results(
    db: Database,
    results_path: Path,
    metric_name: str,
    baselines: List[str],
    only_paper_datasets: bool = True,
    omit_scores: bool = False,
) -> None:
    path = Path(results_path).resolve()
    if not path.exists() or path.is_file():
        raise ValueError(f"Path to the TimeEval result folder ({path}) is invalid!")

    metric = METRIC_MAPPING[metric_name]
    baselines = [b.lower() for b in baselines]

    df = pd.read_csv(results_path / "results.csv")
    df["overall_runtime"] = (
        df["train_preprocess_time"].fillna(0.0)
        + df["train_main_time"].fillna(0.0)
        + df["execute_preprocess_time"].fillna(0.0)
        + df["execute_main_time"].fillna(0.0)
        + df["execute_postprocess_time"].fillna(0.0)
    )
    df["orig_collection"] = df["collection"].copy()
    # adapt dataset names for GutenTAG datasets
    gt_collections = [
        "univariate-anomaly-test-cases",
        "multivariate-anomaly-test-cases",
        "multivariate-test-cases",
        "variable-length-test-cases",
        "correlation-anomalies",
    ]
    df["dataset_name"] = df["dataset"]
    gt_mask = df["collection"].isin(gt_collections)
    df.loc[gt_mask, "dataset_name"] = df.loc[gt_mask, "dataset"].str.split(".").str[0]

    # alias GutenTAG collection names:
    df_tmp = df[gt_mask].copy()
    df_tmp["collection"] = "GutenTAG"
    df = pd.concat([df, df_tmp], ignore_index=True)
    df_datasets = df[["collection", "dataset_name"]].drop_duplicates()

    with db.begin() as conn:
        df_available_datasets = pd.read_sql_table(
            table_name="dataset", con=conn, schema="autotsad"
        )
        if only_paper_datasets:
            df_available_datasets = df_available_datasets[
                df_available_datasets["paper"] == True
            ]
    df_datasets = pd.merge(
        df_datasets,
        df_available_datasets,
        left_on=["collection", "dataset_name"],
        right_on=["collection", "name"],
    )
    df_datasets = df_datasets[["collection", "dataset_name", "hexhash"]]

    print(f"Loading TimeEval results for datasets:")
    old_c = None
    for c, d in df_datasets[["collection", "dataset_name"]].values:
        if old_c != c:
            print(f"  {c}:")
        old_c = c
        print(f"    {d}")

    # only consider relevant datasets:
    df = pd.merge(df, df_datasets, on=["collection", "dataset_name"])
    # reduce to relevant columns
    metrics = [
        "RangePrAUC",
        "RangeRocAUC",
        "PrecisionAtK",
        "RangePrecision",
        "RangeRecall",
    ]
    metrics = [
        METRIC_MAPPING[m] for m in metrics if METRIC_MAPPING[m].name in df.columns
    ]
    columns = [
        "collection",
        "orig_collection",
        "dataset",
        "dataset_name",
        "hexhash",
        "algorithm",
        "hyper_params",
        "hyper_params_id",
        "overall_runtime",
    ]
    columns.extend([m.name for m in metrics])
    df = df[columns]

    if BASELINE_MAX_NAME in baselines:
        name = BASELINE_MAX_NAME
        print(f"Processing results for {name} baseline...")
        df_max = (
            df.sort_values(metric.name, ascending=False)
            .groupby(["collection", "hexhash"])
            .agg(
                {
                    **{m.name: "first" for m in metrics},
                    "algorithm": "first",
                    "hyper_params": "first",
                    "hyper_params_id": "first",
                    "overall_runtime": "first",
                    "dataset": "first",
                    "orig_collection": "first",
                }
            )
            .reset_index()
        )
        df_max["name"] = name
        print(f"{name} baseline results:")
        print(df_max)
        # df_save = df_max[["collection", "dataset", "algorithm", "hyper_params", "hyper_params_id", *[m.name for m in metrics], "overall_runtime"]]
        # df_save.to_csv("best-algo-iops-norma.csv", index=False)
        # print(df_save)

        with db.begin() as conn:
            for _, row in df_max.iterrows():
                upload_single_result(
                    name, row, results_path, conn, omit_scores=omit_scores
                )

    if BASELINE_MEAN_NAME in baselines:
        name = BASELINE_MEAN_NAME
        print(f"Processing results for {name} baseline...")
        df_mean = (
            df.groupby(["collection", "orig_collection", "dataset_name", "hexhash"])[
                [*[m.name for m in metrics], "overall_runtime"]
            ]
            .mean()
            .reset_index()
        )
        df_mean = df_mean.rename(columns={"dataset_name": "dataset"})
        df_mean["name"] = name
        print(f"{name} baseline results:")
        print(df_mean)

        with db.begin() as conn:
            for _, row in df_mean.iterrows():
                upload_single_result(
                    name, row, results_path, conn, omit_scores=omit_scores
                )

    if "kmeans" in baselines:
        name = BASELINE_KMEANS_NAME
        print(f"Processing results for {name} baseline...")
        df_kmeans = df[df["algorithm"] == "k-Means"].copy()
        df_kmeans["name"] = name
        print(f"{name} baseline results:")
        print(df_kmeans)

        with db.begin() as conn:
            for _, row in df_kmeans.iterrows():
                upload_single_result(
                    name, row, results_path, conn, omit_scores=omit_scores
                )

    if "sand" in baselines:
        name = BASELINE_SAND_NAME
        print(f"Processing results for {name} baseline...")
        df_kmeans = df[df["algorithm"] == "SAND"].copy()
        df_kmeans["name"] = name
        print(f"{name} baseline results:")
        print(df_kmeans)

        with db.begin() as conn:
            for _, row in df_kmeans.iterrows():
                upload_single_result(
                    name, row, results_path, conn, omit_scores=omit_scores
                )
