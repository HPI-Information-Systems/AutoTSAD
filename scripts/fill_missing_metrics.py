from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Dict

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select, update


sys.path.append(".")

from autotsad.util import format_time
from autotsad.database.database import Database
from autotsad.database.autotsad_connection import DB_METRIC_NAME_MAPPING
from autotsad.evaluation import compute_metrics
from autotsad.system.execution.aggregation import aggregate_scores, normalize_scores, ensure_finite_scores

ISOLATION_LEVEL = "READ COMMITTED"
METRIC_NAMES = list(DB_METRIC_NAME_MAPPING.values())


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate and insert missing metrics for AutoTSAD.")
    parser.add_argument("--db-host", type=str, default="172.17.17.32:5432",
                        help="Database hostname (and port)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs.")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Use all datasets instead of just the datasets described in the paper.")
    parser.add_argument("--autotsad-version", type=str, default=None,
                        help="Only consider experiments with the specified AutoTSAD version."
                             "Default: use all versions.")
    parser.add_argument("--config-id", type=str, default=None,
                        help="Only consider experiments with the specified AutoTSAD config ID."
                             "Default: use all config IDs.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force new metrics, overwriting existing results, even if the new metrics differ from the "
                             "existing ones.")
    return parser.parse_args(args)


def create_db_url(args: argparse.Namespace) -> str:
    db_user = "autotsad"
    db_pw = "holistic-tsad2023"
    db_database_name = "akita"

    return f"postgresql+psycopg2://{db_user}:{db_pw}@{args.db_host}/{db_database_name}"


def main(sys_args: List[str]) -> None:
    args = parse_args(sys_args)
    db = Database(create_db_url(args), ISOLATION_LEVEL)
    n_jobs = args.n_jobs
    only_paper_datasets = not args.all_datasets
    autotsad_version = args.autotsad_version
    config_id = args.config_id
    force_overwrite = args.force

    # load experiment result entries
    print("Loading AutoTSAD execution results...")
    df_experiments = _load_experiments(db, only_paper_datasets, autotsad_version, config_id)
    print(df_experiments)
    print(f"... found {len(df_experiments)} results with missing metrics.")

    # calculate missing metrics in parallel
    print("Calculating missing metrics...")
    dataset_groups = df_experiments.groupby("dataset_id")
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_calc_missing_metrics_for_dataset)(dataset_id, df, force_overwrite=force_overwrite, db_url=db.url)
        for dataset_id, df in iter(dataset_groups)
    )
    print("Done.")


def _load_experiments(db: Database,
                      only_paper_datasets: bool = True,
                      autotsad_version: Optional[str] = None,
                      config_id: Optional[str] = None) -> pd.DataFrame:
    with db.begin() as conn:
        query = select(db.autotsad_execution_table).where(
            (db.autotsad_execution_table.c.pr_auc.is_(None)) |
            (db.autotsad_execution_table.c.roc_auc.is_(None)) |
            (db.autotsad_execution_table.c.range_pr_auc.is_(None)) |
            (db.autotsad_execution_table.c.range_roc_auc.is_(None)) |
            (db.autotsad_execution_table.c.range_pr_vus.is_(None)) |
            (db.autotsad_execution_table.c.range_roc_vus.is_(None)) |
            (db.autotsad_execution_table.c.precision_at_k.is_(None)) |
            (db.autotsad_execution_table.c.precision.is_(None)) |
            (db.autotsad_execution_table.c.recall.is_(None)) |
            (db.autotsad_execution_table.c.fscore.is_(None)) |
            (db.autotsad_execution_table.c.range_precision.is_(None)) |
            (db.autotsad_execution_table.c.range_recall.is_(None)) |
            (db.autotsad_execution_table.c.range_fscore.is_(None))
        )
        if only_paper_datasets:
            query = query.where(db.autotsad_execution_table.c.dataset_id == db.dataset_table.c.hexhash,
                                db.dataset_table.c.paper == True)
        if autotsad_version is not None:
            query = query.where(db.autotsad_execution_table.c.autotsad_version == autotsad_version)
        if config_id is not None:
            query = query.where(db.autotsad_execution_table.c.config_id == config_id)

        df_experiments = pd.read_sql_query(query, con=conn)
    df_experiments[METRIC_NAMES] = df_experiments[METRIC_NAMES].astype(np.float_)
    return df_experiments.set_index("id")


def _calc_missing_metrics_for_dataset(dataset_id: str, df: pd.DataFrame, db_url: str,
                                      force_overwrite: bool = False) -> None:
    db = Database(db_url, ISOLATION_LEVEL)
    t0 = time.time_ns()
    dataset = db.load_test_dataset(dataset_id)
    duration = (time.time_ns() - t0)
    print(f"  loaded dataset {dataset_id} ({format_time(duration, is_ns=True, precision=2)})")

    for i, s in df.iterrows():
        _calc_missing_metrics(i, s, dataset, db, force_overwrite)


def _calc_missing_metrics(execution_id: int, s: pd.Series, dataset: "TestDataset", db: Database,
                          force_overwrite: bool = False) -> None:
    missing_metrics = [m for m in METRIC_NAMES if np.isnan(s[m])]
    print(f"  processing execution {execution_id}: {missing_metrics} ...")

    dataset_id = s["dataset_id"]
    assert dataset.hexhash == dataset_id, f"Dataset ID mismatch: {dataset.hexhash} != {dataset_id}"
    ranking_id = s["algorithm_ranking_id"]
    normalization_method = s["normalization_method"]
    aggregation_method = s["aggregation_method"]

    # load algorithm executions
    with db.begin() as conn:
        df_scorings = pd.read_sql_query(
            select(db.scoring_table)
            .where(db.scoring_table.c.algorithm_scoring_id == db.ranking_entry_table.c.algorithm_scoring_id,
                   db.ranking_entry_table.c.ranking_id == ranking_id),
            con=conn
        )

    df_scorings = df_scorings.pivot(index="time", columns="algorithm_scoring_id", values="score")
    print(f"    loaded scorings (shape={df_scorings.shape})")

    scores = normalize_scores(df_scorings.values, normalization_method)
    # check for NaNs and Infs again in case normalization did something strange
    scores, instances = ensure_finite_scores(scores, df_scorings.columns)
    df_scorings = pd.DataFrame(scores, columns=instances, dtype=np.float_)
    print(f"    normalized scores with {normalization_method}")

    # Combine scores
    combined_score = aggregate_scores(df_scorings.values, agg_method=aggregation_method)

    # compute metrics
    print(f"    computing metrics...")
    metrics = compute_metrics(dataset.label, combined_score, combined_score > 0)
    print(f"      aggregated score has RangePrAUC={metrics['RangePrAUC']}")
    print(f"    ... done computing {len(metrics)} metrics")
    try:
        if np.isfinite(s["range_pr_auc"]):
            np.testing.assert_allclose(metrics["RangePrAUC"], s["range_pr_auc"],
                                       rtol=1e-4,
                                       atol=1e-6,
                                       err_msg=f"New RangePrAUC did not match old metric: {metrics['RangePrAUC']} != {s['range_pr_auc']}")
        if np.isfinite(s["range_roc_auc"]):
            np.testing.assert_allclose(metrics["RangeRocAUC"], s["range_roc_auc"],
                                       rtol=1e-4,
                                       atol=1e-6,
                                       err_msg=f"New RangeRocAUC did not match old metric: {metrics['RangeRocAUC']} != {s['range_roc_auc']}")
    except AssertionError as e:
        if not force_overwrite:
            print(f"  ... skipping execution {execution_id} due to error: {e}")
            return
        else:
            print(f"  ... overwriting execution {execution_id} despite differences!")

    # translate column names
    entry = {}
    for m in metrics:
        _add_if_missing(entry, metrics, s, m, force=force_overwrite)
    print(f"    new metrics: {','.join(entry.keys())}")

    # upload ranking to DB
    with db.begin() as conn:
        conn.execute(
            update(db.autotsad_execution_table)
            .where(db.autotsad_execution_table.c.id == execution_id)
            .values(entry)
        )
    print(f"  ... finished processing execution {execution_id}")


def _add_if_missing(entry: Dict[str, float], metrics: Dict[str, float], s: pd.Series, metric_name: str,
                    force: bool = False) -> None:
    if force or np.isnan(s[DB_METRIC_NAME_MAPPING[metric_name]]):
        entry[DB_METRIC_NAME_MAPPING[metric_name]] = metrics[metric_name]


if __name__ == '__main__':
    main(sys.argv[1:])
