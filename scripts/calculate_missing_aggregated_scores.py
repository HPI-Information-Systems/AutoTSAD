import argparse
import sys
from typing import List

import joblib
import pandas as pd
from sqlalchemy import select, distinct, insert, update


sys.path.append(".")

from autotsad.evaluation import compute_metrics
from autotsad.system.execution.aggregation import aggregate_scores, normalize_scores
from autotsad.database.database import Database


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load AutoTSAD results from DB, calculate aggregated scorings, "
                                                 "and write them back to DB.")
    parser.add_argument("--db-host", type=str, default="172.17.17.32:5432",
                        help="Database hostname (and port)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs to use for calculating the aggregated scorings.")
    parser.add_argument("--update-metrics", action="store_true",
                        help="Recompute the metrics of the AutoTSAD execution based on the aggregated scoring.")
    return parser.parse_args(args)


def create_db_url(args: argparse.Namespace) -> str:
    db_user = "autotsad"
    db_pw = "holistic-tsad2023"
    db_database_name = "akita"

    return f"postgresql+psycopg2://{db_user}:{db_pw}@{args.db_host}/{db_database_name}"


def main(sys_args: List[str]) -> None:
    args = parse_args(sys_args)
    db = Database(create_db_url(args), "READ COMMITTED")
    update_metrics = args.update_metrics

    # load results with missing aggregated scorings
    print("Loading AutoTSAD execution results...")
    with db.begin() as conn:
        df_results = pd.read_sql_query(
            select(
                db.autotsad_execution_table.c.id, db.autotsad_execution_table.c.dataset_id,
                db.autotsad_execution_table.c.ranking_method, db.autotsad_execution_table.c.normalization_method,
                db.autotsad_execution_table.c.aggregation_method, db.autotsad_execution_table.c.algorithm_ranking_id,
                db.autotsad_execution_table.c.experiment_id
            ).where(db.autotsad_execution_table.c.aggregated_scoring_id == None)
            .order_by(db.autotsad_execution_table.c.dataset_id, db.autotsad_execution_table.c.normalization_method),
            con=conn
        )
    print(f"... found {len(df_results)} results with missing aggregated scorings.")

    # calculate aggregated scorings in parallel
    print("Calculating aggregated scorings...")
    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(_calc_combined_scoring)(
            e_id, dataset_id, rmethod, nmethod, amethod, ranking_id, experiment_id, update_metrics, db.url
        )
        for _, (e_id, dataset_id, rmethod, nmethod, amethod, ranking_id, experiment_id) in df_results.iterrows()
    )
    print("Done.")


def _calc_combined_scoring(e_id: int, dataset_id: str, rmethod: str, nmethod: str, amethod: str, ranking_id: int,
                           experiment_id: int, update_metrics: bool, db_url: str) -> None:
    db = Database(db_url, "READ COMMITTED")
    print(f"  processing execution {e_id} ({rmethod} {nmethod} {amethod}) from experiment {experiment_id}")

    with db.begin() as conn:
        # fetch scores
        scoring_ids = conn.execute(
            select(distinct(db.ranking_entry_table.c.algorithm_scoring_id)).where(
                db.ranking_table.c.id == ranking_id,
                db.ranking_table.c.id == db.ranking_entry_table.c.ranking_id,
                db.ranking_entry_table.c.algorithm_scoring_id == db.algorithm_scoring_table.c.id
            )
        ).fetchall()
        scoring_ids = [s[0] for s in scoring_ids]
        df_scorings = pd.read_sql(
            select(db.scoring_table)
            .where(db.scoring_table.c.algorithm_scoring_id.in_(scoring_ids))
            .order_by(db.scoring_table.c.algorithm_scoring_id, db.scoring_table.c.time),
            conn
        )

        # compute aggregated scoring
        scores = df_scorings.pivot(index="time", columns="algorithm_scoring_id", values="score").values
        scores = normalize_scores(scores, normalization_method=nmethod)
        combined_scoring = aggregate_scores(scores, agg_method=amethod)

        if update_metrics:
            dataset = db.load_test_dataset(dataset_id)
            metrics = compute_metrics(dataset.label, combined_scoring, combined_scoring > 0)
            metrics = {
                "range_pr_auc": metrics["RangePrAUC"],
                "range_roc_auc": metrics["RangeRocAUC"],
                "precision_at_k": metrics["RangePrecision"],
                "precision": metrics["RangeRecall"],
                "recall": metrics["PrecisionAtK"],
            }
        else:
            metrics = {}

        # write back to DB: aggregated scoring
        res = conn.execute(insert(db.aggregated_scoring_table).values({
            "dataset_id": dataset_id,
            "experiment_id": experiment_id
        }))
        aggregated_scoring_id = res.inserted_primary_key[0]

        # write back to DB: aggregated scoring scores
        df_combined_scoring_scores = pd.DataFrame({"score": combined_scoring})
        df_combined_scoring_scores["aggregated_scoring_id"] = aggregated_scoring_id
        df_combined_scoring_scores.index.name = "time"
        df_combined_scoring_scores = df_combined_scoring_scores.reset_index()
        df_combined_scoring_scores.to_sql(
            con=conn, **db.aggregated_scoring_scores_table_meta, if_exists="append", index=False
        )

        # write back to DB: update execution
        conn.execute(update(db.autotsad_execution_table).where(db.autotsad_execution_table.c.id == e_id).values({
            "aggregated_scoring_id": aggregated_scoring_id,
            **metrics
        }))


if __name__ == '__main__':
    main(sys.argv[1:])
