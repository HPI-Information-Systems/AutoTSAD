import argparse
import json
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select, insert

sys.path.append(".")

from autotsad.database.database import Database
from autotsad.evaluation import compute_metrics
from autotsad.rra import trimmed_partial_borda, minimum_influence
from autotsad.system.execution.algo_selection import select_algorithm_instances
from autotsad.system.execution.aggregation import aggregate_scores, normalize_scores, ensure_finite_scores, \
    algorithm_instances


ISOLATION_LEVEL = "READ COMMITTED"
ROBUST_BORDA_NAME = "aggregated-robust-borda"
MIM_NAME = "aggregated-minimum-influence"
RANKING_METHODS = (
    # "training-coverage",
    "training-quality",
    "training-result",
    "affinity-propagation-clustering",
    "kmedoids-clustering",
    "greedy-euclidean", "greedy-annotation-overlap",
    "mmq-euclidean", "mmq-annotation-overlap",
    # "interchange-euclidean", "interchange-annotation-overlap"
)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute missing rank aggregation ensembling strategies for AutoTSAD "
                                                 "and write them back to DB.")
    parser.add_argument("--db-host", type=str, default="172.17.17.32:5432",
                        help="Database hostname (and port)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs.")
    parser.add_argument("--max-instances", type=int, default=6,
                        help="Maximum number of instances to select.")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Use all datasets instead of just the datasets described in the paper.")
    parser.add_argument("--autotsad-version", type=str, default="0.2.1",
                        help="Only consider experiments with the specified AutoTSAD version.")
    parser.add_argument("--config-id", type=str, default="26a8aa1409ae87761d9d405ff0e49f9e",
                        help="Only consider experiments with the specified AutoTSAD config ID.")
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
    max_instances = args.max_instances
    only_paper_datasets = not args.all_datasets

    # load results with missing aggregated rankings
    print("Loading AutoTSAD execution results...")
    missing_rb_eids = _load_experiments_for_missing(db, ROBUST_BORDA_NAME, only_paper_datasets, args.autotsad_version, args.config_id)
    missing_mim_eids = _load_experiments_for_missing(db, MIM_NAME, only_paper_datasets, args.autotsad_version, args.config_id)
    df_tmp = pd.concat([missing_rb_eids, missing_mim_eids])
    missing_both_eids = set(set(missing_rb_eids.index) & set(missing_mim_eids.index))
    missing_rb_eids = missing_rb_eids.loc[list(set(missing_rb_eids.index) - missing_both_eids), :]
    missing_mim_eids = missing_mim_eids.loc[list(set(missing_mim_eids.index) - missing_both_eids), :]
    missing_both_eids = df_tmp.loc[list(missing_both_eids), :].drop_duplicates()
    print(f"... found {len(missing_rb_eids)} results with missing {ROBUST_BORDA_NAME}.")
    print(f"... found {len(missing_mim_eids)} results with missing {MIM_NAME}.")
    print(f"... found {len(missing_both_eids)} results with missing {ROBUST_BORDA_NAME} and {MIM_NAME}.")

    # create jobs
    jobs = [(s.name, *s, (ROBUST_BORDA_NAME, MIM_NAME)) for _, s in missing_both_eids.iterrows()]
    jobs += [(s.name, *s, (ROBUST_BORDA_NAME,)) for _, s in missing_rb_eids.iterrows()]
    jobs += [(s.name, *s, (MIM_NAME,)) for _, s in missing_mim_eids.iterrows()]

    # calculate new rankings in parallel
    print("Calculating rank aggregation methods...")
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_calc_aggregated_rankings)(*args, max_instances=max_instances, db_url=db.url)
        for args in jobs
    )
    print("Done.")


def _load_experiments_for_missing(db: Database, method_name: str,
                                  only_paper_datasets: bool = True,
                                  autotsad_version: str = "0.2.1",
                                  config_id: str = "26a8aa1409ae87761d9d405ff0e49f9e") -> pd.DataFrame:
    with db.begin() as conn:
        experiments_with_method = conn.execute(
            select(db.autotsad_execution_table.c.experiment_id).distinct()
            .where(
                db.autotsad_execution_table.c.dataset_id == db.dataset_table.c.hexhash,
                db.dataset_table.c.paper == only_paper_datasets,
                db.autotsad_execution_table.c.autotsad_version == autotsad_version,
                db.autotsad_execution_table.c.config_id == config_id,
                db.autotsad_execution_table.c.ranking_method == method_name
            )
        ).fetchall()
        experiments_with_method = [e[0] for e in experiments_with_method]
        df_experiments = pd.read_sql_query(
            select(
                db.autotsad_execution_table.c.experiment_id,
                db.autotsad_execution_table.c.dataset_id,
                db.autotsad_execution_table.c.config_id,
                db.autotsad_execution_table.c.autotsad_version,
                db.autotsad_execution_table.c.runtime,
            ).distinct()
            .where(
                db.autotsad_execution_table.c.dataset_id == db.dataset_table.c.hexhash,
                db.dataset_table.c.paper == only_paper_datasets,
                db.autotsad_execution_table.c.autotsad_version == autotsad_version,
                db.autotsad_execution_table.c.config_id == config_id,
                db.autotsad_execution_table.c.experiment_id.not_in(experiments_with_method)
            ),
            con=conn
        )
    return df_experiments.set_index("experiment_id")


def _calc_aggregated_rankings(experiment_id: int, dataset_id: str, config_id: int, autotsad_version: str,
                              runtime: float, methods: Tuple[str, ...], max_instances: int, db_url: str) -> None:
    db = Database(db_url, ISOLATION_LEVEL)
    print(f"  processing execution {experiment_id}: {methods}...")

    # load algorithm executions
    with db.begin() as conn:
        df_results = pd.read_sql_query(
            select(db.algorithm_execution_table).where(db.algorithm_execution_table.c.experiment_id == experiment_id),
            conn
        )
        df_results = df_results[~df_results["quality"].isna()]
        # dummy algorithm instance ID:
        df_results = df_results.rename(columns={"id": "algorithm"})
        df_results["algorithm"] = df_results["algorithm"].astype(str)
        df_results["params"] = "{}"
        df_results["params"] = df_results["params"].apply(json.loads)
        scoring_ids = df_results["algorithm_scoring_id"].unique().tolist()
        df_scorings = pd.read_sql_query(
            select(db.scoring_table)
            .where(db.scoring_table.c.algorithm_scoring_id.in_(scoring_ids))
            .order_by(db.scoring_table.c.algorithm_scoring_id, db.scoring_table.c.time),
            conn
        )
    all_scores = df_scorings.pivot(index="time", columns="algorithm_scoring_id", values="score").values
    # check for NaNs and Infs in case some algorithm did something strange
    all_scores, instances = ensure_finite_scores(all_scores, algorithm_instances(df_results))
    all_results = df_results[algorithm_instances(df_results).isin(instances)]
    all_instances = sorted(instances)
    print(f"    loaded {len(all_instances)} instances")

    for normalization_method in ["minmax", "gaussian"]:
        scores = normalize_scores(all_scores, normalization_method)
        # check for NaNs and Infs again in case normalization did something strange
        scores, instances = ensure_finite_scores(scores, all_instances)
        results = all_results[algorithm_instances(all_results).isin(instances)].copy()
        scores = pd.DataFrame(scores, columns=instances, dtype=np.float_)
        print(f"    normalized scores with {normalization_method}")

        # compute individual rankings
        ranks = pd.DataFrame(len(instances), columns=RANKING_METHODS, index=instances, dtype=np.int_)
        for ranking_method in RANKING_METHODS:
            maxi = results.shape[0]
            if ranking_method in ["kmedoids-clustering", "affinity-propagation-clustering"]:
                maxi = max_instances
            df = select_algorithm_instances(results, scores, selection_method=ranking_method, max_instances=maxi)
            ranked_instances = algorithm_instances(df.reset_index(drop=True))
            ranks.loc[ranked_instances.values, ranking_method] = ranked_instances.index
        print(f"      computed {len(RANKING_METHODS)} individual rankings")

        for method in methods:
            # compute aggregated ranking
            if method == ROBUST_BORDA_NAME:
                aggregated_ranks = trimmed_partial_borda(
                    ranks=ranks.values.T, aggregation_type="borda", metric="influence"
                )
            else:  # method == MIM_NAME
                aggregated_ranks = minimum_influence(ranks.values.T, aggregation_type="borda")

            selected_instances = pd.Series(aggregated_ranks, index=instances, dtype=np.int_)
            selected_instances = selected_instances[selected_instances < max_instances].sort_values()
            print(f"      best instances using {method}: {selected_instances.index.tolist()}")

            selected_results = all_results[algorithm_instances(all_results).isin(selected_instances.index)].copy()
            selected_results["name"] = algorithm_instances(selected_results)
            selected_results["rank"] = selected_results["name"].map(selected_instances)
            selected_results = selected_results.sort_values("rank")
            selected_scores = scores[selected_results["name"]]
            selected_results = selected_results.drop(columns=["name"])
            if selected_results["rank"].unique().shape[0] != selected_results.shape[0]:
                selected_results["rank"] = np.arange(selected_results.shape[0])
                print(f"      duplicate ranks detected and fixed")

            execution_entries = {}
            combined_scores = {}
            for aggregation_method in ["max", "custom"]:
                # Combine scores
                combined_score = aggregate_scores(selected_scores.values, agg_method=aggregation_method)

                # compute metrics
                dataset = db.load_test_dataset(dataset_id)
                metrics = compute_metrics(dataset.label, combined_score, combined_score > 0)
                print(f"      {aggregation_method} aggregated score quality: {metrics['RangePrAUC']}")

                combined_scores[aggregation_method] = combined_score
                execution_entries[aggregation_method] = {
                    "range_pr_auc": metrics["RangePrAUC"],
                    "range_roc_auc": metrics["RangeRocAUC"],
                    "precision_at_k": metrics["RangePrecision"],
                    "precision": metrics["RangeRecall"],
                    "recall": metrics["PrecisionAtK"],
                }

            # upload ranking to DB
            with db.begin() as conn:
                # create ranking
                res = conn.execute(insert(db.ranking_table).values({"experiment_id": experiment_id}))
                ranking_id = res.inserted_primary_key[0]
                # create ranking entries
                df_ranking = selected_results[["algorithm_scoring_id", "rank"]].copy()
                df_ranking["ranking_id"] = ranking_id
                df_ranking.to_sql(con=conn, **db.ranking_entry_table_meta, if_exists="append", index=False)
                print(f"      uploaded {len(df_ranking)} ranking entries to new ranking {ranking_id}")

                entries = []
                for aggregation_method in execution_entries:
                    entries.append({
                        "experiment_id": experiment_id,
                        "dataset_id": dataset_id,
                        "config_id": config_id,
                        "autotsad_version": autotsad_version,
                        "ranking_method": method,
                        "normalization_method": normalization_method,
                        "aggregation_method": aggregation_method,
                        "runtime": runtime,
                        "algorithm_ranking_id": ranking_id,
                        **execution_entries[aggregation_method],
                    })
                conn.execute(insert(db.autotsad_execution_table).values(entries))
                print(f"      [{normalization_method},{method}] added 2 executions for experiment {experiment_id}")
    print(f"  ... finished processing execution {experiment_id}: {methods}")


if __name__ == '__main__':
    main(sys.argv[1:])
