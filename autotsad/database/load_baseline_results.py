import json
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select, insert, delete
from timeeval.utils.hash_dict import hash_dict

from autotsad.database.database import Database

CONFIG_FILEPATH = "config.json"
DATASET_NAME_FILEPATH = "dataset.txt"
VERSION_FILEPATH = "version.txt"
HOSTNAME_FILEPATH = "hostname.txt"
METRIC_FILEPATH = "metrics.json"
SCORES_FILEPATH = "scores.csv"
RUNTIME_FILEPATH = "runtime.txt"


def _process(baseline_name: str, dataset_id: str, folder: Path, db_url: str, delete_existing: bool = False) -> None:
    db = Database(db_url)
    print(f"Processing {baseline_name} on {dataset_id}...")

    metric_filepath = folder / METRIC_FILEPATH
    score_filepath = folder / SCORES_FILEPATH
    if not metric_filepath.exists() or not score_filepath.exists():
        print(f"ERROR: Could not find baseline results in folder {folder}! "
              f"Missing {METRIC_FILEPATH} or {SCORES_FILEPATH}!")
        return

    metrics = metric_filepath.read_text(encoding="UTF-8")
    metrics = json.loads(metrics)
    print("  loaded metrics")

    scores = np.genfromtxt(score_filepath, delimiter=",")
    print("  loaded scores")

    config_filepath = folder / CONFIG_FILEPATH
    if config_filepath.exists():
        config = config_filepath.read_text(encoding="UTF-8")
        config = json.loads(config)
        print("  loaded config")
    else:
        print("  WARNING: No configuration file found!")
        config = {}

    runtime_filepath = folder / RUNTIME_FILEPATH
    if runtime_filepath.exists():
        runtime = float(runtime_filepath.read_text(encoding="UTF-8"))
        runtime /= 1e9
        print("  loaded runtime")
    else:
        print("  WARNING: No runtime file found!")
        runtime = None

    # perform uploads
    with db.begin() as conn:
        if delete_existing:
            res1 = conn.execute(delete(db.algorithm_scoring_table).where(
                db.algorithm_scoring_table.c.dataset_id == dataset_id,
                db.algorithm_scoring_table.c.algorithm == baseline_name
            )).rowcount
            res2 = conn.execute(delete(db.baseline_execution_table).where(
                db.baseline_execution_table.c.dataset_id == dataset_id,
                db.baseline_execution_table.c.name == baseline_name
            )).rowcount
            assert res1 <= 1 and res2 <= 1, "Tried to delete more than one execution!"
            if res1 == 1 or res2 == 1:
                print("  deleted existing execution and scoring")

        res = conn.execute(insert(db.algorithm_scoring_table).values({
            "dataset_id": dataset_id,
            "algorithm": baseline_name,
            "hyper_params_id": hash_dict(config),
            "hyper_params": config,
            "range_pr_auc": metrics["RangePrAUC"],
            "range_roc_auc": metrics["RangeRocAUC"],
            "precision_at_k": metrics["PrecisionAtK"],
            "runtime": runtime
        }).returning(db.algorithm_scoring_table.c.id)).first()
        scoring_id = res[0]

        df_scoring = pd.DataFrame()
        df_scoring["score"] = scores
        df_scoring["time"] = df_scoring.index
        df_scoring["algorithm_scoring_id"] = scoring_id
        df_scoring.to_sql(con=conn, **db.scoring_table_meta, if_exists="append", index=False)

        conn.execute(insert(db.baseline_execution_table).values({
            "dataset_id": dataset_id,
            "name": baseline_name,
            "algorithm_scoring_id": scoring_id,
            "runtime": runtime,
            "pr_auc": metrics["PrAUC"],
            "roc_auc": metrics["RocAUC"],
            "range_pr_auc": metrics["RangePrAUC"],
            "range_roc_auc": metrics["RangeRocAUC"],
            "range_pr_vus": metrics["RangePrVUS"],
            "range_roc_vus": metrics["RangeRocVUS"],
            "precision_at_k": metrics["PrecisionAtK"],
            "range_precision": metrics["RangePrecision"],
            "range_recall": metrics["RangeRecall"],
            "range_fscore": metrics["RangeFScore"],
            "precision": metrics["Precision"],
            "recall": metrics["Recall"],
            "fscore": metrics["FScore"],
        }))
    print(f"  uploaded scoring and added execution for {baseline_name} - {dataset_id}")
    print("... done.")


def load_baseline_results(db: Database,
                          result_path: Path,
                          baseline_name: Optional[str] = None,
                          skip_existing_experiments: bool = False) -> None:
    path = Path(result_path).resolve()
    if not path.exists() or path.is_file():
        raise ValueError(f"Path to the result folder ({path}) is invalid!")

    # load existing results from the DB
    with db.begin() as conn:
        df_existing = pd.read_sql_query(
            select(db.baseline_execution_table.c.name, db.baseline_execution_table.c.dataset_id).distinct(),
            con=conn
        )
        print(f"Found {len(df_existing)} existing baseline executions in the database.")
    df_existing = df_existing.set_index(["name", "dataset_id"])

    # parse folder structure to discover baselines results
    results_to_load = []
    for folder in path.iterdir():
        if "failed" in folder.name:
            print(f"Skipping failed experiment {folder.name}")
            continue
        if not folder.is_dir() or not re.match("(.*-)?[0-9a-f]{32}-.*", folder.name):
            continue

        name = folder.name.split('-')[0].strip()
        if len(list(filter(str.isdigit, name))) > 0:
            name = baseline_name
        if name == "" or name is None:
            raise ValueError(f"Invalid baseline name '{name}'! You can overwrite the baseline name with the --name "
                             "argument.")
        name = f"{name}-{folder.name.split('-')[-1].strip()}"
        dataset_id = folder.name.split('-')[-2].strip()

        if skip_existing_experiments and (name, dataset_id) in df_existing.index:
            print(f"Skipping existing experiment {name} - {dataset_id}")
            continue

        results_to_load.append((name, dataset_id, folder))
    print(f"Found {len(results_to_load)} baseline executions to load.")

    joblib.Parallel()(
        joblib.delayed(_process)(baseline_name, dataset_id, folder, db.url, delete_existing=not skip_existing_experiments)
        for baseline_name, dataset_id, folder in results_to_load
    )
