import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select, insert, text, delete
from sqlalchemy.exc import OperationalError
from timeeval.utils.hash_dict import hash_dict

from .database import Database
from ..config import ALGORITHM_SELECTION_METHODS, SCORE_NORMALIZATION_METHODS, SCORE_AGGREGATION_METHODS, METRIC_MAPPING
from ..dataset import TestDataset
from ..evaluation import evaluate_result
from ..system.execution.algo_selection import select_algorithm_instances
from ..util import format_time_ns


CONFIG_FILEPATH = "config.json"
CONFIG_FILEPATH_ALT1 = "autotsad.yaml"
CONFIG_FILEPATH_ALT2 = "autotsad-exp-config.yaml"
VERSION_FILEPATH = "version.txt"
EXECUTION_RESULTS_FILEPATH = "execution-results.csv"
METRIC_FILEPATH = "metrics.csv"
RUNTIME_FILEPATH = "runtimes.csv"
CACHE_FOLDERPATH = "tmp/cache"
SCORING_FOLDERPATH = "scores"
RANKING_FOLDERPATH = "rankings"


def _parse_name_for_timestamp(name: str) -> str:
    # name format: YYYY-MM-dd[-_]<experiment_name>.tar.gz
    timestamp_parts = name.replace("_", "-").split("-")[:3]
    if len(timestamp_parts) != 3:
        print(f"No timestamp at beginning of the experiment name ({name}) found (format: YYYY-MM-DD.*)!")
        return ""

    timestamp = "-".join(timestamp_parts)
    return f"{timestamp} 12:00:00+2"


def _clean_config(config: Dict[str, Any]) -> None:
    general_config: Dict[str, Any] = config["general"]
    keys = ["cache_key", "tmp_path", "result_path", "TIMESTAMP", "algorithm_selection_method",
            "score_normalization_method", "score_aggregation_method", "plot_final_scores", "compute_all_combinations",]
    for key in keys:
        if key in general_config:
            del general_config[key]

    if "data_gen_plotting" in config:
        del config["data_gen_plotting"]

    if "plot" in config["consolidation"]:
        del config["consolidation"]["plot"]


def _process_config(db: Database, tmpdir: Path) -> str:
    print("Processing AutoTSAD configuration...")
    print("  loading configuration...")
    config_file = tmpdir / CONFIG_FILEPATH
    if config_file.exists():
        with config_file.open("r") as fh:
            config_json = fh.read()
    else:
        print(f"  no {config_file}")
        config_file = tmpdir / CONFIG_FILEPATH_ALT1
        if not config_file.exists():
            print(f"  no {config_file}")
            config_file = tmpdir / CONFIG_FILEPATH_ALT2
        if not config_file.exists():
            print(f"  no {config_file}")
            raise ValueError(f"Could not find configuration file!")

        print(f"  using {config_file}")

        from nx_config import fill_config_from_path
        from autotsad.config import config

        fill_config_from_path(config, path=config_file)
        config_json = config.to_json()
    print("  ...successfully loaded AutoTSAD configuration!")

    print("  cleaning configuration...")
    config_json = json.loads(config_json)
    _clean_config(config_json)
    print("  ...cleaning done.")

    with db.begin() as conn:
        config_id = hash_dict(config_json)
        configs = conn.execute(select(db.configuration_table).where(db.configuration_table.c.id == config_id)).first()
        if configs:
            config_id = configs[0]
            print(f"Configuration already exists in the database! Using config {config_id}")
        else:
            res = conn.execute(insert(db.configuration_table), {"id": config_id, "config": config_json})
            config_id = res.inserted_primary_key[0]
            print(f"Configuration successfully uploaded to the database! Using config {config_id}")
    return config_id


def _extract_version(tmpdir: Path) -> str:
    print("Extracting AutoTSAD version...")
    version = "unknown"
    path = tmpdir / VERSION_FILEPATH
    if path.exists():
        with path.open("r") as fh:
            version = fh.read().strip()
        print(f"...reading AutoTSAD version from {path}: {version=}")
    else:
        print(f"...no version file found at path {path}: {version=}!")

    return version


def _load_execution_results(tmpdir: Path) -> pd.DataFrame:
    print("Loading AutoTSAD execution results...")
    results_path = tmpdir / EXECUTION_RESULTS_FILEPATH
    if not results_path.exists():
        raise ValueError(f"Expected exactly one result file, none found at {results_path}!")
    return pd.read_csv(results_path)


def _load_metrics(tmpdir: Path) -> pd.DataFrame:
    print("Loading ranking metrics...")
    metrics_path = tmpdir / METRIC_FILEPATH
    if not metrics_path.exists():
        raise ValueError(f"Expected exactly one metric file, none found at {metrics_path}!")
    df = pd.read_csv(metrics_path, index_col=0)
    df.index.name = "name"
    df = df.reset_index()
    columns = list(set().union(df.columns, METRIC_MAPPING.keys()))
    df = df.reindex(columns=columns, fill_value=np.nan)
    return df


def _calculate_runtime(tmpdir: Path) -> Optional[int]:
    path = tmpdir

    def _extract_runtime(file: Path) -> Optional[int]:
        df = pd.read_csv(file)
        try:
            return df.loc[(df["Timer"] == "autotsad") & (df["Type"] == "END"), "Duration (ns)"].item()
        except ValueError:
            return None

    def _extract_start_end(file: Path) -> Tuple[Optional[int], Optional[int]]:
        df = pd.read_csv(file)
        try:
            s = df.loc[(df["Timer"] == "Base TS generation") & (df["Type"] == "START"), "Begin (ns)"].item()
        except ValueError:
            s = None
        try:
            e = df.loc[(df["Timer"] == "Execution") & (df["Type"] == "END"), "End (ns)"].item()
        except ValueError:
            e = None
        return s, e

    start, end = None, None
    base_runtime_file = path / RUNTIME_FILEPATH
    if base_runtime_file.exists():
        runtime = _extract_runtime(base_runtime_file)
        if runtime is not None:
            return runtime

        start, end = _extract_start_end(base_runtime_file)

    path = tmpdir / CACHE_FOLDERPATH
    path = list(path.iterdir())[0]
    for file in path.glob("runtimes*.csv"):
        if start is not None and end is not None:
            return end - start
        start, end = _extract_start_end(file)

    return None


def _process_runtime_trace(db: Database, tmpdir: Path, experiment_id: int) -> Optional[int]:
    base_runtime_file = tmpdir / RUNTIME_FILEPATH
    print("Processing AutoTSAD runtime trace...")
    if not base_runtime_file.exists():
        print("...no runtime trace found!")

    df = pd.read_csv(base_runtime_file)
    df = df.rename(columns={
        "Timer": "trace_name",
        "Type": "trace_type",
        "Begin (ns)": "begin_ns",
        "End (ns)": "end_ns",
        "Duration (ns)": "duration_ns",
    })
    df["position"] = df.index.values
    df["experiment_id"] = experiment_id

    with db.begin() as conn:
        res = conn.execute(delete(db.runtime_trace_table).where(
            db.runtime_trace_table.c.experiment_id == experiment_id
        ))
        if res.rowcount != 0:
            print(f"  deleted {res.rowcount} existing runtime trace entries")

        n_uploaded = df.to_sql(con=conn, **db.runtime_trace_table_meta, if_exists="append", index=False)
        if n_uploaded != 0:
            print(f"  uploaded {n_uploaded} new runtime trace entries")

    print("  extracting overall runtime information")
    try:
        runtime = df.loc[(df["trace_name"] == "autotsad") & (df["trace_type"] == "END"), "duration_ns"].item()
    except ValueError:
        runtime = None
    print(f"...{n_uploaded} runtime traces uploaded (overall runtime={format_time_ns(runtime) if runtime else 'unknown'})")
    return runtime


def _process_scoring(scoring_id: str, dataset: str, algorithm: str, hyper_params_id: str, tmpdir: Path,
                     db_url: str) -> None:
    engine = Database.create_engine(db_url)

    scoring_path = tmpdir / SCORING_FOLDERPATH / f"{dataset}-{algorithm}-{hyper_params_id}.csv"
    if not scoring_path.exists():
        print("    ... missing scoring file, skipping!")
        # raise ValueError(f"Could not find scoring file {path}!")
        return

    t0 = time.time_ns()
    df_scoring = pd.DataFrame()
    df_scoring["score"] = np.genfromtxt(scoring_path, delimiter=",")
    df_scoring["time"] = df_scoring.index
    df_scoring["algorithm_scoring_id"] = scoring_id
    t1 = time.time_ns()

    with engine.begin() as conn:
        df_scoring.to_sql(con=conn, **Database.scoring_table_meta, if_exists="append", index=False)
    t2 = time.time_ns()
    print(f"    ... {dataset}-{algorithm}-{hyper_params_id} done (local {(t1 - t0) / 1e9:.2f}s, "
          f"DB {(t2 - t1) / 1e9:.2f}s)")


def _upload_ranking_results(db: Database, experiment_id: int, algorithm_scoring_ids: np.ndarray, entry: Dict[str, Any],
                            max_retries: int = 3) -> None:
    name = f"{entry['ranking_method']}-{entry['normalization_method']}-{entry['aggregation_method']}"

    def _upload_results(retries: int) -> None:
        try:
            with db.begin() as conn:
                # create ranking
                res = conn.execute(insert(db.ranking_table).values({"experiment_id": experiment_id}))
                ranking_id = res.inserted_primary_key[0]
                print(f"      {name}: created ranking with ID {ranking_id}")

                # create ranking entries
                df_ranking = pd.DataFrame(algorithm_scoring_ids, columns=["algorithm_scoring_id"])
                df_ranking["rank"] = df_ranking.index
                df_ranking["ranking_id"] = ranking_id
                df_ranking.to_sql(con=conn, **db.ranking_entry_table_meta, if_exists="append", index=False)
                print(f"      {name}: added {len(df_ranking)} ranking entries to ranking {ranking_id}")

                # create execution
                entry["algorithm_ranking_id"] = ranking_id
                if "experiment_id" not in entry:
                    entry["experiment_id"] = experiment_id
                conn.execute(insert(db.autotsad_execution_table).values(entry))
                print(f"      {name}: added execution ({experiment_id=})")
        except OperationalError as e:
            if "SerializationFailure" in repr(e):
                print(f"      {name}: serialization failure while uploading to DB, retrying {retries + 1}/{max_retries}!")
                if retries + 1 < max_retries:
                    time.sleep(0.5)
                    _upload_results(retries=retries + 1)
            else:
                raise

    _upload_results(retries=0)


def _process_ranking_method(selection_m: str, normalization_m: str, aggregation_m: str, dataset_id: str,
                            experiment_id: int, config_id: str, runtime: float, test_dataset: TestDataset,
                            df_algorithm_executions: pd.DataFrame, tmpdir: Path, version: str, db_url: str) -> None:
    db = Database(db_url)

    print(f"    processing {selection_m}-{normalization_m}-{aggregation_m}...")
    from autotsad.config import config
    from nx_config.test_utils import update_section

    update_section(config.general, tmp_path=tmpdir / "tmp")
    update_section(config.general, algorithm_selection_method=selection_m)
    update_section(config.general, score_normalization_method=normalization_m)
    update_section(config.general, score_aggregation_method=aggregation_m)
    update_section(config.general, cache_key=dataset_id)
    selected_results = select_algorithm_instances(df_algorithm_executions).reset_index(drop=True)
    method_metrics = evaluate_result(test_dataset, selected_results, tmpdir / SCORING_FOLDERPATH)

    _upload_ranking_results(
        db,
        experiment_id=experiment_id,
        algorithm_scoring_ids=selected_results["algorithm_scoring_id"].values,
        entry={
            "dataset_id": dataset_id,
            "config_id": config_id,
            "autotsad_version": version,
            "ranking_method": selection_m,
            "normalization_method": normalization_m,
            "aggregation_method": aggregation_m,
            "runtime": None if runtime is None else runtime / 1e9,  # convert to seconds
            "pr_auc": method_metrics["PrAUC"],
            "roc_auc": method_metrics["RocAUC"],
            "range_pr_auc": method_metrics["RangePrAUC"],
            "range_roc_auc": method_metrics["RangeRocAUC"],
            "range_pr_vus": method_metrics["RangePrVUS"],
            "range_roc_vus": method_metrics["RangeRocVUS"],
            "range_precision": method_metrics["RangePrecision"],
            "range_recall": method_metrics["RangeRecall"],
            "range_fscore": method_metrics["RangeFScore"],
            "precision_at_k": method_metrics["PrecisionAtK"],
            "precision": method_metrics["Precision"],
            "recall": method_metrics["Recall"],
            "fscore": method_metrics["FScore"],
        },
        max_retries=3
    )
    print(f"    ...done {selection_m}.")


def load_result_backup(db: Database,
                       result_backup_path: Path,
                       name: Optional[str] = None,
                       description: Optional[str] = None,
                       timestamp: Optional[str] = None,
                       selection_methods: List[str] = ["all"],
                       normalization_methods: List[str] = ["all"],
                       aggregation_methods: List[str] = ["all"],
                       skip_existing_experiments: bool = False) -> None:
    path = Path(result_backup_path).resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Path to the result backup ({path}) is invalid!")
    if not path.suffixes[-2:] == [".tar", ".gz"] or not tarfile.is_tarfile(path):
        raise ValueError(f"Result backup file ({path}) must be a .tar.gz file!")

    if selection_methods == ["all"]:
        selection_methods = ALGORITHM_SELECTION_METHODS
    if normalization_methods == ["all"]:
        normalization_methods = SCORE_NORMALIZATION_METHODS
    if aggregation_methods == ["all"]:
        aggregation_methods = SCORE_AGGREGATION_METHODS

    unknown_methods = set(selection_methods) - set(ALGORITHM_SELECTION_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown selection methods: {unknown_methods}")
    unknown_methods = set(normalization_methods) - set(SCORE_NORMALIZATION_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown normalization methods: {unknown_methods}")
    unknown_methods = set(aggregation_methods) - set(SCORE_AGGREGATION_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown aggregation methods: {unknown_methods}")

    experiment_name = name or path.name.split(".")[0]
    experiment_description = description or ""
    experiment_timestamp = timestamp or _parse_name_for_timestamp(path.name)

    # create experiment entry
    print(f"Creating experiment {experiment_name} ({experiment_timestamp})...")
    print(f"  description: {experiment_description}")
    with db.begin() as conn:
        res = conn.execute(
            select(db.experiment_table.c.id).where(db.experiment_table.c.name == experiment_name)).first()
        if res and skip_existing_experiments:
            print(f"Experiment already exists in the database! Skipping!")
            return
        elif res and not skip_existing_experiments:
            experiment_id = res[0]
            print(f"Experiment already exists in the database! Using experiment ID {experiment_id}")
        else:
            res = conn.execute(insert(db.experiment_table), {"name": experiment_name,
                                                             "description": experiment_description,
                                                             "date": experiment_timestamp})
            experiment_id = res.inserted_primary_key[0]
            print(f"Experiment successfully uploaded to the database! Using experiment ID {experiment_id}")

    # extract backup to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Extracting result backup to {tmpdir}")
        with tarfile.open(path) as tar:
            tar.extractall(path=tmpdir)
        tmpdir = Path(tmpdir).resolve()

        # load (or create) configuration
        config_id = _process_config(db, tmpdir)

        # load AutoTSAD version
        version = _extract_version(tmpdir)

        # load AutoTSAD runtime trace
        runtime = _process_runtime_trace(db, tmpdir, experiment_id)

        # load AutoTSAD execution results
        results = _load_execution_results(tmpdir)
        print(f"...loaded {results.shape[0]} results, processing ...")

        dataset_id = results["dataset_id"].unique().item()

        # upload scorings
        df_algorithm_scoring = results[["dataset_id", "algorithm"]].copy()
        df_algorithm_scoring["experiment_id"] = experiment_id
        df_algorithm_scoring["hyper_params"] = results["params"]
        df_algorithm_scoring["hyper_params_id"] = df_algorithm_scoring["hyper_params"].apply(lambda x: hash_dict(json.loads(x)))
        df_algorithm_scoring["range_pr_auc"] = results["quality"]
        # df_algorithm_scoring["range_roc_auc"] = ...
        # df_algorithm_scoring["precision_at_k"] = ...
        # df_algorithm_scoring["runtime"] = ...

        with db.begin() as conn:
            # filter out existing scorings
            df_existing_scorings = pd.read_sql_query(select(
                db.algorithm_scoring_table.c.id, db.algorithm_scoring_table.c.dataset_id,
                db.algorithm_scoring_table.c.algorithm, db.algorithm_scoring_table.c.hyper_params_id
            ).where(
                db.algorithm_scoring_table.c.experiment_id == experiment_id
            ), con=conn, index_col="id")
            df_algorithm_scoring = df_algorithm_scoring[~(
                    (df_algorithm_scoring["dataset_id"].isin(df_existing_scorings["dataset_id"])) &
                    (df_algorithm_scoring["algorithm"].isin(df_existing_scorings["algorithm"])) &
                    (df_algorithm_scoring["hyper_params_id"].isin(df_existing_scorings["hyper_params_id"]))
            )]

            # only upload new scorings
            if len(df_algorithm_scoring) != 0:
                print(f"  uploading {len(df_algorithm_scoring)} new algorithm scoring metadata")
                # create scoring metadata
                df_algorithm_scoring.to_sql(con=conn, **db.algorithm_scoring_table_meta, if_exists="append",
                                            index=False)

        # also upload scorings (but we need the IDs)
        with db.begin() as conn:
            missing_scorings = conn.execute(
                text(f"""select distinct id, dataset_id, algorithm, hyper_params_id
                             from algorithm_scoring
                             where id not in (select distinct algorithm_scoring_id from scoring)
                             and experiment_id = '{experiment_id}'
                             and range_pr_auc is not null;""")
            ).fetchall()
        if len(missing_scorings) != 0:
            print(f"  found {len(missing_scorings)} missing scorings, uploading...")
            joblib.Parallel()(
                joblib.delayed(_process_scoring)(
                    scoring_id, dataset, algorithm, hyper_params_id, tmpdir, db.url
                )
                for scoring_id, dataset, algorithm, hyper_params_id in missing_scorings
            )

            print("  uploaded all missing scorings")
        else:
            print("  all algorithm scorings already exist in the database")

        # upload algorithm executions
        print("  uploading algorithm execution metadata...")
        with db.begin() as conn:
            res = conn.execute(delete(db.algorithm_execution_table).where(
                db.algorithm_execution_table.c.experiment_id == experiment_id
            ))
            if res.rowcount != 0:
                print(f"    deleted {res.rowcount} existing algorithm execution entries")

            df_algorithm_executions = results[
                ["dataset_id", "algorithm", "no_datasets", "mean_train_quality", "quality"]].copy()
            df_algorithm_executions["hyper_params_id"] = results["params"].apply(lambda x: hash_dict(json.loads(x)))
            df_algorithm_executions["experiment_id"] = experiment_id

            # read scoring IDs
            df_scorings = pd.read_sql_query(select(
                db.algorithm_scoring_table.c.id, db.algorithm_scoring_table.c.dataset_id,
                db.algorithm_scoring_table.c.algorithm, db.algorithm_scoring_table.c.hyper_params_id
            ).where(
                db.algorithm_scoring_table.c.experiment_id == experiment_id
            ), con=conn)

            df_algorithm_executions = pd.merge(
                df_algorithm_executions.set_index(["dataset_id", "algorithm", "hyper_params_id"]),
                df_scorings.set_index(["dataset_id", "algorithm", "hyper_params_id"]),
                how="left", left_index=True, right_index=True
            ).reset_index()
            df_algorithm_executions = df_algorithm_executions.rename(columns={"id": "algorithm_scoring_id"})
            df_tmp = df_algorithm_executions.drop(columns=["dataset_id", "algorithm", "hyper_params_id"])
            df_tmp.to_sql(con=conn, **db.algorithm_execution_table_meta, if_exists="append", index=False)
        print(f"  ...uploaded algorithm execution metadata ({len(df_tmp)})")

        # load existing rankings
        df_metrics = _load_metrics(tmpdir)
        requested_ranking_methods = [(selection, normalization, aggregation)
                               for selection in selection_methods
                               for normalization in normalization_methods
                               for aggregation in aggregation_methods]
        existing_rankings = df_metrics["name"].unique().tolist()
        pending_rankings = [
            (s, n, a)
            for s, n, a in requested_ranking_methods
            if f"{s}-{n}-{a}" not in existing_rankings
        ]

        # runtime = _calculate_runtime(tmpdir)
        # print(f"  extracted runtime of {'unknown' if runtime is None else format_time_ns(runtime)} from the results")

        if len(df_metrics) != 0 or len(pending_rankings) != 0:
            print("  deleting existing rankings in DB...")
            with db.begin() as conn:
                res = conn.execute(delete(db.autotsad_execution_table).where(
                    db.autotsad_execution_table.c.experiment_id == experiment_id
                ))
                if res.rowcount != 0:
                    print(f"    deleted {res.rowcount} existing execution entries")
                res = conn.execute(delete(db.ranking_table).where(
                    db.ranking_table.c.experiment_id == experiment_id
                ))
                if res.rowcount != 0:
                    print(f"    deleted {res.rowcount} existing rankings")

        if len(df_metrics) != 0:
            print("  loading existing rankings...")
            for i, s in df_metrics.iterrows():
                print(f"    ... processing ranking for {s['name']}")
                # lookup algorithm_scoring_ids
                instances_path = tmpdir / RANKING_FOLDERPATH / s["name"] / "selected-instances.csv"
                df_instances = pd.read_csv(instances_path)[["dataset_id", "algorithm", "params"]]
                df_instances["hyper_params_id"] = df_instances["params"].apply(lambda x: hash_dict(json.loads(x)))
                df_instances = pd.merge(
                    df_instances.set_index(["dataset_id", "algorithm", "hyper_params_id"]),
                    df_scorings.set_index(["dataset_id", "algorithm", "hyper_params_id"]),
                    how="left", left_index=True, right_index=True
                ).reset_index()
                df_instances = df_instances.rename(columns={"id": "algorithm_scoring_id"})

                # upload to database
                _upload_ranking_results(
                    db,
                    experiment_id=experiment_id,
                    algorithm_scoring_ids=df_instances["algorithm_scoring_id"].values,
                    entry={
                        "dataset_id": dataset_id,
                        "config_id": config_id,
                        "autotsad_version": version,
                        "ranking_method": s["selection-method"],
                        "normalization_method": s["normalization-method"],
                        "aggregation_method": s["aggregation-method"],
                        "runtime": None if runtime is None else runtime / 1e9,  # convert to seconds
                        "pr_auc": s["PrAUC"],
                        "roc_auc": s["RocAUC"],
                        "range_pr_auc": s["RangePrAUC"],
                        "range_roc_auc": s["RangeRocAUC"],
                        "range_pr_vus": s["RangePrVUS"],
                        "range_roc_vus": s["RangeRocVUS"],
                        "range_precision": s["RangePrecision"],
                        "range_recall": s["RangeRecall"],
                        "range_fscore": s["RangeFScore"],
                        "precision_at_k": s["PrecisionAtK"],
                        "precision": s["Precision"],
                        "recall": s["Recall"],
                        "fscore": s["FScore"],
                    },
                    max_retries=3
                )
                print(f"    ...done for {s['name']}.")

        # compute remaining rankings
        if len(pending_rankings) != 0:
            print("  precomputing rankings...")
            print("    loading test dataset to compute the metrics")
            df_algorithm_executions["params"] = results["params"].apply(json.loads)
            df_algorithm_executions = df_algorithm_executions[~df_algorithm_executions["quality"].isna()]
            test_dataset = db.load_test_dataset(dataset_id)
            df_algorithm_executions["dataset"] = test_dataset.name

            joblib.Parallel()(
                joblib.delayed(_process_ranking_method)(
                    s, n, a, dataset_id, experiment_id, config_id, runtime, test_dataset, df_algorithm_executions,
                    tmpdir, version, db.url
                )
                for s, n, a in pending_rankings
            )
            print("  ...done precomputing rankings.")
        print(f"...done processing result backup {tmpdir} ({experiment_name=}, {experiment_id=}).")
