import argparse
from pathlib import Path

import joblib

from ..config import (
    ALGORITHM_SELECTION_METHODS,
    BASELINE_MAX_NAME,
    BASELINE_MEAN_NAME,
    METRIC_MAPPING,
    SCORE_AGGREGATION_METHODS,
    SCORE_NORMALIZATION_METHODS,
)


def register_db_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-user", type=str, default="autotsad", help="Database user")
    parser.add_argument(
        "--db-password",
        type=str,
        default="holistic-tsad2023",
        help="Database user password",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default="172.17.17.32:5432",
        help="Database hostname (and port)",
    )
    parser.add_argument(
        "--db-database-name", type=str, default="akita", help="Database name."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs to use."
    )

    # subsubparsers for the individual tasks
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    parser_load_dataset = subparsers.add_parser(
        "load-dataset", help="Load a dataset into the database."
    )
    parser_load_dataset.add_argument(
        "dataset_path", type=Path, help="Path to the dataset file!"
    )
    parser_load_dataset.add_argument("--name", type=str, help="Name of the dataset.")
    parser_load_dataset.add_argument(
        "--collection", type=str, help="Collection of the dataset."
    )
    parser_load_dataset.add_argument(
        "--paper", action="store_true", help="Whether the dataset is used in a paper."
    )

    parser_load_result_backup = subparsers.add_parser(
        "load-result-backup", help="Load results from a backup into the database."
    )
    parser_load_result_backup.add_argument(
        "result_backup_path", type=Path, help="Path to the result backup (tar.gz) file!"
    )
    parser_load_result_backup.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the experiment. If not specified, the name is read from the "
        "backup filename.",
    )
    parser_load_result_backup.add_argument(
        "--experiment-description", type=str, help="Description of the experiment."
    )
    parser_load_result_backup.add_argument(
        "--experiment-timestamp",
        type=str,
        help="Timestamp of the experiment. If not specified, the backup filename is "
        "analyzed for a timestamp.",
    )
    choices = list(ALGORITHM_SELECTION_METHODS) + ["all"]
    parser_load_result_backup.add_argument(
        "--selection-methods",
        type=str,
        nargs="*",
        default=["all"],
        choices=choices,
        help="Precompute the AutoTSAD rankings using the given strategies if not"
        "already present.",
    )
    choices = list(SCORE_NORMALIZATION_METHODS) + ["all"]
    parser_load_result_backup.add_argument(
        "--normalization-methods",
        type=str,
        nargs="*",
        default=["all"],
        choices=choices,
        help="Precompute the AutoTSAD rankings using the given strategies if not"
        "already present.",
    )
    choices = list(SCORE_AGGREGATION_METHODS) + ["all"]
    parser_load_result_backup.add_argument(
        "--aggregation-methods",
        type=str,
        nargs="*",
        default=["all"],
        choices=choices,
        help="Precompute the AutoTSAD rankings using the given strategies if not"
        "already present.",
    )

    parser_load_all_result_backups = subparsers.add_parser(
        "load-all-result-backups",
        help="Load all result backups from a folder into the database.",
    )
    parser_load_all_result_backups.add_argument(
        "result_backup_folder",
        type=Path,
        help="Path to the folder containing the result backup files (tar.gz) !",
    )
    parser_load_all_result_backups.add_argument(
        "--experiment-description",
        type=str,
        help="Description used for all experiments.",
    )
    parser_load_all_result_backups.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments, whose name already exists in the database."
        "Otherwise, the experiment is overwritten.",
    )

    parser_delete_experiment = subparsers.add_parser(
        "delete-experiment",
        help="Delete all results of an experiment from the database. "
        "Either the experiment name or the experiment ID must be "
        "specified.",
    )
    parser_delete_experiment.add_argument(
        "--experiment-name", type=str, help="Name of the experiment to delete."
    )
    parser_delete_experiment.add_argument(
        "--experiment-id", type=int, help="ID of the experiment to delete."
    )

    parser_load_baselines = subparsers.add_parser(
        "load-baseline-results",
        help="Load baseline results from a result folder into the database.",
    )
    parser_load_baselines.add_argument(
        "result_folder",
        type=Path,
        help="Path to the folder containing the baseline result folders. Naming scheme: "
        "`<baseline-name>-<dataset_id>-<parameters>`.",
    )
    parser_load_baselines.add_argument(
        "--name",
        type=str,
        help="Name of the baseline. Overwrites the " "name in the folder name.",
    )
    parser_load_baselines.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip baseline executions, whose name-dataset combination already exists "
        "in the database. Otherwise, the results are overwritten.",
    )

    parser_timeeval_baselines = subparsers.add_parser(
        "load-timeeval-results",
        help="Load TimeEval algorithm executions as baseline results from a "
        "TimeEval result folder into the database.",
    )
    parser_timeeval_baselines.add_argument(
        "results_path",
        type=Path,
        help="Path to the TimeEval results folder (needs to contain the"
        "'results.csv'-file and the individual anomaly scorings!",
    )
    parser_timeeval_baselines.add_argument(
        "--metric",
        type=str,
        choices=list(METRIC_MAPPING.keys()),
        default="RangePrAUC",
        help="The metric used for computing the max and mean quality.",
    )
    parser_timeeval_baselines.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=[BASELINE_MAX_NAME, BASELINE_MEAN_NAME, "kmeans", "sand"],
        help="The baselines to compute and upload to the DB.",
    )
    parser_timeeval_baselines.add_argument(
        "--only-paper-datasets",
        action="store_true",
        help="Use only the datasets described in the paper.",
    )
    parser_timeeval_baselines.add_argument(
        "--omit-scores",
        action="store_true",
        help="Do not upload the algorithm's anomaly scores to the DB.",
    )

    parser_missing_metrics = subparsers.add_parser(
        "calc-missing-baseline-metrics",
        help="Calculate missing metrics for the baseline executions by loading the "
        "dataset and scorings from the database, inserts the results back into the DB",
    )
    parser_missing_metrics.add_argument(
        "--all-datasets", action="store_true",
        help="Use all datasets instead of just the datasets described in the paper."
    )
    parser_missing_metrics.add_argument(
        "--name", type=str, default=None,
        help="Only consider experiments with the specified baseline name."
             "Default: use the baselines 'best-algo', 'k-Means (TimeEval)' and "
             "'SAND (TimeEval)'."
    )
    parser_missing_metrics.add_argument(
        "-f", "--force", action="store_true",
        help="Force new metrics, overwriting existing results, even if the new metrics "
             "differ from the existing ones (comparison is done via RANGE_PR_AUC and "
             "RANGE_ROC_AUC."
    )


def main(args: argparse.Namespace) -> None:
    # speed up startup by importing modules lazily
    from .database import Database

    db = Database.from_args(args)

    command = args.subcommand
    if command == "load-dataset":
        from .load_dataset import load_dataset

        load_dataset(db, args.dataset_path, args.name, args.collection, args.paper)

    elif command == "load-result-backup":
        from .load_result_backup import load_result_backup

        with joblib.parallel_backend("loky", n_jobs=args.n_jobs):
            load_result_backup(
                db,
                args.result_backup_path,
                name=args.experiment_name,
                description=args.experiment_description,
                timestamp=args.experiment_timestamp,
                selection_methods=args.selection_methods,
                normalization_methods=args.normalization_methods,
                aggregation_methods=args.aggregation_methods,
            )

    elif command == "load-all-result-backups":
        from .load_result_backup import load_result_backup

        with joblib.parallel_backend("loky", n_jobs=args.n_jobs):
            result_backup_path: Path
            for result_backup_path in args.result_backup_folder.glob("*.tar.gz"):
                if "failed" in str(result_backup_path):
                    print(
                        "\n-----\n"
                        f"Skipping {result_backup_path.name} because it contains 'failed' in its name."
                        "\n-----\n"
                    )
                    continue

                print(f"\n-----\nProcessing {result_backup_path.name}")
                load_result_backup(
                    db, result_backup_path, description=args.experiment_description
                )

    elif command == "delete-experiment":
        from .delete_experiment_results import delete_experiment_results

        delete_experiment_results(
            db, experiment_id=args.experiment_id, name=args.experiment_name
        )

    elif command == "load-baseline-results":
        from .load_baseline_results import load_baseline_results

        with joblib.parallel_backend("loky", n_jobs=args.n_jobs):
            load_baseline_results(
                db,
                args.result_folder,
                baseline_name=args.name,
                skip_existing_experiments=args.skip_existing,
            )
    elif command == "load-timeeval-results":
        from .load_timeeval_baseline_results import load_timeeval_baseline_results

        load_timeeval_baseline_results(
            db,
            args.results_path,
            args.metric,
            args.baselines,
            args.only_paper_datasets,
            args.omit_scores,
        )

    elif command == "calc-missing-baseline-metrics":
        from .calculate_baseline_metrics import calculate_baseline_metrics

        with joblib.parallel_backend("loky", n_jobs=args.n_jobs):
            calculate_baseline_metrics(
                db,
                all_datasets=args.all_datasets,
                baseline_name=args.name,
                force_overwrite=args.force,
            )

    else:
        raise ValueError(f"Unknown DB command: {command}")
