import argparse
import joblib

from pathlib import Path
from ..config import ALGORITHM_SELECTION_METHODS, SCORE_NORMALIZATION_METHODS, SCORE_AGGREGATION_METHODS


def register_db_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-user", type=str, default="autotsad", help="Database user")
    parser.add_argument("--db-password", type=str, default="holistic-tsad2023", help="Database user password")
    parser.add_argument("--db-host", type=str, default="172.17.17.32:5432", help="Database hostname (and port)")
    parser.add_argument("--db-database-name", type=str, default="akita", help="Database name.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to use.")

    # subsubparsers for the individual tasks
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    parser_load_dataset = subparsers.add_parser("load-dataset", help="Load a dataset into the database.")
    parser_load_dataset.add_argument("dataset_path", type=Path, help="Path to the dataset file!")
    parser_load_dataset.add_argument("--name", type=str, help="Name of the dataset.")
    parser_load_dataset.add_argument("--collection", type=str, help="Collection of the dataset.")
    parser_load_dataset.add_argument("--paper", action="store_true", help="Whether the dataset is used in a paper.")

    parser_load_result_backup = subparsers.add_parser("load-result-backup",
                                                      help="Load results from a backup into the database.")
    parser_load_result_backup.add_argument("result_backup_path", type=Path,
                                           help="Path to the result backup (tar.gz) file!")
    parser_load_result_backup.add_argument("--experiment-name", type=str,
                                           help="Name of the experiment. If not specified, the name is read from the "
                                                "backup filename.")
    parser_load_result_backup.add_argument("--experiment-description", type=str,
                                           help="Description of the experiment.")
    parser_load_result_backup.add_argument("--experiment-timestamp", type=str,
                                           help="Timestamp of the experiment. If not specified, the backup filename is "
                                                "analyzed for a timestamp.")
    choices = list(ALGORITHM_SELECTION_METHODS) + ["all"]
    parser_load_result_backup.add_argument("--selection-methods", type=str, nargs="*",
                                           default=["all"], choices=choices,
                                           help="Precompute the AutoTSAD rankings using the given strategies if not"
                                                "already present.")
    choices = list(SCORE_NORMALIZATION_METHODS) + ["all"]
    parser_load_result_backup.add_argument("--normalization-methods", type=str, nargs="*",
                                           default=["all"], choices=choices,
                                           help="Precompute the AutoTSAD rankings using the given strategies if not"
                                                "already present.")
    choices = list(SCORE_AGGREGATION_METHODS) + ["all"]
    parser_load_result_backup.add_argument("--aggregation-methods", type=str, nargs="*",
                                           default=["all"], choices=choices,
                                           help="Precompute the AutoTSAD rankings using the given strategies if not"
                                                "already present.")

    parser_load_all_result_backups = subparsers.add_parser("load-all-result-backups",
                                                           help="Load all result backups from a folder into the database.")
    parser_load_all_result_backups.add_argument("result_backup_folder", type=Path,
                                                help="Path to the folder containing the result backup files (tar.gz) !")
    parser_load_all_result_backups.add_argument("--experiment-description", type=str,
                                                help="Description used for all experiments.")
    parser_load_all_result_backups.add_argument("--skip-existing", action="store_true",
                                                help="Skip experiments, whose name already exists in the database."
                                                     "Otherwise, the experiment is overwritten.")

    parser_delete_experiment = subparsers.add_parser("delete-experiment",
                                                     help="Delete all results of an experiment from the database. "
                                                          "Either the experiment name or the experiment ID must be "
                                                          "specified.")
    parser_delete_experiment.add_argument("--experiment-name", type=str, help="Name of the experiment to delete.")
    parser_delete_experiment.add_argument("--experiment-id", type=int, help="ID of the experiment to delete.")

    parser_load_baselines = subparsers.add_parser("load-baseline-results",
                                                  help="Load baseline results from a result folder into the database.")
    parser_load_baselines.add_argument("result_folder", type=Path,
                                       help="Path to the folder containing the baseline result folders. Naming scheme: "
                                            "`<baseline-name>-<dataset_id>-<parameters>`.")
    parser_load_baselines.add_argument("--name", type=str, help="Name of the baseline. Overwrites the "
                                                                "name in the folder name.")
    parser_load_baselines.add_argument("--skip-existing", action="store_true",
                                       help="Skip baseline executions, whose name-dataset combination already exists "
                                            "in the database. Otherwise, the results are overwritten.")


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
            load_result_backup(db, args.result_backup_path,
                               name=args.experiment_name,
                               description=args.experiment_description,
                               timestamp=args.experiment_timestamp,
                               selection_methods=args.selection_methods,
                               normalization_methods=args.normalization_methods,
                               aggregation_methods=args.aggregation_methods)

    elif command == "load-all-result-backups":
        from .load_result_backup import load_result_backup
        with joblib.parallel_backend("loky", n_jobs=args.n_jobs):
            result_backup_path: Path
            for result_backup_path in args.result_backup_folder.glob("*.tar.gz"):
                if "failed" in str(result_backup_path):
                    print(f"\n-----\nSkipping {result_backup_path.name} because it contains 'failed' in its name.\n-----\n")
                    continue

                print(f"\n-----\nProcessing {result_backup_path.name}")
                load_result_backup(db, result_backup_path, description=args.experiment_description)

    elif command == "delete-experiment":
        from .delete_experiment_results import delete_experiment_results
        delete_experiment_results(db, experiment_id=args.experiment_id, name=args.experiment_name)

    elif command == "load-baseline-results":
        from .load_baseline_results import load_baseline_results
        with joblib.parallel_backend("loky", n_jobs=args.n_jobs):
            load_baseline_results(db, args.result_folder,
                                  baseline_name=args.name,
                                  skip_existing_experiments=args.skip_existing)
    else:
        raise ValueError(f"Unknown DB command: {command}")
