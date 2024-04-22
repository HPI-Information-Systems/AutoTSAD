import argparse
import sys
from typing import List

import shtab
from periodicity_detection.__main__ import main as estimate_periodicity
from periodicity_detection.__main__ import register_periodicity_arguments

from ._version import __version__
from .baselines.__main__ import main as run_baseline
from .baselines.__main__ import register_baseline_arguments
from .database.cli import main as manage_db
from .database.cli import register_db_arguments
from .system.main import main as run_autotsad
from .system.main import register_autotsad_arguments


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "autotsad",
        description="Unsupervised anomaly detection system for univariate time series.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"AutoTSAD v{__version__}",
        help="Show version number of AutoTSAD.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_completion = subparsers.add_parser(
        "completion", help="Output shell completion script"
    )
    shtab.add_argument_to(parser_completion, option_string="completion", parent=parser)

    parser_run = subparsers.add_parser("run", help="Run AutoTSAD on a given dataset.")
    register_autotsad_arguments(parser_run)

    parser_db = subparsers.add_parser("db", help="Manage AutoTSAD result database.")
    register_db_arguments(parser_db)

    parser_period = subparsers.add_parser(
        "estimate-period",
        help="Estimate the period size of a given time series " "dataset.",
    )
    register_periodicity_arguments(parser_period)

    parser_baselines = subparsers.add_parser(
        "baselines", help="Run a baseline algorithm on a given dataset."
    )
    register_baseline_arguments(parser_baselines)
    return parser


def main(argv: List[str] = sys.argv[1:]) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_autotsad(args)

    elif args.command == "db":
        manage_db(args)

    elif args.command == "estimate-period":
        period = estimate_periodicity(args)
        print(period)

    elif args.command == "baselines":
        run_baseline(args)

    else:
        raise ValueError(f"Unknown command '{args.command}' for AutoTSAD!")


if __name__ == "__main__":
    main(sys.argv[1:])
