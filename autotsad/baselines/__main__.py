import argparse
import sys

from autotsad._version import __version__
from autotsad.baselines.cae_ensemble import register_cae_ensemble_arguments, main as run_cae_ensemble
from autotsad.baselines.select import register_select_arguments, main as run_select
from autotsad.baselines.tsadams import register_tsadams_arguments, main as run_tsadams


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("autotsad baselines",
                                     description="AutoTSAD: Unsupervised anomaly detection system for univariate time "
                                                 "series. This is the baselines module!")
    parser.add_argument("--version", action="version",
                        version=f"AutoTSAD v{__version__} (baselines module)",
                        help="Show version number of AutoTSAD.")
    register_baseline_arguments(parser)
    return parser


def register_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="baseline", required=True)

    parser_select = subparsers.add_parser("select", help="Run SELECT baseline algorithm on a given dataset.")
    register_select_arguments(parser_select)

    parser_tsadams = subparsers.add_parser("tsadams", help="Run tsadams baseline algorithm on a given dataset.")
    register_tsadams_arguments(parser_tsadams)

    parser_case_ensemble = subparsers.add_parser("cae-ensemble", help="Run cae-ensemble baseline algorithm on a given dataset.")
    register_cae_ensemble_arguments(parser_case_ensemble)


def main(args: argparse.Namespace):
    if args.baseline == "select":
        run_select(args)
    elif args.baseline == "tsadams":
        run_tsadams(args)
    elif args.baseline == "cae-ensemble":
        run_cae_ensemble(args)
    else:
        raise ValueError(f"Unknown baseline '{args.baseline}' for AutoTSAD baselines!")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
