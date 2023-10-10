import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from nx_config import add_cli_options, fill_config_from_path, resolve_config_path
from nx_config.test_utils import update_section

from .logging import setup_logging_from_config
from .._version import __version__
from ..config import config
from ..dataset import TestDataset
from ..util import format_time_ns


def register_autotsad_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path, help="Path to the target time series to be analyzed for anomalies.")
    parser.add_argument("--use-gt-for-cleaning", action="store_true",
                        help="Use ground truth information for cleaning step. Just for debugging!")
    add_cli_options(parser, config_t=type(config))


def main(args: argparse.Namespace) -> None:
    fill_config_from_path(config, path=resolve_config_path(cli_args=args), env_prefix="AUTOTSAD")
    dataset_path: Path = args.dataset
    use_gt_for_cleaning: bool = args.use_gt_for_cleaning
    autotsad(dataset_path, use_gt_for_cleaning=use_gt_for_cleaning)


def autotsad(dataset_path: Path, testdataset: Optional[TestDataset] = None, use_gt_for_cleaning: bool = False) -> np.ndarray:
    if testdataset is None:
        testdataset = TestDataset.from_file(dataset_path)

    # potentially dangerous, but we need to update the cache key in the config:
    update_section(config.general, cache_key=testdataset.hexhash)

    print(f"AutoTSAD v{__version__}")
    print("------------------------")
    print(f"CACHING directory={config.general.cache_dir()}")
    print(f"RESULT directory={config.general.result_dir()}")
    print(f"Configuration=\n{repr(config)}")
    config.general.cache_dir().mkdir(parents=True, exist_ok=True)
    config.general.result_dir().mkdir(parents=True, exist_ok=True)

    # speed up startup by importing local modules lazily
    # this also prevents the automatic loading of these modules in the remote_funcs modules
    # import after the configuration has been loaded!
    from .data_gen.main import main as generate_training_data
    from .execution.main import execute_and_rank
    from .optimization.main import optimize_algorithms
    from .timer import Timers

    with setup_logging_from_config(config.general, config.optimization):
        log = logging.getLogger("autotsad")

        log.info(f"Loaded configuration:\n{repr(config)}")
        log.info(f"Processing dataset {testdataset.name} (ID={testdataset.hexhash}): {dataset_path}")
        (config.general.result_dir() / "dataset.txt").write_text(str(dataset_path), encoding="UTF-8")
        (config.general.result_dir() / "config.json").write_text(config.to_json(), encoding="UTF-8")
        (config.general.result_dir() / "version.txt").write_text(__version__, encoding="UTF-8")
        start_time = time.time_ns()
        Timers.start("autotsad")

        try:
            train_collection = generate_training_data(testdataset, use_gt_for_cleaning=use_gt_for_cleaning)
            algorithm_instances = optimize_algorithms(train_collection)
            return execute_and_rank(train_collection, algorithm_instances)

        finally:
            Timers.stop("autotsad")
            Timers.save_trace(config.general.result_dir() / "runtimes.csv")
            end_time = time.time_ns()
            duration = end_time - start_time
            print(f"AutoTSAD total time: {duration/1e9}s ({format_time_ns(duration)})")
