import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt

from autotsad._version import __version__
from autotsad.dataset import TestDataset, get_hexhash
from . import runner
from .assessment import compute_metrics_for_tsadams
from .util import ENTITY_MAPPING

default_config_path = Path(__file__).parent / "config.yml"


def register_tsadams_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path,
                        help="Path to the target time series to be analyzed for anomalies.")
    parser.add_argument("--results-path", type=Path, default=Path("results"),
                        help="Path to store the results in.")
    parser.add_argument("--config-path", type=Path, default=default_config_path,
                        help="Path to the config file for tsadams.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs to use for execution.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the final ranking and scores.")
    parser.add_argument("--recreate-env", action="store_true",
                        help="Recreate the virtual environment for tsadams.")


def main(args: argparse.Namespace) -> None:
    dataset_path: Path = args.dataset
    dataset_path = dataset_path.resolve()
    results_path: Path = args.results_path
    config_path: Path = args.config_path
    config_path = config_path.resolve()
    n_jobs: int = args.n_jobs
    recreate_env: bool = args.recreate_env

    dataset_hash = get_hexhash(dataset_path)

    # prepare results folder
    results_folder = results_path
    results_path = results_folder / f"tsadams-{dataset_hash}-mim"
    results_path = results_path.resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "dataset.txt").write_text(str(dataset_path), encoding="UTF-8")
    (results_path / "version.txt").write_text(__version__, encoding="UTF-8")

    with (results_path / "config.json").open("w") as f:
        with config_path.open("r") as c:
            config = yaml.safe_load(c)
        json.dump(config, f)
    print(f"RESULT directory={results_path}")

    # execute tsadams (in its own process)
    # tsadams should produce a ranking object dump in a results-folder
    print("\nExecuting tsadams:")
    tsadams_tmp_path = results_folder / "tsadams"  # same as in the config.yaml
    start_time = time.time_ns()
    runner.main(dataset_hash, results_path / "execution.log", use_existing_env=not recreate_env)
    end_time = time.time_ns()
    print(f"\nExecution time: {(end_time - start_time) / 1e9:.2f}s")
    (results_path / "runtime.txt").write_text(f"{(end_time - start_time):f}", encoding="UTF-8")

    # load results (scoring and log)
    print("\nLoading results:")
    entity = ENTITY_MAPPING[dataset_hash]
    scoring_path = tsadams_tmp_path / "results" / "autotsad" / f"scores_{entity}.csv"
    if not scoring_path.exists():
        raise ValueError("Failed to process tsadams results! See execution.log for details.")

    final_scores = np.genfromtxt(scoring_path, delimiter=",")
    # write a copy into the results folder
    np.savetxt(results_path / "scores.csv", final_scores, delimiter=",")

    # parse the log file and write it into the results folder as well
    log_path = tsadams_tmp_path / "results" / "autotsad" / f"execution_{entity}.log"
    if log_path.exists():
        with log_path.open("r") as f:
            log = f.read()
        with (results_path / "execution.log").open("a") as f:
            f.write(log)

    # compute metrics
    print("\nComputing metrics:")
    data = TestDataset.from_file(dataset_path)
    downsampling = config.get("downsampling", None)
    metrics = compute_metrics_for_tsadams(data, final_scores, downsampling=downsampling, n_jobs=n_jobs)
    with (results_path / "metrics.json").open("w") as f:
        json.dump(metrics, f)

    for m in metrics:
        print(f"  {m}: {metrics[m]:.4f}")

    if args.plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 3), sharex="col")
        # plot data
        data.plot(ax=axs[0])
        axs[0].set_title(f"Result for {data.name}")

        # plot aggregated scoring
        axs[1].plot(final_scores, color="green", label="tsadams scoring")
        axs[1].set_title("tsadams MIM")

        fig.savefig(results_path / "plot.png")
        fig.savefig(results_path / "plot.pdf")
        plt.show()
