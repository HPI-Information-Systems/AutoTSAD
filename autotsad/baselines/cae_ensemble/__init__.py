import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from timeeval.metrics.thresholding import SigmaThresholding

from autotsad._version import __version__
from autotsad.dataset import TestDataset, get_hexhash
from autotsad.evaluation import compute_metrics
from . import runner


def register_cae_ensemble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path,
                        help="Path to the target time series to be analyzed for anomalies.")
    parser.add_argument("--results-path", type=Path, default=Path("results"),
                        help="Path to store the results in.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs to use for execution.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the final ranking and scores.")
    parser.add_argument("--recreate-env", action="store_true",
                        help="Recreate the virtual environment for cae-ensemble.")


def main(args: argparse.Namespace) -> None:
    dataset_path: Path = args.dataset
    dataset_path = dataset_path.resolve()
    results_path: Path = args.results_path
    n_jobs: int = args.n_jobs
    recreate_env: bool = args.recreate_env

    dataset_hash = get_hexhash(dataset_path)

    # prepare results folder
    results_folder = results_path
    results_path = results_folder / f"cae-{dataset_hash}-ensemble"
    results_path = results_path.resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "dataset.txt").write_text(str(dataset_path), encoding="UTF-8")
    (results_path / "version.txt").write_text(__version__, encoding="UTF-8")

    print(f"RESULT directory={results_path}")

    # execute cae-ensemble (in its own process)
    # cae-ensemble should produce an anomaly score in the results folder
    print("\nExecuting cae-ensemble:")
    cae_ensemble_tmp_path = Path.cwd() / "cae-ensemble"
    start_time = time.time_ns()
    runner.main(dataset_hash, dataset_path, results_path / "execution.log", use_existing_env=not recreate_env)
    end_time = time.time_ns()
    print(f"\nExecution time: {(end_time - start_time) / 1e9:.2f}s")
    (results_path / "runtime.txt").write_text(f"{(end_time - start_time):f}", encoding="UTF-8")

    # copy results to the results folder
    shutil.copy2(cae_ensemble_tmp_path / "save_outputs" / "scores" / "-1" / f"CAE_ENSEMBLE_{dataset_path.stem}.csv",
                 results_path / "scores.csv")
    shutil.copy2(cae_ensemble_tmp_path / "save_configs" / "-1" / f"Config_CAE_ENSEMBLE_pid=0.json",
                 results_path / "config.json")

    # load results (scoring and log)
    print("\nLoading results:")
    scoring_path = results_path / "scores.csv"
    if not scoring_path.exists():
        raise ValueError("Failed to process cae-ensemble results! See execution.log for details.")

    final_scores = np.genfromtxt(scoring_path, delimiter=",")
    # write a copy into the results folder
    np.savetxt(results_path / "scores.csv", final_scores, delimiter=",")

    # compute metrics
    print("\nComputing metrics:")
    data = TestDataset.from_file(dataset_path)
    t = SigmaThresholding(factor=2)
    final_preds = t.fit_transform(data.label, final_scores.reshape(-1, 1)).ravel()
    metrics = compute_metrics(data.label, final_scores, final_preds, n_jobs=n_jobs)
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
        axs[1].plot(final_scores, color="green", label="cae-ensemble scoring")
        axs[1].set_title("CAE-ENSEMBLE diversity")

        fig.savefig(results_path / "plot.png")
        fig.savefig(results_path / "plot.pdf")
        plt.show()
