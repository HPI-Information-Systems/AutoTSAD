import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from timeeval.metrics.thresholding import SigmaThresholding

from autotsad.dataset import TestDataset
from autotsad._version import __version__
from .base_algorithms import execute_base_algorithms, ranks_to_scores
from .baseline_select import Select
from ...config import ALGORITHMS
from ...evaluation import compute_metrics


def register_select_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path,
                        help="Path to the target time series to be analyzed for anomalies.")
    parser.add_argument("--results-path", type=Path, default=Path("results"),
                        help="Path to store the results in.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs to use for execution.")
    parser.add_argument("--em-cut-off", type=int, default=20,
                        help="Cut-off value for EM algorithm.")
    parser.add_argument("--em-iterations", type=int, default=100,
                        help="Number of iterations to use for EM algorithm.")
    parser.add_argument("--strategy", type=str, default="horizontal",
                        choices=["horizontal", "vertical"],
                        help="Selection strategy to use for selecting the final ranking.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the final ranking and scores.")


def main(args: argparse.Namespace) -> None:
    dataset_path: Path = args.dataset
    dataset_path = dataset_path.resolve()
    results_path: Path = args.results_path
    n_jobs: int = args.n_jobs
    em_cut_off: int = args.em_cut_off
    em_iterations: int = args.em_iterations
    selection_strategy: str = args.strategy

    start_time = time.time_ns()
    data = TestDataset.from_file(dataset_path)
    print("Loaded data")

    # prepare results folder
    results_path = results_path / f"select-{data.hexhash}-{selection_strategy}"
    results_path = results_path.resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "dataset.txt").write_text(str(dataset_path), encoding="UTF-8")
    (results_path / "version.txt").write_text(__version__, encoding="UTF-8")
    with (results_path / "config.json").open("w") as f:
        json.dump({
            "base_algorithms": ALGORITHMS,
            "n_jobs": n_jobs,
            "em_cut_off": em_cut_off,
            "em_iterations": em_iterations,
            "selection_strategy": selection_strategy
        }, f)
    print(f"RESULT directory={results_path}")

    scores, ranks = execute_base_algorithms(data, n_jobs=n_jobs)
    print("Executed base algorithms")

    final_ranks = Select(cut_off=em_cut_off, iterations=em_iterations, n_jobs=n_jobs)\
        .run(scores, ranks, strategy=selection_strategy)
    final_scores = ranks_to_scores(final_ranks)
    np.savetxt(results_path / "scores.csv", final_scores)
    print("Executed SELECT algorithm")

    print("\nEvaluating results:")
    t = SigmaThresholding(factor=2)
    final_preds = t.fit_transform(data.label, final_scores.reshape(-1, 1)).ravel()
    metrics = compute_metrics(data.label, final_scores, final_preds, n_jobs=n_jobs)
    with (results_path / "metrics.json").open("w") as f:
        json.dump(metrics, f)

    for m in metrics:
        print(f"  {m}: {metrics[m]:.4f}")

    end_time = time.time_ns()
    print(f"\nExecution time: {(end_time - start_time) / 1e9:.2f}s")
    (results_path / "runtime.txt").write_text(f"{(end_time - start_time):f}", encoding="UTF-8")

    if args.plot:
        n = 2 + len(scores)
        fig, axs = plt.subplots(n, 1, figsize=(10, 1.5*n), sharex="col")
        # plot data
        data.plot(ax=axs[0])
        axs[0].set_title(f"Result for {data.name}")

        # plot aggregated scoring
        axs[1].plot(final_scores, color="green", label="SELECT scoring")
        axs[1].set_title(f"SELECT {selection_strategy}")

        # plot individual base algorithm scorings
        algos = list(ALGORITHMS)
        for i in range(len(scores)):
            axs[i + 2].plot(scores[i, :], color="black", label=algos[i])
            axs[i + 2].legend()
        fig.savefig(results_path / "plot.png")
        fig.savefig(results_path / "plot.pdf")
        plt.show()
