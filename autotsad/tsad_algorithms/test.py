from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from timeeval.metrics import RangePrAUC

from .interop import exec_algo
from ..dataset import TrainingDatasetCollection, TrainingTSDataset, Dataset


def plot_algo_scores(dataset: Dataset, scores: np.ndarray, algo: str):
    fig, axs = plt.subplots(2, 1, sharex="col")
    if isinstance(dataset, TrainingTSDataset):
        dataset.plot(ax=axs[0], cuts=True, annotations=True)
    else:
        dataset.plot(ax=axs[0])
    axs[0].legend()
    axs[1].plot(scores, label=algo)
    axs[1].legend()
    plt.savefig(f"{algo}.png")


def main():
    train_collection = TrainingDatasetCollection.load(Path("tmp") / "cache" / "gt-3" / "train-ts-collection.pkl")
    # dataset = train_collection[1]
    dataset = [d for d in train_collection if "A" in d.name][0]
    assert isinstance(dataset, TrainingTSDataset)
    # dataset = train_collection.test_data
    # dataset.period_size = 100
    # n_jobs = joblib.cpu_count(only_physical_cores=True)
    n_jobs = 1
    ignore_cuts = True
    algorithms = [
        # "subsequence_lof",
        # "subsequence_lof",
        # "subsequence_knn",
        # "subsequence_if",
        # "stomp",
        "kmeans",
        "kmeans",
        # "torsk",
        # "dwt_mlead",
        # "dwt_mlead",
        # "grammarviz",
    ]

    print("data", dataset.data.shape)
    params = None
    scores = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(exec_algo)(dataset, algo, params, ignore_cuts) for algo in algorithms
    )
    scores = np.vstack(scores).T
    scores = MinMaxScaler().fit_transform(scores)
    print("scores", scores.shape)

    metrics = []
    m = RangePrAUC(buffer_size=dataset.period_size)
    for i in range(len(algorithms)):
        metric = m(dataset.label, scores[:, i])
        metrics.append(metric)
    metrics = np.asarray(metrics, dtype=np.float_)

    # joblib.dump(scores[:, 0], "scores.pkl")
    precision = 15
    old_score = joblib.load("scores.pkl")
    old_metric = m(dataset.label, old_score)
    print("metrics", metrics)
    print("old metric", old_metric)

    for i, j in combinations(np.arange(metrics.shape[0]), 2):
        np.testing.assert_allclose(metrics[i], metrics[j], rtol=1e-15, atol=10**-precision)
    np.testing.assert_allclose(metrics[0], old_metric, rtol=1e-15, atol=10**-precision)

    for i, j in combinations(np.arange(scores.shape[1]), 2):
        np.testing.assert_allclose(scores[:, i], scores[:, j], rtol=1e-15, atol=10**-precision)
    np.testing.assert_allclose(scores[:, 0], old_score, rtol=1e-15, atol=10**-precision)

    fig, axs = plt.subplots(2, 1, sharex="col")
    if isinstance(dataset, TrainingTSDataset):
        dataset.plot(ax=axs[0], cuts=True, annotations=True)
    else:
        dataset.plot(ax=axs[0])
    axs[0].legend()
    for s, name, pr_auc in zip(scores.T, algorithms, metrics):
        axs[1].plot(s, label=f"{name} score ({pr_auc=:.2f})")
    axs[1].legend()

    # # run on original test dataset
    # scores = np.c_[
    #     subsequence_lof.main(train_collection.test_data.data, params=subsequence_lof.CustomParameters(window_size=dataset.period_size)),
    #     subsequence_knn.main(train_collection.test_data.data, params=subsequence_knn.CustomParameters(window_size=dataset.period_size)),
    #     subsequence_if.main(train_collection.test_data.data, params=subsequence_if.CustomParameters(window_size=dataset.period_size)),
    #     stomp.main(train_collection.test_data.data, params=stomp.CustomParameters(anomaly_window_size=dataset.period_size)),
    # ]
    # scores = MinMaxScaler().fit_transform(scores)
    # metrics = []
    # m = RangePrAUC(buffer_size=dataset.period_size)
    # for i in range(len(names)):
    #     metric = m(train_collection.test_data.labels, scores[:, i])
    #     metrics.append(metric)
    #
    # fig, axs = plt.subplots(2, 1, sharex="col")
    # axs[0].plot(train_collection.test_data.data, label="test data")
    # for b, e in _mask_to_slices(train_collection.test_data.labels):
    #     y0, y1 = axs[0].get_ylim()
    #     axs[0].add_patch(Rectangle((b, y0), e-b, y1-y0, edgecolor="orange", facecolor="yellow", alpha=0.5))
    # axs[0].legend()
    # for s, name, pr_auc in zip(scores.T, names, metrics):
    #     axs[1].plot(s, label=f"{name} score ({pr_auc=:.2f})")
    # axs[1].legend()
    plt.show()

    # for s, name in zip(scores.T, algorithms):
    #     name = f"{name}-{'ignore_cuts' if ignore_cuts else 'with_cuts'}"
    #     plot_algo_scores(dataset, s, name)


if __name__ == "__main__":
    main()
