from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from timeeval.metrics.thresholding import SigmaThresholding

from autotsad.dataset import TestDataset
from autotsad.evaluation import compute_metrics
from autotsad.util import mask_to_slices
from .base_algorithms import execute_base_algorithms, ranks_to_scores
from .baseline_select import Select


def main():
    data = TestDataset.from_file(Path("data/synthetic/gt-3.csv"))
    print("Loaded data")
    # l = 1000
    # data.data = data.data[2500:2500+l]
    # data.label = data.label[2500:2500+l]
    data.plot(cuts=True)

    scores, ranks = execute_base_algorithms(data, n_jobs=2)

    print(scores)
    print(ranks)
    assert not np.any(ranks == 0), f"Ranks should not contain 0: {np.meshgrid(*np.where(ranks == 0))}"
    # pd.DataFrame(scores.T).plot(subplots=True, title="scores")
    # pd.DataFrame(ranks.T).plot(subplots=True, title="ranks")

    # final_ranks, final_scores = inverse_ranking(ranks)
    # final_ranks, final_scores, _, _ = robust_rank_aggregation(ranks)  # broken
    # final_ranks = borda(ranks) + 1
    # final_scores = 1 / final_ranks
    # final_ranks, final_scores = kemeny_young(ranks)  # memory error
    # final_ranks, final_scores = mixture_modeling(scores, aggregation=np.mean)
    # final_ranks, final_scores = unification(scores, aggregation=np.mean)

    final_ranks = Select().run(scores, ranks, strategy="horizontal")
    assert not np.any(final_ranks == 0), f"Ranks should not contain 0: {np.meshgrid(*np.where(final_ranks == 0))}"
    final_scores = ranks_to_scores(final_ranks)

    t = SigmaThresholding(factor=2)
    final_preds = t.fit_transform(data.label, final_scores.reshape(-1, 1)).ravel()

    print("FinalRank:", final_ranks)
    print("FinalPreds:", final_preds)
    print("Scores:", final_scores)

    metrics = compute_metrics(data.label, final_scores, final_preds)
    print("Metrics:")
    for m in metrics:
        print(m, metrics[m])

    pd.DataFrame(final_ranks).plot(title="final ranks")
    pd.DataFrame(final_scores).plot(title="final scores")
    plt.hlines(t.threshold, 0, len(final_scores), color="red", label="threshold")
    anomalies = mask_to_slices(final_preds)
    for b, e in anomalies:
        plt.axvspan(b, e, color="yellow", alpha=0.6)
    plt.show()


if __name__ == '__main__':
    main()
