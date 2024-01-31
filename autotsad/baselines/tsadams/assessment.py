from typing import Dict, Optional

import numpy as np
from timeeval.metrics.thresholding import SigmaThresholding

from autotsad.dataset import TestDataset
from autotsad.evaluation import compute_metrics


def compute_metrics_for_tsadams(data: TestDataset, scores: np.ndarray,
                                downsampling: Optional[int] = None,
                                n_jobs: int = 1) -> Dict[str, float]:
    # Downsampling
    if downsampling is not None:
        Y = data.data.reshape(1, -1)
        labels = data.label
        n_features, n_t = Y.shape
        right_padding = downsampling - n_t % downsampling

        Y = np.pad(Y, ((0, 0), (right_padding, 0)))
        labels = np.pad(labels, (right_padding, 0))

        Y = Y.reshape(n_features, Y.shape[-1] // downsampling, downsampling).max(axis=2)
        labels = labels.reshape(labels.shape[0] // downsampling, downsampling).max(axis=1)
        data = TestDataset(data=Y.ravel(), labels=labels, name=data.name, hexhash=data.hexhash)

    t = SigmaThresholding(factor=2)
    final_preds = t.fit_transform(data.label, scores.reshape(-1, 1)).ravel()
    metrics = compute_metrics(data.label, scores, final_preds, n_jobs=n_jobs)
    return metrics
