import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import pandas as pd

from autotsad.dataset import TestDataset
from autotsad.evaluation import compute_metrics
from autotsad.system.execution.aggregation import load_scores, algorithm_instances, aggregate_scores
from autotsad.system.execution.algo_selection import select_algorithm_instances
from autotsad.util import save_serialized_results


def test_single_combination(dataset: TestDataset, results: pd.DataFrame, result_dir: Path, selection: str,
                            normalization: str, aggregation: str, max_instances: int,
                            scores_path: Path) -> Dict[str, Any]:
    log = logging.getLogger(f"autotsad.execution.selection.{selection}-{normalization}-{aggregation}")

    # prepare result folder for this combination
    result_dir = result_dir / f"{selection}-{normalization}-{aggregation}"
    result_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Selecting {max_instances} instances with {selection} and {normalization} normalization and "
             f"{aggregation} aggregation")
    log.debug(f"Loading scores from {scores_path}")
    instances = algorithm_instances(results)
    scores = load_scores(instances,
                         dataset_id=dataset.hexhash,
                         base_scores_path=scores_path,
                         normalization_method=normalization,
                         ensure_finite=True)

    # also filter instances that have non-finite results (in addition to scores)
    n_invalid = len(instances) - scores.shape[1]
    if n_invalid > 0:
        invalid_instances = list(set(instances) - set(scores.columns))
        log.warning(f"{selection}-{normalization}-{aggregation}: Filtering out {n_invalid} instances with non-finite "
                    f"scores: {invalid_instances}!")
    results = results[instances.isin(scores.columns)].copy()

    log.debug(f"Performing algorithm selection with {selection} and max_instances={max_instances}")
    results = select_algorithm_instances(results, scores, selection_method=selection, max_instances=max_instances)
    save_serialized_results(results, result_dir / "selected-instances.csv")

    log.debug(f"Combining scores with {aggregation} aggregation")
    scores = scores[algorithm_instances(results)]
    combined_score = aggregate_scores(scores.values, agg_method=aggregation)
    np.savetxt(result_dir / "combined-score.csv", combined_score, delimiter=",")

    metrics: Dict[str, Union[str, float]] = {"selection-method": selection, "normalization-method": normalization,
                                             "aggregation-method": aggregation}
    metrics.update(compute_metrics(dataset.label, combined_score, combined_score > 0))
    log.debug(f"{selection}-{normalization}-{aggregation} metrics: {metrics}")
    with (result_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f)

    return metrics
