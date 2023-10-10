import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from timeeval import Metric
from timeeval.utils.hash_dict import hash_dict

from autotsad.config import GeneralSection
from autotsad.dataset import Dataset
from autotsad.system.pynisher import pynish_func, PynisherException
from autotsad.tsad_algorithms.interop import params_from_dict, exec_algo


def execution_job(algorithm: str,
                  dataset: Dataset,
                  params: Dict[str, Any],
                  metric: Metric,
                  score_dirpath: Optional[Path],
                  config: GeneralSection) -> Tuple[str, Dict[str, Any], str, float, float]:
    def objective() -> float:
        algo_params = params_from_dict(params, algorithm)
        scores = exec_algo(dataset, algorithm, algo_params)
        dataset_id = getattr(dataset, "hexhash", dataset.name)
        if score_dirpath is not None:
            np.savetxt(str(score_dirpath / f"{dataset_id}-{algorithm}-{hash_dict(params)}.csv"),
                       scores,
                       delimiter=",")

        if np.any(dataset.label):
            return metric(dataset.label, scores)
        else:
            return -1.

    result: float = np.nan
    duration: float = np.nan
    try:
        t0 = time.time()
        result = pynish_func(objective, **config.default_testing_limits(), name=f"EXEC: {dataset.name}-{algorithm}")()
        duration = time.time() - t0
    except PynisherException as e:
        print(f"RESOURCE-LIMIT: Execution of {algorithm} with params '{params}' failed on {dataset.name} because {repr(e)}")
    except Exception as e:
        print(f"ERROR: Execution of {algorithm} with params '{params}' failed on {dataset.name} because {repr(e)}")

    return algorithm, params, dataset.name, result, duration
