import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Sequence

import gutenTAG.api.bo as gt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from autotsad.tsad_algorithms.interop import params_from_dict, exec_algo
from autotsad.system.anomaly.transforms import VMirrorTransform, ScaleTransform, LocalPointOutlierTransform


def generate_intro_example_data(
        N: int = 5000,
        rng: np.random.Generator = np.random.default_rng(42),
        anomaly_region_offset: int = 2000,
) -> pd.DataFrame:
    # Create synthetic ECG signal
    data = gt.ecg(rng, length=N, frequency=2.3, amplitude=0.85, ecg_sim_method="ecgsyn")
    labels = np.zeros(data.shape[0], dtype=np.int_)

    idx = anomaly_region_offset
    t = VMirrorTransform(strength=1, rng=rng)
    data[idx + 200:idx + 250], labels[idx + 200:idx + 250] = t(data[idx + 200:idx + 250])
    # t = StretchTransform(strength=1, rng=rng)
    # data[offset+200:offset+300], labels[offset+200:offset+300] = t(data[offset+200:offset+250])

    t = ScaleTransform(strength=0.55, rng=rng)
    data[idx + 500:idx + 600], labels[idx + 500:idx + 600] = t(data[idx + 500:idx + 600])

    t = LocalPointOutlierTransform(strength=1, rng=rng)
    data[idx + 790:idx + 800], labels[idx + 790:idx + 800] = t(data[idx + 790:idx + 800])

    df = pd.DataFrame({
        "timestamp": np.arange(len(data)),
        "value": data,
        "labels": labels,
    })
    return df


def run_algorithms_on(data: np.ndarray,
                      instances: Sequence[Tuple[str, Dict[str, Any]]] = ("kmeans", {})
                      ) -> pd.DataFrame:
    algo_names = [a for a, p in instances]
    df_results = pd.DataFrame(index=np.arange(data.shape[0]), columns=algo_names)
    for algo, params in instances:
        cust_params = params_from_dict(params, algo)
        scores = exec_algo(data, algo, cust_params, ignore_cuts=False)
        df_results[algo] = scores
    return df_results
