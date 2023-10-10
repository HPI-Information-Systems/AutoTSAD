import sys
from pathlib import Path

import gutenTAG.api as gt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

sys.path.append(".")

from holistic_tsad.anomaly.transforms import LocalPointOutlierTransform, HMirrorTransform, NoiseTransform


def white_noise(length: int, variance: float, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    return rng.normal(0, variance, length)


def generate_synthetic_data(i: int = 0, plot: bool = False, save: bool = True) -> pd.DataFrame:
    n = 1000
    f = 2

    if i == 0:
        rng = np.random.default_rng(42)
        data = np.r_[
            gt.cosine(length=10*n, frequency=f/2, amplitude=1.78, freq_mod=0.2),
            gt.dirichlet(length=10*n, frequency=f/2, amplitude=1.78),
        ]
        data *= gt.sine(length=20*n, frequency=0.05, amplitude=.1)
        data += white_noise(length=20*n, variance=0.002, rng=rng)
        labels = np.zeros_like(data, dtype=np.bool_)

        data[2460:2560], labels[2460:2560] = LocalPointOutlierTransform(strength=0.7, rng=rng)(data[2460:2560])
        data[7500:7660], labels[7500:7660] = HMirrorTransform(strength=0.7, rng=rng)(data[7500:7660])
        data[12600:12650], labels[12600:12650] = LocalPointOutlierTransform(strength=0.3, rng=rng)(data[12600:12650])
        data[15633:15722], labels[15633:15722] = HMirrorTransform(strength=0.5, rng=rng)(data[15633:15722])
        data[17500:17600], labels[17500:17600] = NoiseTransform(strength=1, rng=rng)(data[17500:17600])
        df = pd.DataFrame({"value": data, "is_anomaly": labels})

    elif i == 1:
        path = Path("../data/global-temperature-mean-monthly.csv")
        data = (pd.read_csv(path, skiprows=1, index_col=None)
                .iloc[:-2, 1]
                .str.strip()
                .astype(np.float_)
                .values)
        df = pd.DataFrame({"value": data, "is_anomaly": np.zeros_like(data, dtype=np.bool_)})

    elif i == 2:
        data = np.r_[
            gt.sawtooth(length=n, frequency=f, width=0),
            gt.square(length=n, frequency=f),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f / 2),
        ]
        data = np.tile(data, reps=3)
        data = np.r_[data[:3000], gt.mls(rng=np.random.default_rng(2), length=50), data[3000:]]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        labels[3000:3100] = 1
        df = pd.DataFrame({"value": data, "is_anomaly": labels})

    elif i == 3:
        data = np.r_[
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            # gt.square(length=n, frequency=f),
            # gt.dirichlet(length=n, frequency=f, amplitude=.5),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f, ecg_sim_method="ecgsyn"),

            # gt.square(length=n, frequency=f),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f, ecg_sim_method="ecgsyn"),
            gt.mls(rng=np.random.default_rng(2), length=100),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f, ecg_sim_method="ecgsyn"),
            # gt.square(length=n, frequency=f),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        labels[3000:3100] = 1
        df = pd.DataFrame({"value": data, "is_anomaly": labels})

    elif i == 4:
        rng = np.random.default_rng(1)
        data = np.r_[
            gt.dirichlet(length=n, frequency=f, amplitude=.55),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=rng, length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.dirichlet(length=n, frequency=f, amplitude=.5),
            gt.ecg(rng=rng, length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.dirichlet(length=n, frequency=f, amplitude=.6),
            gt.sawtooth(length=n, frequency=f / 2, width=0.05),
            gt.ecg(rng=np.random.default_rng(2), length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.2,
            gt.sawtooth(length=n, frequency=f / 2, width=0, amplitude=0.8),
            gt.dirichlet(length=n, frequency=f, amplitude=.45),
            gt.ecg(rng=np.random.default_rng(3), length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.sawtooth(length=n, frequency=f / 2, width=0.1, amplitude=1.2),
            gt.dirichlet(length=n, frequency=f, amplitude=.65),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        # labels[2300:2400] = 1  # existing variation in ECG signal

        data[3650] = data[3650] - 0.5
        labels[3650] = 1

        data[6200:6400] = data[6200:6400] + norm.pdf(np.linspace(-5, 5, 200), loc=0, scale=1)
        labels[6200:6400] = 1

        data[10200:10300] = data[10200:10300] + (rng.random(100) - 0.5) / 3
        labels[10200:10300] = 1
        df = pd.DataFrame({"value": data, "is_anomaly": labels})
    else:
        raise ValueError(f"Invalid dataset index {i}.")

    if save:
        df.to_csv(Path("data").resolve() / "synthetic")

    print(df)

    if plot:
        fig, axs = plt.subplots(2, 1, sharex="col")
        axs[0].plot(df.iloc[:, 0].values, label="TS")
        axs[1].plot(df.iloc[:, -1].values, label="label", color="orange")
        # axs[0].legend()
        # axs[1].legend()
        plt.show()
    return df


if __name__ == '__main__':
    generate_synthetic_data(i=0, plot=True, save=False)
