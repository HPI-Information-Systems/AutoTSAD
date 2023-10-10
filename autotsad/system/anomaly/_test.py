from pathlib import Path

import numpy as np
import gutenTAG.api as gt
import matplotlib.pyplot as plt

from ...dataset import TrainingDatasetCollection
from .injection import inject_anomalies
from .transforms import ReplaceWithGutenTAGPattern


def main():
    # load data
    dataset = TrainingDatasetCollection.load(Path("tmp/training-dataset-collection.pkl"))[1]
    data = dataset.data.copy()

    # inject anomalies
    data, label, annotations = inject_anomalies(data,
                                                anomaly_types=["compress", "stretch", "smoothing", "hmirror", "vmirror", "scale", "pattern"],
                                                n_anomalies=6,
                                                length=dataset.period_size,
                                                random_state=42)

    # plot
    fig, axs = plt.subplots(3, 1, sharex="col")
    axs[0].plot(dataset.data, label="original")
    axs[1].plot(data, label="anomaly injected")
    axs[2].plot(label, label=f"label (contamination={np.sum(label)/label.shape[0]:.2f})", color="orange")
    for idx, text in annotations.items():
        axs[2].annotate(text, (idx, 1), color="black", ha="center", va="top")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()


def test_anomaly_transforms():
    # transform = LocalPointOutlierTransform(strength=0.2, random_state=42)
    # transform = CompressTransform(strength=0, random_state=42)
    # transform = StretchTransform(strength=1, random_state=42)
    # transform = NoiseTransform(strength=0.5, random_state=42)
    # transform = SmoothingTransform(strength=0.5, random_state=42)
    # transform = HMirrorTransform()
    # transform = VMirrorTransform()
    # transform = ScaleTransform(strength=1, random_state=42)
    transform = ReplaceWithGutenTAGPattern(strength=0.75, random_state=42)

    # load data
    data = gt.sine(length=300, frequency=1, amplitude=1)
    data += np.random.normal(scale=0.05, size=len(data))
    subsequence = data[100:175]
    anomaly_subsequence = transform(subsequence)

    plt.figure()
    plt.plot(data, label="original")
    plt.plot(np.r_[data[:100], anomaly_subsequence, data[200:]], label="anomaly injected")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
