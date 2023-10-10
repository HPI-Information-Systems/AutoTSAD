import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeeval.metrics.thresholding import TopKRangesThresholding


sys.path.append(".")

from holistic_tsad.autotsad.util import mask_to_slices


def main():
    dataset_path = Path.cwd().parent / "data" / "sand-data" / "processed" / "SED.csv"
    scores_path = Path.cwd().parent / "data" / "sand-data" / "results" / "SED.csv"
    ws = 75

    df = pd.read_csv(dataset_path)
    df["scores"] = np.genfromtxt(scores_path)

    # apply thresholding and relax predicted ranges by anomaly window size
    thresholding = TopKRangesThresholding()
    predictions = thresholding.fit_transform(df["is_anomaly"], df["scores"])

    slices = mask_to_slices(predictions)
    slices[:, 0] -= ws // 2
    slices[:, 1] += ws // 2
    predictions = np.zeros_like(predictions, dtype=np.bool_)
    for b, e in slices:
        predictions[b:e] = True
    df["predictions"] = predictions

    print(f"Threshold: {thresholding.threshold}")
    print(f"Buffer size: {ws // 2}")
    print(f"#Anomalies: {thresholding._k}")
    print(f"Detected #Anomalies: {slices.shape[0]}")

    # compute quality metric
    combined = df["is_anomaly"] & df["predictions"]
    precision = combined.sum() / df["is_anomaly"].sum()
    print(f"Precision@k={precision}")

    fig, axs = plt.subplots(4, 1, sharex="col")
    axs[0].set_title(f"SAND on {dataset_path.stem} Precision@k={precision}")
    axs[0].plot(df["value"], label="timeseries", color="black")
    axs[1].plot(df["is_anomaly"], color="red")
    axs[2].plot(df["scores"], color="blue")
    axs[2].hlines([thresholding.threshold], 0, df.shape[0], color="red")
    axs[2].plot(df["predictions"], color="orange")
    axs[3].plot(combined, color="black")
    plt.show()


if __name__ == '__main__':
    main()
