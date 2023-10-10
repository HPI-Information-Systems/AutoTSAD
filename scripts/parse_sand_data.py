from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import wfdb


def read_ann_WFDB(patient, patient_path: Path):
    return wfdb.rdann(str(patient_path / patient / patient), 'atr')


def convert_wfdb_ann(ann, tot_length, normal_symbol='N'):
    tmp_ann = []
    for i in range(len(ann.symbol)):
        if ann.sample[i] > tot_length:
            break
        if ann.symbol[i] != normal_symbol:
            tmp_ann.append(ann.sample[i])
    return np.array(tmp_ann, dtype=np.int_)


def convert_sed_ann(ann, tot_length: int) -> np.ndarray:
    ann = np.array(ann, dtype=np.int_)
    ann = ann[ann <= tot_length]
    return ann


def read_all_ts(max_rows: int) -> Dict[str, np.ndarray]:
    return {"803": np.genfromtxt(ts_803_file, max_rows=max_rows),
            # "804": np.genfromtxt(ts_804_file, max_rows=max_rows),
            "805": np.genfromtxt(ts_805_file, max_rows=max_rows),
            "806": np.genfromtxt(ts_806_file, max_rows=max_rows),
            "820": np.genfromtxt(ts_820_file, max_rows=max_rows),
            "SED": np.genfromtxt(ts_SED_file, max_rows=max_rows)}


def read_all_annotations(max_rows: int) -> Dict[str, np.ndarray]:
    return {"annotations_803": convert_wfdb_ann(read_ann_WFDB("803", PATH / "ANNOTATIONS"), tot_length=max_rows),
            # "annotations_804": convert_ann(read_ann_WFDB("804", PATH / "ANNOTATIONS"), tot_length=max_rows).
            "annotations_805": convert_wfdb_ann(read_ann_WFDB("805", PATH / "ANNOTATIONS"), tot_length=max_rows),
            "annotations_806": convert_wfdb_ann(read_ann_WFDB("806", PATH / "ANNOTATIONS"), tot_length=max_rows),
            "annotations_820": convert_wfdb_ann(read_ann_WFDB("820", PATH / "ANNOTATIONS"), tot_length=max_rows),
            "annotations_SED": convert_sed_ann(np.genfromtxt(PATH / "ANNOTATIONS" / "SED_Annotations.txt"),
                                               tot_length=max_rows)}


def create_combinations(max_rows: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ts_all = read_all_ts(max_rows)
    dict_anom = read_all_annotations(max_rows)

    # Create annotations for the concatenations
    timeseries = {}
    annotations = {}
    for dataset_ann in all_dataset_combinations:
        # create annotation name
        name = "_".join(map(str, dataset_ann))
        # concatenate annotations
        anomaly_indices = []
        timeseries_parts = []
        for i, d_ref in enumerate(dataset_ann):
            timeseries_parts.append(ts_all[str(d_ref)])
            # Specific problem with 805 (the labels are duplicated)
            if d_ref == 805:
                origin = dict_anom[f"annotations_{d_ref}"] + i * max_rows
                to_add = [origin[0]]
                for ann in origin[1:]:
                    if ann - to_add[-1] > 150:
                        to_add.append(ann)
                anomaly_indices.append(np.array(to_add, dtype=np.int_))
            # General case
            else:
                anomaly_indices.append(dict_anom[f"annotations_{d_ref}"] + i * max_rows)

        timeseries[name] = np.concatenate(timeseries_parts, axis=0)
        annotations[name] = np.concatenate(anomaly_indices, axis=0)
    return timeseries, annotations


def sand_processing(target_path: Path, max_rows: int) -> None:
    timeseries, annotations = create_combinations(max_rows)

    for name in annotations:
        print(f"Creating dataframe for {name}")
        ts = timeseries[name]
        label_indices = annotations[name]
        df = pd.DataFrame(ts, columns=["value"])
        df["is_anomaly"] = 0
        df.iloc[label_indices, -1] = 1
        df.index.name = "timestamp"

        df.to_csv(target_path / f"{name}.csv", index=True)
        print("... written to disk!")


def timeeval_processing(target_path: Path, max_rows: int, anom_length: int) -> None:
    timeseries, annotations = create_combinations(max_rows)

    # test
    # test_combine_name = all_dataset_combinations[9]
    # test_combine_name = "_".join(map(str, test_combine_name))
    # print(test_combine_name)
    # print(annotations[test_combine_name])
    # print(timeseries[test_combine_name].shape)

    datasets = []
    for name in annotations:
        print(f"Creating dataframe for {name}")
        ts = timeseries[name]
        label_indices = annotations[name]
        df = pd.DataFrame(ts, columns=["value"])
        df["is_anomaly"] = 0
        for idx in label_indices:
            df.iloc[idx:idx+anom_length, -1] = 1
        df.index.name = "timestamp"

        datasets.append({
            "collection_name": "SAND",
            "dataset_name": name,
            "train_path": None,
            "test_path": f"{name}.csv",
            "dataset_type": "synthetic" if "_" in name else "real",
            "datetime_index": False,
            "split_at": None,
            "train_type": "unsupervised",
            "train_is_normal": True,
            "input_type": "univariate",
            "length": df.shape[0],
            "dimensions": 1,
            "contamination": df["is_anomaly"].sum() / df.shape[0],
            "num_anomalies": label_indices.shape[0],
            "min_anomaly_length": anom_length,
            "median_anomaly_length": anom_length,
            "max_anomaly_length": anom_length,
            "mean": df["value"].mean(),
            "stddev": df["value"].std(),
            "trend": None,
            "stationarity": None,
            "period_size": None,
        })

        df.to_csv(target_path / f"{name}.csv", index=True)
        print("... written to disk!")

    print("Saving overview file")
    pd.DataFrame(datasets).to_csv(target_path / "datasets.csv", index=False)


def compare_complexity() -> None:
    orig_path = PATH / "processed" / "original"
    timeeval_path = PATH / "processed" / "timeeval"
    df_datasets = pd.read_csv(timeeval_path / "datasets.csv")

    for d in all_dataset_combinations:
        name = "_".join(map(str, d))
        df = pd.read_csv(orig_path / f"{name}.csv")
        sand_n_anomalies = df["is_anomaly"].sum()
        sand_length = df.shape[0]
        sand_perc = sand_n_anomalies / sand_length * 100
        timeeval_n_anomalies = df_datasets.loc[df_datasets["dataset_name"] == name, "num_anomalies"].item()
        timeeval_length = df_datasets.loc[df_datasets["dataset_name"] == name, "length"].item()
        timeeval_perc = timeeval_n_anomalies / timeeval_length * 100

        print(f"{name}\t{sand_n_anomalies:3.0f} -> {timeeval_n_anomalies:3.0f} ({timeeval_n_anomalies/sand_n_anomalies:4.0%}) "
              f"{sand_length/1000:3.0f}k -> {timeeval_length/1000:3.0f}k ({timeeval_length/sand_length:4.0%}) "
              f"{sand_perc:0.2f} -> {timeeval_perc:0.2f}")


if __name__ == '__main__':
    PATH = Path.cwd().parent / "data" / "sand-data"
    TARGET_PATH = PATH / "processed"
    MAX_ROWS = 100_000
    ANOM_LENGTH = 75

    ts_803_file = PATH / "MBA_ECG_803.ts"
    # ts_804_file = PATH / "MBA_ECG_804.ts"
    ts_805_file = PATH / "MBA_ECG_805.ts"
    ts_806_file = PATH / "MBA_ECG_806.ts"
    ts_820_file = PATH / "MBA_ECG_820.ts"
    ts_SED_file = PATH / "SED.ts"

    all_dataset_combinations = [
        [803],
        [805],
        [806],
        [820],
        ["SED"],
        [803, 805],
        [803, 806],
        [803, 820],
        [805, 806],
        [805, 820],
        [806, 820],
        [803, "SED"],
        [805, "SED"],
        [806, "SED"],
        [820, "SED"],
        [803, 805, 806],
        [803, 805, 820],
        [803, 806, 820],
        [805, 806, 820],
        [803, 805, "SED"],
        [803, 806, "SED"],
        [803, 820, "SED"],
        [805, 806, "SED"],
        [805, 820, "SED"],
        [806, 820, "SED"],
        [803, 805, 806, 820],
        [803, 805, 806, "SED"],
        [803, 806, 820, "SED"],
        [805, 806, 820, "SED"],
        [803, 805, 806, 820, "SED"]
    ]

    # original processing of SAND paper:
    # sand_processing(TARGET_PATH / "original", max_rows=MAX_ROWS)

    # preprocess to TimeEval format
    timeeval_processing(TARGET_PATH / "timeeval", max_rows=100_000, anom_length=ANOM_LENGTH)

    compare_complexity()
