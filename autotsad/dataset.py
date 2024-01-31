from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Iterator, Dict, Any, Union, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from timeeval.utils.window import ReverseWindowing

if TYPE_CHECKING:
    import matplotlib.axis
    from .system.anomaly import Injection, AnomalyAnnotation

from .system.anomaly import encode_annotations
from .util import majority_vote


def get_hexhash(dataset_path: Union[Path, Dataset]) -> str:
    """Compute the MD5 hash of a (dataset)-file.

    Parameters
    ----------
    dataset_path : Union[Path, Dataset]
        The path to the file or the dataset object. If a dataset object is provided, it is first written to a
        temporary file to compute its hash.

    Returns
    -------
    hexhash : str
        The MD5 hash of the file.
    """

    def _hash_file(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    if isinstance(dataset_path, Dataset):
        import tempfile

        with tempfile.NamedTemporaryFile() as fh:
            filepath = Path(fh.name)
            dataset_path.to_csv(filepath)
            hexhash = _hash_file(filepath)
    else:
        hexhash = _hash_file(dataset_path)

    return hexhash


@dataclass(init=False, repr=True, order=True)
class Dataset(abc.ABC):
    name: str
    data: np.ndarray
    label: np.ndarray

    @abc.abstractmethod
    def cut_points(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def plot(self, ax: Optional[matplotlib.axis.Axis] = None, cuts: bool = False) -> None:
        ...

    @abc.abstractmethod
    def __sizeof__(self) -> int:
        ...

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def length(self) -> int:
        return self.shape[0]

    def __len__(self) -> int:
        return self.length

    def __eq__(self, other) -> bool:
        return (self.name == other.name
                and np.array_equal(self.data, other.data)
                and np.array_equal(self.label, other.label))

    def sliding_window_view(self, window_size: int) -> DatasetSlidingWindowView:
        return DatasetSlidingWindowView(self, window_size)

    def tumbling_window_view(self, window_size: int, train_window_size: int, prediction_window_size: int) -> DatasetTumblingWindowView:
        return DatasetTumblingWindowView(self, window_size, train_window_size, prediction_window_size)

    def nan_separated_view(self) -> DatasetNaNSeparatedView:
        return DatasetNaNSeparatedView(self)

    def reverse_windowing(self, scores: np.ndarray, window_size: int) -> np.ndarray:
        # filter out windows that contain a cut point
        cuts = self.cut_points()
        idxs = np.arange(len(scores))
        idxs = np.repeat(idxs, len(cuts)).reshape(-1, len(cuts))
        idxs = idxs[np.any((idxs < cuts) & (cuts < idxs + window_size), axis=1), 0]
        # print("invalid window indices (cross a cut):", idxs)
        window_scores = scores.copy()
        window_scores[idxs] = np.nan

        # compute point scores
        pad_n = (window_size - 1, window_size - 1)
        point_scores = np.pad(window_scores, pad_n, "constant", constant_values=(np.nan, np.nan))
        for i in range(len(window_scores) - (window_size - 1)):
            point_scores[i] = np.nanmean(point_scores[i:i + window_size]).item()
        point_scores = point_scores[:-(window_size - 1)]
        return point_scores

    def to_csv(self, path: Path) -> Path:
        filename = path / f"{self.name}.csv"
        df = pd.DataFrame({
            "timestamp": np.arange(self.length),
            "data": self.data,
            "is_anomaly": self.label
        })
        df.to_csv(filename, index=False)
        return filename


@dataclass(init=False, repr=True, order=True)
class TestDataset(Dataset):
    name: str
    data: np.ndarray
    label: np.ndarray
    hexhash: str

    def __init__(self, data: np.ndarray, labels: np.ndarray, hexhash: str, name: Optional[str] = None) -> None:
        assert len(data.shape) == 1, "Currently, only univariate TS are supported!"
        assert data.shape[0] == labels.shape[0], "Data and labels must have the same length!"
        assert data.dtype == np.float_, "Data must be float!"
        assert labels.dtype == np.bool_, "Labels must be boolean!"

        self.data = data
        self.label = labels
        self.hexhash = hexhash
        self.name = name or hexhash

    def __sizeof__(self) -> int:
        return self.data.__sizeof__() + self.label.__sizeof__()

    def __eq__(self, other) -> bool:
        return (self.name == other.name
                and self.hexhash == other.hexhash)

    def cut_points(self) -> np.ndarray:
        return np.array([], dtype=np.int_)

    def plot(self, ax: Optional[matplotlib.axis.Axis] = None, cuts: bool = False) -> None:
        if ax is None:
            from matplotlib import pyplot as plt

            ax = plt.gca()
        y = self.data
        ax.plot(y, label=f"Test dataset")
        if np.any(self.label):
            from matplotlib.pyplot import Rectangle
            from autotsad.util import mask_to_slices

            label_slices = mask_to_slices(self.label)
            y0, y1 = ax.get_ylim()
            for b, e in label_slices:
                height = y1 - y0
                ax.add_patch(
                    Rectangle((b, y0), e-b, height, edgecolor="orange", facecolor="yellow", alpha=0.5)
                )

    @staticmethod
    def from_df(df: pd.DataFrame, hexhash: str, name: Optional[str] = None) -> TestDataset:
        data = df.iloc[:, 1].values.astype(np.float_)
        label = df.iloc[:, -1].values.astype(np.bool_)
        return TestDataset(data, label, hexhash, name=name)

    @staticmethod
    def from_file(filepath: Path) -> TestDataset:
        hexhash = get_hexhash(filepath)
        df = pd.read_csv(filepath)
        return TestDataset.from_df(df, hexhash, name=filepath.stem)


@dataclass(init=False, repr=True, order=True)
class TrainDataset(Dataset, abc.ABC):
    name: str
    data: np.ndarray   # float64, shape=(n,)
    label: np.ndarray  # bool,    shape=(n,)
    period_size: int
    annotations: List[AnomalyAnnotation]

    def __eq__(self, other) -> bool:
        return (self.name == other.name
                and np.array_equal(self.data, other.data)
                and np.array_equal(self.label, other.label)
                and self.period_size == other.period_size
                and all(a == b for a, b in zip(self.annotations, other.annotations)))

    @property
    def contamination(self) -> float:
        return self.label.sum() / self.length


@dataclass(init=False, repr=True, order=True)
class BaseTSDataset(TrainDataset):
    name: str
    data: np.ndarray   # float64, shape=(n,)
    label: np.ndarray  # bool,    shape=(n,)
    mask: np.ndarray   # bool,    shape=(m,)
    period_size: int   # original window size
    annotations: List[AnomalyAnnotation]

    def __init__(self, name: str, array: np.ndarray, mask: np.ndarray, period_size: int) -> None:
        self.name = name
        self.data = array[mask]
        self.label = np.zeros_like(self.data, dtype=np.bool_)
        self.mask = mask
        self.period_size = period_size
        self.annotations = []

    def __sizeof__(self) -> int:
        size_annotations = self.annotations.__sizeof__() + sum(a.__sizeof__() for a in self.annotations)
        return (self.name.__sizeof__() + self.data.__sizeof__() + self.label.__sizeof__() + self.mask.__sizeof__() +
                self.period_size.__sizeof__() + size_annotations)

    @property
    def index_mapping(self) -> Dict[int, int]:
        index = np.arange(self.mask.shape[0])
        data_index = np.cumsum(self.mask, dtype=np.int_) - 1
        idx_mapping = np.c_[data_index, index][self.mask, :]
        return dict(idx_mapping)

    @property
    def reverse_index_mapping(self) -> Dict[int, int]:
        reverse_idx_mapping = dict((v, k) for k, v in self.index_mapping.items())
        return reverse_idx_mapping

    def cut_points(self) -> np.ndarray:
        orig_cut_idxs = np.nonzero(np.diff(np.r_[0, self.mask, 0]) == 1)[0]
        cut_points = np.array([
            self.reverse_index_mapping[idx] for idx in orig_cut_idxs if self.reverse_index_mapping[idx] > 0
        ], dtype=np.int_)
        return cut_points

    def remove_slices(self, slices: np.ndarray) -> None:
        idx_mapping = self.index_mapping
        keep_mask = np.ones_like(self.data, dtype=np.bool_)
        for b, e in slices:
            keep_mask[b:e] = False
            begin = idx_mapping[b]
            end = idx_mapping[e-1]+1
            self.mask[begin:end] = False
        self.data = self.data[keep_mask]
        self.label = self.label[keep_mask]

    def inject_anomalies(self, injection: Injection) -> TrainingTSDataset:
        data, label, annotations, cut_points = injection(
            self.data.copy(),
            self.label.copy(),
            [],
            self.cut_points()
        )
        name = f"train-ts-{'-'.join(self.name.split('-')[-2:])}-{encode_annotations(annotations)}"
        return TrainingTSDataset(
            name=name,
            data=data,
            label=label,
            period_size=self.period_size,
            annotations=annotations,
            cuts=cut_points
        )

    def plot(self, ax: Optional[matplotlib.axis.Axis] = None, cuts: bool = False) -> None:
        if ax is None:
            from matplotlib import pyplot as plt

            ax = plt.gca()
        y = self.data
        ax.plot(y, label=f"TS with period {self.period_size}")
        if cuts:
            ax.vlines(self.cut_points(), y.min() - y.std(), y.max() + y.std(), color="red", label="Cuts")


@dataclass(init=True, repr=True, order=True)
class TrainingTSDataset(TrainDataset):
    name: str
    data: np.ndarray   # float64, shape=(n,)
    label: np.ndarray  # bool,    shape=(n,)
    period_size: int
    annotations: List[AnomalyAnnotation]
    cuts: np.ndarray

    def __sizeof__(self) -> int:
        size_annotations = self.annotations.__sizeof__() + sum(a.__sizeof__() for a in self.annotations)
        return (self.name.__sizeof__() + self.data.__sizeof__() + self.label.__sizeof__() +
                self.period_size.__sizeof__() + size_annotations + self.cuts.__sizeof__())

    def __eq__(self, other) -> bool:
        return (self.name == other.name
                and np.array_equal(self.data, other.data)
                and np.array_equal(self.label, other.label)
                and self.period_size == other.period_size
                and all(a == b for a, b in zip(self.annotations, other.annotations))
                and np.array_equal(self.cuts, other.cuts))

    @property
    def opt_dims(self) -> Dict[str, Any]:
        base_ts_name = "-".join(self.name.split("-")[:-1])
        anomaly_type = majority_vote([a.anomaly_type for a in self.annotations])
        anomaly_length = majority_vote([a.length for a in self.annotations])
        return {
            "base": base_ts_name,
            "anomaly_type": anomaly_type,
            "anomaly_length": int(anomaly_length),
        }

    def cut_points(self) -> np.ndarray:
        return self.cuts.copy()

    def plot(self, ax: Optional[matplotlib.axis.Axis] = None, cuts: bool = False, annotations: bool = False) -> None:
        if ax is None:
            from matplotlib import pyplot as plt

            ax = plt.gca()
        y = self.data
        ax.plot(y, label=f"TS with period {self.period_size}")
        y0, y1 = ax.get_ylim()
        if cuts:
            ax.vlines(self.cuts, y0, y1, color="red", label="Cuts")
        if annotations:
            from matplotlib.pyplot import Rectangle

            for a in self.annotations:
                height = y1 - y0
                ax.add_patch(
                    Rectangle((a.position, y0), a.length, height, edgecolor="orange", facecolor="yellow", alpha=0.5)
                )
                ax.annotate(a.text, (a.display_idx, y1), color="black", ha="center", va="top")


@dataclass(repr=True, order=True)
class DatasetSlidingWindowView:
    window_size: int
    data: np.ndarray

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self.window_size = window_size

        cuts = dataset.cut_points()
        cuts = cuts[cuts > 0]
        region_slices = np.array(list(zip(np.r_[0, cuts], np.r_[cuts, dataset.length])), dtype=np.int_)
        # print("region slices", region_slices)
        result_region_slices = []
        windows = []
        offset = 0
        current_skip_offset = -1
        for i in range(region_slices.shape[0]):
            b, e = region_slices[i]
            if e - b >= window_size:
                # print(b, e, "computing sliding windows", offset, current_skip_offset)
                windows.append(sliding_window_view(dataset.data[b:e], window_shape=window_size))
                result_region_slices.append((offset, offset + e - b - window_size + 1, e - b - window_size + 1))
                offset = offset + e - b - window_size + 1
            else:
                # print(b, e, "skipping", offset, current_skip_offset)
                if offset == current_skip_offset:
                    result_region_slices[-1] = (offset, offset, result_region_slices[-1][-1] + e - b)
                else:
                    result_region_slices.append((offset, offset, e - b))
                    current_skip_offset = offset
        self.data = np.concatenate(windows)
        self.length = self.data.shape[0]
        self._window_region_slices = np.array(result_region_slices)

    def __len__(self) -> int:
        return self.length

    @property
    def shape(self) -> tuple:
        return self.length, self.window_size

    def reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        score_windows = []
        # print("region slices", self._window_region_slices)
        for b, e, length in self._window_region_slices:
            if e - b == length:
                # print(b, e, "--> reverse windowing")
                score_windows.append(ReverseWindowing(window_size=self.window_size).fit_transform(scores[b:e]))
            else:
                # print(b, e, f"--> adding {length} zeros")
                score_windows.append(np.full(length, fill_value=np.nan))
        result = np.concatenate(score_windows)
        # set skipped regions to minimum scores
        result[np.isnan(result)] = np.nanmin(result)
        return result


@dataclass(repr=True, order=True)
class DatasetTumblingWindowView:
    data: np.ndarray

    def __init__(self, dataset: Dataset, window_size: int, train_window_size: int, prediction_window_size: int) -> None:
        series = dataset.data
        self.padding_size = 0
        self.window_size = window_size
        self.train_window_size = train_window_size
        self.prediction_window_size = prediction_window_size

        cuts = dataset.cut_points()
        cuts = cuts[cuts > 0]
        # print("Cuts", cuts)

        if series.shape[0] % window_size != 0:
            slices = series.shape[0] // window_size
            self.padding_size = (slices + 1) * window_size - series.shape[0]
            # print(f"Series not divisible by context window size, adding {self.padding_size} padding points")
            series = np.concatenate([series, np.full(self.padding_size, fill_value=0)], axis=0)
        data = series.reshape((series.shape[0] // window_size, window_size, 1))
        self.mask = np.ones(data.shape[0], dtype=np.bool_)
        self.mask[cuts // window_size] = 0
        self.data = data[self.mask]
        # print("data shape", self.data.shape)

    # @staticmethod
    # def _reverse_windowing_vectorized_entire_3d(scores: np.ndarray, window_size: int) -> np.ndarray:
    #     dim = scores.shape[-1]
    #     unwindowed_length = (window_size - 1) + len(scores)
    #     print(f"from ({len(scores)},{dim}) to ({unwindowed_length},{dim})")
    #     mapped = np.full(shape=(window_size, unwindowed_length, dim), fill_value=np.nan)
    #     mapped[0, :len(scores), :] = scores
    #
    #     for w in range(1, window_size):
    #         mapped[w, :, :] = np.roll(mapped[0, :, :], shift=w, axis=0)
    #
    #     return np.nanmean(mapped, axis=0)

    @staticmethod
    def _reverse_windowing_vectorized_entire_tumbling(scores: np.ndarray, window_size: int, tumbling_window_size: int) -> np.ndarray:
        unwindowed_length = (window_size-1)*tumbling_window_size + len(scores)
        # print(f"from ({len(scores)},) to ({unwindowed_length},)")
        mapped = np.full(shape=(window_size*tumbling_window_size, unwindowed_length), fill_value=np.nan)
        mapped[0, :len(scores)] = scores

        for w in range(tumbling_window_size, window_size*tumbling_window_size):
            mapped[w-tumbling_window_size, :] = np.roll(mapped[0, :], shift=w, axis=0)

        return np.nanmean(mapped, axis=0)

    def reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        scores = 1 - scores
        parts = [
            np.full((self.train_window_size, self.window_size), fill_value=np.nan),
        ]

        cut_indices = np.arange(self.mask.shape[0])[~self.mask]
        cut_indices = np.r_[cut_indices, self.mask.shape[0]]
        n_cuts = 0
        last_end = 0
        for e in cut_indices:
            begin = last_end
            end = min(e - self.train_window_size - n_cuts, scores.shape[0])
            if end > 0:
                parts.append(scores[begin:end, :])
                last_end = end
            parts.append(np.full((1, self.window_size), fill_value=np.nan, dtype=np.float_))
            n_cuts += 1
        # remove last nan part
        parts.pop()

        s = np.concatenate(parts)
        # print(f"{s.shape=} (parts)")

        # s = self._reverse_windowing_vectorized_entire_3d(s, self.prediction_window_size + 1)
        # print(f"{s.shape=} (cuts)")

        s = s.ravel()
        # print(f"{s.shape=} (ravel)")
        s = self._reverse_windowing_vectorized_entire_tumbling(s, self.prediction_window_size + 1, self.window_size)
        # print(f"{s.shape=} (reverse windowing)")

        if self.padding_size:
            # remove padding points
            # print("Removing padding from scores ...")
            s = s[:-self.padding_size]
        # print(f"{s.shape=} (padding)")

        # set skipped regions to minimum scores
        s[np.isnan(s)] = np.nanmin(s)
        return s


@dataclass(repr=True, order=True)
class DatasetNaNSeparatedView:
    data: np.ndarray

    def __init__(self, dataset: Dataset) -> None:
        region_slices = np.array(list(zip(
            np.r_[0, dataset.cut_points()],
            np.r_[dataset.cut_points(), dataset.length]
        )), dtype=np.int_)
        # print("region slices", region_slices)
        regions = []
        result_region_slices = []
        offset = 0
        for b, e in region_slices:
            if e - b == 0:
                continue
            regions.append(np.r_[dataset.data[b:e], np.nan])
            result_region_slices.append((offset, offset + e - b))
            offset = offset + e - b + 1
        self.data = np.concatenate(regions)
        self.length = self.data.shape[0]
        self._window_region_slices = np.array(result_region_slices)

    def __len__(self) -> int:
        return self.length

    @property
    def shape(self) -> tuple:
        return self.length,

    def reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        score_windows = []
        # print("region slices", self._window_region_slices)
        for b, e in self._window_region_slices:
            score_windows.append(scores[b:e])
        result = np.concatenate(score_windows)
        # set skipped to regions to minimum scores
        result[~np.isfinite(result)] = np.min(result[np.isfinite(result)], initial=0.)
        return result


@dataclass(init=True, repr=True, order=True)
class TrainingDatasetCollection:
    test_data: TestDataset
    # TODO: refactor to a dictionary (name -> dataset) and inherit from collections.abc.Mapping
    datasets: List[TrainDataset] = field(default_factory=list)

    def __iter__(self) -> Iterator[TrainDataset]:
        for d in self.datasets:
            yield d

    def __getitem__(self, i: Union[int, str]) -> Dataset:
        if isinstance(i, int):
            return self.datasets[i]
        elif isinstance(i, str):
            if i == self.test_data.name:
                return self.test_data
            else:
                return self.find(i)
        else:
            raise KeyError(f"Index type ({type(i)} of {i} is not supported!")

    def __len__(self) -> int:
        return self.size

    def __sizeof__(self) -> int:
        return self.test_data.__sizeof__() + self.datasets.__sizeof__() + sum([d.__sizeof__() for d in self.datasets])

    @property
    def size(self) -> int:
        return len(self.datasets)

    @property
    def training_datasets(self) -> List[TrainingTSDataset]:
        return [d for d in self.datasets if isinstance(d, TrainingTSDataset)]

    @property
    def cache_key(self) -> str:
        return self.test_data.hexhash

    def append(self, dataset: TrainDataset) -> None:
        self.datasets.append(dataset)

    def remove(self, dataset: TrainDataset) -> None:
        self.datasets.remove(dataset)

    def remove_named(self, name: str) -> TrainDataset:
        dataset = self.find(name)
        self.remove(dataset)
        return dataset

    def add_base_ts(self, mask: np.ndarray, period_size: int) -> BaseTSDataset:
        name = f"base-ts-{self.test_data.hexhash}-{self.size}"
        dataset = BaseTSDataset(name, self.test_data.data, mask, period_size)
        self.datasets.append(dataset)
        return dataset

    def serialize_datasets_to_csv(self, path: Path) -> Iterator[Path]:
        for d in self.datasets:
            yield d.to_csv(path)

    def save(self, path: Path) -> None:
        self.datasets = sorted(self.datasets, key=lambda d: d.name)
        with path.open("wb") as fh:
            joblib.dump(self, fh)

    def get_base_optimization_series(self,
                                     tpe: str,
                                     base: Optional[str] = None,
                                     alength: Optional[int] = None,
                                     atype: Optional[str] = None) -> List[TrainingTSDataset]:
        fix_base = base or self.training_datasets[0].opt_dims["base"]
        fix_length = alength or 100
        fix_type = atype or "scale"

        if tpe == "base":
            return self.base_optimization_series(fix_length, fix_type)
        elif tpe == "anomaly_length":
            return self.length_optimization_series(fix_base, fix_type)
        elif tpe == "anomaly_type":
            return self.type_optimization_series(fix_base, fix_length)
        else:
            raise ValueError(f"Optimization series type '{tpe}' is not defined!")

    def base_optimization_series(self, anomaly_length: int, anomaly_type: str) -> List[TrainingTSDataset]:
        datasets = self.training_datasets
        datasets = [d for d in datasets
                    if d.opt_dims["anomaly_length"] == anomaly_length
                    and d.opt_dims["anomaly_type"] == anomaly_type]
        return datasets

    def length_optimization_series(self, base: str, anomaly_type: str) -> List[TrainingTSDataset]:
        datasets = self.training_datasets
        datasets = [d for d in datasets
                    if d.opt_dims["base"] == base
                    and d.opt_dims["anomaly_type"] == anomaly_type]
        return datasets

    def type_optimization_series(self, base: str, anomaly_length: int) -> List[TrainingTSDataset]:
        datasets = self.training_datasets
        datasets = [d for d in datasets
                    if d.opt_dims["base"] == base
                    and d.opt_dims["anomaly_length"] == anomaly_length]
        return datasets

    def find(self, dataset_name: str) -> TrainingTSDataset:
        datasets = [d for d in self.training_datasets if d.name == dataset_name]
        if len(datasets) < 1:
            raise KeyError(f"Dataset with name {dataset_name} not found!")
        return datasets[0]

    @staticmethod
    def load(path: Path) -> TrainingDatasetCollection:
        with path.open("rb") as fh:
            return joblib.load(fh)

    @staticmethod
    def from_base_timeseries(
            test_data: TestDataset,
            initial_ts: Optional[List[Tuple[np.ndarray, int]]] = None
    ) -> TrainingDatasetCollection:
        collection = TrainingDatasetCollection(test_data)
        if initial_ts is not None:
            for mask, period in initial_ts:
                collection.add_base_ts(mask, period)
        return collection
