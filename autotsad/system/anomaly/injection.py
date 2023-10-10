from dataclasses import dataclass
from itertools import cycle
from typing import Optional, Tuple, List

import numpy as np

from .annotation import AnomalyAnnotation
from .transforms import CompressTransform, StretchTransform, AnomalyTransform
from ...config import config, ANOMALY_TYPES, AnomalyGenerationSection

InjectionResult = Tuple[np.ndarray, np.ndarray, List[AnomalyAnnotation], np.ndarray]
"""Type alias for the result of the anomaly injection function."""


@dataclass(init=True, frozen=True)
class Injection:
    """Class that captures the parameters for an anomaly injection. The injection can be performed by calling the
    instance.

    Parameters
    ----------
    anomaly_types : List[str]
        Types of anomalies to inject. One of ``"outlier"``, ``"compress"``, ``"stretch"``, ``"noise"``,
        ``"smoothing"``, ``"hmirror"``, ``"vmirror"``, ``"scale"``, or ``"pattern"``.
    n_anomalies : int
        Number of anomalies to inject.
    length : int
        Length of the anomaly to inject.
    random_state : int, optional
        Seed used by the random number generator.
    rng : np.random.RandomState, optional
        Random number generator.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, List[AnomalyAnnotation], np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[int, str], np.ndarray]]
        Function that injects anomalies into a time series.
    """
    anomaly_types: List[str]
    n_anomalies: int
    length: int = 100
    random_state: Optional[int] = None
    rng: Optional[np.random.Generator] = None
    anomaly_config: AnomalyGenerationSection = config.anomaly

    def __call__(self, data: np.ndarray, labels: np.ndarray, annotations: List[AnomalyAnnotation], cut_points: np.ndarray) -> InjectionResult:
        """Inject the anomalies into a time series.

        Parameters
        ----------
        data : np.ndarray
            Time series data.
        labels : np.ndarray
            Existing anomaly labels.
        annotations : List[AnomalyAnnotation]
            Existing anomaly annotations.
        cut_points : np.ndarray
            Cut points of the time series.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[AnomalyAnnotation], np.ndarray]
            Tuple of the time series data, anomaly labels, anomaly annotations, and cut points with the anomalies
            injected.
        """
        return inject_anomalies(
            # first the data to operate on:
            data, labels, annotations, cut_points,
            # then the parameters:
            self.anomaly_types, self.n_anomalies, self.length, self.random_state, self.rng,
            # and the config
            self.anomaly_config
        )


def inject_anomalies(
    data: np.ndarray,
    labels: np.ndarray,
    annotations: List[AnomalyAnnotation],
    cut_points: np.ndarray,
    anomaly_types: List[str],
    n_anomalies: int,
    anomaly_length: int = 100,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    anomaly_config: AnomalyGenerationSection = config.anomaly
) -> InjectionResult:
    """Inject anomalies into a time series.

    Parameters
    ----------
    data : np.ndarray
        Time series.
    labels : np.ndarray
        Existing anomaly labels.
    annotations : List[AnomalyAnnotation]
        Existing anomaly annotations.
    cut_points : np.ndarray
        Cut points of the time series.
    anomaly_types : List[str]
        Types of anomalies to inject. One of ``"outlier"``, ``"compress"``, ``"stretch"``, ``"noise"``,
        ``"smoothing"``, ``"hmirror"``, ``"vmirror"``, ``"scale"``, or ``"pattern"``.
    n_anomalies : int
        Number of anomalies to inject.
    anomaly_length : int
        Length of the anomaly to inject.
    random_state : int, optional
        Seed used by the random number generator.
    rng : np.random.RandomState, optional
        Random number generator.
    anomaly_config : AnomalyGenerationSection, optional
        Configuration for the anomaly generation process.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[AnomalyAnnotation], np.ndarray]
        Tuple of the time series data, anomaly labels, anomaly annotations, and cut points with the anomalies
        injected.
    """
    unknown_anomaly_types = set(anomaly_types) - set(ANOMALY_TYPES)
    if len(unknown_anomaly_types) > 0:
        raise ValueError(f"Unknown anomaly types: {unknown_anomaly_types}")

    if rng is None:
        rng = np.random.default_rng(random_state)
    rng.shuffle(anomaly_types)
    margin = anomaly_length // 4

    for i, anomaly_type in zip(range(n_anomalies), cycle(anomaly_types)):
        # stop when we are above contamination threshold
        if np.sum(labels) / len(labels) > anomaly_config.contamination_threshold:
            print(f"Stopping anomaly injection after {i+1} anomalies, because we are above "
                  f"{anomaly_config.contamination_threshold:.0%} contamination.")
            break

        strength = AnomalyTransform.sample_strength(anomaly_type, rng)

        # adjust length for stretch/compress anomalies to ensure that final anomaly length is the desired one
        if anomaly_type == "stretch":
            length = int(np.ceil(anomaly_length / StretchTransform.factor(strength)))
        elif anomaly_type == "compress":
            length = int(np.ceil(anomaly_length / CompressTransform.factor(strength)))
        else:
            length = anomaly_length

        # get anomaly position
        section_size = len(data) // 3
        position = len(data)
        max_tries = anomaly_config.find_position_max_retries

        while (
                position + length > len(data)
                or np.any(labels[position-margin:position + length + margin])
                or np.any((position - margin < cut_points) & (cut_points < position + length + margin))
        ) and max_tries > 0:
            position_idx = rng.choice([0, 1, 2], p=anomaly_config.anomaly_section_probas)
            position_within_section = rng.integers(0, max(1, section_size - length))
            position = position_idx * section_size + position_within_section
            max_tries -= 1
        if max_tries == 0:
            print(f"Could not find a position for the anomaly {i+1}/{n_anomalies} of type {anomaly_type}, skipping!")
            break

        # get anomaly subsequence
        anomaly_subsequence = data[position:position + length]
        anomaly_subsequence, label_subsequence = _anomaly_transformer(anomaly_type, strength, rng)(anomaly_subsequence)
        anom_length = len(anomaly_subsequence)

        if np.allclose(data[position:position + anom_length], anomaly_subsequence):
            print(f"Anomaly {i+1}/{n_anomalies} of type {anomaly_type} at position {position} with strength "
                  f"{strength:.2f} and length {anomaly_length} is not significant, skipping!")
            continue

        print(f"Injecting anomaly {i+1}/{n_anomalies} of type {anomaly_type} at position {position} with strength "
              f"{strength:.2f} and length {anomaly_length}.")
        data = np.r_[data[:position], anomaly_subsequence, data[position + length:]]
        labels = np.r_[labels[:position], label_subsequence, labels[position + length:]]

        if length != anom_length:
            # fix annotation positions
            annotations = [a.adjust_position(anom_length - length) if a.position > position else a for a in annotations]
            # fix cut points
            cut_points[cut_points > position] += anom_length - length

        # add new annotation
        if np.sum(label_subsequence) < len(label_subsequence):
            display_idx = position + np.nonzero(label_subsequence)[0][0]
            position = display_idx
            anom_length = np.sum(label_subsequence)
        else:
            display_idx = position + anom_length//2
            # less than by 3 off is close enough:
            assert np.abs(anomaly_length - anom_length) <= 3
            anom_length = anomaly_length
        annotations.append(AnomalyAnnotation(position=position, length=anom_length, anomaly_type=anomaly_type,
                                             strength=strength, display_idx=display_idx))

    mask = np.isnan(data)
    data = data[~mask]
    labels = labels[~mask]
    return data, labels, annotations, cut_points


def _anomaly_transformer(anomaly_type: str, strength: float, rng: np.random.Generator) -> AnomalyTransform:
    return AnomalyTransform.get_factory(anomaly_type)(strength=strength, rng=rng)
