#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from optuna import Trial

from .torsk import Params, train_predict_esn
from .torsk.anomaly import sliding_score
from .torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
from .torsk.models.numpy_esn import NumpyESN as ESN
from ..util import random_seed, default_reverse_windowing, supress_output

MODEL_PATH = Path("model")

supports_sliding_window_view = False
supports_tumbling_window_view = True
supports_nan_view = False


@dataclass
class CustomParameters:
    input_map_size: int = 10
    input_map_scale: float = 0.001
    context_window_size: int = 10  # this is a tumbling window creating the slices
    # -----
    # These create the subsequences (sliding window of train_window_size + prediction_window_size + 1 slices of shape (context_window_size, dim))
    train_window_size: int = 20
    prediction_window_size: int = 10
    transient_window_size: int = 10
    # -----
    spectral_radius: float = 1.0
    density: float = 1
    reservoir_representation: str = "sparse"  # sparse is significantly faster
    imed_loss: bool = False  # both work
    train_method: str = "pinv_svd"  # options: "pinv_lstsq", "pinv_svd", "tikhonov"
    tikhonov_beta: Optional[float] = None  # float; only used when train_method="tikhonov"
    verbose: int = 2
    scoring_small_window_size: int = 10
    scoring_large_window_size: int = 50
    random_state: int = 42

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            context_window_size=trial.suggest_int("context_window_size", 10, 1000),
            density=trial.suggest_float("density", 0.0001, 1, log=True),
            imed_loss=trial.suggest_categorical("imed_loss", [True, False]),
            input_map_scale=trial.suggest_float("input_map_scale", 0.001, 1.0),
            input_map_size=trial.suggest_int("input_map_size", 10, 2000),
            scoring_large_window_size=trial.suggest_int("scoring_large_window_size", 50, 1000),
            scoring_small_window_size=trial.suggest_int("scoring_small_window_size", 10, 500),
            spectral_radius=trial.suggest_float("spectral_radius", 1.0, 10.0),
            train_window_size=trial.suggest_int("train_window_size", 10, 1000),
            prediction_window_size=trial.suggest_int("prediction_window_size", 10, 1000),
            transient_window_size=trial.suggest_int("transient_window_size", 10, 1000),
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            context_window_size=10,
            density=0.01,
            imed_loss=False,
            input_map_scale=0.125,
            input_map_size=100,
            prediction_window_size=5,
            scoring_large_window_size=100,
            scoring_small_window_size=10,
            spectral_radius=2.0,
            train_window_size=100,
            transient_window_size=20
        )


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def configure_logging(params: CustomParameters) -> logging.Logger:
    verbosity = params.verbose
    level = "ERROR"
    if verbosity == 1:
        level = "WARNING"
    elif verbosity == 2:
        level = "INFO"
    elif verbosity > 2:
        level = "DEBUG"

    base_logger_name = ".".join(__name__.split(".")[:-1])
    logger = logging.getLogger(base_logger_name)
    logger.setLevel(level)
    return logger


def create_torsk_params(custom_params: CustomParameters, ndims: int) -> Params:
    # check additional invariants:
    assert custom_params.input_map_size >= custom_params.context_window_size, \
        "Hidden size must be larger than or equal to input window size!"

    params = Params()

    # add custom parameters while rewriting some key-names:
    custom_params_dict = asdict(custom_params)
    params.train_length = custom_params_dict["train_window_size"]
    params.pred_length = custom_params_dict["prediction_window_size"]
    params.transient_length = custom_params_dict["transient_window_size"]
    del custom_params_dict["train_window_size"]
    del custom_params_dict["prediction_window_size"]
    del custom_params_dict["transient_window_size"]
    del custom_params_dict["random_state"]
    params.update(custom_params_dict)

    # fixed values:
    params.input_map_specs = [{
        "type": "random_weights",
        "size": [custom_params.input_map_size],
        "input_scale": custom_params.input_map_scale
    }]
    params.input_shape = (custom_params.context_window_size, ndims)
    params.dtype = "float64"  # no need to change
    params.backend = "numpy"  # torch does not work, bh not implemented!
    params.debug = False  # must always be False
    params.timing_depth = custom_params.verbose  # verbosity for timing output
    return params


def main(data: np.ndarray,
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True):
    # disable logging
    params.verbose = 0

    with random_seed(params.random_state), supress_output(stderr=True):
        logger = configure_logging(params)
        ndims = 1

        t_params = create_torsk_params(params, ndims)
        logger.debug(t_params)

        if not cut_mode:
            series = data
            padding_size = 0
            padding_needed = series.shape[0] % t_params.input_shape[0] != 0
            if padding_needed:
                slices = series.shape[0] // t_params.input_shape[0]
                padding_size = (slices + 1) * t_params.input_shape[0] - series.shape[0]
                logger.info(f"Series not divisible by context window size, adding {padding_size} padding points")
                if ndims == 1:
                    series = np.concatenate([series, np.zeros(padding_size)])
                else:
                    series = np.concatenate([series, np.zeros((padding_size, ndims))], axis=0)
            data = series.reshape((series.shape[0] // t_params.input_shape[0], t_params.input_shape[0], ndims))
            dataset = ImageDataset(images=data, scale_images=True, params=t_params)
        else:
            dataset = ImageDataset(images=data, scale_images=True, params=t_params)

        steps = dataset.nr_sequences
        logger.debug(f"Input shape\t: {data.shape}")
        logger.debug(f"Target input shape\t: {t_params.input_shape}")
        logger.debug(f"Steps\t\t: {steps}")
        logger.debug(f"Dataset length\t: {len(dataset)}")

        model = ESN(t_params)
        predictions, targets = train_predict_esn(model, dataset, steps=steps, step_length=1, step_start=0)

        errors = []
        for preds, labels in zip(predictions, targets):
            error = np.abs(labels - preds).mean(axis=-1).mean(axis=0)
            errors.append(error)
        logger.debug(f"{len(predictions)}x error shape: {errors[0].shape}")
        scores, _, _, _ = sliding_score(np.array(errors),
                                        small_window=t_params.scoring_small_window_size,
                                        large_window=t_params.scoring_large_window_size)
        if postprocess:
            scores = np.concatenate([
                # the first batch of training samples has no predictions --> no scores
                np.full(shape=(t_params.train_length, t_params.context_window_size), fill_value=np.nan),
                scores
            ], axis=0)

            scores = 1 - scores.ravel()
            if padding_needed:
                # remove padding points
                logger.info("Removing padding from scores ...")
                scores = scores[:-padding_size]
            scores = default_reverse_windowing(scores, t_params,
                                               window_size=t_params.pred_length * t_params.context_window_size + 1)

            # set skipped regions to minimum scores
            scores[np.isnan(scores)] = np.nanmin(scores)

    logger.debug(f"Scores shape={scores.shape}")
    # plot(series, scores, config)
    return scores


def plot(series, scores, params: CustomParameters):
    import matplotlib.pyplot as plt

    scores_post = default_reverse_windowing(scores, params,
                                            window_size=params.prediction_window_size * params.context_window_size + 1)
    fig, ax = plt.subplots()
    ax.plot(series[:, 0], label="series", color="blue", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax2 = ax.twinx()
    ax2.plot(scores, label="window-score", color="red", alpha=0.25)
    ax2.plot(scores_post, label="point-score", color="green", alpha=0.7)
    ax2.set_ylabel("score")
    plt.legend()
    plt.show()
