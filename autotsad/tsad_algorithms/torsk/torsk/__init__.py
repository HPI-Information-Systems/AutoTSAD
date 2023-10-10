import logging
import pathlib

import joblib
import numpy as np

from .params import Params, default_params
from .imed import imed_metric, eucd_metric
from .numpy_accelerate import bh, to_np
from .numpy_accelerate import before_storage, after_storage, numpyize

__all__ = ["Params", "default_params", "load_model", "save_model"]

logger = logging.getLogger(__name__)


def _save_numpy_model(model_pth, model, prefix):
    old_state = before_storage(model)
    joblib.dump(model, model_pth)
    after_storage(model,old_state)


def _load_numpy_model(model_pth):
    from .models.numpy_esn import NumpyESN
    loaded = joblib.load(model_pth)
    model = NumpyESN(loaded.params)
    numpyize(model)
    if model.params.reservoir_representation == "dense":
        model.esn_cell.weight_hh[:] = loaded.esn_cell.weight_hh[:]
    else:
        model.esn_cell.weight_hh.values[:] = loaded.esn_cell.weight_hh.values[:]
        model.esn_cell.weight_hh.col_idx[:] = loaded.esn_cell.weight_hh.col_idx[:]
    after_storage(model)
    return model


def _fix_prefix(prefix):
    if prefix is not None:
        prefix = prefix.strip("-") + "-"
    else:
        prefix = ""
    return prefix


def save_model(modeldir, model, prefix=None):
    if not isinstance(modeldir, pathlib.Path):
        modeldir = pathlib.Path(modeldir)
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
    prefix = _fix_prefix(prefix)

    params_json = modeldir / f"{prefix}params.json"
    logger.info(f"Saving model parameters to {params_json}")
    model.params.save(params_json)

    if model.params.backend == "numpy":
        modelfile = modeldir / f"{prefix}model.pkl"
        logger.info(f"Saving model to {modelfile}")
        _save_numpy_model(modelfile, model, prefix)
    else:
        raise ValueError(f"Unkown backend: {model.params.backend}")


def load_model(modeldir, prefix=None):
    # TODO: fix circular import
    if not isinstance(modeldir, pathlib.Path):
        modeldir = pathlib.Path(modeldir)
    prefix = _fix_prefix(prefix)

    params = Params(modeldir / f"{prefix}params.json")

    if params.backend == "numpy":
        model_pth = modeldir / f"{prefix}model.pkl"
        model = _load_numpy_model(model_pth)
        after_storage(model)
    else:
        raise ValueError(f"Unkown backend: {params.backend}")

    return model


def initial_state(hidden_size, dtype, backend):
    if backend == "numpy":
        zero_state = bh.zeros([hidden_size], dtype=np.float64)
    else:
        raise ValueError(f"Unkown backend: {backend}")
    return zero_state


def dump_cycles(dst, dataset):
    dst.createDimension("cycle_length", dataset.params.cycle_length)
    dst.createDimension("three", 3)
    dst.createVariable(
        "quadratic_trend", float, ["image_height", "image_width", "three"])
    dst.createVariable(
        "mean_cycle", float, ["image_height", "image_width", "cycle_length"])

    dst["quadratic_trend"][:] = dataset.quadratic_trend
    dst["mean_cycle"][:] = dataset.mean_cycle

    dst.setncatts({
        "cycle_timescale": dataset.cycle_timescale,
        "cycle_length": dataset.params.cycle_length
    })


def train_esn(model, dataset):

    inputs, labels, _ = dataset[0]

    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    zero_state = initial_state(hidden_size, dtype, backend)
    _, states = model.forward(inputs, zero_state, states_only=True)

    logger.info("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])
    return inputs, states, labels


def train_predict_esn(model, dataset, steps=1, step_length=1, step_start=0):
    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    predictions = np.empty((steps, model.params.pred_length, model.params.input_shape[0], model.params.input_shape[1]))
    targets = np.empty((steps, model.params.pred_length, model.params.input_shape[0], model.params.input_shape[1]))

    for ii in range(steps):
        model.timer.reset()

        logger.info(f"--- Train/Predict Step Nr. {ii+1} ---")
        idx = ii * step_length + step_start
        inputs, labels, pred_targets = dataset[idx]

        logger.debug(f"Creating {inputs.shape[0]} training states")
        zero_state = initial_state(hidden_size, dtype, backend)
        _, states = model.forward(inputs, zero_state, states_only=True)

        logger.debug("Optimizing output weights")
        model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

        logger.debug(f"Predicting the next {model.params.pred_length} frames")
        init_inputs = labels[-1]
        outputs, out_states = model.predict(
            init_inputs, states[-1], nr_predictions=model.params.pred_length)

        logger.debug(model.timer.pretty_print())

        predictions[ii, :, :, :] = outputs
        targets[ii, :, :, :] = pred_targets

    logger.info(f"Done")
    return predictions, targets
