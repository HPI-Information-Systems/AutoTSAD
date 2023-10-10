from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, TypeVar, Sequence, Iterator

import numpy as np
import pandas as pd


def mask_to_slices(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean vector mask to index slices (inclusive start, exclusive end) where the mask is ``True``.

    Parameters
    ----------
    mask : numpy.ndarray
        A boolean 1D array

    Returns
    -------
    slices : numpy.ndarray
        (-1, 2)-shaped array of slices. Each slice consists of the start index (inclusive) and the end index (exclusive)
        of a continuous region of ``True``-values.
    """
    tmp = np.r_[0, mask, 0]
    slices = np.c_[
        np.nonzero(np.diff(tmp) == 1)[0],
        np.nonzero(np.diff(tmp) == -1)[0]
    ]
    return slices


def slices_to_mask(slices: np.ndarray, length: int) -> np.ndarray:
    """Convert a vector of index slices to a boolean mask of the specified length. The mask is ``True`` for all indices
    that are contained in any of the slices.

    Parameters
    ----------
    slices : numpy.ndarray
        A 2D integer array containing a vector of (inclusive start, exclusive end)-slices.
    length : int
        The length of the resulting mask.

    Returns
    -------
    mask : numpy.ndarray
        A boolean 1D array of the specified length.
    """
    mask = np.zeros(length, dtype=np.bool_)
    for start, end in slices:
        mask[start:end] = True
    return mask


def invert_slices(slices: np.ndarray, first_last: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Inverts the slices, such that the new first slice describes the region between the end of the first slices and
    the beginning of the second slice, and so forth. If ``first_last`` is provided, the first and last points
    are used to construct the first and last slices, respectively.

    Parameters
    ----------
    slices : np.ndarray
        A 2D integer array containing a vector of (start, end)-slices.
    first_last : Tuple[int, int], optional
        The first and last (exclusive) index within the original array.

    Returns
    -------
    inv_slices : np.ndarray
        A 2D integer array containing a vector of (end_i, start_i-1)-slices.
    """
    if first_last is not None:
        first, last = first_last
        slices = np.r_[slices, [[last, first]]]
    ends = slices[:, 1]
    starts = np.roll(slices[:, 0], -1)
    inv_slices = np.c_[ends, starts]
    return inv_slices


def slice_lengths(slices: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Compute the (absolute) length of each slice (start, end) by computing length = end - start. If ``absolute`` is
    True, the absolute length is computed.

    Parameters
    ----------
    slices : np.ndarray
        A 2D integer array containing a vector of (start, end)-slices.
    absolute : bool, optional
        Whether to return the absolute length (default: False).

    Returns
    -------
    lengths : np.ndarray
        A 1D integer array containing the length of each slice (end - start).
    """
    length = np.diff(slices, axis=1).ravel()
    if absolute:
        return np.abs(length, dtype=np.int_)
    else:
        return length.astype(np.int_)


def getsize(obj: object, decimals: int = 3) -> str:
    """Get the size of an object in human-readable format.

    Parameters
    ----------
    obj : object
        The object to get the size of.
    decimals : int, optional
        Number of decimals to use in the output.

    Returns
    -------
    str
        Human-readable string representation of the size of the object.
    """
    return pretty_print_size(sys.getsizeof(obj), decimals=decimals)


def pretty_print_size(nbytes: int, decimals: int = 3) -> str:
    """Convert a size in number of bytes to a human-readable string (using B, kiB, MiB, and GiB).

    Parameters
    ----------
    nbytes : int
        Number of bytes
    decimals : int, optional
        Number of decimals to use in the output.

    Returns
    -------
    str
        Human-readable string representation of the size.
    """
    suffixes = ["B", "kiB", "MiB", "GiB"]
    suffix_index = 0
    while nbytes >= 1024 and suffix_index < 4:
        suffix_index += 1
        nbytes /= 1024
    return f"{nbytes:.{decimals}f} {suffixes[suffix_index]}"


def format_time_ns(time: float) -> str:
    """Format a time in nanoseconds to a human-readable string (using ms, s, min, and h).

    Parameters
    ----------
    time : float
        Time in seconds

    Returns
    -------
    str
        Human-readable string representation of the time.
    """
    return format_time(time, is_ns=True)


def format_time(time: float, is_ns: bool = False) -> str:
    """Format a time in seconds or nanoseconds to a human-readable string (using ms, s, min, and h).

    Parameters
    ----------
    time : float
        Time in seconds
    is_ns : bool, optional
        Whether the time is in nanoseconds.

    Returns
    -------
    str
        Human-readable string representation of the time.
    """
    suffixes = ["ns", "ms", "s", "min", "h", "d"]
    suffix_index = 2
    if is_ns:
        time /= 1e9

    # less than a millisecond
    if time < 0.001:
        suffix_index = 0
        time *= 1e9

    # less than a second
    elif time < 0:
        suffix_index = 1
        time *= 1000

    else:
        while time >= 60 and suffix_index < 4:
            suffix_index += 1
            time /= 60
        if suffix_index == 4 and time >= 24:
            suffix_index += 1
            time /= 24

    return f"{time:.3f} {suffixes[suffix_index]}"


_T = TypeVar("_T")


def majority_vote(x: Sequence[_T]) -> _T:
    """Return the majority vote of the elements in ``x``. This is equal to the single mode of ``x``.

    Parameters
    ----------
    x : ArrayLike[_T]
        A 1D array-like object.

    Returns
    -------
    mode : _T
        The element that occurs most often in ``x``.
    """
    types, counts = np.unique(x, return_counts=True)
    return types[np.argwhere(counts == np.max(counts))].flatten()[0]


def save_serialized_results(results: pd.DataFrame, filename: Path) -> None:
    """Save a Pandas DataFrame to disk. If the DataFrame contains a column named ``params``, the values in this column
    are serialized to JSON before saving.

    Parameters
    ----------
    results : pd.DataFrame
        The DataFrame to save.
    filename : Path
        The path to save the DataFrame to (containing path and file name).

    Returns
    -------
    None
    """
    df_serialized = results.copy()
    if "params" in df_serialized.columns:
        df_serialized["params"] = df_serialized["params"].apply(json.dumps)
    df_serialized.to_csv(filename, index=False)


def load_serialized_results(filename: Path) -> pd.DataFrame:
    """Load a Pandas DataFrame from disk. If the DataFrame contains a column named ``params``, the values in this column
    are deserialized from JSON after loading.

    Parameters
    ----------
    filename : Path
        The path to load the DataFrame from (containing path and file name).

    Returns
    -------
    results : pd.DataFrame
        The loaded DataFrame.
    """
    results = pd.read_csv(filename)
    if "params" in results.columns:
        # try:
        results["params"] = results["params"].apply(json.loads)
        # except JSONDecodeError:
        #     results["params"] = results["params"].apply(eval)
    return results


@contextlib.contextmanager
def show_full_df(max_rows: Optional[int] = None) -> Iterator[None]:
    """Context manager to temporarily show full DataFrames when printing.

    Returns
    -------
    None
    """
    with pd.option_context("display.width", None, "display.max_columns", None, "display.max_colwidth", None,
                           "display.max_rows", max_rows):
        yield
