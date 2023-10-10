import contextlib
import os
import random
from typing import Any, Generator, Optional

import numpy as np


@contextlib.contextmanager
def random_seed(random_state: Any) -> Generator[None, None, None]:
    old_np_state: Any = np.random.get_state()
    old_random_state: Any = random.getstate()
    np.random.seed(random_state)
    random.seed(random_state)
    try:
        yield
    finally:
        np.random.set_state(old_np_state)
        random.setstate(old_random_state)


@contextlib.contextmanager
def supress_output(stderr: bool = False) -> Generator[None, None, None]:
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if stderr:
            with contextlib.redirect_stderr(f):
                yield
        else:
            yield


def default_reverse_windowing(scores: np.ndarray, params: Any, window_size: Optional[int] = None) -> np.ndarray:
    from timeeval.utils.window import ReverseWindowing

    if window_size is None:
        ws = params.window_size
    else:
        ws = window_size
    return ReverseWindowing(window_size=ws).fit_transform(scores)
