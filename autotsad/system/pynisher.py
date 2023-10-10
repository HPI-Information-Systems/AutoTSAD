from typing import Callable, Optional

import pynisher

# explicitly re-export exception for nicer imports
from pynisher import PynisherException # noqa


def pynish_func(func: Callable,
                enabled: bool = True,
                time_limit: Optional[int] = None,
                memory_limit: Optional[int] = None,
                name: Optional[str] = None) -> Callable:
    """Use pynisher to limit a function's memory and runtime.

    Parameters
    ----------
    func : Callable
        Function to execute in a separate process and limit its memory and runtime.
    enabled : bool
        If `False`, the function is not limited and just executed in the current process.
    time_limit : Optional[int]
        Time limit in seconds. If `None`, runtime is not limited.
    memory_limit : Optional[int]
        Memory limit in megabytes. If `None`, memory is not limited.
    name : str
        Optional process name.

    Returns
    -------
    limited_func : Callable
        A function with the same signature as the incoming function, but it's executed in a separate process and limited
        in its memory and runtime.
    """
    if enabled:
        return pynisher.limit(
            func,
            name=name,
            wall_time=(time_limit, "s") if time_limit else None,
            memory=(memory_limit, "MB") if memory_limit else None,
            memory_limit_type="rss",
        )
    else:
        return func
