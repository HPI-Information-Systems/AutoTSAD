from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Dict, Union, Sequence, Optional

from ..config import config, GeneralSection
from ..util import format_time_ns

SEPARATOR: str = "-%-"


class TimersMngr:
    def __init__(self, timer_name: str = "Timer", general_config: Optional[GeneralSection] = None):
        if general_config is None:
            general_config = config.general
        self.enabled = general_config.use_timer
        self.stack: List[Timer] = []
        self.mapping: Dict[str, Timer] = {}
        self._timer_logging_level = general_config.timer_logging_level
        if timer_name != "Timer":
            self._trace_filename = general_config.cache_dir() / f"runtimes-{timer_name}.csv"
        else:
            self._trace_filename = general_config.cache_dir() / "runtimes.csv"
        self._log = logging.getLogger(f"autotsad.{timer_name}")
        self._trace: List[str] = []

    def start(self, name: Union[str, Sequence[str]]) -> None:
        if not self.enabled:
            return

        if isinstance(name, str):
            name = [name]
        for n in name:
            self._start_timer(n)

    def _start_timer(self, name: str) -> None:
        if name in self.mapping:
            raise ValueError(f"Timer {name} already exists!")
        try:
            outer_name = self.stack[-1].name
            fullname = f"{outer_name}{SEPARATOR}{name}"
        except (IndexError, KeyError):
            fullname = name
        timer = Timer(fullname)
        self.mapping[name] = timer
        self.stack.append(timer)
        self._trace.append(timer.start_entry())

    def stop(self, name: str) -> None:
        if not self.enabled:
            return

        timer = self.mapping[name]
        idx = self.stack.index(timer)
        while len(self.stack) > idx:
            t = self.stack.pop(-1)
            del self.mapping[t.name.split(SEPARATOR)[-1]]
            t.stop()
            self._trace.append(t.end_entry())
            self._log.log(self._timer_logging_level, str(t))

        # after each major trace, save timings in the cache folder
        if len(self.stack) <= 1:
            self.save_trace()

    def save_trace(self, filepath: Optional[Path] = None) -> None:
        if not self.enabled:
            return

        if filepath is None:
            filepath = self._trace_filename
        i = 0
        new_filepath = filepath
        while new_filepath.exists():
            new_filepath = filepath.parent / f"{filepath.stem}-{i}{filepath.suffix}"
            i += 1
        with new_filepath.open("w") as fh:
            fh.write("Timer,Type,Begin (ns),End (ns),Duration (ns)\n")
            for entry in self._trace:
                fh.write(entry + "\n")


class Timer:
    def __init__(self, name: str, fmt_text: str = "{} took {}"):
        self.name = name
        self._begin = time.time_ns()
        self._end: Optional[int] = None
        self._elapsed_ns: Optional[int] = None
        self._fmt_text = fmt_text

    def stop(self) -> None:
        self._end = time.time_ns()
        self._elapsed_ns = self._end - self._begin

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self) -> str:
        return self._fmt_text.format(self.name, format_time_ns(self._elapsed_ns))

    def start_entry(self, start_marker: str = "START") -> str:
        return f"{self.name},{start_marker},{self._begin},,"

    def end_entry(self, end_marker: str = "END") -> str:
        return f"{self.name},{end_marker},,{self._end},{self._elapsed_ns}"


Timers = TimersMngr()


if __name__ == '__main__':
    Timers.start(["outer", "inner-1", "inner-2"])
    time.sleep(1)
    Timers.stop("inner-2")

    Timers.stop("outer")
    time.sleep(1)

    Timers.start("outer2")
    Timers.start("inner2-1")
    Timers.stop("inner2-1")
    Timers.start("inner2-2")
    Timers.stop("inner2-2")
    Timers.stop("outer2")

    print("timer,TYPE,begin,end,duration")
    for entry in Timers._trace:
        print(entry)
    Timers.save_trace(Path("test.csv"))
