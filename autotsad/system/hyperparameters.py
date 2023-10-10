from __future__ import annotations

import collections.abc
import copy
from functools import reduce, singledispatch
from operator import xor
from types import MappingProxyType
from typing import Any, Mapping, Iterator, Hashable, Sequence, List


DEFAULT_FLOAT_PRECISION = 6
DEFAULT_FLOAT_TOL = 10 ** -DEFAULT_FLOAT_PRECISION


class ParamSetting(Mapping[str, Any], Hashable):

    def __init__(self, param_mapping: Mapping[str, Any]):
        self._dict = MappingProxyType(copy.deepcopy(param_mapping))

    def __getitem__(self, name: str) -> Any:
        return self._dict.__getitem__(name)

    def __len__(self) -> int:
        return self._dict.__len__()

    def __iter__(self) -> Iterator[str]:
        return self._dict.__iter__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(dict(self._dict))})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(make_hashable(self._dict))

    def __eq__(self, other: Any, float_tol: float = DEFAULT_FLOAT_TOL) -> bool:
        """Compare two parameter dicts for equality, allowing for some tolerance in float and integer values."""
        if not isinstance(other, collections.abc.Mapping):
            return False

        return compare_param_settings(self, other, float_tol)

    @property
    def bin(self, float_precision: int = DEFAULT_FLOAT_PRECISION) -> BinnedParamSetting:
        return BinnedParamSetting(self, float_precision)

    @staticmethod
    def default() -> ParamSetting:
        return ParamSetting({})


class BinnedParamSetting(ParamSetting):

    def __init__(self, param_mapping: Mapping[str, Any], float_precision: int = DEFAULT_FLOAT_PRECISION):
        # bin parameter values
        params = {}
        for k, v in param_mapping.items():
            if isinstance(v, float):
                params[k] = float(int(v * 10 ** float_precision) * 10 ** -float_precision)
            else:
                params[k] = v
        super().__init__(params)

    def __eq__(self, other: Any, **kwargs: Any) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return False

        return dict(self) == dict(other)

    def __hash__(self) -> int:
        return hash(make_hashable(self._dict))


# @numba.jit(nopython=True)
def compare_param_settings(obj: collections.abc.Mapping,
                           other: collections.abc.Mapping,
                           float_tol: float = DEFAULT_FLOAT_TOL) -> bool:
    """Compare two parameter dicts for equality, allowing for some tolerance in float values."""
    keys = other.keys() | obj.keys()
    if len(keys) != len(obj.keys()) != len(other.keys()):
        return False

    for k in keys:
        v1 = obj[k]
        v2 = other[k]
        if isinstance(v1, float) and isinstance(v2, float):
            if abs(v1 - v2) > float_tol:
                return False
        else:
            if v1 != v2:
                return False
    return True


# @numba.jit(nopython=True)
def param_setting_list_intersection(params1: Sequence[ParamSetting], params2: Sequence[ParamSetting]) -> List[ParamSetting]:
    result = []
    for p1 in params1:
        match_found = False
        for p2 in params2:
            if p1 == p2:
                match_found = True
                break
        if match_found:
            result.append(p1)
    return result


def param_setting_binned_list_intersection(params1: Sequence[ParamSetting], params2: Sequence[ParamSetting]) -> List[ParamSetting]:
    compare = set(p.bin for p in params1)
    result = []
    for p in params2:
        if p.bin in compare:
            result.append(p)
    return result


@singledispatch
def make_hashable(o):
    raise TypeError(f"Cannot make {o} (type {type(o)}) hashable!")


@make_hashable.register
def _(o: collections.abc.Hashable):
    return o


@make_hashable.register
def _(o: list):
    return tuple(make_hashable(e) for e in o)


@make_hashable.register
def _(o: tuple):
    return tuple(make_hashable(e) for e in o)


@make_hashable.register
def _(o: collections.abc.Set):
    return frozenset(make_hashable(e) for e in o)


@make_hashable.register
def _(o: collections.abc.Mapping):
    def _hash(kv):
        k, v = kv
        return hash((k, make_hashable(v)))

    return reduce(xor, map(_hash, o.items()), 0)
