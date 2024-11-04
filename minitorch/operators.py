"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable, Union

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


Number = Union[int, float]

def mul(x: Number, y: Number) -> Number:
    return x * y

def id(x: Number) -> Number:
    return x

def add(x: Number, y: Number) -> Number:
    return x + y

def neg(x: Number) -> Number:
    return -x

def lt(x: Number, y: Number) -> bool:
    return x < y

def eq(x: Number, y: Number) -> bool:
    return x == y

def max(x: Number, y: Number) -> Number:
    return x if x > y else y

def is_close(x: Number, y: Number, tol: float = 1e-2) -> bool:
    return abs(x - y) < tol

def sigmoid(x: Number) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

def relu(x: Number) -> Number:
    return x if x > 0 else 0

def log(x: Number) -> float:
    return math.log(x)

def exp(x: Number) -> float:
    return math.exp(x)

def inv(x: Number) -> float:
    return 1.0 / x

def log_back(x: Number, d: Number) -> float:
    return d / x

def inv_back(x: Number, d: Number) -> float:
    return -d / (x ** 2)

def relu_back(x: Number, d: Number) -> Number:
    return d if x > 0 else 0


EPS = 1e-6


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in lst]

def zipWith(fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    return [fn(x, y) for x, y in zip(lst1, lst2)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float], initial: float) -> float:
    result = initial
    for x in lst:
        result = fn(result, x)
    return result

def negList(lst: Iterable[float]) -> Iterable[float]:
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, lst1, lst2)

def sum(lst: Iterable[float]) -> float:
    return reduce(add, lst, 0.0)

def prod(lst: Iterable[float]) -> float:
    return reduce(mul, lst, 1.0)

