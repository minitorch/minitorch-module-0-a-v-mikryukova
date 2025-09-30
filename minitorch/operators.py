"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

def id(x: float) -> float:
    """Return the input unchanged."""
    return x

def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if one number is less than another."""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """Check if two numbers are equal."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close in value."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function.

    For x >= 0: f(x) = 1.0 / (1.0 + e^(-x))
    For x < 0: f(x) = e^x / (1.0 + e^x)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """Apply the ReLU activation function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculate the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function."""
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    """Compute the derivative of log times a second argument."""
    return (1.0 / x) * d

def inv(x: float) -> float:
    """Calculate the reciprocal."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of reciprocal times a second argument."""
    return (-1.0 / (x * x)) * d


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of ReLU times a second argument."""
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Apply a given function to each element of an iterable."""
    return [fn(x) for x in iterable]


def zipWith(fn: Callable[[float, float], float],
            iter1: Iterable[float],
            iter2: Iterable[float]) -> Iterable[float]:
    """Combine  elements from two iterables using a given function."""
    return [fn(x, y) for x, y in zip(iter1, iter2)]


def reduce(fn: Callable[[float, float], float],
           iterable: Iterable[float],
           initial: float = 0.0) -> float:
    """Reduce an iterable to a single value using a given function."""
    result = initial
    for x in iterable:
        result = fn(result, x)
    return result


def negList(iterable: Iterable[float]) -> Iterable[float]:
    """ Negate all elements in a list."""
    return map(neg, iterable)

def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add, iter1, iter2)


def sum(iterable: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, iterable, 0.0)


def prod(iterable: Iterable[float]) -> float:
    """Calculate the product of all elements in a list ."""
    return reduce(mul, iterable, 1.0)
