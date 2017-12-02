from random import random
from typing import Callable, Union

import numpy as np
from decimal import *


# Supplementary implementations
def solve_with_stats(problem, k, method):
    x = method(problem, k)
    print(f"For {method.__name__}:")
    print(f"Got {x} after {k} iterations")
    dx = x - problem.x_optimal
    print(f"It is {np.linalg.norm(dx)} away from optimum coordinate of {problem.x_optimal}")
    dy = abs(problem(x) - problem(problem.x_optimal))
    print(f"and {dy} away from optimum of {problem(problem.x_optimal)}")
    print("_" * 179)

    return x


def pi():
    """Compute Pi to the current precision.
    """
    getcontext().prec += 2  # extra digits for intermediate steps
    three = Decimal(3)      # substitute "three=3.0" for regular floats
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n+na, na+8
        d, da = d+da, da+32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s               # unary plus applies the new precision


def cos(x):
    """Return the cosine of x as measured in radians.
    """
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s


# Task-related implementations
# noinspection PyPep8Naming
class Func:
    def __init__(self,
                 func: Callable[[np.ndarray], Decimal],
                 arity: int,
                 gradient: Callable[[np.ndarray], np.ndarray],
                 m: Union[Decimal, None],
                 M: Union[Decimal, None],
                 x_optimal: np.ndarray):
        self.func = func
        self.arity = arity
        self.gradient = gradient
        self.m = self.__approx_m__() if m is None else m
        self.M = self.__approx_M__() if M is None else M
        self.x_optimal = x_optimal

    def __approx_m__(self) -> Decimal:
        m = None
        for i in range(RANDOM_POINTS):
            x = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            y = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            if not np.array_equal(x, y):
                cur_m = 2 * abs(self(y) - self(x) - self.gradient(x).dot(y - x)) / np.linalg.norm(x - y)
                if m is None or cur_m < m:
                    m = cur_m
        return m / 2

    def __approx_M__(self) -> Decimal:
        m = None
        for i in range(RANDOM_POINTS):
            x = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            y = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            if not np.array_equal(x, y):
                cur_m = np.linalg.norm(self.gradient(x) - self.gradient(y)) / np.linalg.norm(x - y)
                if m is None or cur_m > m:
                    m = cur_m
        return Decimal(m) * 2

    def __call__(self, x: np.ndarray) -> Decimal:
        return self.func(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.gradient(x)


def heavy_ball_method(
        f: Func,
        iterations: int
) -> np.ndarray:

    alpha = 4 / (np.sqrt(f.m) + np.sqrt(f.M)) ** 2
    beta = (np.sqrt(f.M) - np.sqrt(f.m)) / (np.sqrt(f.M) + np.sqrt(f.m))
    x = x_prev = X0.copy()
    for iteration in range(iterations):
        p = x - x_prev if iteration else np.zeros(X0.shape, dectype)
        x_next = x - alpha * f.gradient(x) + beta * p
        x, x_prev = x_next, x
    return x


def nesterov_method(
        f: Func,
        iterations: int
) -> np.ndarray:
    #  a x ^ 2 + b x + c = 0
    def solve_quadratic_equation(a: Decimal, b: Decimal, c: Decimal) -> Decimal:
        d = b ** 2 - 4 * a * c
        if d < 0:
            raise ValueError
        return (-b + np.sqrt(d)) / 2 / a

    y = x = X0.copy()
    alpha = np.sqrt(f.m / f.M) if f.m else Decimal(0.5)
    for _ in range(iterations):
        x_new = y - f.gradient(y) / f.M
        alpha_new = solve_quadratic_equation(Decimal(1), alpha ** 2 - f.m / f.M, -alpha ** 2)
        beta = alpha * (1 - alpha) / (alpha ** 2 + alpha_new)
        y = x + beta * (x_new - x)
        x = x_new
        alpha = alpha_new
    return x


def chebyshev_method(
        f: Func,
        iterations: int
) -> np.ndarray:
    # noinspection PyShadowingNames
    def a(i: int, k: int) -> Decimal:
        return 1 / ((f.M + f.m) / 2 + (f.M - f.m) / 2 * cos(pi() / 2 * (2 * i + 1) / k))

    x = X0.copy()
    for i in range(iterations):
        x = x - a(i, iterations) * f.gradient(x)
    return x


dectype = np.dtype(Decimal)
EPS: Decimal = Decimal(1e-5)
X0: np.ndarray = np.array([Decimal(1)] * 2, dectype)
# Used in approximate M and argmin counting
# setting it pretty low, because we know answer is somewhere in this bounds
INF: Decimal = Decimal(2)
RANDOM_POINTS = 10 ** 3
