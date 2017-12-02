from random import random
from typing import Callable, Union

import numpy as np
from decimal import Decimal

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
                cur_m = 2 * (self(y) - self(x) - self.gradient(x).dot(y - x)) / np.linalg.norm(x - y)
                if m is None or cur_m < m:
                    m = cur_m
        return m

    def __approx_M__(self) -> Decimal:
        m = None
        for i in range(RANDOM_POINTS):
            x = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            y = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            if not np.array_equal(x, y):
                cur_m = np.linalg.norm(self.gradient(x) - self.gradient(y)) / np.linalg.norm(x - y)
                if m is None or cur_m > m:
                    m = cur_m
        return Decimal(m)

    def __approx_argmin_calc__(self, x: np.ndarray) -> Decimal:
        def func_to_minimize(alpha: Decimal) -> Decimal:
            return self(x - alpha * self.gradient(x))

        left = -INF
        right = INF
        while left + EPS < right:
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            if func_to_minimize(mid1) > func_to_minimize(mid2):
                left = mid1
            else:
                right = mid2
        return (left + right) / 2

    def __call__(self, x: np.ndarray) -> Decimal:
        return self.func(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.gradient(x)

def heavy_ball_method(
        f: Func,
        iterations: int
) -> np.ndarray:

    alpha = 4 / (f.m.sqrt() + f.M.sqrt()) ** 2
    beta = (f.M.sqrt() - f.m.sqrt()) / (f.M.sqrt() + f.m.sqrt())
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
        return (-b + d.sqrt()) / 2 / a

    y = x = X0.copy()
    alpha = (f.m / f.M).sqrt() if f.m else Decimal(0.5)
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
    raise NotImplemented


dectype = np.dtype(Decimal)
EPS: Decimal = Decimal(1e-5)
X0: np.ndarray = np.array([Decimal(1)] * 2, dectype)
# Used in approximate M and argmin counting
# setting it pretty low, because we know answer is somewhere in this bounds
INF: Decimal = Decimal(2)
RANDOM_POINTS = 10 ** 3
