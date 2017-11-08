from typing import Tuple, Callable, Union
from random import random
from math import log
from decimal import Decimal
import numpy as np


class ConvexFunc:
    def __init__(self,
                 func: Callable[[np.ndarray], Decimal],
                 arity: int,
                 gradient: Callable[[np.ndarray], np.ndarray],
                 m: Union[Decimal, None],
                 x_optimal: np.ndarray,
                 argmin_calc: Union[Callable[[np.ndarray], Decimal], None]):
        self.func = func
        self.arity = arity
        self.gradient = gradient
        self.m = self.__approx_m__() if m is None else m
        self.x_optimal = x_optimal
        self.argmin_calc = self.__approx_argmin_calc__ if argmin_calc is None else argmin_calc

    def __approx_m__(self) -> Decimal:
        m = None
        for i in range(RANDOM_POINTS):
            x = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            y = np.array([Decimal(random() * 2 - 1) for _ in range(2)], dectype) * INF
            if not np.array_equal(x, y):
                cur_m = np.linalg.norm(self.gradient(x) - self.gradient(y)) / np.linalg.norm(x - y)
                if m is None or cur_m > m:
                    m = cur_m
        return m

    def __approx_argmin_calc__(self, x: np.ndarray) -> Decimal:
        def func_to_minimize(alpha: Decimal) -> Decimal:
            return self.apply(x - alpha * self.gradient(x))

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

    def apply(self, x: np.ndarray) -> Decimal:
        return self.func(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.gradient(x)


def gradient_method(func: ConvexFunc,
                    x0: np.ndarray,
                    eps: Decimal,
                    alpha: Callable[[ConvexFunc, np.ndarray], Decimal]) -> Tuple[Decimal, np.ndarray]:
    x: np.ndarray = x0

    iteration: int = 0
    while func.apply(x) - func.apply(func.x_optimal) >= eps:
        x = x - alpha(func, x) * func.gradient(x)
        iteration += 1
        print("After iteration {}: x = {}, ||x - x*|| = {}, f(x) - f(x*) = {}".format(
            iteration,
            x,
            np.linalg.norm(x - func.x_optimal),
            func.apply(x) - func.apply(func.x_optimal)))

    return func.apply(x), x


def gradient_method_const(func: ConvexFunc,
                          x0: np.ndarray,
                          eps: Decimal) -> Tuple[Decimal, np.ndarray]:
    return gradient_method(func, x0, eps, lambda f, x: Decimal(1) / func.m)


def gradient_method_argmin(func: ConvexFunc,
                           x0: np.ndarray,
                           eps: Decimal) -> Tuple[Decimal, np.ndarray]:
    return gradient_method(func, x0, eps, lambda f, x: f.argmin_calc(x))


def gradient_method_backtracking(func: ConvexFunc,
                                 x0: np.ndarray,
                                 eps: Decimal) -> Tuple[Decimal, np.ndarray]:
    def backtracking_line_search(f, x):
        alpha = Decimal(1)
        beta = Decimal(0.5)
        gamma = Decimal(0.25)
        while f.apply(x - alpha * f.gradient(x)) > f.apply(x) - gamma * alpha * np.linalg.norm(f.gradient(x)) ** 2:
            alpha *= beta
        return alpha

    return gradient_method(func, x0, eps, backtracking_line_search)


dectype = np.dtype(Decimal)
EPS: Decimal = Decimal(1e-5)
INF: Decimal = Decimal(2) # Used in approximate M and argmin counting -- setting it pretty low, because we know answer is somewhere in this bounds
X0: np.ndarray = np.array([Decimal(1)] * 2, dectype)  # X0 should be also set accurately for the above reason
RANDOM_POINTS = 10 ** 3

if __name__ == '__main__':
    first_function = ConvexFunc(
        lambda a: (a[0] ** 2 + 69 * a[1] ** 2) / 2,
        2,
        lambda a: np.array([a[0], 69 * a[1]], dectype),
        Decimal(69),
        np.array([Decimal(0), Decimal(0)], dectype),
        lambda a: (a[0] ** 2 + 4761 * a[1] ** 2) / (a[0] ** 2 + 328509 * a[1] ** 2)
    )

    print(gradient_method_const(first_function, X0, EPS))
    print(gradient_method_argmin(first_function, X0, EPS))
    print(gradient_method_backtracking(first_function, X0, EPS))

    second_function = ConvexFunc(
        lambda a: ((a[0] + 3 * a[1]).exp() + (a[0] - 3 * a[1]).exp() + (-a[0]).exp()),
        2,
        lambda a: np.array([(a[0] + 3 * a[1]).exp() + (a[0] - 3 * a[1]).exp() - (-a[0]).exp(),
                            3 * ((a[0] + 3 * a[1]).exp() - (a[0] - 3 * a[1]).exp())], dectype),
        None,  # m will be calculated in constructor
        np.array([Decimal(-log(2) / 2), Decimal(0)], dectype),
        None  # argmin_calc will be calculated in constructor
    )

    print(gradient_method_const(second_function, X0, EPS))
    print(gradient_method_argmin(second_function, X0, EPS))
    print(gradient_method_backtracking(second_function, X0, EPS))
