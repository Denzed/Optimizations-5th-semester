from gradient_method_implementations import *
from math import log
from decimal import Decimal
import numpy as np

X0: np.ndarray = np.array([Decimal(1)] * 2, dectype)  # X0 should be also set accurately for the above reason

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
