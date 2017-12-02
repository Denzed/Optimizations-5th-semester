from gradient_method_implementations import *
from math import log
from decimal import Decimal
import numpy as np

X0: np.ndarray = np.array([Decimal(1)] * 2, dectype)  # X0 should be also set accurately for the above reason

if __name__ == '__main__':
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
