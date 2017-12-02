from hw8_optimized_gradient_method.implementations import *
from math import log
from decimal import Decimal
import numpy as np

if __name__ == '__main__':
    problem = Func(
        lambda a: (np.exp(a[0] + 3 * a[1]) + np.exp(a[0] - 3 * a[1]) + np.exp(-a[0])),
        2,
        lambda a: np.array([np.exp(a[0] + 3 * a[1]) + np.exp(a[0] - 3 * a[1]) - np.exp(-a[0]),
                            3 * (np.exp(a[0] + 3 * a[1]) - np.exp(a[0] - 3 * a[1]))], dectype),
        None,  # m will be calculated in constructor
        None,  # M will be calculated in constructor
        np.array([Decimal(-log(2) / 2), Decimal(0)], dectype)
    )

    NESTEROV_ITERATIONS = 100
    solve_with_stats(
        problem,
        NESTEROV_ITERATIONS,
        nesterov_method
    )
