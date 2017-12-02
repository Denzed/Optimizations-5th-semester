from hw8_optimized_gradient_method.implementations import *
from decimal import Decimal
import numpy as np

if __name__ == '__main__':
    problem = Func(
        lambda a: (a[0] ** 2 + 69 * a[1] ** 2) / 2,
        2,
        lambda a: np.array([a[0], 69 * a[1]], dectype),
        Decimal(1),
        Decimal(69),
        np.array([Decimal(0), Decimal(0)], dectype)
    )

    HEAVY_BALL_ITERATIONS = 50
    solve_with_stats(
        problem,
        HEAVY_BALL_ITERATIONS,
        heavy_ball_method
    )

    NESTEROV_ITERATIONS = 100
    solve_with_stats(
        problem,
        NESTEROV_ITERATIONS,
        nesterov_method
    )

    CHEBYSHEV_ITERATIONS = 50
    solve_with_stats(
        problem,
        CHEBYSHEV_ITERATIONS,
        chebyshev_method
    )
