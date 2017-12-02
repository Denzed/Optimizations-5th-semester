from hw8_optimized_gradient_method.implementations import *
from decimal import Decimal
import numpy as np

if __name__ == '__main__':
    function = Func(
        lambda a: (a[0] ** 2 + 69 * a[1] ** 2) / 2,
        2,
        lambda a: np.array([a[0], 69 * a[1]], dectype),
        Decimal(1),
        Decimal(69),
        np.array([Decimal(0), Decimal(0)], dectype)
    )

    def print_stats(method: Callable[[Func, int], np.ndarray], k: int, name: str):
        x = method(function, k)
        print(f"For {name} method:")
        print(f"Got {x} after {k} iterations")
        dx = x - function.x_optimal
        print(f"It is {np.linalg.norm(dx)} away from optimum coordinate of {function.x_optimal}")
        dy = abs(function(x) - function(function.x_optimal))
        print(f"and {dy} away from optimum of {function(function.x_optimal)}")
        print("_" * 179)

    HEAVY_BALL_ITERATIONS = 50
    print_stats(
        heavy_ball_method,
        HEAVY_BALL_ITERATIONS,
        "heavy ball"
    )

    NESTEROV_ITERATIONS = 100
    print_stats(
        nesterov_method,
        NESTEROV_ITERATIONS,
        "nesterov"
    )