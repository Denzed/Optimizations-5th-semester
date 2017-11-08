from typing import Union
import numpy as np


def cholesky(a: np.matrix) -> Union[np.matrix, None]:
    if not np.array_equal(a.T, a):
        return None

    n = a.shape[0]

    def helper(n: int, a: np.matrix) -> Union[np.matrix, None]:
        corner = a.A[0][0]
        if corner < 0:
            return None
        sqrt_corner = corner ** 0.5
        if n == 1:
            return np.matrix([[sqrt_corner]])
        l = a[1:, 0]
        _a = a[1:, 1:]
        hl = helper(n - 1, _a - l * l.T / corner)
        return None if hl is None else \
            np.bmat([
                [np.matrix([[sqrt_corner]]),     np.matrix([[0] * (n - 1)])],
                [l / sqrt_corner,                hl]
            ])

    return helper(n, a)


if __name__ == '__main__':
    a = np.matrix([
        [2, 1],
        [1, 2]
    ])
    l = cholesky(a)
    print("not positive definite" if l is None else l)
