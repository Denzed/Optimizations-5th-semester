import numpy as np
from numpy.polynomial import Polynomial as Poly
import matplotlib.pyplot as plt


def solve_linear_system(n: int, a: np.matrix, b: np.ndarray) -> np.ndarray:
    if a.shape != (n, n):
        raise NotImplemented
    elif np.allclose(a, np.tril(a)):
        x = np.zeros(n)
        for i in range(n):
            x[i] = (b[i] - np.dot(a[i], x)) / a[i][i]
        return x
    elif np.allclose(a, np.triu(a)):
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(a[i], x)) / a[i][i]
        return x
    else:
        raise NotImplemented


def least_squares(n: int, x: np.ndarray, y: np.ndarray) -> Poly:
    def generate_a():
        a = np.zeros((n + 1, n + 1))
        cur = np.ones(x.shape[0])
        for diagonal in range(2 * n + 1):
            value: float = np.sum(cur)
            for k in range(max(0, diagonal - n),
                           min(n, diagonal) + 1):
                a[k][diagonal - k] = value
            cur *= x.T
        return a

    def generate_b():
        b = np.ones(n + 1)
        cur = np.array(y)
        for row in range(n + 1):
            b[row] = np.sum(cur)
            cur *= x.T
        return b

    a = generate_a()
    l: np.matrix = np.linalg.cholesky(a)

    b = generate_b()
    p_intermediate = solve_linear_system(n + 1, l, b)
    p = solve_linear_system(n + 1, l.T, p_intermediate)

    return Poly(p)


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5.3, 6.3, 4.8, 3.8, 3.3])
    n = 1

    poly = least_squares(n, x, y)

    plt.scatter(x, y)
    bounds = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)), 100)
    plt.plot(bounds, poly(bounds))
    plt.show()
