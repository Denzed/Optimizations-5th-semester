import numpy as np


def conjugate_gradient_method(a: np.matrix, b: np.matrix) -> np.matrix:
    def get_gradient(a: np.matrix, b: np.matrix, x: np.matrix) -> np.matrix:
        return a * x - b

    x = np.matrix(np.zeros(b.shape))
    v = get_gradient(a, b, x)
    d = v
    for i in range(b.shape[0]):
        if not np.linalg.norm(d.T * a * d):
            break
        x = x - d * (d.T * (a * x - b) / (d.T * a * d))
        v = get_gradient(a, b, x)
        d = v - d * (d.T * a * v / (d.T * a * d))
    return x


if __name__ == '__main__':
    a = np.matrix([
        [3, 1],
        [2, 1]
    ])
    b = np.matrix([
        [16],
        [1]
    ])
    print(conjugate_gradient_method(a, b))