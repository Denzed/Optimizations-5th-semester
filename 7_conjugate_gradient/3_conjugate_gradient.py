import numpy as np


def conjugate_gradient_method(a: np.matrix, b: np.matrix) -> np.matrix:
    def get_gradient(a: np.matrix, b: np.matrix, x: np.matrix) -> np.matrix:
        return a * x - b

    x = np.matrix(np.zeros(b.shape))
    v = get_gradient(a, b, x)
    d = v
    # for i in range(b.shape[0]):
    steps = 0
    while True:
        if not np.linalg.norm(d.T * a * d):
            break
        steps += 1
        new_x = x - d * (d.T * (a * x - b) / (d.T * a * d))
        v_prev, v = v, get_gradient(a, b, new_x)
        d = v - d * (v.T * v / (v_prev.T * v_prev))
        if not np.linalg.norm(x - new_x):
            x = new_x
            break
        x = new_x
    print(steps)
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