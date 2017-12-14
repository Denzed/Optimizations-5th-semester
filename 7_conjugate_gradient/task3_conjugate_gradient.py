import numpy as np


# A
def conjugate_gradient_method(a: np.matrix, b: np.matrix) -> np.matrix:
    def get_gradient(a: np.matrix, b: np.matrix, x: np.matrix) -> np.matrix:
        return a * x - b

    x = np.matrix(np.zeros(b.shape))
    v = get_gradient(a, b, x)
    d = v
    i = 0
    while i < b.shape[0] and np.linalg.norm(d.T * a * d):
        x = x - d * (d.T * (a * x - b) / (d.T * a * d))
        v = get_gradient(a, b, x)
        d = v - d * (d.T * a * v / (d.T * a * d))
        i += 1
    return x


EPS = 1e-5


# B
def biconjugate_gradient_method(a: np.matrix, b: np.matrix) -> np.matrix:
    x = np.matrix(np.zeros((a.shape[1], 1)))
    s = z = p = r = b - a * x
    while True:
        alpha = (p.T * r) / (s.T * a * z)
        if (np.linalg.norm(z * alpha)) < EPS:
            break
        x = x + z * alpha
        new_r = r - a * z * alpha
        new_p = p - a.T * s * alpha
        beta = (new_p.T * new_r) / (p.T * r)
        r = new_r
        p = new_p
        z = r + z * beta
        s = p + s * beta
    return x


if __name__ == '__main__':
    a = np.matrix([
        [5, 2],
        [2, 1]
    ])
    b = np.matrix([
        [16],
        [1]
    ])
    print(conjugate_gradient_method(a, b))
    print(biconjugate_gradient_method(a, b))
