import numpy as np
import sys


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm else v


# Gram-Schmidt process
def orthogonal_basis(v: [np.ndarray]) -> [np.ndarray]:
    def proj(u: np.ndarray):
        if np.linalg.norm(u):
            tmp = u / u.dot(u)

            def proj_u(v: np.ndarray):
                return u.dot(v) * tmp
        else:
            def proj_u(v: np.ndarray):
                return np.zeros(v.shape)

        return proj_u

    projections = []
    u = []
    for i in range(len(v)):
        next_u = v[i] - sum(map(lambda pr: pr(v[i]), projections), np.zeros(v[i].shape))
        u.append(next_u)
        projections.append(proj(next_u))
    return [normalize(x) for x in u if np.linalg.norm(x)]


def a_orthogonal_basis(v: [np.ndarray], a: np.matrix) -> [np.ndarray]:
    def proj(u: np.ndarray):
        u = np.matrix(u)
        left = u * a
        if np.linalg.norm(left * u.T):
            sys.stdout.flush()
            right = u / (left * u.T)

            def proj_u(v: np.ndarray):
                return np.array(left * v[None].T * right)
        else:
            def proj_u(v: np.ndarray):
                return np.zeros(v.shape)

        return proj_u

    projections = []
    u = []
    for i in range(len(v)):
        next_u = v[i] - sum(map(lambda pr: pr(v[i]), projections), np.zeros(len(v)))
        u.append(next_u)
        projections.append(proj(next_u))
    return [normalize(x) for x in u if np.linalg.norm(x)]


if __name__ == '__main__':
    v = [np.array([3, 1]), np.array([2, 2])]
    print(orthogonal_basis(v))  # prints [[3, 1], [-1, 3]] / sqrt(10)
    a = np.array([
        [1, 0],
        [0, 1]
    ])
    print(a_orthogonal_basis(v, a)) # should be the same as previous
