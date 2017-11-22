from typing import Union
import numpy.matlib as np


# rank(A) = rank(A * A.T)
# so to be of full rank matrix must be positive definite
# and this property we can effectively find with Cholesky
# decomposition
def has_full_rank(a: np.matrix) -> bool:
    try:
        np.linalg.cholesky(a.T * a)
    except np.linalg.LinAlgError:
        return False
    return True


def is_atomic(a: np.matrix) -> bool:
    diagonal = a.diagonal()
    return np.allclose(diagonal, np.ones(diagonal.shape))


def invert_square_matrix(a: np.matrix) -> np.matrix:
    if np.allclose(a, np.diag(np.diag(a))):
        return np.mat(np.diag(np.reciprocal(np.diag(a))))
    elif np.allclose(a, np.tril(a)): # Following the idea from https://math.stackexchange.com/a/1008675
        if is_atomic(a):
            return np.mat(2 * np.identity(a.shape[0]) - a)
        lam = np.mat(np.diag(np.diag(a)))
        inv_lam = invert_square_matrix(lam)
        tmp = np.matrix(np.identity(a.shape[0]) +
                        inv_lam * (a - lam))
        return invert_square_matrix(tmp) * inv_lam
    elif np.allclose(a, np.triu(a)):
        return invert_square_matrix(a.T).T
    else:
        l = np.mat(np.linalg.cholesky(a))
        inv_l = invert_square_matrix(l)
        return inv_l.T * inv_l


def find_optimum(a: np.matrix, b: np.ndarray) -> np.ndarray:
    if not has_full_rank(a):
        raise np.linalg.LinAlgError("Matrix A must have full rank")
    return invert_square_matrix(a.T * a) * a.T * b


if __name__ == '__main__':
    print(find_optimum(np.mat([
        [2, 0],
        [-1, 1],
        [0, 3]
    ]), np.array([[1], [0], [-1]])))