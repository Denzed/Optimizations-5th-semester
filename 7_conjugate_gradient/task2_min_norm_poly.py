import numpy as np
import numpy.matlib
from task3_conjugate_gradient import conjugate_gradient_method, EPS

# Calculations:
# 1 + 2 * sum + sum ^ 2 =
# 1 + sum(i)[2 * ai x ^ i + ai ^ 2 x ^ 2i] + sum(i != j) [ai aj x ^ (i + j)]
# f(x) := x
#   + sum(i)[2 * ai x ^ (i + 1) / i + ai ^ 2 x ^ (2i + 1) / 2i]
#   + 2 * sum(i != j) [ai aj x ^ (i + j + 1) / (i + j)]
# norm = f(1) - f(-1)
#   = 2 + a.T [2 / i (1 + (-1) ^ i) for i in range(1, n + 1)]
#       + a.T diag(1 / i for i in range(1, n + 1)) a
#       + 2 a.T [ (1 + (-1) ^ (i + j)) / (i + j) for i in range(1, n + 1) for j in range(1, n + 1)] a
#       - 2 a.T diag(1 / i for i in range(1, n + 1)) a

# Given polynomial norm can be converted to an expression of form 2 * (1 + a ^ T (-b) + a ^ T A a)
# (where A is a symmetric positive-[semi]definite matrix)
# which has an extremum point at A a = b which (by occasion!) is a solution. So we have to generate
# A and b and solve the linear system.
def generate_a(n: int) -> np.matrix:
    return np.matrix([
        [(1 + (-1) ** (i + j)) / (i + j) for j in range(1, n + 1)] for i in range(1, n + 1)
    ]) - np.matlib.diag([1 / 2 / i for i in range(1, n + 1)])

def generate_b(n: int) -> np.matrix:
    return np.matrix([[(1 + (-1) ** i) / i] for i in range(1, n + 1)])

def poly_to_string(coefficients: [float]) -> str:
    def power_to_string(power: int) -> str:
        return ''.join(map(lambda c: "⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c) - ord('0')], str(power)))

    # joiner[first, negative] = str
    joiner = {
        (True, True): '-',
        (True, False): '',
        (False, True): ' - ',
        (False, False): ' + '
    }

    result = []
    for power, coefficient in reversed(list(enumerate(coefficients))):
        j = joiner[not result, coefficient < 0]
        coefficient = abs(coefficient)
        if abs(coefficient) < EPS:
            continue
        if abs(coefficient - 1) < EPS and power != 0:
            coefficient = ''

        if power == 0:
            result.append(f"{j}{coefficient}")
        elif power == 1:
            result.append(f"{j}{coefficient}x")
        else:
            result.append(f"{j}{coefficient}x{power_to_string(power)}")

    return ''.join(result) or '0'

if __name__ == '__main__':
    n = 2
    if n > 0:
        a = generate_a(n)
        b = generate_b(n)
        poly = [1] + conjugate_gradient_method(a, b).flatten().tolist()[0]
    else:
        poly = [1]

    print(poly_to_string(poly))