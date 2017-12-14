from collections import defaultdict


class Vector:
    def __init__(self, n, data=None):
        self.n = n
        if data is None:
            self.data = [0] * n
        elif isinstance(data, list):
            self.data = data
        else:
            raise TypeError

    def __add__(self, other):
        if isinstance(other, Vector):
            if other.n == self.n:
                return Vector(self.n, [self.data[i] + other.data[i] for i in range(self.n)])
            raise ValueError
        raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Vector(self.n, [self.data[i] * other for i in range(self.n)])
        elif isinstance(other, Vector):
            if other.n == self.n:
                return sum(self.data[i] * other[i] for i in range(self.n))
            raise ValueError
        elif isinstance(other, Matrix):
            return NotImplemented
        raise TypeError

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if other:
                return Vector(self.n, [self.data[i] / other for i in range(self.n)])
            raise ZeroDivisionError
        raise TypeError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__neg__() + other

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        raise TypeError

    @property
    def norm(self):
        return sum(self.data[i] ** 2 for i in range(self.n)) ** 0.5

    @property
    def T(self):
        return Matrix(1, self.n, [self.data])

    def __str__(self):
        return "Vector" + str(self.data)


class Matrix:
    def __init__(self, n, m, data=None):
        self.n = n
        self.m = m
        if data is None:
            self.data = defaultdict(float)
        elif isinstance(data, list):
            self.data = defaultdict(float, {(i, j): data[i][j] for i in range(n) for j in range(m) if data[i][j]})
        elif isinstance(data, dict):
            self.data = defaultdict(float, {key: data[key] for key in data if data[key]})
        elif isinstance(data, defaultdict):
            self.data = data
        else:
            raise TypeError

    def __add__(self, other):
        if isinstance(other, Matrix):
            if other.n == self.n and other.m == self.m:
                return Matrix(
                    self.n,
                    self.m,
                    {
                        (i, j): self.data[(i, j)] + other.data[(i, j)]
                        for (i, j) in (self.data.keys() | other.data.keys())
                    }
                )
            raise ValueError
        raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(
                self.n,
                self.m,
                {key: value * other for (key, value) in self.data.items()}
            )
        elif isinstance(other, Vector):
            if other.n == self.m:
                data = [0] * self.n
                for (i, j) in self.data.keys():
                    data[i] += self.data[(i, j)] * other[j]
                result = Vector(self.n, data)
                return result if result.n > 1 else result[0]
            raise ValueError
        elif isinstance(other, Matrix):
            if self.m == other.n:
                result = Matrix(
                    self.n,
                    other.m,
                    [
                        [
                            sum(self.data[(i, k)] * other[(k, j)] for k in range(self.m))
                            for j in range(other.m)
                        ]
                        for i in range(self.n)
                    ]
                )
                return result if result.n > 1 or result.m > 1 else result[(0, 0)]
        raise TypeError

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.__mul__(other)
        elif isinstance(other, Vector):
            if self.n == 1:
                return Matrix(other.n, 1, list(map(lambda k: [k], other.data))) * self
            raise ValueError
        raise TypeError

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__neg__() + other

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.data[item]
        raise TypeError

    @property
    def T(self):
        return Matrix(self.m, self.n, {(j, i): self.data[(i, j)] for (i, j) in self.data})

    @property
    def norm(self):
        return sum(self.data[(i, j)] ** 2 for i in range(self.n) for j in range(self.m)) ** 0.5

    def __str__(self):
        return "Matrix[\n\t" + "\n\t".join(str([self.data[(i, j)] for j in range(self.m)]) for i in range(self.n)) + "\n]"


# A
def conjugate_gradient_method(a: Matrix, b: Vector) -> Vector:
    def get_gradient(a: Matrix, b: Vector, x: Vector) -> Vector:
        return a * x - b

    x = Vector(b.n)
    v = get_gradient(a, b, x)
    d = v
    i = 0
    while i < b.n and norm(d.T * a * d):
        x = x - d * d.T * (a * x - b) / (d.T * a * d)
        v = get_gradient(a, b, x)
        d = v - d * (d.T * a * v) / (d.T * a * d)
        i += 1
    return x


EPS = 1e-5


# B
def biconjugate_gradient_method(a: Matrix, b: Vector) -> Matrix:
    x = Vector(a.m)
    p = r = p_ = r_ = b - a * x
    while True:
        print(a, p, p_, sep="\n")
        alpha = (r * r_) / ((p * a) * p_)
        if norm(p * alpha) < EPS:
            break
        x = x + p * alpha
        new_r = r - a * p * alpha
        new_r_ = r_ - a.T * p_ * alpha
        beta = (new_r.T * new_r_) / (r * r_)
        r, r_ = new_r, new_r_
        p = r + p * beta
        p_ = r_ + p_ * beta
    return x


def norm(obj):
    if isinstance(obj, float) or isinstance(obj, int):
        return obj
    elif isinstance(obj, Vector) or isinstance(obj, Matrix):
        return obj.norm
    raise TypeError


if __name__ == '__main__':
    a = Matrix(2, 2, [
        [5, 2],
        [2, 1]
    ])
    b = Vector(2, [16, 1])
    print(conjugate_gradient_method(a, b))
    print(biconjugate_gradient_method(a, b))
