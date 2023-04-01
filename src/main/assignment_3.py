import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)


def func(t: float, w: float):
    return t - (w ** 2)


def euler_method(t0, t1, it, y0):
    h = (t1 - t0) / it

    for i in range(0, it):
        t = t0
        w = y0

        next_it = w + (h * func(t, w))

        t0 = t + h
        y0 = next_it

    # print(y0, end="\n\n")
    print("%.5f" % y0, end="\n\n")


def runge_method(t0, t1, it, y0):
    h = (t1 - t0) / it

    for i in range(0, it):
        w = y0

        k1 = h * func(t0, w)
        k2 = h * func(t0 + h / 2, w + k1 / 2)
        k3 = h * func(t0 + h / 2, w + k2 / 2)
        k4 = h * func(t0 + h, w + k3)

        y0 = w + (1 / 6) * (k1 + (k2 * 2) + (k3 * 2) + k4)
        t0 = t0 + h

    print("%.5f" % y0, end="\n\n")


def g_elimination():
    a = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]], dtype=float)
    b = np.array([6, 0, -3])
    n = len(b)
    Ab = np.concatenate((a, b.reshape(n, 1)), axis=1)
    n = len(Ab)
    for i in range(n):
        max_row = i
        for j in range(i + 1, n):
            if abs(Ab[j][i]) > abs(Ab[max_row][i]):
                max_row = j

        Ab[[i, max_row], :] = Ab[[max_row, i], :]

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, :] -= factor * Ab[i, :]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, n] / Ab[i, i]
        for j in range(i - 1, -1, -1):
            Ab[j, n] -= Ab[j, i] * x[i]

    print(x, end="\n\n")


def lu_decomp():
    mat = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
    n = len(mat)
    mat_cpy_u = np.matrix.copy(mat)
    mat_cpy_l = np.matrix.copy(mat)

    for i in range(n):
        for j in range(i + 1, n):
            cr_u = mat_cpy_u[j][i] / mat_cpy_u[i][i]
            for k in range(n):
                mat_cpy_u[j][k] = mat_cpy_u[j][k] - cr_u * mat_cpy_u[i][k]

    for i in range(n):
        mat_cpy_l[i][i] = 1
        for j in range(i + 1, n):
            mat_cpy_l[i][j] = 0

    det = float(1)
    for i in range(n):
        det *= mat_cpy_u[i][i]
    print("%.5f" % det, end="\n\n")
    print(mat_cpy_l, end="\n\n")
    print(mat_cpy_u, end="\n\n")


def diag_dom():
    mat = np.array([[9, 0, 5, 2, 1],
                    [3, 9, 1, 2, 1],
                    [0, 1, 7, 2, 3],
                    [4, 2, 3, 12, 2],
                    [3, 2, 4, 0, 8]])
    n = len(mat)
    b = 0

    for i in range(n):
        a = abs(mat[i][i])
        for j in range(n):
            if j != i:
                b += abs(mat[i][j])
        if a < b:
            return "False"
        b = 0
    return "True"


def positive_definite():
    mat = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    n = len(mat)
    mat_t = np.zeros([n, n])

    for i in range(n):
        mat_t[i][i] = mat[i][i]

    for i in range(n):
        for j in range(n):
            if j != i:
                mat_t[i][j] = mat[j][i]

    for i in range(n):
        for j in range(n):
            if mat[i][j] != mat_t[j][i]:
                return "False"
    return "True"


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    euler_method(0, 2, 10, 1)

    runge_method(0, 2, 10, 1)

    g_elimination()

    lu_decomp()

    print(diag_dom(), end="\n\n")

    print(positive_definite())
