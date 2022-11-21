import numpy as np
import sys


def ApproxMinEvecPower(M, n, q):
    if M.isnumeric():
        M = lambda x: M * x

    Mnorm, cnt = powermethod(M, n, 0.1)
    shift = 2 * Mnorm
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    Mv = M(v)
    cnt = cnt + 1

    for iter in range(0, q):
        v = shift * v - Mv
        v = v / np.linalg.norm(v)
        Mv = M(v)
        cnt = cnt + 1
    xi = v.getH() * Mv
    return v, xi, cnt


def powermethod(S, n, tol):
    cnt = 0
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    e = 1
    e0 = 0
    while abs(e - e0) > tol * e:
        e0 = e
        Sx = S(x)
        if np.count_nonzero(Sx) == 0:
            Sx = np.random.rand(np.size(Sx))
        x = S(Sx)
        normx = np.linalg.norm(x)
        e = normx / np.linalg.norm(Sx)
        x = x / normx
        cnt = cnt + 1
        if cnt > 100:
            print("Power method did not converge")

    return e, cnt


def cgal_eig(X):
    try:
        V, D = np.linalg.eig(X)
    except:
        V, D, W = np.linalg.svd(X)
        D = np.diagonal(D).T * np.sign(np.real(np.dot(V, W, 1)))
        D, ind = np.sort(D)
        V = V[:, ind]
    return V, D


def ApproxMinEvecLanczosSE(M, n, q):
    q = min(q, n - 1)
    if M.isnumeric():
        M = lambda x: M * x

    Q = np.zeros((n, q + 1))

    aleph = np.zeros(q)
    beth = np.zeros(q)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    vi = v
    vim1 = 1
    i = 0
    for i in range(q):
        vip1 = M(vi)
        aleph[i] = np.real(vi.T * vip1)
        if i == 0:
            vip1 -= aleph[i] * vi
        else:
            vip1 -= aleph[i] * vi - beth[i - 1] * vim1
        beth[i] = np.linalg.norm(vip1)
        if abs(beth[i]) < np.sqrt(n) * sys.float_info.epsilon:
            break
        vip1 /= beth[i]
        vim1 = vi
        vi = vip1

    B = (
        np.diagonal(aleph[:i], 0)
        + np.diagonal(beth[1 : [i - 1]], +1)
        + np.diagonal(beth[1 : [i - 1]], -1)
    )
    U, D = cgal_eig(0.5 * (B + np.transpose(B)))
    [xi, ind] = min(D)
    Uind1 = U[:, ind]
    aleph = np.zeros(q)
    beth = np.zeros(q)
    vi = v
    v = np.zeros(n)
    for i in range(len(Uind1)):
        v = v + vi * Uind1(i)

        vip1 = M(vi)
        aleph[i] = np.real(vi.T * vip1)
        if i == 1:
            vip1 = vip1 - aleph[i] * vi
        else:
            vip1 = vip1 - aleph[i] * vi - beth[i - 1] * vim1
        beth[i] = np.linalg.norm(vip1)
        vip1 = vip1 / beth[i]
        vim1 = vi
        vi = vip1
    i = 2 * i

    nv = np.linalg.norm(v)
    xi = xi * nv
    v = v / nv
    return v, xi, i


def ApproxMinEvecLanczos(M, n, q):
    q = min(q, n - 1)
    if M.isnumeric():
        M = lambda x: M * x

    Q = np.zeros((n, q + 1))

    aleph = np.zeros(q)
    beth = np.zeros(q)

    Q[:, 0] = np.random.randn(n)
    Q[:, 0] = Q[:, 1] / np.linalg.norm(Q)

    i = 0
    for i in range(q):
        Q[:, i + 1] = M(Q[:, i])
        aleph[i] = np.real(Q[:, i].getH() * Q[: i + 1])
        if i == 0:
            Q[:, i + 1] -= aleph[i] * Q[:, i]
        else:
            Q[:, i + 1] = Q[:, i + 1] - aleph[i] * Q[:, i] - beth[i - 1] * Q[:, i - 1]

        beth[i] = np.linalg.norm(Q[:, i + 1])

        if abs(beth[i]) < np.sqrt(n) * sys.float_info.epsilon:
            break

        Q[:, i + 1] /= beth[i]

    B = (
        np.diagonal(aleph[0:i], 0)
        + np.diagonal(beth[1 : [i - 1]], +1)
        + np.diagonal(beth[1 : [i - 1]], -1)
    )

    U, D = cgal_eig(0.5 * (B + np.transpose(B)))
    [xi, ind] = min(D)
    v = Q[:, :i] * U[:, ind]
    nv = np.linalg.norm(v)
    xi = xi * nv
    v = v / nv
    return v, xi, i
