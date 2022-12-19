import sys
import numpy as np
import math
from numba import jit

def mrdivide(A, B):
        """

        :param A:
        :param B:
        :return:
        """
        C = A.dot(B.T.dot(np.linalg.inv(B.dot(B.T))))
        return C

@jit(nopython=True)
def reconstruct(S, Omega, eps):
        """

        :param S:
        :param Omega:
        :param eps:
        :return:
        """
        n = len(S)
        norms = []
        for col in S.reshape(S.shape[1], S.shape[0]):
            norms.append(np.linalg.norm(col))
        sigma = math.sqrt(n) * eps * max(norms)
        S = S + sigma * Omega
        B = Omega.conj().T.dot(S)
        B = 0.5 * (B + B.conj().T)

        # Where the code is failing! - Cholesky in numpy not giving the upper triangle.
        C = np.linalg.cholesky(B)
        U, Sigma, temp = np.linalg.svd(S.dot(C.T.dot(np.linalg.inv(C.dot(C.T)))), full_matrices=False)
        Delta = np.power(Sigma, 2) - sigma * np.eye(len(Sigma), 10)
        Delta = np.where(Delta < 0, 0, Delta)
        return U, Delta
