import numpy as np
import sys
import math
from scipy.linalg import cholesky

class NystromSketch:
    def __init__(self, n, R, field):
        if R > n:
            print("Sketch-size cannot be larger than the problem size.")
        if field == "real":
            self.Omega = np.random.normal(0,1, (n, R))
        elif field == "complex":
            self.Omega = np.random.randn(n, R) + 1j * np.random.normal(0,1, (n, R))
        else:
            print("Should be real or complex")
        self.S = np.zeros((n, R))

    def mrdivide(self, A, B):
        C = A.dot(B.T.dot(np.linalg.inv(B.dot(B.T))))
        return C

    def reconstruct(self):
        eps = sys.float_info.epsilon
        S = self.S
        n = len(S)
        sigma = math.sqrt(n) * eps * max(np.linalg.norm(S, axis=-1))
        S = S + sigma * self.Omega
        B = self.Omega.conj().T.dot(S)
        B = 0.5 * (B + B.conj().T)
        if not np.any(B):
            Delta = 0
            U = np.zeros(n, 1)
        else:
            C = cholesky(B)
            U, Sigma, temp = np.linalg.svd(self.mrdivide(S, C), full_matrices=False)
            Delta = np.power(Sigma, 2) - sigma * np.eye(len(Sigma), 10)
            Delta = np.where(Delta < 0, 0, Delta)
        return U, Delta

    def rank_one_update(self, v, eta):
        self.S = (1 - eta) * self.S + eta * (
            v.reshape(len(v), 1).dot((v.conj().T.reshape(1, len(v)).dot(self.Omega)))
        )

    def set(self, val):
        if np.size(val) == np.size(self.S) or not self.S:
            self.S = val
        else:
            print("Size of input does not match with sketch size")
