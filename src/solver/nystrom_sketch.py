import numpy as np
import sys
import math
from scipy.linalg import cholesky
from numba import jit
from utils.utils import *
class NystromSketch:
    """

    """
    def __init__(self, n, R, field):
        """

        :param n:
        :param R:
        :param field:
        """
        if R > n:
            print("Sketch-size cannot be larger than the problem size.")
        if field == "real":
            self.Omega = np.random.normal(0,1, (n, R)) # Draw and fix random test matrix
        elif field == "complex":
            self.Omega = np.random.randn(n, R) + 1j * np.random.normal(0,1, (n, R))
        else:
            print("Should be real or complex")
        self.S = np.zeros((n, R)) # Form sketch of zero matrix

    def reconstruct(self):
        """
        Reconsturction of the Nystrom Sketch
        :return: U, Delta
        """
        eps = sys.float_info.epsilon
        U, Delta = reconstruct(self.S, self.Omega, eps)
        return U, Delta

    def rank_one_update(self, v, eta):
        """
        Update sketch of matrix
        :param v: v ∈ F
        :param eta: η ∈ [0, 1]
        :return: None
        """
        self.S = (1 - eta) * self.S + eta * (
            v.reshape(len(v), 1).dot((v.conj().T.reshape(1, len(v)).dot(self.Omega)))
        )

    def set(self, val):
        """

        :param val:
        :return:
        """
        if np.size(val) == np.size(self.S) or not self.S:
            self.S = val
        else:
            print("Size of input does not match with sketch size")
