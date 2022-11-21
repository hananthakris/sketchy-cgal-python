import numpy as np
import sys
class NystromSketch:
    def __init__(self, n, R, field):
        if R > n:
            print('Sketch-size cannot be larger than the problem size.')
        if field == 'real':
            self.Omega = np.random.randn(n, R)
        elif field == 'complex':
            self.Omega = np.random.randn(n, R) + 1j *np.random.randn(n,R)
        else:
            print('Should be real or complex')
        self.S = np.zeros(n, R)

    def reconstruct(self):
        eps = sys.float_info.epsilon
        S = self.S
        n = np.size(self.S,1)
        sigma = np.sqrt(n)* eps *max(np.linalg.norm(S, axis=-1))
        S = S + sigma*self.Omega
        B = self.Omega.T * S
        B = 0.5*(B+B.T)
        if ~any(B[:]):
            Delta = 0
            U = np.zeros(n,1)
        else:
            C = np.linalg.cholesky(B)
            U, Sigma, temp = np.linalg.svd( S / C )
            Delta = max(np.power(Sigma, 2) - sigma*np.eye(np.size(Sigma)), 0)
        return U, Delta

    def rank_one_update(self, v, eta):
        self.S = (1-eta)*self.S + eta*(v*(v.T*self.Omega))


    def set(self, val):
            if np.size(val) == np.size(self.S) or not self.S:
                self.S = val
            else:
                print("Size of input does not match with sketch size")
