import numpy as np
from solver.CGAL import CGAL
from pymatreader import read_mat
from scipy.sparse import spdiags
from scipy.sparse.linalg import norm
import argparse
parser = argparse.ArgumentParser(
                    prog = 'Sketchy-CGAL-Python',
                    description = 'Storage optimized SDP solver',
                    epilog = 'Text at the bottom of help')
parser.add_argument('--data_store', type=str, help='Folder where all your GSET files are stored')
parser.add_argument('--R', type=int, help='Rank/Sketch size parameter')
parser.add_argument('--max_it',  type=int, help='max iterations, enter a power of 10. Ex: 6 would mean 10^6')
args = parser.parse_args()

def cutvalue(C, u):
    """

    :param C:
    :param u:
    :return:
    """
    cutvalue = 0
    for t in range(np.shape(u)[1]):
        sign_evec = np.sign(u[:, t])
        rankvalue = -(sign_evec.conj().T.dot(C*sign_evec))
        cutvalue = max(cutvalue, rankvalue)
    return cutvalue


if __name__ == "__main__":
    R = args.R  # rank/sketch size parameter
    beta0 = 1  # we didn't tune - choose 1 - you can tune this!
    K = float("inf")
    maxit = 10** args.max_it  # limit on number of iterations
    mat_fname = f"{args.data_store}/G/G1.mat"
    data = read_mat(mat_fname)
    A = data["Problem"]["A"]
    n = A.shape[0]
    C = spdiags(A * np.ones((n)), 0, n, n) - A
    C = 0.5 * (C + C.getH())  # symmetrize if not symmetric
    C = (-0.25) * C

    del data
    a = n
    b = np.ones((n))
    SCALE_X = 1 / n
    SCALE_C = 1 / norm(C, ord="fro")

    errfunc = {}
    if SCALE_X != 1:
        Primitive1 = lambda x: C.dot(x) * SCALE_X
    else:
        Primitive1 = lambda x: C.dot(x)

    Primitive2 = lambda y, x: y.dot(x)
    Primitive3 = lambda x: np.sum(np.power(x, 2))
    errfunc['cutvalue'] = lambda u: cutvalue(C, u)

    obj = CGAL(
        n=n,
        Primitive1=Primitive1,
        Primitive2=Primitive2,
        Primitive3=Primitive3,
        errfunc=errfunc,
        a=a,
        b=b,
        R=R,
        T=maxit,
        beta0=beta0,
        K=K,
        FLAG_MULTIRANK_P1=True,
        FLAG_MULTIRANK_P3=True,
        SCALE_X=SCALE_X,
        SCALE_C=SCALE_C,
        stoptol=0.1,
    )
    obj.solve() # Call to the solve() function inside the CGAL file.
