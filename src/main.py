import math
import time
import numpy as np
from solver.CGAL import CGAL
import scipy.io as sio
from pymatreader import read_mat
from scipy.sparse import spdiags
from scipy.sparse.linalg import norm
from os.path import dirname, join as pjoin


def Primitive1MultRank(U):
    pass


def Primitive2MultRank(U):
    pass


def Primitive3MultRank(U):
    pass


def cutvalue(c, U):
    """function cutvalue = round(C,u)
        cutvalue = 0;
        for t = 1:size(u,2)
        sign_evec = sign(u(:,t));
        rankvalue = -(sign_evec'*(C*sign_evec));
        cutvalue = max(cutvalue, rankvalue);
    end
    end"""
    pass


if __name__ == "__main__":
    R = 10  # rank/sketch size parameter
    beta0 = 1  # we didn't tune - choose 1 - you can tune this!
    K = float("inf")
    maxit = 1e6  # limit on number of iterations
    # mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')
    mat_fname = "/Users/haritha/Desktop/SketchyCGAL/FilesMaxcut/data/G/G1.mat"
    data = read_mat(mat_fname)
    A = data["Problem"]["A"]
    n = A.shape[0]
    C = spdiags(A * np.ones((n)), 0, n, n) - A
    C = 0.5 * (C + C.getH())  # symmetrize if not symmetric
    C = (-0.25) * C

    del data

    Primitive1 = lambda x: C * x
    Primitive2 = lambda y, x: y * x
    Primitive3 = lambda x: np.sum(np.power(x, 2), 2)
    a = n
    b = np.ones((n))
    SCALE_X = 1 / n
    SCALE_C = 1 / norm(C, ord="fro")

    varargins = dict()

    obj = CGAL(
        n=n,
        Primitive1=Primitive1,
        Primitive2=Primitive2,
        Primitive3=Primitive3,
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
    """ 
                     
cputimeEnd = cputime;
totalTime = toc(timer);

out.totalTime = totalTime;
out.totalCpuTime = cputimeEnd - cputimeBegin;"""
    # Evaluate Errors
    # Save results
