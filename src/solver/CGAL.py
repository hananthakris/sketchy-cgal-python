import math
import numpy as np

class CGAL:
    """
    CGAL This function implements CGAL - SketchyCGAL - ThinCGAL for solving semidefinite programs of the following form:

      minimize  <C,X>  subj.to   X is symmetric positive semidefinite
                                 al <= tr(X) <= au
                                 bl_i <= <A_i,X> <= bu_i
    """

    def __init__(
        self,
        n,
        Primitive1,
        Primitive2,
        Primitive3,
        a,
        b,
        R,
        T,
        beta0=1,
        K=float("inf"),
        walltime=float("inf"),
        stoptol=[],
        stopcond=[],
        FLAG_INCLUSION=False,
        FLAG_LANCZOS=2,
        FLAG_TRACECORRECTION=True,
        FLAG_CAREFULLSTOPPING=False,
        FLAG_METHOD=1,
        FLAG_EVALSURROGATEGAP=False,
        FLAG_MULTIRANK_P1=False,
        FLAG_MULTIRANK_P3=False,
        SKETCH_FIELD="real",
        ERR={},
        SCALE_A=1,
        SCALE_C=1,
        SCALE_X=1,
        NORM_A=1,
    ):
        # Initialize the dual
        # Choose the linear minimization oracle
        self.n = n
        self.Primitive1 = Primitive1
        self.Primitive2 = Primitive2
        self.Primitive3 = Primitive3
        self.a = a
        self.b = b
        self.R = R
        self.T = T
        self.beta0 = beta0
        self.K = K
        self.walltime = walltime
        self.stoptol = stoptol
        self.stopcond = stopcond
        self.FLAG_INCLUSION = FLAG_INCLUSION
        self.FLAG_LANCZOS = FLAG_LANCZOS
        self.FLAG_TRACECORRECTION = FLAG_TRACECORRECTION
        self.FLAG_CAREFULLSTOPPING = FLAG_CAREFULLSTOPPING
        self.FLAG_METHOD = FLAG_METHOD
        self.FLAG_EVALSURROGATEGAP = FLAG_EVALSURROGATEGAP
        self.FLAG_MULTIRANK_P1 = FLAG_MULTIRANK_P1
        self.FLAG_MULTIRANK_P3 = FLAG_MULTIRANK_P3
        self.SKETCH_FIELD = SKETCH_FIELD
        self.ERR = ERR
        self.SCALE_A = SCALE_A
        self.SCALE_C = SCALE_C
        self.SCALE_X = SCALE_X
        self.NORM_A = NORM_A

        arr = [2^i for i in range(math.floor(math.log(T,2)))]
        arr.append(T)
        self.SAVEHIST = np.unique(arr)

    #  Scale the problem

    # Check convergence
    def scale(self):
        b_org = self.b
        a_org = self.a
        Primitive1_original = lambda x: self.Primitive1(x)
        Primitive2_original = lambda y, x: self.Primitive2(y,x)
        Primitive3_original = lambda x: self.Primitive3(x)
        RESCALE_FEAS=1
        RESCALE_OBJ=1

        if self.SCALE_A != 1:
            self.b = self.b * self.SCALE_A
            Primitive3 = lambda x: Primitive3(x).multiply(self.SCALE_A)
            Primitive2 = lambda y,x: Primitive2(y.multiply(self.SCALE_A), x)
            RESCALE_FEAS = RESCALE_FEAS / self.SCALE_A

        if self.SCALE_C != 1:
            Primitive1 = lambda x: Primitive1(x).multiply(self.SCALE_C)
            RESCALE_OBJ = RESCALE_OBJ / self.SCALE_C

        if self.FLAG_INCLUSION:
            PROJBOX = lambda y: min(max(y,self.b[:,1]),self.b[:, 2])





    # Start iterations and solve

    def solve(self):
        pass

    def update(self):
        pass

    # Create and maintain structures where we store optimization information
    def create_err_structs(self):
        pass

    def updateErrStructs(self):
        pass

    def clearErrStructs(self):
        pass

    def store_errors(self):
        pass
