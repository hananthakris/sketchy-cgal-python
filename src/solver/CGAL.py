import math
import numpy as np
from src.solver.nystrom_sketch import NystromSketch
from src.utils.utils import updateThinSVDrank1
from src.utils.approx_eigenvectors import (
    ApproxMinEvecLanczosSE,
    ApproxMinEvecLanczos,
    ApproxMinEvecPower,
)
from scipy.sparse.linalg import eigs


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
        stoptol=0,
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
        arr = [2 ** i for i in range(0, math.floor(math.log(self.T, 2)))]
        arr.append(self.T)
        self.SAVEHIST = np.unique(arr)

        self.scale()
        self.choose_method()
        self.choose_linear_minimization_oracle()

        self.z = np.zeros(self.b.shape[1], 1)
        self.y0 = np.zeros(self.b.shape[1], 1)
        self.y = self.y0
        self.pobj = 0

    def choose_method(self):
        # Choose method - cgal, sketchy cgal, thin cgal
        if self.FLAG_METHOD == 0:
            self.X = np.zeros(self.n, self.n)
        elif self.FLAG_METHOD == 1:
            self.mySketch = NystromSketch(self.n, self.R, self.SKETCH_FIELD)
        elif self.FLAG_METHOD == 2:
            self.UTHIN = np.zeros(self.n, 1)
            self.DTHIN = 0

    def choose_linear_minimization_oracle(self):
        if self.FLAG_LANCZOS == 2:
            self.ApproxMinEvec = lambda x, t: ApproxMinEvecLanczosSE(
                x, self.n, math.ceil((t ** 0.25) * math.log(self.n, 2))
            )
        elif self.FLAG_LANCZOS == 1:
            self.ApproxMinEvec = lambda x, t: ApproxMinEvecLanczos(
                x, self.n, math.ceil((t ** 0.25) * math.log(self.n, 2))
            )
        else:
            self.ApproxMinEvec = lambda x, t: ApproxMinEvecPower(
                x, self.n, math.ceil(8 * (t ** 0.5)(math.log(self.n, 2)))
            )

    # Check convergence
    def scale(self):
        self.b_org = self.b
        self.RESCALE_FEAS = 1
        self.RESCALE_OBJ = 1

        if self.SCALE_A != 1:
            self.b = self.b * self.SCALE_A
            self.Primitive3 = lambda x: self.Primitive3(x).multiply(self.SCALE_A)
            self.Primitive2 = lambda y, x: self.Primitive2(y.multiply(self.SCALE_A), x)
            self.RESCALE_FEAS = self.RESCALE_FEAS / self.SCALE_A

        if self.SCALE_C != 1:
            self.Primitive1 = lambda x: self.Primitive1(x).multiply(self.SCALE_C)
            self.RESCALE_OBJ = self.RESCALE_OBJ / self.SCALE_C

        if self.FLAG_INCLUSION:
            self.PROJBOX = lambda y: min(max(y, self.b[:, 1]), self.b[:, 2])

    # Start iterations and solve

    def ApplyMinEvecApply(self, t):
        self.u, self.sig, cntInner = self.ApproxMinEvec(self.eigsArg, t)
        self.cntTotal += cntInner
        if self.sig > 0:
            self.a_t = min(self.a)
        else:
            self.a_t = max(self.a)
            self.u = np.sqrt(self.a_t) * self.u

    def getObjCond(self):
        AHk = self.Primitive3(self.u)
        ObjCond = (
            (
                self.pobj
                - self.Primitive1(self.u).getH() * self.u
                + self.y.T * (self.b - AHk)
                + self.beta * (self.z - self.b).getH() * (self.z - AHk)
                - 0.5 * self.beta * np.linalg.norm(self.z - self.b)
                ^ 2
            )
            * self.RESCALE_OBJ
            / max(abs(self.pobj * self.RESCALE_OBJ), 1)
        )

        return AHk, ObjCond

    def check_stopping_criteria(self, t):
        if self.stoptol:
            if self.stopcond:
                if self.FLAG_METHOD == 0:
                    self.U, self.Delt = eigs(self.X, self.R, which="LM")
                elif self.FLAG_METHOD == 1:
                    self.U, self.Delt = self.mySketch.Reconstruct()
                    if self.FLAG_TRACECORRECTION:
                        self.Delt = self.Delt + (
                            (self.TRACE - np.trace(self.Delt)) / self.R
                        ) * np.eye(np.shape(self.Delt)[0])
                elif self.FLAG_METHOD == 2:
                    self.U = self.UTHIN
                    self.Delt = self.DTHIN
                self.U = self.U * np.sqrt(self.Delt)

                if self.stopcond(self.U / np.sqrt(self.SCALE_X)) <= self.stoptol:
                    self.implement_stopping_criterion(msg="stopcond stopping")
                    return True

            else:
                FeasOrg = np.linalg.norm((self.z - self.b) * self.RESCALE_FEAS)
                FeasCond = FeasOrg / max(np.linalg.norm(self.b_org), 1)
                AHk, ObjCond = self.getObjCond()

                if FeasCond <= self.stoptol and ObjCond <= self.stoptol:
                    if self.FLAG_CAREFULLSTOPPING:
                        if not hasattr(self, "LastCheckpoint"):
                            self.LastCheckpoint = t
                        if t > self.LastCheckpoint:
                            self.LastCheckpoint = t
                            self.ApplyMinEvecApply(
                                max(t ** 2, math.ceil(1 / self.stoptol ** 2))
                            )
                        AHk, ObjCond = self.getObjCond()
                        if ObjCond <= self.stoptol:
                            self.implement_stopping_criterion(msg="accurate stopping")
                            return True

                    else:
                        self.implement_stopping_criterion(msg="stopping criteria met")
                        return True

        return False

    def implement_stopping_criterion(self, msg="accurate stopping"):
        if self.SAVEHIST:
            self.updateErrStructs()
            self.printError()
            self.clearErrStructs()
        self.status = msg

    def update(self):
        pass

    # Create and maintain structures where we store optimization information
    def create_err_structs(self):
        pass

    def printError(self):
        pass

    def updateErrStructs(self):
        pass

    def clearErrStructs(self):
        pass

    def store_errors(self):
        pass

    def solve(self):
        self.cntTotal = 0
        self.TRACE = 0
        for t in range(self.T):
            self.beta = self.beta0 * math.sqrt(self.T + 1)
            eta = 2 / (self.T + 1)
            if self.FLAG_INCLUSION:
                self.vt = self.y + self.beta * (
                    self.z - self.PROJBOX(self.z + (1 / self.beta) * self.y)
                )
            else:
                self.vt = self.y + self.beta * (self.z - self.b)

            self.eigsArg = lambda u: self.Primitive1(u) + self.Primitive2(self.vt, u)
            self.ApplyMinEvecApply(t)
            if self.check_stopping_criteria(t):
                break
            else:
                self.zEvec = self.Primitive3(self.u)
                self.z = (1 - eta) * self.z + eta * self.zEvec
                self.TRACE = (1 - eta) * self.TRACE + eta * self.a_t

                objEvec = self.u.getH() * self.Primitive1(self.u)
                self.pobj = (1 - eta) * self.pobj + eta * objEvec

                if self.FLAG_METHOD == 0:
                    self.X = (1 - eta) * self.X + eta * (self.u * self.u.getH())
                elif self.FLAG_METHOD == 1:
                    self.mySketch.RankOneUpdate(self.u, eta)
                elif self.FLAG_METHOD == 2:
                    self.UTHIN, self.DTHIN = updateThinSVDrank1(
                        self.UTHIN, (1 - eta) * self.DTHIN, self.u, eta
                    )

                beta_p = self.beta0 * math.sqrt(t + 2)

                if self.FLAG_INCLUSION:
                    dualUpdate = self.z - self.PROJBOX(self.z + (1 / beta_p) * (self.y))
                else:
                    dualUpdate = self.z - self.b

                sigma = min(
                    self.beta0,
                    4 * beta_p * eta
                    ^ 2 * max(self.a) ** 2 * self.NORM_A
                    ^ 2 / np.linalg.norm(dualUpdate) ** 2,
                )

                # Update the DUAL
                yt1 = self.y + np.multiply(sigma, dualUpdate)
                if np.linalg.norm(yt1 - self.y0) <= self.K:
                    self.y = yt1

                # # Measure the runtime
                # totTime = toc(timer);
                # totCpuTime = cputime - cputime0;
                # if totCpuTime > self.WALLTIME:
                #     self.implement_stopping_criterion(msg = 'wall time achieved')
                #     break

                # Update OUT
                if any(t == self.SAVEHIST):
                    self.updateErrStructs()
                    self.printError()

        if self.FLAG_METHOD == 0:
            self.U = np.divide(self.X, self.SCALE_X)
            self.Delt = []
        elif self.FLAG_METHOD == 1:
            if self.FLAG_TRACECORRECTION:
                self.Delt = self.Delt + (
                    (self.TRACE - np.trace(self.Delt)) / self.R
                ) * np.eye(np.shape(self.Delt)[0])
            self.Delt = np.divide(self.Delt, self.SCALE_X)
        elif self.FLAG_METHOD == 2:
            self.U = self.UTHIN
            self.Delt = np.divide(self.DTHIN, self.SCALE_X)
        self.y = np.divide(np.multiply(self.SCALE_A, self.y), self.SCALE_C)
        self.z = np.multiply(self.z, self.RESCALE_FEAS)
        self.pobj = np.multiply(self.pobj, self.RESCALE_OBJ)
