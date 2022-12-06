import math
import numpy as np
from solver.nystrom_sketch import NystromSketch
from utils.approx_eigenvectors import (
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
        errfunc={},
        SCALE_A=1,
        SCALE_C=1,
        SCALE_X=1,
        NORM_A=1,
    ):
        self.final_print = []
        self.out = {}
        self.out["info"] = {}
        self.out["params"] = {}
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
        self.errfunc = errfunc
        self.SCALE_A = SCALE_A
        self.SCALE_C = SCALE_C
        self.SCALE_X = SCALE_X
        self.NORM_A = NORM_A
        arr = [2 ** i for i in range(0, math.floor(math.log(self.T, 2)))]
        arr.append(self.T)
        self.SAVEHIST = np.unique(arr)
        self.scale()
        self.choose_method()
        self.errNames, self.errNamesPrint, self.ptr = self.create_err_structs()
        self.choose_linear_minimization_oracle()
        self.z = np.zeros(self.b.shape[0])
        self.y0 = np.zeros(self.b.shape[0])
        self.y = self.y0
        self.pobj = 0

    def choose_method(self):
        """
        Chooses which method to use - cgal, sketchy cgal
        :return:  <Modiifies self>
        """

        if self.FLAG_METHOD == 0:
            self.X = np.zeros(self.n, self.n)
        elif self.FLAG_METHOD == 1:
            self.mySketch = NystromSketch(self.n, self.R, self.SKETCH_FIELD)

    def choose_linear_minimization_oracle(self):
        """
        Chooses which approximation method to use
        :return: <Modiifies self>
        """
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
        """
        Scales the matrices to desired levels
        :return:  <Modiifies self>
        """
        self.b_org = self.b
        self.RESCALE_FEAS = 1
        self.RESCALE_OBJ = 1

        if self.SCALE_A != 1:
            self.b = self.b * self.SCALE_A
            self.Primitive3_ = lambda x: self.Primitive3(x) * self.SCALE_A
            self.Primitive2_ = lambda y, x: self.Primitive2(y * self.SCALE_A, x)
            self.RESCALE_FEAS = self.RESCALE_FEAS / self.SCALE_A

        if self.SCALE_C != 1:
            self.Primitive1_ = lambda x: self.Primitive1(x) * self.SCALE_C
            self.RESCALE_OBJ = self.RESCALE_OBJ / self.SCALE_C

        if self.FLAG_INCLUSION:
            self.PROJBOX = lambda y: min(max(y, self.b[:, 1]), self.b[:, 2])

    # Start iterations and solve

    def ApplyMinEvecApply(self, t):
        """
        Applies the minimization function
        :param t: time step
        :return:  <Modiifies self>
        """
        self.u, self.sig, cntInner = self.ApproxMinEvec(self.eigsArg, t)
        self.cntTotal += cntInner
        if self.sig > 0:
            self.a_t = min([self.a])
        else:
            self.a_t = max([self.a])
            self.u = np.sqrt(self.a_t) * self.u

    def getObjCond(self):
        """
        Written to reduce code redundancy
        :return:  <Modiifies self>
        """
        AHk = self.Primitive3(self.u)
        var = (
            self.pobj
            - self.Primitive1_(self.u).conj().T.dot(self.u)
            + self.y.T.dot(self.b - AHk)
            + self.beta * (self.z - self.b).conj().T.dot(self.z - AHk)
            - 0.5 * self.beta * np.linalg.norm(self.z - self.b) ** 2
        )

        ObjCond = var * self.RESCALE_OBJ / max(abs(self.pobj * self.RESCALE_OBJ), 1)

        return AHk, ObjCond

    def check_stopping_criteria(self, t):
        """
        Checks if the given loop meets stopping criteria, if yes breaks
        :param t: time step
        :return:  <Modiifies self>
        """
        if self.stoptol:
            if self.stopcond:
                if self.FLAG_METHOD == 0:
                    self.Delt, self.U = eigs(self.X, self.R, which="LM")
                elif self.FLAG_METHOD == 1:
                    self.U, self.Delt = self.mySketch.reconstruct()
                    if self.FLAG_TRACECORRECTION:
                        self.Delt = self.Delt + (
                            (self.TRACE - np.trace(self.Delt)) / self.R
                        ) * np.eye(np.shape(self.Delt)[0])
                self.U = self.U * np.sqrt(self.Delt)

                # if self.stopcond(self.U / np.sqrt(self.SCALE_X)) <= self.stoptol:
                #     self.implement_stopping_criterion(t=t, msg="stopcond stopping")
                #     return True

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
                            self.implement_stopping_criterion(
                                t, msg="accurate stopping"
                            )
                            return True

                    else:
                        self.implement_stopping_criterion(
                            t, msg="stopping criteria met"
                        )
                        return True

        return False

    def implement_stopping_criterion(self, t, msg="accurate stopping"):
        """
        Implements stopping criteria
        :param t: time step
        :param msg: what kind of stopping/ reason for stoppig
        :return:  <Modiifies self>
        """
        if len(self.SAVEHIST) > 0:
            self.updateErrStructs(t)
            self.printError(t)
            self.clearErrStructs()
        self.status = msg

    def update(self):
        pass

    def Primitive3MultRank(self):
        """
        Primitive3MultRank
        :return: AUU
        """
        if self.FLAG_MULTIRANK_P3:
            AUU = self.Primitive3(self.U)
        else:
            AUU = np.zeros(len(self.b))
            for ind in range(self.U.shape[1]):
                AUU += self.Primitive3(self.U[:, ind])

        return AUU

    def Primitive1MultRank(self):
        """
        Primitive1MultRank
        :return: CU
        """
        if self.FLAG_MULTIRANK_P1:
            CU = self.Primitive1(self.U)
        else:
            CU = np.zeros(len(self.b))
            for ind in range(self.U.shape[1]):
                CU[:, ind] = self.Primitive1(self.U[:, ind])
        return CU

    # Create and maintain structures where we store optimization information
    def create_err_structs(self):
        """
        Create final structures to display
        :return:
        """
        if self.stoptol or self.FLAG_EVALSURROGATEGAP:
            if self.stopcond:
                self.out["info"]["stopCond"] = np.empty(len(self.SAVEHIST))
                if self.FLAG_EVALSURROGATEGAP:
                    self.out["info"]["stopObj"] = np.empty(len(self.SAVEHIST))
                    self.out["info"]["stopFeas"] = np.empty(len(self.SAVEHIST))
            else:
                self.out["info"]["stopObj"] = np.empty(len(self.SAVEHIST))
                self.out["info"]["stopFeas"] = np.empty(len(self.SAVEHIST))

        self.out["info"]["primalObj"] = np.empty(len(self.SAVEHIST))
        self.out["info"]["primalFeas"] = np.empty(len(self.SAVEHIST))
        if self.FLAG_METHOD == 1:
            self.out["info"]["skPrimalObj"] = np.empty(len(self.SAVEHIST))
            self.out["info"]["skPrimalFeas"] = np.empty(len(self.SAVEHIST))
        for k, val in self.errfunc.items():
            self.out["info"][k] = np.empty(len(self.SAVEHIST))
        self.out["info"]["cntTotal"] = np.empty(len(self.SAVEHIST))
        self.out["iteration"] = np.empty(len(self.SAVEHIST))
        self.out["time"] = np.empty(len(self.SAVEHIST))
        self.out["cputime"] = np.empty(len(self.SAVEHIST))
        errNames = list(self.out["info"].keys())
        errNamesPrint = errNames
        for pIt in range(len(errNamesPrint)):
            if len(errNamesPrint[pIt]) > 10:
                errNamesPrint[pIt] = errNamesPrint[pIt][:10]
        ptr = 0
        self.status = "running"
        self.out["params"]["ALPHA"] = self.a
        self.out["params"]["BETA0"] = self.beta0
        self.out["params"]["R"] = self.R
        self.out["params"]["FLAG_LANCZOS"] = self.FLAG_LANCZOS
        self.out["params"]["FLAG_TRACECORRECTION"] = self.FLAG_TRACECORRECTION
        self.out["params"]["FLAG_SKETCH"] = self.FLAG_METHOD
        self.status = "running"
        return errNames, errNamesPrint, ptr

    def printError(self, t):
        """
        print required errors
        :param t: t
        :return:
        """
        str = f"| {t} |"
        for p in range(len(self.errNames)):
            if self.out["info"].get(self.errNames[p], None) is not None:
                str += f" {self.out['info'][self.errNames[p]][self.ptr]} |"
            else:
                str += "Nan |"
        print(str)

    def printHeader(self):
        """
        Print the header
        :return:
        """
        str = "|    iter | "
        for p in self.errNamesPrint:
            str += f" {p}     |"
        print(str)

    def updateErrStructs(self, t):
        """
        Update final print structure with each SAVEHIST timestep
        :param t:
        :return:
        """
        if self.ptr % 20 == 0:
            self.printHeader()
        self.ptr += 1
        self.out["iteration"][self.ptr] = t
        # self.time[self.ptr] = self.totTime
        # self.time[self.ptr] = self.totCpuTime
        self.out["info"]["primalObj"][self.ptr] = self.pobj * self.RESCALE_OBJ
        if self.FLAG_INCLUSION:
            FEAS = np.linalg.norm((self.z - self.PROJBOX(self.z)) * self.RESCALE_FEAS)
        else:
            FEAS = np.linalg.norm((self.z - self.b) * self.RESCALE_FEAS) / (
                1 + np.linalg.norm(self.b_org)
            )
        self.out["info"]["primalFeas"][self.ptr] = FEAS
        self.out["info"]["cntTotal"][self.ptr] = self.cntTotal
        if self.FLAG_METHOD == 1:
            self.U, self.Delt = self.mySketch.reconstruct()
            if self.FLAG_TRACECORRECTION:
                self.Delt += ((self.TRACE - np.trace(self.Delt)) / self.R) * np.eye(
                    len(self.Delt)
                )
            self.U = self.U.dot(np.sqrt(self.Delt))
            self.out["info"]["skPrimalObj"][self.ptr] = (
                np.trace(self.U.conj().T.dot(self.Primitive1MultRank()))
                * self.RESCALE_OBJ
            )
            if self.FLAG_INCLUSION:
                self.AUU = self.Primitive3MultRank()
                self.out["info"]["skPrimalFeas"][self.ptr] = np.linalg.norm(
                    (self.AUU - self.PROJBOX(self.AUU)) * self.RESCALE_FEAS
                )
            else:
                self.out["info"]["skPrimalFeas"][self.ptr] = np.linalg.norm(
                    (self.Primitive3MultRank() - self.b) * self.RESCALE_FEAS
                ) / (1 + np.linalg.norm(self.b_org))

        elif self.FLAG_METHOD == 0:
            self.Delt, self.U = eigs(self.X, self.R, which="LM")
            self.U *= np.sqrt(self.Delt)

        else:
            print("Unknown FLAG_METHOD.")

        if not self.stoptol:
            FeasOrg = np.linalg.norm((self.z - self.b) * self.RESCALE_FEAS)
            FeasCond = FeasOrg / max(np.linalg.norm(self.b_org), 1)
            AHk, ObjCond = self.getObjCond()

            self.out["info"]["stopObj"][self.ptr] = ObjCond
            self.out["info"]["stopFeas"][self.ptr] = FeasCond

        for k, val in self.errfunc.items():
            self.out["info"][k][self.ptr] = val(self.U / math.sqrt(self.SCALE_X))

    def clearErrStructs(self):
        self.out["iteration"] = np.empty(len(self.SAVEHIST))
        self.out["time"] = np.empty(len(self.SAVEHIST))
        self.out["cputime"] = np.empty(len(self.SAVEHIST))
        info_fields = self.out["info"].keys()
        param_fields = self.out["params"].keys()
        for k in info_fields:
            self.out["info"][k] = np.empty(len(self.SAVEHIST))
        for k in param_fields:
            self.out["params"][k] = 0

    def solve(self):
        self.cntTotal = 0
        self.TRACE = 0
        for t in range(1, int(self.T) + 1):
            self.beta = self.beta0 * math.sqrt(self.T + 1)
            eta = 2 / (self.T + 1)
            if self.FLAG_INCLUSION:
                self.vt = self.y + self.beta * (
                    self.z - self.PROJBOX(self.z + (1 / self.beta) * self.y)
                )
            else:
                self.vt = self.y + self.beta * (self.z - self.b)

            self.eigsArg = lambda u: self.Primitive1_(u) + self.Primitive2(self.vt, u)
            self.ApplyMinEvecApply(t)
            if self.check_stopping_criteria(t):
                break
            else:
                self.zEvec = self.Primitive3(self.u)
                self.z = (1 - eta) * self.z + eta * self.zEvec
                self.TRACE = (1 - eta) * self.TRACE + eta * self.a_t

                objEvec = np.dot(self.u.conj().T, self.Primitive1_(self.u))
                self.pobj = (1 - eta) * self.pobj + eta * objEvec

                if self.FLAG_METHOD == 0:
                    self.X = (1 - eta) * self.X + eta * np.dot(self.u, self.u.conj().T)
                elif self.FLAG_METHOD == 1:
                    self.mySketch.rank_one_update(self.u, eta)

                beta_p = self.beta0 * math.sqrt(t + 2)

                if self.FLAG_INCLUSION:
                    dualUpdate = self.z - self.PROJBOX(self.z + (1 / beta_p) * (self.y))
                else:
                    dualUpdate = self.z - self.b

                sigma = min(
                    self.beta0,
                    4
                    * beta_p
                    * eta ** 2
                    * max([self.a]) ** 2
                    * self.NORM_A ** 2
                    / np.linalg.norm(dualUpdate) ** 2,
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
                if t in self.SAVEHIST:
                    self.updateErrStructs(t)
                    self.printError(t)

        if self.FLAG_METHOD == 0:
            self.U = np.divide(self.X, self.SCALE_X)
            self.Delt = []
        elif self.FLAG_METHOD == 1:
            if self.FLAG_TRACECORRECTION:
                self.Delt = self.Delt + (
                    (self.TRACE - np.trace(self.Delt)) / self.R
                ) * np.eye(np.shape(self.Delt)[0])
            self.Delt = np.divide(self.Delt, self.SCALE_X)
        self.y = np.divide(np.multiply(self.SCALE_A, self.y), self.SCALE_C)
        self.z = np.multiply(self.z, self.RESCALE_FEAS)
        self.pobj *= self.RESCALE_OBJ
