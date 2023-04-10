from typing import Optional

import cvxpy as cp
import numpy as np

import torch


def thresh(X, tol: float = 1.0e-6):
    M = np.abs(X) > tol
    return M * X, M.size - M.sum()


class LinearSubspaceProjectionNaive:
    def __init__(self, E: int, H: int, D: Optional[int] = None) -> None:

        self.E, self.H = E, H

        if (H > 1) or (D is None):
            assert E % H == 0
            self.D = E // H
        else:
            self.D = D

        DxD = (self.D, self.D)
        DxE = (self.D, self.E)

        self.S = cp.Variable(DxD)
        self._UKS = cp.Parameter(DxD)
        self._UQS = cp.Parameter(DxD)
        self._VKT = cp.Parameter(DxE)
        self._VQT = cp.Parameter(DxE)
        self._dWQ = cp.Parameter(DxE)
        self._dWK = cp.Parameter(DxE)

        self.obj = cp.sum(
            cp.abs(self._dWQ - self._UKS @ self.S @ self._VQT) + cp.abs(self._dWK + self._UQS @ self.S.T @ self._VKT)
        )

        self.problem = cp.Problem(cp.Minimize(self.obj))

    def solve(
        self,
        WQ: torch.Tensor,
        WK: torch.Tensor,
        dWQ: torch.Tensor,
        dWK: torch.Tensor,
    ):
        D, E, H = self.D, self.E, self.H

        ndWQ = torch.linalg.norm(dWQ).numpy(force=True)
        ndWK = torch.linalg.norm(dWK).numpy(force=True)

        status = 0.0
        ZK, ZQ = np.zeros(6), np.zeros(6)
        zk, zq = np.zeros(6), np.zeros(6)
        resolves = np.zeros(H)
        for h in range(H):

            UQ, SQ, VQT = torch.linalg.svd(WQ[D * h : D * (h + 1), :], full_matrices=False)
            UK, SK, VKT = torch.linalg.svd(WK[D * h : D * (h + 1), :], full_matrices=False)

            self._UQS.value = (UQ / SQ).detach().numpy(force=True)
            self._UKS.value = (UK / SK).detach().numpy(force=True)
            self._VQT.value = VQT.detach().numpy(force=True)
            self._VKT.value = VKT.detach().numpy(force=True)

            self._dWQ.value = dWQ[D * h : D * (h + 1), :].detach().clone().numpy(force=True) / ndWQ
            self._dWK.value = dWK[D * h : D * (h + 1), :].detach().clone().numpy(force=True) / ndWK

            self.problem.solve(solver=cp.ECOS)
            status += int(self.problem.status == "optimal")

            DWQ = self._UKS @ self.S @ self._VQT
            DWK = -self._UQS @ self.S.T @ self._VKT

            self._dWQ.value = self._dWQ.value - DWQ.value
            self._dWK.value = self._dWK.value - DWK.value

            for n in range(6):
                N = 10.0 ** (-(n + 6.0))  # -6 to -12
                _, zq[n] = thresh(self._dWQ.value, tol=N)
                _, zk[n] = thresh(self._dWK.value, tol=N)

            ZQ, ZK = ZQ + zq, ZK + zk

            self.problem.solve(solver=cp.ECOS)
            resolves[h] = np.linalg.norm(self.S.value)

        return status / H, resolves, (ZQ + ZK) / (H * D * D), 1.0 - (ZQ + ZK) / (2 * E * E)


class LinearSubspaceProjectionConstr:
    def __init__(self, E: int, H: int, D: Optional[int] = None) -> None:

        self.E, self.H = E, H

        if (H > 1) or (D is None):
            assert E % H == 0
            self.D = E // H
        else:
            self.D = D

        DxD = (self.D, self.D)
        DxE = (self.D, self.E)

        self.S = cp.Variable(DxD)
        self.A = cp.Variable(DxE)
        self.B = cp.Variable(DxE)
        self._UKS = cp.Parameter(DxD)
        self._UQS = cp.Parameter(DxD)
        self._VKT = cp.Parameter(DxE)
        self._VQT = cp.Parameter(DxE)
        self._dWQ = cp.Parameter(DxE)
        self._dWK = cp.Parameter(DxE)

        self.obj = cp.sum(cp.abs(self.A) + cp.abs(self.B))

        # Note: This is a "DPP" version, but that may not
        # be super important... unless we share this
        # problem instance across all calls to solve...
        self.problem = cp.Problem(
            cp.Minimize(self.obj),
            constraints=[
                self.A == self._dWQ - self._UKS @ self.S @ self._VQT,
                self.B == self._dWK + self._UQS @ self.S.T @ self._VKT,
            ],
        )

    def solve(
        self,
        WQ: torch.Tensor,
        WK: torch.Tensor,
        dWQ: torch.Tensor,
        dWK: torch.Tensor,
    ):
        D, E, H = self.D, self.E, self.H

        ndWQ = torch.linalg.norm(dWQ).numpy(force=True)
        ndWK = torch.linalg.norm(dWK).numpy(force=True)

        status = 0.0
        ZK, ZQ = np.zeros(6), np.zeros(6)
        zk, zq = np.zeros(6), np.zeros(6)
        resolves = np.zeros(H)
        for h in range(H):

            UQ, SQ, VQT = torch.linalg.svd(WQ[D * h : D * (h + 1), :], full_matrices=False)
            UK, SK, VKT = torch.linalg.svd(WK[D * h : D * (h + 1), :], full_matrices=False)

            self._UQS.value = (UQ / SQ).detach().numpy(force=True)
            self._UKS.value = (UK / SK).detach().numpy(force=True)
            self._VQT.value = VQT.detach().numpy(force=True)
            self._VKT.value = VKT.detach().numpy(force=True)

            self._dWQ.value = dWQ[D * h : D * (h + 1), :].detach().clone().numpy(force=True) / ndWQ
            self._dWK.value = dWK[D * h : D * (h + 1), :].detach().clone().numpy(force=True) / ndWK

            self.problem.solve(solver=cp.ECOS)
            status += int(self.problem.status == "optimal")

            DWQ = self._UKS @ self.S @ self._VQT
            DWK = -self._UQS @ self.S.T @ self._VKT

            self._dWQ.value = self._dWQ.value - DWQ.value
            self._dWK.value = self._dWK.value - DWK.value

            for n in range(6):
                N = 10.0 ** (-(n + 6.0))  # -6 to -12
                _, zq[n] = thresh(self._dWQ.value, tol=N)
                _, zk[n] = thresh(self._dWK.value, tol=N)

            ZQ, ZK = ZQ + zq, ZK + zk

            self.problem.solve(solver=cp.ECOS)
            resolves[h] = np.linalg.norm(self.S.value)

        return status / H, resolves, (ZQ + ZK) / (H * D * D), 1.0 - (ZQ + ZK) / (2 * E * E)


class LinearSubspaceProjectionDPP:
    def __init__(self, E: int, H: int, D: Optional[int] = None) -> None:

        self.E, self.H = E, H

        if (H > 1) or (D is None):
            assert E % H == 0
            self.D = E // H
        else:
            self.D = D

        DxD = (self.D, self.D)
        DxE = (self.D, self.E)

        self.S = cp.Variable(DxD)
        self.A = cp.Variable(DxE)
        self.B = cp.Variable(DxE)
        self.P = cp.Variable(DxE)
        self.Q = cp.Variable(DxE)
        self._UKS = cp.Parameter(DxD)
        self._UQS = cp.Parameter(DxD)
        self._VKT = cp.Parameter(DxE)
        self._VQT = cp.Parameter(DxE)
        self._dWQ = cp.Parameter(DxE)
        self._dWK = cp.Parameter(DxE)

        self.obj = cp.sum(cp.abs(self.A) + cp.abs(self.B))

        # Note: This is a "DPP" version, but that may not
        # be super important... unless we share this
        # problem instance across all calls to solve...
        self.problem = cp.Problem(
            cp.Minimize(self.obj),
            constraints=[
                self.A == self._dWQ - self._UKS @ self.P,
                self.B == self._dWK + self._UQS @ self.Q,
                self.P == self.S @ self._VQT,
                self.Q == self.S.T @ self._VKT,
            ],
        )

    def solve(
        self,
        WQ: torch.Tensor,
        WK: torch.Tensor,
        dWQ: torch.Tensor,
        dWK: torch.Tensor,
    ):
        D, E, H = self.D, self.E, self.H

        ndWQ = torch.linalg.norm(dWQ).numpy(force=True)
        ndWK = torch.linalg.norm(dWK).numpy(force=True)

        status = 0.0
        ZK, ZQ = np.zeros(6), np.zeros(6)
        zk, zq = np.zeros(6), np.zeros(6)
        resolves = np.zeros(H)
        for h in range(H):

            UQ, SQ, VQT = torch.linalg.svd(WQ[D * h : D * (h + 1), :], full_matrices=False)
            UK, SK, VKT = torch.linalg.svd(WK[D * h : D * (h + 1), :], full_matrices=False)

            self._UQS.value = (UQ / SQ).detach().numpy(force=True)
            self._UKS.value = (UK / SK).detach().numpy(force=True)
            self._VQT.value = VQT.detach().numpy(force=True)
            self._VKT.value = VKT.detach().numpy(force=True)

            self._dWQ.value = dWQ[D * h : D * (h + 1), :].detach().clone().numpy(force=True) / ndWQ
            self._dWK.value = dWK[D * h : D * (h + 1), :].detach().clone().numpy(force=True) / ndWK

            self.problem.solve(solver=cp.ECOS)
            status += int(self.problem.status == "optimal")

            DWQ = self._UKS @ self.S @ self._VQT
            DWK = -self._UQS @ self.S.T @ self._VKT

            self._dWQ.value = self._dWQ.value - DWQ.value
            self._dWK.value = self._dWK.value - DWK.value

            for n in range(6):
                N = 10.0 ** (-(n + 6.0))  # -6 to -12
                _, zq[n] = thresh(self._dWQ.value, tol=N)
                _, zk[n] = thresh(self._dWK.value, tol=N)

            ZQ, ZK = ZQ + zq, ZK + zk

            self.problem.solve(solver=cp.ECOS)
            resolves[h] = np.linalg.norm(self.S.value)

        return status / H, resolves, (ZQ + ZK) / (H * D * D), 1.0 - (ZQ + ZK) / (2 * E * E)
