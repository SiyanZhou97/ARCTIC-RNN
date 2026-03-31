import copy
import math
try:
    import cupy as np
    a=np.array([0])
except:
    import numpy as np


class Rls():
    def __init__(self,M,N=None,reg_ratio=None,lr=1):
        """initialize the inv corr matrix P for RLS-based RNN training

        Args:
            M (int): number of source neurons
            N (int, optional): number of target neurons. Needed if reg_ratio is not None
            reg_ratio (scalar,optional):
                apply extra regularization on the [M-N:] columns of the weight, through P0.
                In practice, these are cue & beh feedback weights
            lr (scalar,optional): learning rate
        """
        self.P0=1
        self.M=M
        self.P = self.P0 * np.eye(self.M, self.M)
        if reg_ratio is not None:
            self.P[N:, N:] = self.P[N:, N:] / reg_ratio
        self.lr=lr

    def update(self, weight, input, err):
        """

        Args:
            weight (ndarray of shape (N,M)): weight matrix being optimized
            input (ndarray of shape (M, ))
            err (ndarray of shape (N, ))

        Returns:
            updated weight matrix

        """
        Pu = np.dot(self.P, input)
        k = Pu / (1 + np.dot(input, Pu))
        self.P = self.P - np.outer(k, np.dot(input, self.P))
        weight = weight - self.lr * np.outer(err, k)

        return weight


class RlsZeroOutDiagonal():
    """
    This optimization algorithm is modified from the vanilla RLS.
    It is designed for optimizing weight matrix while not updating the zero-initialized diagonal.
    """
    def __init__(self,M,N,reg_ratio=None,lr=10):
        """initialize the inv corr matrix P for RLS-based RNN training

        Args:
            M (int): number of source neurons
            N (int): number of target neurons
            reg_ratio (scalar,optional):
                apply extra regularization on the [M-N:] columns of the weight, through P0.
                In practice, these are cue & beh feedback weights
            lr (scalar,optional): learning rate

        C: correlation matrix
        P: inverse correlation matrix
        """
        self.P0 = 1
        self.M = M
        self.P = self.P0 * np.eye(self.M)
        self.C = (1 / self.P0) * np.eye(self.M)
        self.N = N
        if reg_ratio is not None:
            self.P[N:, N:] = self.P[N:, N:] / reg_ratio
            self.C[N:, N:] = self.C[N:, N:] * reg_ratio
        self.lr=lr

    def update(self, weight, input, err):
        """

        Args:
            weight (ndarray of shape (N,M)): weight matrix being optimized
            input (ndarray of shape (M, ))
            err (ndarray of shape (N, ))

        Returns:
            updated weight matrix

        """
        self.C = self.C + np.outer(input, input)

        Pu = np.dot(self.P, input)
        k = Pu / (1 + np.dot(input, Pu))
        self.P = self.P - np.outer(k, np.dot(input, self.P))

        # PU, VPU and V matrixs
        PU = np.zeros((self.N, self.M, 2))
        PU[:, :, 0] = self.P[:self.N, :]
        np.fill_diagonal(PU[:, :, 1], 1)
        PU[:, :, 1] = PU[:, :, 1] - self.P[:self.N, :] / self.P0

        VPU = np.zeros((self.N, 2, 2))
        VPU[:, 0, 0] = 1
        VPU[:, 0, 0] = VPU[:, 0, 0] - np.multiply(np.diag(self.C), np.diag(self.P))[:self.N]
        VPU[:, 0, 1] = np.multiply(np.diag(self.C), np.diag(self.P))[:self.N] / self.P0 - 1 / self.P0
        VPU[:, 1, 0] = np.diag(self.P)[:self.N]
        VPU[:, 1, 1] = 1 - np.diag(self.P)[:self.N] / self.P0

        V = np.zeros((self.N, 2, self.M))
        V[:, 0, :] = self.C[:self.N, :]
        np.fill_diagonal(V[:, 1, :self.N], 1)
        np.fill_diagonal(V[:, 0, :self.N], 0)
        # mask
        mask = (1 - np.eye(self.M))[:self.N, :]

        # weight update based on the compact matrix formula
        partA = np.matmul(PU, np.linalg.inv(np.eye(2) - VPU))
        partB = np.matmul(V, np.matmul(self.P, input))[:, :, np.newaxis]
        partApartB = np.matmul(partA, partB)[:, :, 0]
        deltaJ = self.lr * np.multiply(mask,
                                  np.multiply(err[:, np.newaxis],
                                              (np.matmul(self.P, input) + partApartB)))
        weight = weight - deltaJ

        return weight




