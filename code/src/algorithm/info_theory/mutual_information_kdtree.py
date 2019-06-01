# @Author: mario
# @Date:   2019-01-07T17:03:06+01:00
# @Last modified by:   mario
# @Last modified time: 2019-01-07T17:16:55+01:00



import abc
import numpy as np
from functools import partial
from scipy.special import digamma
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
from sklearn.preprocessing import scale, normalize
from multiprocessing import Pool


class MIEstimator(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def estimateConditionalMI(self, X, Y, Z):
        """
        In the context of feature selection, X is the feature to be added
        or deleted, Y is the response (either categorical for classification
        or numerical for regression) and Z is the matrix of the other features
        """
        pass

    @abc.abstractmethod
    def estimateMI(self, X, Y, Z):
        pass


class EntropyMIEstimator(MIEstimator):
    """
    Basic 3H MIEstimator.
    """
    def __init__(self, entropyEstimator):
        self.entropyEstimator = entropyEstimator

    def estimateMI(self, X, Y):
        h_x = self.entropyEstimator.estimateFromData(X)
        h_y = self.entropyEstimator.estimateFromData(Y)
        h_xy = self.entropyEstimator.estimateJoint([X, Y])
        return h_x + h_y - h_xy

    def estimateConditionalMI(self, X, Y, Z):
        h_xz = self.entropyEstimator.estimateJoint([X, Z])
        h_yz = self.entropyEstimator.estimateJoint([Y, Z])
        h_xyz = self.entropyEstimator.estimateJoint([X, Y, Z])
        h_z = self.entropyEstimator.estimateFromData(Z)
        return h_xz + h_yz - h_xyz - h_z


def distanceInNorm(x, y, norm):
    return np.linalg.norm(x - y, norm)


def computeMIforSample(i, XYZ, XZ, YZ, Z, norm, k):
    dists = np.array(list(map(
            lambda z: np.linalg.norm(XYZ[i] - z, norm),
            np.delete(XYZ, i, axis=0))))
    idx = np.argpartition(dists, k-1)[k-1]
    epsI = dists[idx]

    distsXZ = np.array(list(map(
            lambda z: np.linalg.norm(XZ[i] - z, norm),
            np.delete(XZ, i, axis=0))))
    nXZ = np.sum(distsXZ < epsI) + 1

    distsYZ = np.array(list(map(
            lambda z: np.linalg.norm(YZ[i] - z, norm),
            np.delete(XZ, i, axis=0))))
    nYZ = np.sum(distsYZ < epsI) + 1
    distsZ = np.array(list(map(
            lambda z: np.linalg.norm(Z[i] - z, norm),
            np.delete(Z, i, axis=0))))
    nZ = np.sum(distsZ < epsI) + 1
    print("Old eps: {0}, nXZ: {1}, nYZ: {2}, nZ:{3}".format(epsI, nXZ, nYZ, nZ))
    return digamma(nXZ) + digamma(nYZ) - digamma(nZ)


class MixedRvMiEstimator(MIEstimator):
    """
    An estimator of the mutual information based on the local estimate of
    the Radon-Nikodym derivative. For more information see:
    https://papers.nips.cc/paper/7180-estimating-mutual-information-for-discrete-continuous-mixtures.pdf
    """
    def __init__(self, num_neighbors, norm=2, nproc=1):
        super().__init__()
        self.k = num_neighbors
        self.norm = norm
        self.nproc = nproc

    @staticmethod
    def firstNonZero(vec):
        mask = (vec != 0)
        return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)

    def estimateConditionalMI(self, X, Y, Z):
        """
        I(X;Y|Z) = I(X,Z; Y) - I(Z; Y)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        XYZ = np.hstack((X, Y, Z))
        XZ = np.hstack((X, Z))
        YZ = np.hstack((Y, Z))
        nSamples = X.shape[0]

        XYZdists = squareform(pdist(XYZ))
        XZTree = KDTree(XZ)
        YZTree = KDTree(YZ)
        ZTree = KDTree(Z)
        partialMis = np.zeros(nSamples)

        for i in range(nSamples):
            dists = np.array(XYZdists[i, ])
            idx = np.argpartition(dists, self.k-1)[self.k-1]
            epsI = dists[idx]

            nXZ = XZTree.query_radius(XZ[i].reshape(1, -1), epsI, count_only=True)
            nYZ = YZTree.query_radius(YZ[i].reshape(1, -1), epsI, count_only=True)
            nZ = ZTree.query_radius(Z[i].reshape(1, -1), epsI, count_only=True)

            partialMis[i] = digamma(nXZ) + digamma(nYZ) - digamma(nZ)

        return digamma(self.k) - np.sum(partialMis) / nSamples

    def estimateMI(self, X, Y):
        nSamples = X.shape[0]
        out = 0
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        for i in range(nSamples):
            x_i = X[i]
            y_i = Y[i]
            distsX = np.array(list(map(
                lambda z: np.linalg.norm(x_i - z, self.norm),
                np.delete(X, i, axis=0))))
            distsY = np.array(list(map(
                lambda z: np.linalg.norm(y_i - z, self.norm),
                np.delete(Y, i, axis=0))))
            dists = np.maximum(distsX, distsY)
            idx = np.argpartition(dists, self.k-1)[self.k-1]
            distK = dists[idx]
            k = self.k if distK > 0 else self.firstNonZero(distK)
            nX = sum(distsX <= distK-1e-15) + 1
            # print('nX: {0}'.format(nX))
            nY = sum(distsY <= distK-1e-15) + 1
            # print('nY: {0}'.format(nY))
            out += (digamma(k) - digamma(nX) - digamma(nY)) / nSamples
        return out + np.log(nSamples)
