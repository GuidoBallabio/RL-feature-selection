import numpy as np
from npeet import entropy_estimators as ee
from numpy import pi
from scipy.special import gamma, psi
from sklearn.neighbors import KDTree, NearestNeighbors

from src.algorithm.info_theory.it_estimator import ItEstimator
from src.algorithm.info_theory.mutual_information import MixedRvMiEstimator

class NpeetEstimator(ItEstimator):
    def __init__(self):
        pass

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropy(X)

    def mi(self, X, Y):
        np.random.seed(0)
        return ee.mi(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return ee.mi(X.copy(order='C'), Y.copy(order='C'), z=Z.copy(order='C'))

    def flags(self):
        return True, False, True

class CmiEstimator(ItEstimator):
    def __init__(self):
        self.est = MixedRvMiEstimator(3)

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropy(X)

    def mi(self, X, Y):
        np.random.seed(0)
        return self.est.estimateMI(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return self.est.estimateConditionalMI(X.copy(order='C'), Y.copy(order='C'), Z.copy(order='C'))

    def flags(self):
        return True, False, True


class FastNNEntropyEstimator(ItEstimator):
    EPS = np.finfo(np.float).eps

    def __init__(self, kfold=10):
        self.kfold = kfold

    def estimateFromData(self, datapoints):
        entropy = 0.0
        n, d = datapoints.shape
        ma = np.ones(n, dtype=np.bool)
        unit = n // self.kfold
        rem = n % self.kfold

        start = 0
        end = unit + rem
        for i in range(self.kfold):
            sel = np.arange(start, end)
            ma[start:end] = False
            arr = datapoints[ma, :]

            neighbors = KDTree(arr)
            dist, _ = neighbors.query(datapoints[sel, :], 1)
            entropy += np.log((unit * dist).sum() + self.EPS)

            ma[:] = True
            start += unit
            end = min(unit + end, n)

        return entropy / n + np.log(2) + np.euler_gamma

    def entropy(self, X):
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False


