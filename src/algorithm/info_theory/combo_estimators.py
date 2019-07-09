import numpy as np
from npeet import entropy_estimators as ee
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity

from src.algorithm.info_theory.entropy import NNEntropyEstimator
from src.algorithm.info_theory.it_estimator import ItEstimator
from src.algorithm.info_theory.mutual_information import MixedRvMiEstimator


class NpeetEstimator(ItEstimator):
    def __init__(self):
        pass

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropy(X.copy(order='C'))

    def mi(self, X, Y):
        np.random.seed(0)
        return ee.mi(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return ee.mi(X.copy(order='C'), Y.copy(order='C'), z=Z.copy(order='C'))

    def flags(self):
        return True, False, True


class CmiEstimator(ItEstimator):
    def __init__(self, nproc=1):
        self.h_est = NNEntropyEstimator(nproc=nproc)
        self.mi_est = MixedRvMiEstimator(3)

    def entropy(self, X):
        np.random.seed(0)
        return self.h_est.entropy(X.copy(order='C'))

    def mi(self, X, Y):
        np.random.seed(0)
        return self.mi_est.estimateMI(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return self.mi_est.estimateConditionalMI(X.copy(order='C'), Y.copy(order='C'), Z.copy(order='C'))

    def flags(self):
        return True, False, True


class FastNNEntropyEstimator(ItEstimator):
    EPS = np.finfo(np.float).eps

    def __init__(self, kfold=10):
        self.kfold = kfold

    def estimateFromData(self, datapoints):
        entropy = 0.0
        n, d = datapoints.shape
        k = 1

        ma = np.ones(n, dtype=np.bool)
        unit = n // self.kfold
        rem = n % self.kfold

        start = 0
        end = unit + rem
        for i in range(self.kfold):
            sel = np.arange(start, end)
            ma[start:end] = False
            curr = datapoints[ma, :]

            nn = NearestNeighbors(n_neighbors=k).fit(curr)
            dist, _ = nn.kneighbors(datapoints[sel, :])
            entropy += np.log(n * dist + self.EPS).sum()

            ma[:] = True
            start = end
            end = min(unit + end, n)

        return entropy / n + np.log(2) + np.euler_gamma

    def entropy(self, X):
        np.random.seed(0)
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False


class KDEntropyEstimator(ItEstimator):
    def __init__(self, kernel="gaussian",  min_log_proba=-500, bandwith=1.0, kfold=10):
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwith)
        self.min_log_proba = min_log_proba
        self.kfold = kfold

    def estimateFromData(self, datapoints):
        if len(datapoints.shape) == 1:
            datapoints = np.expand_dims(datapoints, 1)

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
            curr = datapoints[ma, :]

            self.kde.fit(curr)
            score = self.kde.score(datapoints[sel, :])

            ma[:] = True
            start = end
            end = min(unit + end, n)

            if score < self.min_log_proba:
                continue

            entropy -= score

        return entropy / n

    def entropy(self, X):
        np.random.seed(0)
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False
