import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors.kde import KernelDensity

from src.algorithm.info_theory.it_estimator import ItEstimator


class LeveOneOutEntropyEstimator(ItEstimator):
    """
    Leave One Out cross-validation entropy estimation from datapoints by
    using kernel estimation of the probability density
    See More:
    Ivanov A. V. and Rozhkova . Properties of the statistical estimate of the
    entropy of a random vector with a probability density
    """

    def __init__(self, kernel,  min_log_proba, bandwith=1.0):
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwith)
        self.min_log_proba = min_log_proba

    def estimateFromData(self, datapoints):
        entropy = 0.0
        if len(datapoints.shape) == 1:
            datapoints = np.expand_dims(datapoints, 1)
        for i in range(datapoints.shape[0]):
            curr = np.delete(datapoints, i, axis=0)
            self.kde.fit(curr)
            score = self.kde.score(np.array([datapoints[i, :]]))
            if score < self.min_log_proba:
                print(score)
                continue

            entropy -= score

        return entropy / datapoints.shape[0]

    def entropy(self, X):
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False


class NNEntropyEstimator(ItEstimator):
    EPS = 0.001

    def __init__(self):
        pass

    def estimateFromData(self, datapoints):
        entropy = 0.0
        nPoints = datapoints.shape[0]
        for i in range(nPoints):
            neighbors = KDTree(np.delete(datapoints, i, axis=0))
            dist, nearest = neighbors.query(datapoints[i, :].reshape(1, -1), 1)
            dist = dist[0][0]
            entropy += np.log(nPoints * dist + self.EPS)

        return entropy / nPoints + np.log(2) + np.euler_gamma

    def entropy(self, X):
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False
