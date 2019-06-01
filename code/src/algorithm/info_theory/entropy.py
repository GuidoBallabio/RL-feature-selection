import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KDTree


class LeveOneOutEntropyEstimator(object):
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

    def estimateJoint(self, datas: list):
        for i in range(len(datas)):
            if len(datas[i].shape) == 1:
                datas[i] = np.expand_dims(datas[i], 1)

        joint = np.hstack(datas)
        return self.estimateFromData(joint)


class NNEntropyEstimator(object):
    EPS = 0.001

    def __init__(self):
        pass

    def estimateFromData(self, datapoints):
        """
        N.B. the real estimate is this one + log(2) + C_E (Euler's constant)
        """
        entropy = 0.0
        nPoints = datapoints.shape[0]
        for i in range(nPoints):
            neighbors = KDTree(np.delete(datapoints, i, axis=0))
            dist, nearest = neighbors.query(datapoints[i, :].reshape(1, -1), 1)
            dist = dist[0]
            entropy += np.log(nPoints * dist + self.EPS)

        return entropy / nPoints

    def estimateJoint(self, datas: list):
        for i in range(len(datas)):
            if len(datas[i].shape) == 1:
                datas[i] = np.expand_dims(datas[i], 1)

        joint = np.hstack(datas)
        return self.estimateFromData(joint)
