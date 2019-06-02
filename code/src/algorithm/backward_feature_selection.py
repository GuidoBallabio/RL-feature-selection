import numpy as np
from src.algorithm.feature_selection import FeatueSelector


class BackwardFeatureSelector(FeatueSelector):
    def __init__(self, itEstimator, trajectories, nproc=1):
        super().__init__(self, itEstimator, trajectories, nproc)

        self.idSelected = self.idSet.copy()

    def selectOnError(self, k, gamma, max_error):
        error = 0.0
        self.weights = self._get_weights(k)
        self._prep_data(k)

        while error < max_error and len(self.idSelected) > 1:
            scores = self.scoreFeatures(k, gamma, weights)
            self.res += scores[1][0]
            error = self.computeError()

            if error >= max_error:
                return self.idSelected

            self.idSelected.remove(scores[0][0])

        return self.idSelected

    def _scoreFeatureParallel(self):
        pass

    def _scoreFeatureSequential(self, k, gamma):
        no_S = self.idSet.difference(self.idSelected)  # discarded
        list_ids = list(no_S)
        score_mat = np.zeros((k+1, len(list_ids)))

        for i in list_ids:
            S_no_i = self.idSelected.difference({i})
            no_S_i = discarded_feat.union({i})
            for t in range(k+1):
                score_mat[i, t] = np.sqrt(self.itEstimator.estimateCMI(
                    {self.id_reward}, no_S_i, S_no_i, t=t))
            score_mat[i, k +
                      1] = np.sqrt(self.itEstimator.estimateCH(no_S_i, S_no_i))

        scores = np.einsum('a, ab->b', self.weights, score_mat)
        sorted_idx = np.argsort(scores)

        return list_ids[sorted_idx], scores[sorted_idx]
