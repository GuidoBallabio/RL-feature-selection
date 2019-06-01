import numpy as np
from src.algorithm.feature_selection import FeatueSelector


class BackwardFeatureSelector(FeatueSelector):
    def __init__(self, itEstimator, trajectories, nproc=1):
        super().__init__(self, itEstimator, trajectories, nproc)

    def selectOnError(self, k, gamma, max_error):
        error = 0.0
        self.weights = self._get_weights(k)

        while error < max_error and len(self.idSet) > 1:
            scores = self.scoreFeatures(k, gamma, weights)
            self.res += scores[1][0]
            error = self.computeError()

            if error >= max_error:
                return self.idSet

            self.idSet.remove(scores[0][0])

        return self.idSet

    def _scoreFeatureParallel(self):
        pass

    def _scoreFeatureSequential(self, k, gamma):
        list_ids = list(self.idSet)
        score_mat = np.zeros((k+1, len(self.idSet)))
        
        for i in list_ids:
            for t in range(k+1):
                score_mat[i,t] = np.sqrt(self.itEstimator.estimateCMI(self.id_reward, i, self.idSet.difference([i])))
            score_mat[i,k+1] = np.sqrt(self.itEstimator.estimateCH(i, self.idSet.difference([i]))

        scores = np.einsum('a, ab->b', self.weights, score_mat)
        
        sorted_idx = np.argsort(scores)
        
        return list_ids[sorted_idx], scores[sorted_idx]
