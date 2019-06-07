import numpy as np
from src.algorithm.feature_selection import FeatureSelector
from tqdm.autonotebook import tqdm


class BackwardFeatureSelector(FeatureSelector):
    def __init__(self, itEstimator, trajectories, nproc=1):
        super().__init__(itEstimator, trajectories, nproc)

        self.idSelected = set(self.idSet)

    def selectOnError(self, k, gamma, max_error, show_progress=True):
        assert k <= self.max_k, f"k {k} is larger than the shortest trajectory (len {self.max_k})"

        error = 0.0
        self.weights = self._get_weights(k, gamma)
        self._prep_data(k)

        with tqdm(total=100, disable=not show_progress) as pbar:  # tqdm problem with floats
            while error <= max_error and len(self.idSelected) > 1:
                scores = self.scoreFeatures(
                    k, gamma, show_progress=show_progress)
                self.residual_error += scores[1][0]
                error = self.computeError()

                perc_of_max = int(100*error/max_error)
                pbar.update(min(perc_of_max, pbar.total) - pbar.n)

                if error >= max_error:
                    return self.idSelected

                self.idSelected.remove(scores[0][0])

        return self.idSelected

    def _scoreFeatureParallel(self):
        pass

    def _scoreFeatureSequential(self, k, gamma, show_progress):
        no_S = self.idSet.difference(self.idSelected)  # discarded
        list_ids = np.fromiter(self.idSelected, dtype=np.int)
        score_mat = np.zeros((k+1, len(list_ids)))

        for i, id in enumerate(tqdm(list_ids, leave=False, disable=not show_progress)):
            S_no_i = frozenset(self.idSelected.difference({id}))
            no_S_i = no_S.union({id})
            for t in range(k):
                score_mat[t, i] = np.sqrt(self.itEstimator.estimateCMI(
                    frozenset({self.id_reward}), no_S_i, S_no_i, t=t))
            score_mat[t, i] = np.sqrt(
                self.itEstimator.estimateCH(no_S_i, S_no_i))

        scores = np.einsum('a, ab->b', self.weights, score_mat)
        sorted_idx = np.argsort(scores)

        return list_ids[sorted_idx], scores[sorted_idx]
