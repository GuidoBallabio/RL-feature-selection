import numpy as np

from src.algorithm.feature_selection import Bound, FeatureSelector
from tqdm.autonotebook import tqdm


class BackwardFeatureSelector(FeatureSelector):
    def __init__(self, itEstimator, trajectories, nproc=1):
        super().__init__(itEstimator, trajectories, nproc)

        self.idSelected = set(self.idSet)

    def selectNfeatures(self, n, k, gamma, bound=Bound.cmi, show_progress=True):
        assert k <= self.max_k, f"k {k} is larger than the shortest trajectory (len {self.max_k})"
        assert n <= self.n_features, f"Features to be selected {n} must be less than  the total" \
            f"number of feature: {self.n_features}"

        self.reset()

        self.weights = self._get_weights(k, gamma, bound)

        error = 0.0
        self._prep_data(k)

        for i in tqdm(range(self.n_features - n), disable=not show_progress):
            scores = self.scoreFeatures(
                k, gamma, bound, show_progress=show_progress)

            self.idSelected.remove(scores[0][0])
            self.residual_error += scores[1][0]
            self.correction_term = scores[2][0]
            error = self.computeError(bound)

        return self.idSelected.copy(), error

    def try_remove_all(self, k, gamma, bound=Bound.cmi, show_progress=True):
        assert k <= self.max_k, f"k {k} is larger than the shortest trajectory (len {self.max_k})"

        self.reset()

        self.weights = self._get_weights(k, gamma, bound)

        error = 0.0
        self._prep_data(k)

        for i in tqdm(range(self.n_features), disable=not show_progress):
            scores = self.scoreFeatures(
                k, gamma, bound, show_progress=show_progress)

            self.idSelected.remove(scores[0][0])
            self.residual_error += scores[1][0]
            self.correction_term = scores[2][0]
            error = self.computeError(bound)
            yield self.idSelected.copy(), error  # if all is useless move up

    def selectOnError(self, k, gamma, max_error, bound=Bound.cmi, show_progress=True):
        assert k <= self.max_k, f"k {k} is larger than the shortest trajectory (len {self.max_k})"

        self.reset()

        self.weights = self._get_weights(k, gamma, bound)

        error = 0.0
        self._prep_data(k)

        with tqdm(total=100, disable=not show_progress) as pbar:  # tqdm
            while error <= max_error and len(self.idSelected) > 1:
                scores = self.scoreFeatures(
                    k, gamma, bound, show_progress=show_progress)

                new_cmi_term = self.residual_error + scores[1][0]
                new_corr_term = scores[2][0]
                new_error = self.computeError(
                    bound, new_cmi_term, new_corr_term)

                perc_of_max = int(100*new_error/max_error)  # tqdm
                pbar.update(min(perc_of_max, pbar.total) - pbar.n)  # tqdm

                if new_error >= max_error:
                    return self.idSelected.copy(), error

                self.idSelected.remove(scores[0][0])
                self.residual_error = new_cmi_term
                self.correction_term = new_corr_term
                error = new_error

            pbar.update(pbar.total - pbar.n)  # tqdm

        return self.idSelected.copy(), error

    def _scoreFeatureParallel(self):
        pass

    def _scoreFeatureSequential(self, k, gamma, bound, show_progress):
        S = frozenset(self.idSelected)
        no_S = self.idSet.difference(self.idSelected)  # discarded

        list_ids = np.fromiter(S, dtype=np.int)
        score_mat = np.zeros((k+1, len(list_ids)))

        fun_t, fun_k = self._funOfBound(bound)

        for i, id in enumerate(tqdm(list_ids, leave=False, disable=not show_progress)):
            id = frozenset({id})
            S_no_i = S.difference(id)
            no_S_i = no_S.union(id)

            for t in range(k):
                score_mat[t, i] = fun_t(id, S_no_i, t)
            score_mat[k, i] = fun_k(no_S_i, S_no_i)

        cmi_wsum = np.einsum('a, ab->b', self.weights[:-1], score_mat[:-1, :])
        new_cond_entropy = self.weights[-1] * score_mat[-1, :]

        sorted_idx = np.argsort(cmi_wsum + new_cond_entropy)

        return list_ids[sorted_idx], cmi_wsum[sorted_idx],  new_cond_entropy[sorted_idx]

    def reset(self):
        super().reset()
        self.idSelected = set(self.idSet)
