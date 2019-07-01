import numpy as np

from src.algorithm.feature_selection import FeatureSelector
from tqdm.autonotebook import tqdm


class BackwardFeatureSelector(FeatureSelector):
    def __init__(self, itEstimator, trajectories, nproc=1):
        super().__init__(itEstimator, trajectories, nproc)

        self.idSelected = set(self.idSet)

    def reset(self):
        super().reset()
        self.idSelected = set(self.idSet)

    def selectNfeatures(self, n, k, gamma, sampling="frequency", freq=1, sum_cmi=True, use_Rt=True, on_mu=True, show_progress=True):
        assert n <= self.n_features, f"Features to be selected {n} must be less than  the total" \
            f"number of feature: {self.n_features}"

        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        for i in tqdm(range(self.n_features - n), disable=not show_progress):
            scores = self.scoreFeatures(
                steplist, gamma, sum_cmi, show_progress=show_progress)

            self.idSelected.remove(scores[0][0])
            if sum_cmi:
                self.residual_error += scores[1][0]
            else:
                self.residual_error = scores[1][0]
            self.correction_term = scores[2][0]
            error = self.computeError(use_Rt=use_Rt)

        return self.idSelected.copy(), error

    def try_remove_all(self, k, gamma, sampling="frequency", freq=1, sum_cmi=True, use_Rt=True, on_mu=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        for i in tqdm(range(self.n_features), disable=not show_progress):
            scores = self.scoreFeatures(
                steplist, gamma, sum_cmi, show_progress=show_progress)

            self.idSelected.remove(scores[0][0])
            if sum_cmi:
                self.residual_error += scores[1][0]
            else:
                self.residual_error = scores[1][0]
            self.correction_term = scores[2][0]
            error = self.computeError(use_Rt=use_Rt)
            yield self.idSelected.copy(), error

    def selectOnError(self, k, gamma, max_error, sampling="frequency", freq=1, sum_cmi=True, use_Rt=True, on_mu=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        error = 0.0

        with tqdm(total=100, disable=not show_progress) as pbar:  # tqdm
            while error <= max_error and len(self.idSelected) > 1:
                scores = self.scoreFeatures(
                    steplist, gamma, sum_cmi, show_progress=show_progress)

                if sum_cmi:
                    new_cmi_term = self.residual_error + scores[1][0]
                else:
                    new_cmi_term = scores[1][0]
                new_corr_term = scores[2][0]
                new_error = self.computeError(
                    new_cmi_term, new_corr_term, use_Rt)

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

    def _scoreFeatureParallel(self, steplist, gamma, sum_cmi, show_progress):
        k = len(steplist)

        S = frozenset(self.idSelected)
        no_S = self.idSet.difference(self.idSelected)  # discarded

        list_ids = np.fromiter(S, dtype=np.int)

        res = []
        for i, id in enumerate(list_ids):
            id = frozenset({id})
            S_no_i = S.difference(id)
            no_S_i = no_S.union(id)

            if sum_cmi:
                target = id
            else:
                target = no_S_i

            for j, t in enumerate(steplist):
                res.append(self.itEstimator.estimateCMI(
                    frozenset({self.id_reward}), target, S_no_i, t=t))
            res.append(self.itEstimator.estimateCH(no_S_i, S_no_i))

        res = map(lambda x: x.result(), tqdm(
            res, leave=False, disable=not show_progress))
        score_mat = np.fromiter(res, np.float64).reshape(k + 1, -1, order='F')

        cmi_wsum = np.einsum('a, ab->b', self.weights[:-1], score_mat[:-1, :])
        new_cond_entropy = self.weights[-1] * score_mat[-1, :]

        sorted_idx = np.argsort(cmi_wsum + new_cond_entropy)

        return list_ids[sorted_idx], cmi_wsum[sorted_idx],  new_cond_entropy[sorted_idx]

    def _scoreFeatureSequential(self, steplist, gamma, sum_cmi, show_progress):
        k = len(steplist)

        S = frozenset(self.idSelected)
        no_S = self.idSet.difference(self.idSelected)  # discarded

        list_ids = np.fromiter(S, dtype=np.int)
        score_mat = np.zeros((k+1, len(list_ids)))

        for i, id in enumerate(tqdm(list_ids, leave=False, disable=not show_progress)):
            id = frozenset({id})
            S_no_i = S.difference(id)
            no_S_i = no_S.union(id)

            if sum_cmi:
                target = id
            else:
                target = no_S_i

            for j, t in enumerate(steplist):
                score_mat[j, i] = self.itEstimator.estimateCMI(
                    frozenset({self.id_reward}), target, S_no_i, t=t)
            score_mat[k, i] = self.itEstimator.estimateCH(no_S_i, S_no_i)

        cmi_wsum = np.einsum('a, ab->b', self.weights[:-1], score_mat[:-1, :])
        new_cond_entropy = self.weights[-1] * score_mat[-1, :]

        sorted_idx = np.argsort(cmi_wsum + new_cond_entropy)

        return list_ids[sorted_idx], cmi_wsum[sorted_idx],  new_cond_entropy[sorted_idx]
