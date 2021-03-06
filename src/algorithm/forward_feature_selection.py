import numpy as np
from tqdm.autonotebook import tqdm

from src.algorithm.feature_selection import FeatureSelector


class ForwardFeatureSelector(FeatureSelector):
    def __init__(self, itEstimator, trajectories, discrete=False, nproc=None):
        super().__init__(itEstimator, trajectories, discrete, nproc)

        self.forward = True
        self.idSelected = set()

    def reset(self):
        super().reset()
        self.idSelected = set()

    def selectNfeatures(self, n, k, gamma, sampling="frequency", freq=1, use_Rt=True, on_mu=True, sum_cmi=True, show_progress=True):
        assert 0 <= n <= self.n_features, f"Features to be selected {n} must be less than the" \
            f"total number of feature: {self.n_features}"

        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        if sum_cmi:
            self.scores = np.zeros(k+1)

        for i in tqdm(range(self.n_features - n), disable=not show_progress):
            scores = self.scoreFeatures(
                steplist, gamma, sum_cmi, show_progress=show_progress)

            self.idSelected.add(scores[0][0])

            if sum_cmi:
                self.scores += scores[3][:, 0]
                self.residual_error = np.sqrt(
                    self.scores[:-1]) @ self.weights[:-1]
                self.correction_term = self.scores[-1] * self.weights[-1]
            else:
                self.residual_error = scores[1][0]
                self.correction_term = scores[2][0]
            error = self.computeError(use_Rt=use_Rt)

        return self.idSelected.copy(), error

    def try_all(self, k, gamma, all_scores=False, max_n=None, sampling="frequency", freq=1, use_Rt=True, on_mu=True, sum_cmi=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        if max_n is None:
            max_n = self.n_features
        if sum_cmi:
            self.scores = np.zeros(k+1)

        for i in tqdm(range(max_n), disable=not show_progress):
            scores = self.scoreFeatures(
                steplist, gamma, sum_cmi, show_progress=show_progress)

            self.idSelected.add(scores[0][0])

            if sum_cmi:
                self.scores += scores[3][:, 0]
                self.residual_error = np.sqrt(
                    self.scores[:-1]) @ self.weights[:-1]
                self.correction_term = self.scores[-1] * self.weights[-1]
            else:
                self.residual_error = scores[1][0]
                self.correction_term = scores[2][0]
            error = self.computeError(use_Rt=use_Rt)

            if all_scores:
                yield self.idSelected.copy(), error, scores
            else:
                yield self.idSelected.copy(), error

    def selectOnError(self, k, gamma, max_error, sampling="frequency", freq=1, use_Rt=True, on_mu=True, sum_cmi=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        error = 0.0
        if sum_cmi:
            self.scores = np.zeros(k+1)

        with tqdm(total=100, disable=not show_progress) as pbar:  # tqdm
            while error <= max_error and len(self.idSelected) < self.n_features:
                scores = self.scoreFeatures(
                    steplist, gamma, sum_cmi, show_progress=show_progress)

                if sum_cmi:
                    self.scores += scores[3][:, 0]
                    new_cmi_term = np.sqrt(
                        self.scores[:-1]) @ self.weights[:-1]
                    new_corr_term = self.scores[-1] * self.weights[-1]
                else:
                    new_cmi_term = scores[1][0]
                    new_corr_term = scores[2][0]
                new_error = self.computeError(
                    new_cmi_term, new_corr_term, use_Rt)

                perc_of_max = int(100*new_error/max_error)  # tqdm
                pbar.update(min(perc_of_max, pbar.total) - pbar.n)  # tqdm

                if new_error >= max_error:
                    break

                self.idSelected.add(scores[0][0])
                self.residual_error = new_cmi_term
                self.correction_term = new_corr_term
                error = new_error

            pbar.update(pbar.total - pbar.n)  # tqdm

        return self.idSelected.copy(), error
