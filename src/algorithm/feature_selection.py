import abc

import numpy as np

from src.algorithm.info_theory.it_estimator import CachingEstimator
from src.algorithm.utils import independent_roll


class FeatureSelector(metaclass=abc.ABCMeta):
    def __init__(self, itEstimator, trajectories, nproc=1):
        self.trajectories = trajectories
        self.nproc = nproc
        self.itEstimator = CachingEstimator(itEstimator, self, nproc == 1)

        self._setup()

    def _setup(self):
        self.n_features = self.trajectories[0].shape[1] - 1
        self.id_reward = self.n_features
        self.idSet = frozenset(list(range(self.n_features)))

        self.Rmax = np.max([np.max(np.abs(tr[:, self.id_reward]))
                            for tr in self.trajectories])

        self.max_t = min(len(tr) for tr in self.trajectories)

        self.residual_error = 0
        self.correction_term = 0

    def _prep_data(self, max_t):
        if hasattr(self, 't_step_data') and max_t + 1 == self.t_step_data.shape[2]:
            return

        assert max_t < self.max_t, f"max timestep {max_t} is not less than the shortest trajectory (len {self.max_t})"

        shift = np.zeros(self.n_features + 1, dtype=np.int)
        shift[self.id_reward] = -1

        self.t_step_data = []
        for t in range(max_t + 1):
            t_shift = t*shift
            t_step_eps = []
            for ep in self.trajectories:
                t_step_eps.append(independent_roll(ep, t_shift)[0, :])

            self.t_step_data.append(np.vstack(t_step_eps))

        self.t_step_data = np.dstack(self.t_step_data)

    def _get_arrays(self, ids, t):
        if not isinstance(ids, list):
            ids = list(ids)

        return self.t_step_data[:, ids, t]

    def scoreFeatures(self, *args, **kwargs):
        if self.nproc > 1:
            return self._scoreFeatureParallel(*args, **kwargs)
        else:
            return self._scoreFeatureSequential(*args, **kwargs)

    def _generate_steplist(self, k, sampling, freq):
        if sampling == "frequency":
            max_t = (k-1) * freq
            return np.arange(k*freq, step=freq), max_t

        if sampling == "decaying":
            p = np.exp(-np.arange(self.max_t)/freq) / freq
            p = p/p.sum()
            steplist = np.sort(np.random.choice(
                self.max_t, size=k, replace=False, p=p))
            return steplist, steplist[-1]

    def _get_weights_by_steplist(self, steplist, gamma, use_Rt):
        k = len(steplist)
        gamma = gamma**2

        weights = np.ones(k + 1) * gamma
        weights[:-1] = weights[:-1] ** steplist

        weights[k] = 1/(1 - gamma) - weights[:-1].sum()

        if use_Rt:
            Rts = np.abs(self.t_step_data[:, self.id_reward, :]).max(axis=0)
            Rts = Rts[steplist] ** 2

            weights[:-1] *= Rts
            weights[k] *= self.Rmax ** 2

        return weights

    def _prep_all(self, k, gamma, sampling, freq, use_Rt):
        self.reset()
        steplist, max_t = self._generate_steplist(k, sampling, freq)
        self._prep_data(max_t)
        self.weights = self._get_weights_by_steplist(steplist, gamma, use_Rt)

        return steplist

    def computeError(self, residual=None, correction=None, use_Rt=True):
        if residual is None:
            residual = self.residual_error
        if correction is None:
            correction = self.correction_term
        if use_Rt:
            Rmax = 1
        else:
            use_Rt = self.Rmax

        return 2**(1/2) * Rmax * np.sqrt(residual + correction)

    def reset(self):
        self.residual_error = 0
        self.correction_term = 0
        self.weights = None

    def scoreSubset(self, k, gamma, S, sampling="frequency", freq=1, use_Rt=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt)

        S = frozenset(S)
        no_S = self.idSet.difference(S)

        score = np.zeros(k+1)

        for j, t in enumerate(steplist):
            score[j] = self.itEstimator.estimateCMI(
                frozenset({self.id_reward}), no_S, S, t=t)
        score[k] = self.itEstimator.estimateCH(no_S, S)

        self.residual_error = score[:-1] @ self.weights[:-1]
        self.correction_term = score[-1] * self.weights[-1]

        return self.computeError(use_Rt=use_Rt)

    @abc.abstractmethod
    def selectOnError(self, k, gamma, max_error, sampling="frequency", freq=1, sum_cmi=True, use_Rt=True, show_progress=True):
        pass

    @abc.abstractmethod
    def selectNfeatures(self, n, k, gamma, sampling="frequency", freq=1, sum_cmi=True, use_Rt=True, show_progress=True):
        pass
