import abc
from enum import Enum

import numpy as np

from src.algorithm.info_theory.it_estimator import CachingEstimator
from src.algorithm.utils import independent_roll

Bound = Enum("Bound", ['cmi', 'cmi_sqrt', 'entropy'])


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

        self.Rmax = np.max([np.max(np.abs(t[:, self.id_reward]))
                            for t in self.trajectories])
        self.residual_error = 0

        self.max_k = min(len(t) for t in self.trajectories)

        self.residual_error = 0
        self.correction_term = 0

    def _prep_data(self, k):
        if hasattr(self, 'k_step_data') and k == self.k_step_data.shape[2]:
            return

        shift = np.zeros(self.n_features + 1, dtype=np.int)
        shift[self.id_reward] = -1

        self.k_step_data = []
        for t in range(k):
            t_shift = t*shift
            t_step_eps = []
            for ep in self.trajectories:
                t_step_eps.append(independent_roll(ep, t_shift)[0, :])

            self.k_step_data.append(np.vstack(t_step_eps))

        self.k_step_data = np.dstack(self.k_step_data)

    def _get_arrays(self, ids, t):
        if not isinstance(ids, list):
            ids = list(ids)

        return self.k_step_data[:, ids, t]

    def scoreFeatures(self, *args, **kwargs):
        if self.nproc > 1:
            return self._scoreFeatureParallel(*args, **kwargs)
        else:
            return self._scoreFeatureSequential(*args, **kwargs)

    def _get_weights(self, k, gamma, bound):
        if bound is not Bound.cmi_sqrt:
            gamma = gamma**2

        weights = np.ones(k+1) * gamma
        weights = weights ** np.arange(k+1)
            
        if bound is Bound.entropy:
            weights[k] = 1/(1 - gamma)
        else:
            weights[k] /= 1 - gamma

        return weights

    def _funOfBound(self, bound):
        if bound is Bound.cmi_sqrt:
            def fun_t(no_S_i, S_no_i, t): return np.sqrt(self.itEstimator.estimateCMI(
                frozenset({self.id_reward}), no_S_i, S_no_i, t=t))

            def fun_k(no_S_i, S_no_i): return np.sqrt(
                self.itEstimator.estimateCH(no_S_i, S_no_i))
        elif bound is Bound.cmi:
            def fun_t(no_S_i, S_no_i, t): return self.itEstimator.estimateCMI(
                frozenset({self.id_reward}), no_S_i, S_no_i, t=t)

            def fun_k(no_S_i, S_no_i): return self.itEstimator.estimateCH(
                no_S_i, S_no_i)
        elif bound is Bound.entropy:
            def fun_t(no_S_i, S_no_i, t): return -self.itEstimator.estimateCH(no_S_i,
                                        frozenset({self.id_reward}).union(S_no_i), t=t)

            def fun_k(no_S_i, S_no_i): return self.itEstimator.estimateCH(
                no_S_i, S_no_i)

        return fun_t, fun_k

    def computeError(self, bound=Bound.cmi, residual=None, correction=None):
        if residual is None:
            residual = self.residual_error
        if correction is None:
            correction = self.correction_term

        if bound is Bound.cmi_sqrt:
            return 2**(1/2) * self.Rmax * residual
        return 2**(1/2) * self.Rmax * np.sqrt(residual + correction)

    def reset(self):
        self.residual_error = 0
        self.correction_term = 0

    @abc.abstractmethod
    def selectOnError(self, k, gamma, max_error, bound=Bound.entropy, show_progress=True):
        pass

    @abc.abstractmethod
    def selectNfeatures(self, n, k, gamma, bound=Bound.cmi, show_progress=True):
        pass
