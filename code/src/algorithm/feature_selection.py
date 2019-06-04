import abc
import numpy as np
from src.algorithm.utils import independent_roll
from src.algorithm.info_theory.it_estimator import cachingEstimator


class FeatureSelector(metaclass=abc.ABCMeta):
    def __init__(self, itEstimator, trajectories, nproc=1):
        self.trajectories = trajectories
        self.nproc = nproc
        self.itEstimator = cachingEstimator(itEstimator, self, nproc == 1)

        self._setup()

    def _setup(self):
        self.n_features = self.trajectories[0].shape[1] - 1
        self.id_reward = self.n_features
        self.idSet = frozenset(list(range(self.n_features)))

        self.Rmax = np.abs(
            np.max([np.max(t[:, self.id_reward]) for t in self.trajectories]))
        self.residual_error = 0
        
        self.max_k = min(len(t) for t in self.trajectories)

    def _prep_data(self, k):
        # in alternative store only one copy and make a new copy on each call shifted by t (nosense?)
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

    def _get_weights(self, k, gamma):
        weights = np.ones(k+1)
        for t in range(1, k+1):
            weights[t:] *= gamma
        weights[k] /= 1 - gamma

        return weights

    def computeError(self):
        return 2**(1/2) * self.Rmax * self.residual_error

    @abc.abstractmethod
    def selectOnError(self, max_error):
        pass
