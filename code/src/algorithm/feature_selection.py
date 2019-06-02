import abc
import numpy as np
from src.algorithm.utils import independent_roll


class FeatueSelector(metaclass=abc.ABCMeta):
    def __init__(self, itEstimator, trajectories, nproc=1):
        self.itEstimator = itEstimator
        self.trajectories = trajectories
        self.nproc = nproc
        self._setup()
        self._prep_data()

    def _setup(self):
        self.n_features = self.trajectories[0].shape[1] - 1
        self.idx_reward = self.n_features
        self.idSet = set(list(range(self.n_features)))

        self.Rmax = np.abs(
            np.max([np.max(t[:, self.id_reward]) for t in self.trajectories]))
        self.ref = 0

        assert not self.nproc == 1 or self.itEstimator.__hasattr__(
            'cached'), "Or multiprocessing or memoization"

    def _prep_data(self, k):
        # in alternative store only one copy and make a new copy on each call shifted by t (nosense?)
        shift = np.zeros(self.n_features + 1)
        shift[self.idx_reward] = -1

        self.k_step_data = []
        for t in range(k-1):
            t_shift = t*shift
            t_step_eps = []
            for ep in self.trajectories:
                t_step_eps.append(independent_roll(ep, t_shift)[:-k, :])
                
            self.k_step_data.append(np.vstack(k_step_ep))

        self.k_step_data = np.dstack(self.k_step_data)

    def _get_arrays(self, ids, t):
        if not isinstance(ids, list):
            ids = list(ids)

        return self.k_step_data[:, ids, t]

    def scoreFeatures(self):
        if self.nproc > 1:
            return self._scoreFeatureParallel()
        else:
            return self._scoreFeatureSequential()

    def _get_weights(self, k):
        weights = np.ones(k+1)
        for t in range(1, k+1):
            weights[t:] *= gamma
        weights[k] /= 1 - gamma

        return weights

    def computeError(self):
        return 2**(1/2) * self.Rmax * self.res

    @abc.abstractmethod
    def selectOnError(self, max_error):
        pass
