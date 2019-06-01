import abc
import numpy as np

class FeatueSelector(metaclass=abc.ABCMeta):
    def __init__(self, itEstimator, trajectories, nproc=1):
        self.itEstimator = itEstimator
        self.trajectories = trajectories
        self.nproc = nproc
        self._setup()

    def _setup(self):
        self.n_features = self.trajectories[0].shape[1] - 1
        self.idx_reward = self.n_features
        self.all_t = slice(None)
        self.Rmax = np.abs(np.max([np.max(t[:, self.id_reward]) for t in self.trajectories]))
        self.idSet = set(range(self.n_features))
        self.ref = 0
        
        assert not self.nproc == 1 or self.itEstimator.__hasattr__('cached'), "Or multiprocessing or memoization"

    def _get_arrays(ids):
        if not isinstance(ids, (list, tuple)):
            ids = (ids,)
        return [t[self.all_t, ids] for t in self.trajectories]

    def scoreFeatures(self):
        if self.nproc > 1:
            return self._scoreFeatureParallel()
        else:
            return self._scoreFeatureSequential()

    def _get_weights(k):
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
