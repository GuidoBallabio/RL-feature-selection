import abc
import numpy as np
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial


class FeatueSelector(metaclass=abc.ABCMeta):
    def __init__(self, itEstimator, trajectories, nproc=1):
        self.itEstmator = itEstmator
        self.trajectories = trajectories
        self.nproc = nproc
        self.cache = {}
        self._setup()

    def _setup(self):
        self.n_features = self.trajectories[0].shape[1] - 1
        self.id_reward = self.n_features
        self.Rmax = np.abs(np.max([np.max(t[:, self.id_reward]) for t in self.trajectories]))
        self.idMap = set(range(self.n_features))


    def _get_arrays(ids):
        return [t[:, ids] for t in self.trajectories]


    @cachedmethod(lambda self: self.cache, key=partial(hashkey, 'cmi'))
    def estimateCMI(X_ids, Y_ids, Z_ids):
        if not itEstimator.cond_mi:
            return self.estimateMI(X_ids + Y_ids, Z_ids) - self.estimateMI(X_ids, Z_ids)
        
        if not isinstance(X_ids, (list, tuple)):
            X_ids = (X_ids,)
        if not isinstance(Y_ids, (list, tuple)):
            Y_ids = (Y_ids,)
        if not isinstance(Z_ids, (list, tuple)):
            Z_ids = (Z_ids,)
        
        
        X = self._get_arrays(X_ids)
        Y = self._get_arrays(Y_ids)
        Z = self._get_arrays(Z_ids)

        return itEstimator.cmi(X, Y, Z)
    
    @cachedmethod(lambda self: self.cache, key=partial(hashkey, 'ch'))
    def estimateConditionalH(X_ids, Y_ids):
        if not itEstimator.cond_h:
            return self.estimateH(X_ids + Y_ids) - self.estimateH(Y_ids)
        
        if not isinstance(X_ids, (list, tuple)):
            X_ids = (X_ids,)
        if not isinstance(Y_ids, (list, tuple)):
            Y_ids = (Y_ids,)        
        
        X = self._get_arrays(X_ids)
        Y = self._get_arrays(Y_ids)

        return itEstimator.cond_entropy(X, Y)

    @cachedmethod(lambda self: self.cache, key=partial(hashkey, 'mi'))
    def estimateMI(X_ids, Y_ids):
        if not isinstance(X_ids, (list, tuple)):
            X_ids = (X_ids,)
        if not isinstance(Y_ids, (list, tuple)):
            Y_ids = (Y_ids,)
        
        
        X = self._get_arrays(X_ids)
        Y = self._get_arrays(Y_ids)
        
        return itEstimator.mutual_information(X, Y)
    
    @cachedmethod(lambda self: self.cache, key=partial(hashkey, 'h'))
    def estimateH(ids):
        if not isinstance(ids, (list, tuple)):
            ids = (ids,)
        
        X = self._get_arrays(ids)
        
        return itEstimator.entropy(X)

    def scoreFeatures(self):
        if self.nproc > 1:
            return self._scoreFeatureParallel()
        else:
            return self._scoreFeatureSequential()

    def computeError(self):
        return 2**(1/2) * self.Rmax * self.res

    @abc.abstractmethod
    def selectKFeatures(self, num_features):
        pass

    @abc.abstractmethod
    def selectOnError(self, max_error):
        pass

    @abc.abstractmethod
    def selectOnDeltaScore(self, eps):
        pass

    @abc.abstractmethod
    def selectOnFeatureScore(self, threshold):
        pass
