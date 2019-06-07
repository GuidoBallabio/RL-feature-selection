from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial
from operator import attrgetter
import abc


class ItEstimator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def flags(self):
        pass

    def cmi(self, X, Y, Z):
        raise NotImplemented

    def cond_entropy(self, X, Y):
        raise NotImplemented

    def mi(self, X, Y):
        raise NotImplemented

    def entropy(self, X):
        raise NotImplemented


class cachingEstimator():
    def __init__(self, estimator, selector, cached=True):
        self.selector = selector
        self.estimator = estimator
        self.direct_cmi, self.direct_ch, self.direct_mi = self.estimator.flags()

        if cached:
            self.cache = {}

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'cmi'))
    def estimateCMI(self, X_ids, Y_ids, Z_ids, t=0):
        if not self.direct_cmi:
            if self.direct_mi:
                return self.estimateMI(X_ids.union(Y_ids), Z_ids, t) - self.estimateMI(X_ids, Z_ids, t)
            else:
                return self.estimateCH(Y_ids, Z_ids, t) - self.estimateCH(Y_ids, Z_ids.union(X_ids), t)

        X = self.selector._get_arrays(X_ids, t)
        Y = self.selector._get_arrays(Y_ids, t)
        Z = self.selector._get_arrays(Z_ids, t)

        return self.estimator.cmi(X, Y, Z)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'ch'))
    def estimateCH(self, X_ids, Y_ids, t=0):
        if not self.direct_ch:
            return self.estimateH(X_ids.union(Y_ids), t) - self.estimateH(Y_ids, t)

        X = self.selector._get_arrays(X_ids, t)
        Y = self.selector._get_arrays(Y_ids, t)

        return self.estimator.cond_entropy(X, Y)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'mi'))
    def estimateMI(self, X_ids, Y_ids, t=0):
        if not self.direct_mi:
            return self.estimateH(X_ids, t) + self.estimateH(Y_ids, t) - self.estimateH(X_ids.union(Y_ids), t)

        X = self.selector._get_arrays(X_ids, t)
        Y = self.selector._get_arrays(Y_ids, t)

        return self.estimator.mi(X, Y)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'h'))
    def estimateH(self, ids, t=0):
        X = self.selector._get_arrays(ids, t)

        return self.estimator.entropy(X)
