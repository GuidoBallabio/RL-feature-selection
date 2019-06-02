import abc
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial


def caching(*args, ignore=False, **kwargs):
    if ignore:
        return method
    return cachedmethod(*args, **kwargs)


class itEstimator(metaclass=abc.ABCMeta):
    def __init__(cached=True):
        self.selector = selector

        self.direct_cmi = False
        self.direct_ch = False
        self.direct_mi = False

        if cached:
            self.cache = {}

    @abc.abstractmethod
    def cmi(X, Y, Z):
        pass

    @abc.abstractmethod
    def cond_entropy(X, Y):
        pass

    @abc.abstractmethod
    def mi(X, Y):
        pass

    @abc.abstractmethod
    def entropy(X):
        pass

    @caching(lambda self: self.cache, ignore=self.__hasattr__('cache'), key=partial(hashkey, 'cmi'))
    def estimateCMI(X_ids, Y_ids, Z_ids, t=0):
        if not self.direct_cmi
           if self.direct_mi:
                return self.estimateMI(X_ids.union(Y_ids), Z_ids, t) - self.estimateMI(X_ids, Z_ids, t)
            else:
                return self.estimateCH(Y_ids, Z_ids, t) - self.estimateCH(Y_ids, Z_ids.union(X_ids), t)

        X = self._get_arrays(X_ids, t)
        Y = self._get_arrays(Y_ids, t)
        Z = self._get_arrays(Z_ids, t)

        return self.cmi(X, Y, Z)

    @caching(lambda self: self.cache, ignore=self.__hasattr__('cache'), key=partial(hashkey, 'ch'))
    def estimateCH(X_ids, Y_ids, t=0):
        if not self.direct_ch:
            return self.estimateH(X_ids.union(Y_ids), t) - self.estimateH(Y_ids, t)

        X = self._get_arrays(X_ids, t)
        Y = self._get_arrays(Y_ids, t)

        return self.cond_entropy(X, Y)

    @caching(lambda self: self.cache, ignore=self.__hasattr__('cache'), key=partial(hashkey, 'mi'))
    def estimateMI(X_ids, Y_ids, t=0):
        if not self.direct_mi:
            return self.estimateH(X_ids, t) + self.estimateH(Y_ids, t) - self.estimateH(X_ids.union(Y_ids), t)

        X = self._get_arrays(X_ids, t)
        Y = self._get_arrays(Y_ids, t)

        return self.mutual_information(X, Y)

    @caching(lambda self: self.cache, ignore=self.__hasattr__('cache'), key=partial(hashkey, 'h'))
    def estimateH(ids, t=0):
        X = self._get_arrays(ids, t)

        return self.entropy(X)
