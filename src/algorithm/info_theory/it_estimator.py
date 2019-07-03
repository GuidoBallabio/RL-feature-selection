import abc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from operator import attrgetter

from cachetools import cachedmethod
from cachetools.keys import hashkey


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


class CachingEstimator():
    def __init__(self, estimator, data_getter, cached=True):
        self.data_getter = data_getter
        self.estimator = estimator
        self.direct_cmi, self.direct_ch, self.direct_mi = self.estimator.flags()

        if cached:
            self.cache = {}

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'cmi'))
    def estimateCMI(self, X_ids, Y_ids, Z_ids, t=0):
        if not X_ids or not Y_ids:
            return 0
        if not Z_ids:
            return self.estimateMI(X_ids, Y_ids, t)

        if not self.direct_cmi:
            if self.direct_mi:
                return self.estimateMI(X_ids, Y_ids.union(Z_ids), t) - self.estimateMI(X_ids, Z_ids, t)
            else:
                return self.estimateCH(Y_ids, Z_ids, t) - self.estimateCH(Y_ids, Z_ids.union(X_ids), t)

        X = self.data_getter(X_ids, t)
        Y = self.data_getter(Y_ids, t)
        Z = self.data_getter(Z_ids, t)

        return self.estimator.cmi(X, Y, Z)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'ch'))
    def estimateCH(self, X_ids, Y_ids, t=0):
        if not X_ids:
            return 0
        if not Y_ids:
            return self.estimateH(X_ids, t)

        if not self.direct_ch:
            return self.estimateH(X_ids.union(Y_ids), t) - self.estimateH(Y_ids, t)

        X = self.data_getter(X_ids, t)
        Y = self.data_getter(Y_ids, t)

        return self.estimator.cond_entropy(X, Y)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'mi'))
    def estimateMI(self, X_ids, Y_ids, t=0):
        if not X_ids or not Y_ids:
            return 0

        if not self.direct_mi:
            return self.estimateH(X_ids, t) + self.estimateH(Y_ids, t) - self.estimateH(X_ids.union(Y_ids), t)

        X = self.data_getter(X_ids, t)
        Y = self.data_getter(Y_ids, t)

        return self.estimator.mi(X, Y)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'h'))
    def estimateH(self, ids, t=0):
        if not ids:
            return 0

        X = self.data_getter(ids, t)

        return self.estimator.entropy(X)


def zero():
    return 0


def diff(x, y):
    return x.result() - y.result()


def sumdiff(x, y, z):
    return x.result() + y.result() - z.result()


class MPCachingEstimator():
    def __init__(self, estimator, data_getter, nproc, cached=True):
        self.data_getter = data_getter
        self.estimator = estimator
        self.direct_cmi, self.direct_ch, self.direct_mi = self.estimator.flags()
        self.nproc = nproc

        if cached:
            self.cache = {}

        self.proc_pool = ProcessPoolExecutor(max_workers=nproc)
        self.thread_pool = ThreadPoolExecutor()

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'cmi'))
    def estimateCMI(self, X_ids, Y_ids, Z_ids, t=0):
        if not X_ids or not Y_ids:
            return self.thread_pool.submit(zero)
        if not Z_ids:
            return self.estimateMI(X_ids, Y_ids, t)

        if not self.direct_cmi:
            if self.direct_mi:
                return self._defer_diff(self.estimateMI(X_ids, Y_ids.union(Z_ids), t), self.estimateMI(X_ids, Z_ids, t))
            else:
                return self._defer_diff(self.estimateCH(Y_ids, Z_ids, t), self.estimateCH(Y_ids, Z_ids.union(X_ids), t))

        X = self.data_getter(X_ids, t)
        Y = self.data_getter(Y_ids, t)
        Z = self.data_getter(Z_ids, t)

        return self.proc_pool.submit(self.estimator.cmi, X, Y, Z)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'ch'))
    def estimateCH(self, X_ids, Y_ids, t=0):
        if not X_ids:
            return self.thread_pool.submit(zero)
        if not Y_ids:
            return self.estimateH(X_ids, t)

        if not self.direct_ch:
            return self._defer_diff(self.estimateH(X_ids.union(Y_ids), t), self.estimateH(Y_ids, t))

        X = self.data_getter(X_ids, t)
        Y = self.data_getter(Y_ids, t)

        return self.proc_pool.submit(self.estimator.cond_entropy, X, Y)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'mi'))
    def estimateMI(self, X_ids, Y_ids, t=0):
        if not X_ids or not Y_ids:
            return self.thread_pool.submit(zero)

        if not self.direct_mi:
            return self._defer_sumdiff(self.estimateH(X_ids, t), self.estimateH(Y_ids, t), self.estimateH(X_ids.union(Y_ids), t))

        X = self.data_getter(X_ids, t)
        Y = self.data_getter(Y_ids, t)

        return self.proc_pool.submit(self.estimator.mi, X, Y)

    @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'h'))
    def estimateH(self, ids, t=0):
        if not ids:
            return self.thread_pool.submit(zero)

        X = self.data_getter(ids, t)

        return self.proc_pool.submit(self.estimator.entropy, X)

    def _defer_diff(self, a, b):
        return self.thread_pool.submit(diff, a, b)

    def _defer_sumdiff(self, a, b, c):
        return self.thread_pool.submit(sumdiff, a, b, c)
