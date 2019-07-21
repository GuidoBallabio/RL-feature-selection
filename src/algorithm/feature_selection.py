import abc

import numpy as np

from src.algorithm.info_theory.it_estimator import (CachingEstimator,
                                                    MPCachingEstimator)
from src.algorithm.utils import FakeFuture, independent_roll


class FeatureSelector(metaclass=abc.ABCMeta):
    def __init__(self, itEstimator, trajectories, discrete=False, nproc=1):
        self.trajectories = trajectories
        self.nproc = nproc
        self.discrete = discrete

        if nproc != 1:
            self.itEstimator = MPCachingEstimator(
                itEstimator, self._get_arrays, nproc=nproc)
        else:
            self.itEstimator = CachingEstimator(itEstimator, self._get_arrays)

        self.seed()
        self._setup()

    def _setup(self):
        self.n_features = self.trajectories[0].shape[1] - 1
        self.id_reward = self.n_features
        self.idSet = frozenset(list(range(self.n_features)))

        # sqrt of max of square of absolute value max
        self.Rmax = np.max([np.max(np.abs(tr[:, self.id_reward])) ** 2
                            for tr in self.trajectories]) ** (1/2)

        self.tot_t = min(len(tr) for tr in self.trajectories)

        self.data_per_traj = np.dstack(
            [tr[:self.tot_t, :] for tr in self.trajectories])
        self.on_mu = None

        self.residual_error = 0
        self.correction_term = 0

    def _prep_data(self, max_t, on_mu):
        if hasattr(self, 't_step_data') and max_t + 1 == self.t_step_data.shape[2] and on_mu == self.on_mu:
            return

        self.itEstimator.cache.clear()

        assert max_t < self.tot_t, f"max timestep {max_t} is not less than the shortest trajectory (len {self.tot_t})"

        self.on_mu = on_mu
        if on_mu:
            stop_len = 1
        else:
            stop_len = self.tot_t - max_t

        shift = np.zeros(self.n_features + 1, dtype=np.int)
        shift[self.id_reward] = -1

        self.t_step_data = []
        for t in range(max_t + 1):
            t_shift = t*shift
            t_step_eps = []
            for ep in self.trajectories:
                t_step_eps.append(independent_roll(
                    ep, t_shift)[: stop_len, :])

            self.t_step_data.append(np.vstack(t_step_eps))

        self.t_step_data = np.dstack(self.t_step_data)

    def _get_arrays(self, ids, t):
        if not isinstance(ids, list):
            ids = list(ids)

        return self.t_step_data[:, ids, t]

    def _generate_steplist(self, k, sampling, freq):
        if sampling == "frequency":
            max_t = (k-1) * freq
            return np.arange(k*freq, step=freq), max_t

        if sampling == "decaying":
            p = np.exp(-np.arange(self.tot_t)/freq) / freq
            p = p/p.sum()
            steplist = np.sort(self.np_random.choice(
                self.tot_t, size=k, replace=False, p=p))
            return steplist, steplist[-1]

        if sampling == "variance":
            variances = np.var(
                self.data_per_traj[:, self.id_reward, :], axis=1)
            most_var = np.argsort(variances)[::-1][:k]
            steplist = np.sort(most_var)
            return steplist, steplist[-1]

        raise NotImplemented

    def _get_weights_by_steplist(self, steplist, gamma, use_Rt):
        k = len(steplist)
        gamma = gamma**2

        weights = np.ones(k + 1) * gamma
        weights[:-1] = weights[:-1] ** steplist

        weights[k] = 1/(1 - gamma) - weights[:-1].sum()

        if use_Rt:
            Rts = np.abs(self.data_per_traj[:, self.id_reward, :]).max(axis=1)
            Rts = Rts[steplist] ** 2

            weights[:-1] *= Rts
            weights[k] *= self.Rmax ** 2

        return weights

    def _prep_all(self, k, gamma, sampling, freq, use_Rt, on_mu):
        self.reset()
        steplist, max_t = self._generate_steplist(k, sampling, freq)
        self.steplist = steplist
        self._prep_data(max_t, on_mu)
        self.weights = self._get_weights_by_steplist(steplist, gamma, use_Rt)

        return steplist

    def scoreFeatures(self, *args, **kwargs):
        if self.nproc != 1:
            return self._scoreFeatureParallel(*args, **kwargs)
        else:
            return self._scoreFeatureSequential(*args, **kwargs)

    def scoreSubset(self, *args, **kwargs):
        if self.nproc != 1:
            return self._scoreSubsetParallel(*args, **kwargs)
        else:
            return self._scoreSubsetSequential(*args, **kwargs)

    def computeError(self, residual=None, correction=None, use_Rt=True):
        if residual is None:
            residual = self.residual_error
        if correction is None:
            correction = self.correction_term
        if use_Rt:
            Rmax = 1
        else:
            Rmax = self.Rmax

        return 2**(1/2) * Rmax * np.sqrt(residual + correction)

    def reset(self):
        self.residual_error = 0
        self.correction_term = 0
        self.weights = None
        self.steplist = None

    def seed(self, seed=None):
        self.np_random = np.random.seed(seed)
        return

    def _scoreSubsetSequential(self, k, gamma, S, sampling="frequency", freq=1, use_Rt=True, on_mu=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        S = frozenset(S)
        no_S = self.idSet.difference(S)

        score = np.zeros(k+1)

        for j, t in enumerate(steplist):
            score[j] = self.itEstimator.estimateCMI(
                frozenset({self.id_reward}), no_S, S, t=t)

        if self.discrete:
            score[k] = self.itEstimator.estimateCH(no_S, S)
        else:
            score[k] = 4

        self.residual_error = score[:-1] @ self.weights[:-1]
        self.correction_term = score[-1] * self.weights[-1]

        return self.computeError(use_Rt=use_Rt)

    def _scoreSubsetParallel(self, k, gamma, S, sampling="frequency", freq=1, use_Rt=True, on_mu=True, show_progress=True):
        steplist = self._prep_all(k, gamma, sampling, freq, use_Rt, on_mu)

        S = frozenset(S)
        no_S = self.idSet.difference(S)

        res = []
        for t in steplist:
            res.append(self.itEstimator.estimateCMI(
                frozenset({self.id_reward}), no_S, S, t=t))

        if self.discrete:
            res.append(self.itEstimator.estimateCH(no_S, S))
        else:
            res.append(FakeFuture(4))

        res = map(lambda x: x.result(), res)
        score = np.fromiter(res, np.float64)

        self.residual_error = score[:-1] @ self.weights[:-1]
        self.correction_term = score[-1] * self.weights[-1]

        return self.computeError(use_Rt=use_Rt)

    @abc.abstractmethod
    def selectOnError(self, k, gamma, max_error, sampling="frequency", freq=1, use_Rt=True, on_mu=True, show_progress=True):
        pass

    @abc.abstractmethod
    def selectNfeatures(self, n, k, gamma, sampling="frequency", freq=1, use_Rt=True, on_mu=True, show_progress=True):
        pass
