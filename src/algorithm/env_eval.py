from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.autonotebook import tqdm

from src.algorithm.backward_feature_selection import BackwardFeatureSelector
from src.algorithm.forward_feature_selection import ForwardFeatureSelector
from src.algorithm.utils import FakeFuture, episodes_with_len
from src.wenvs import WrapperEnv


class EnvEval:
    def __init__(self, env, itEstimator, estimatorQ, continuous_state=False, continuous_actions=False, nproc=None, backward=True):
        self.env = env
        self.wenv = WrapperEnv(env, continuous_state=continuous_state,
                               continuous_actions=continuous_actions)
        self.itEstimator = itEstimator
        self.estimatorQ = estimatorQ
        self.continuous_state = continuous_state
        self.continuous_actions = continuous_actions
        self.nproc = nproc
        self.backward = backward

        self.discrete = not (continuous_actions or continuous_state)
        self.trajectories = None
        self.fs = None
        self.subsets = {}
        self.mean_return = None

        if self.nproc != 1:
            self.proc_pool = ProcessPoolExecutor(max_workers=nproc)

    def fit_baseline(self, k, gamma, n_trajectories, policy=None, iter_max=50, stop_at_len=True, est_kwargs={}, fs_kwargs={}):
        """To be called before other methods as a reset.

            **kwargs are passed to selectNFeatures
        """
        self.k = k
        self.gamma = gamma
        self.n_trajectories = n_trajectories
        self.policy = policy
        self.iter_max = iter_max
        self.est_kwargs = est_kwargs
        self.fs_kwargs = fs_kwargs

        self.fs = None
        self.subsets = {}

        np.random.seed(0)
        self.wenv.seed(0)
        self.trajectories = episodes_with_len(
            self.wenv, n_trajectories, k, policy=policy, stop_at_len=stop_at_len)

        est = self.itEstimator()

        if self.backward:
            self.fs = BackwardFeatureSelector(est, self.trajectories,
                                              discrete=self.discrete, nproc=self.nproc)
        else:
            self.fs = ForwardFeatureSelector(est, self.trajectories,
                                             discrete=self.discrete, nproc=self.nproc)

        err = self.fs.scoreSubset(self.k, self.gamma, self.fs.idSet)
        Q = self._fitQ(None)
        self.subsets.update({self.fs.idSet: (err, Q)})

    def _fitQ(self, S):
        if S is not None and not S:
            if not self.mean_return:
                self.mean_return = np.mean(
                    [np.polyval(t[:, -1], self.gamma) for t in self.trajectories])
            return (lambda x: self.mean_return)
        return self.estimatorQ(self.gamma).fit(self.trajectories, S, iter_max=self.iter_max, **self.est_kwargs)

    def _fitQ_parallel(self, S):
        if S is not None and not S:
            if not self.mean_return:
                self.mean_return = np.mean(
                    [np.polyval(t[:, -1], self.gamma) for t in self.trajectories])
            return FakeFuture(lambda x: self.mean_return)
        Q = self.estimatorQ(self.gamma)
        return self.proc_pool.submit(Q.fit, self.trajectories, S, iter_max=self.iter_max, **self.est_kwargs)

    def _sequential_try_all(self):
        for S, err in self.fs.try_remove_all(self.k, self.gamma, **self.fs_kwargs):
            Q = self._fitQ(S)
            self.subsets.update({frozenset(S): (err, Q)})

    def _parallel_try_all(self):
        res = []
        for S, err in self.fs.try_all(self.k, self.gamma, **self.fs_kwargs):
            Q = self._fitQ_parallel(S)
            res.append((frozenset(S), err, Q))

        for e in tqdm(res, disable=not self.fs_kwargs.get('show_progress', True)):
            self.subsets.update({e[0]: (e[1], e[2].result())})

    def try_all(self):
        if self.nproc != 1:
            self._parallel_try_all()
        else:
            self._sequential_try_all()

    def try_subset(self, S):
        err = self.fs.scoreSubset(self.k, self.gamma, S)
        Q = self._fitQ(S)
        self.subsets.update({frozenset(S): (err, Q)})
        return np.linalg.norm(self.subsets[self.fs.idSet] - Q, 2), err

    def norm_diff(self):
        if len(self.subsets) == 1:
            return [0]

        _, baseQ = self.subsets[self.fs.idSet]
        self.mu = self.fs.data_per_traj[0, :-1, :].T
        baseQ = baseQ(self.mu)

        res = []
        for S in self.subsets:
            err, Q = self.subsets[S]
            mu = self.mu[:, list(S)]
            Q = Q(mu)
            norm = np.linalg.norm(baseQ - Q, 2)
            res.append((S, norm, err))

        res = sorted(res, key=lambda x: len(x[0]), reverse=True)

        return res

    def run(self, k, gamma, n_trajectories, policy=None, iter_max=50, stop_at_len=True, est_kwargs={}, **fs_kwargs):
        self.fit_baseline(k, gamma, n_trajectories, policy=policy, iter_max=iter_max, stop_at_len=stop_at_len,
                          est_kwargs=est_kwargs, fs_kwargs=fs_kwargs)
        self.try_all()
        return self.norm_diff()
