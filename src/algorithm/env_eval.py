import numpy as np

from src.algorithm.backward_feature_selection import BackwardFeatureSelector
from src.algorithm.utils import episodes_with_len
from src.wenvs import WrapperEnv


class EnvEval:
    def __init__(self, env, itEstimator, estimatorQ, continuous_state=False, continuous_actions=False, nproc=None):
        self.env = env
        self.wenv = WrapperEnv(env, continuous_state=continuous_state,
                               continuous_actions=continuous_actions)
        self.itEstimator = itEstimator
        self.estimatorQ = estimatorQ
        self.continuous_state = continuous_state
        self.continuous_actions = continuous_actions
        self.nproc = nproc

        self.discrete = not (continuous_actions or continuous_state)
        self.trajectories = None
        self.fs = None
        self.subsets = {}
        self.mean_return = None

    def fit_baseline(self, k, gamma, n_trajectories, policy=None, iter_max=50, **kwargs):
        """To be called before other methods as a reset.

            **kwargs are passed to selectNFeatures
        """
        self.k = k
        self.gamma = gamma
        self.n_trajectories = n_trajectories
        self.policy = policy
        self.iter_max = iter_max
        self.kwargs = kwargs

        self.fs = None
        self.subsets = {}

        np.random.seed(0)
        self.wenv.seed(0)
        self.trajectories = episodes_with_len(
            self.wenv, n_trajectories, k, policy=policy)

        est = self.itEstimator()
        self.fs = BackwardFeatureSelector(est, self.trajectories,
                                          discrete=self.discrete, nproc=self.nproc)

        Q = self._fitQ(None)
        self.subsets.update({self.fs.idSet: (0.0, Q)})

    def _fitQ(self, S):
        if S is not None and not S:
            if not self.mean_return:
                self.mean_return = np.mean(
                    [np.polyval(t[:, -1], self.gamma) for t in self.trajectories])
            return (lambda x: self.mean_return)
        return self.estimatorQ(self.gamma).fit(self.trajectories, S, iter_max=self.iter_max)

    def try_all(self):
        for S, err in self.fs.try_remove_all(self.k, self.gamma, **self.kwargs):
            Q = self._fitQ(S)
            self.subsets.update({frozenset(S): (err, Q)})

    def try_subset(self, S):
        err = self.fs.scoreSubset(self.k, self.gamma, S)
        Q = self._fitQ(S)
        self.subsets.update({frozenset(S): (err, Q)})
        return np.linalg.norm(self.subsets[self.fs.idSet] - Q, 2), err

    def norm_diff(self):
        if len(self.subsets) == 1:
            return [0]

        _, baseQ = self.subsets[self.fs.idSet]
        self.mu = self.fs.t_step_data[:, :-1, 0]
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

    def run(self, k, gamma, n_trajectories, policy=None, iter_max=50, **kwargs):
        self.fit_baseline(k, gamma, n_trajectories, **kwargs)
        self.try_all()
        return self.norm_diff()
