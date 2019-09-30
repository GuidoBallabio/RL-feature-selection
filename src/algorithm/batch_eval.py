import pickle
import tempfile

import numpy as np
import ray
from tqdm.autonotebook import tqdm

from src.algorithm.backward_feature_selection import BackwardFeatureSelector
from src.algorithm.forward_feature_selection import ForwardFeatureSelector
from src.algorithm.info_theory.combo_estimators import DiscreteEntropyEstimator
from src.algorithm.utils import env_name, episodes_with_len
from src.policy_eval.fqi import QfunctionFQI
from src.wenvs import WrapperEnv


def _mean_value(gamma, traj):
    return np.mean([np.polyval(t[:, -1], gamma) for t in traj])


@ray.remote
def _evalQ(q, gamma, traj, S, mu, est_kwargs):
    if S is not None and not S:
        return _mean_value(gamma, traj)
    Q = q(gamma).fit(traj, S, **est_kwargs)
    return Q(mu[:, list(S)])


class BatchEval:
    def __init__(self, l_env, l_estimator, l_estimatorQ, l_n, l_k, l_gamma,
                 nproc=None, backward=True, **ray_kwargs):
        """Batch evalutaion of bound on envs.

            l_n, l_k, l_gamma have to be sorted in ascending order
        """
        self.l_env = l_env
        self.l_estimator = l_estimator
        self.l_estimatorQ = l_estimatorQ
        self.l_n = l_n
        self.l_k = l_k
        self.l_gamma = l_gamma
        self.nproc = nproc
        self.backward = backward

        if backward:
            self.FS = BackwardFeatureSelector
        else:
            self.FS = ForwardFeatureSelector

        self.traj_id = None
        self.bounds = {}

        if self.nproc != 1:
            ray.init(ignore_reinit_error=True, **ray_kwargs)

    def run(self, verbose=1, filename=None, stop_at_len=True, est_kwargs={}, fs_kwargs={}):
        """To be called before other methods as a reset.

            **fs_kwargs are passed to fs.try_all
            **est_kwargs are passed to Q.fit
        """
        self.est_kwargs = est_kwargs
        self.fs_kwargs = fs_kwargs

        for f_env in self.l_env:
            env, cs, ca, policy = f_env()
            wenv = WrapperEnv(env, continuous_state=cs, continuous_actions=ca)
            discrete = not (cs or ca)
            # create n traj long k (move in l_k? probelm with policy and stop?)
            self.traj_id = None
            db = {}
            np.random.seed(0)
            wenv.seed(0)
            traj_all = episodes_with_len(
                wenv, self.l_n[-1], self.l_k[-1], policy=policy, stop_at_len=stop_at_len)
            for n in self.l_n:
                traj = traj_all[:n]
                for est in self.l_estimator:
                    if discrete != (est is DiscreteEntropyEstimator):
                        continue
                    fs = self.FS(est(), traj, discrete=discrete,
                                 nproc=self.nproc)
                    for k in self.l_k:
                        for gamma in self.l_gamma:
                            key = (env_name(env), est.__name__, n, k, gamma)
                            if verbose == 2:
                                print('\n', key, flush=True)
                            res = [(set(fs.idSet),
                                    fs.scoreSubset(k, gamma, fs.idSet))]
                            res += list(fs.try_all(k, gamma))
                            db[key] = res
                if verbose == 2:
                    self._dump_dict(db, filename)

            if self.nproc != 1:
                self.traj_id = ray.put(traj_all)

            # move inside l_k? (especially with QasSum)
            # find unique gs
            unique_gs = self._find_unique_gs(db)
            # prepare mu
            mu = fs.data_per_traj[0, :-1, :].T

            if verbose == 2:
                print("Starting Q eval", flush=True)

            for q in self.l_estimatorQ:
                q_name = q.__name__
                # policy_eval
                if self.nproc != 1:
                    db_q = self._fit_all_parallel(unique_gs, mu, q, traj_all)
                else:
                    db_q = self._fit_all(unique_gs, mu, q, traj_all)

                self._norm_diff(q_name, db, db_q, fs.idSet)
                if verbose >= 1:
                    self._dump_dict(self.bounds, filename)
                db_q.clear()

        return self.bounds

    def _find_unique_gs(self, db):
        """Find unique (gamma, Sets) tuples.
        """
        unique_gs = []
        for key, l_bound in db.items():
            gamma = key[-1]
            unique_gs.extend([(gamma, frozenset(S)) for S, _ in l_bound])
        unique_gs = list(set(unique_gs))
        unique_gs.sort(key=lambda x: (x[0], -len(x[1])))
        return unique_gs

    def _fit_all(self, unique_gs, mu, q, traj_all):
        db_q = {}
        for gamma, S in tqdm(unique_gs, disable=not self.fs_kwargs.get('show_progress', True)):
            Q = self._fitQ(q, gamma, traj_all, S)
            db_q[(gamma, S)] = Q(mu[:, list(S)])  # arrays long l_n[-1]
        return db_q

    def _fit_all_parallel(self, unique_gs, mu, q, traj_all):
        db_q = {}
        res = []
        mu_id = ray.put(mu)
        for gamma, S in unique_gs:
            Qest = _evalQ.remote(q, gamma, self.traj_id, S, mu_id,
                                 self.est_kwargs)
            res.append((gamma, S, Qest))

        for x in tqdm(res, disable=not self.fs_kwargs.get('show_progress', True)):
            gamma, S, Qest = x
            db_q[(gamma, S)] = ray.get(Qest)
            # arrays long l_n[-1]
        return db_q

    def _fitQ(self, q, gamma, traj, S):
        if S is not None and not S:
            return _mean_value(gamma, traj)
        return q(gamma).fit(traj, S, **self.est_kwargs)

    def _dump_dict(self, d, file):
        if file is None:
            file = tempfile.mktemp() + ".pkl"
            print("Dict dumped in " + file)
        with open(file, 'wb') as fp:
            pickle.dump(d, fp, pickle.HIGHEST_PROTOCOL)

    def _norm_diff(self, q_name, db, db_q, idSet):
        db_diff = {}

        for k, Q in db_q.items():
            gamma, S = k
            baseQ = db_q[(gamma, idSet)]
            norm = np.linalg.norm(baseQ - Q, 2)
            db_diff[k] = norm

        for key, l_bound in db.items():
            gamma = key[-1]
            key_res = [(S, db_diff[(gamma, frozenset(S))], err)
                       for S, err in l_bound]
            self.bounds[key + (q_name,)] = key_res
