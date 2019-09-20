from pathlib import Path
import pickle

import gym
import numpy as np

from src.algorithm.env_eval import EnvEval
from src.algorithm.info_theory.combo_estimators import (
    DiscreteEntropyEstimator, FastNNEntropyEstimator, NpeetEstimator)
from src.algorithm.utils import (dump_list_to_file, env_name,
                                 episodes_with_len, load_list_from_file)
from src.envs.lqgNdim import LQG_nD
from src.envs.taxi_variants import TaxiBinary, TaxiUnraveled
from src.policy_eval.fqi import QfunctionFQI
from src.policy_eval.k_predictors import QfunctionAsSum, QfunctionAsSumDmu

DIR = Path('benchmarks/all')
OUT_FILE = DIR / 'backward_errors.pkl'


def lqg():
    Q = np.diag([0.9, 0.9, 0.1, 0.1])
    R = Q.copy()
    return LQG_nD(0.9, n_dim=4, Q=Q, R=R), True, True, None


def taxi():
    env = TaxiUnraveled()
    return env, False, False, None


def taxiB():
    env = TaxiBinary()
    return env, False, False, None


def ll():
    env = gym.make('LunarLander-v2')
    return env, True, True, None


# def breakout():
#    model = A2C.load("agents/a2c-BreakoutNoFrameskip-v4.pkl")
#
#    env_orig = gym.make('Breakout-v0')
#    env = FireResetEnv(env_orig)
#    env = GrayRaveler(env)
#    wenv = WrapperEnv(env, continuous_state=False, continuous_actions=False)
#
#    input_shape = env_orig.observation_space.shape[:-1]
#    model_shape = model.observation_space.shape
#    policy = PolicyWrapper(
#        model, frame_size=model_shape[:-1], frame_stacks=model_shape[-1], reshape=input_shape)
#
#    return wenv, False, False, policy

l_env = [lqg, taxi, taxiB, ll]
l_est = [DiscreteEntropyEstimator, FastNNEntropyEstimator, NpeetEstimator]
l_q = [QfunctionFQI]
l_k = [20, 30, 40, 50]
l_gamma = [0.5, 0.9, 0.95]
l_n = [500, 1000, 2000, 5000]

d = {}

for f_env in l_env:
    for est in l_est:
        for q in l_q:
            for k in l_k:
                for gamma in l_gamma:
                    for n_traj in l_n:
                        env, cs, ca, policy = f_env()
                        key = (env_name(env), est.__name__,
                               q.__name__, k, gamma, n_traj)
                        print(key)
                        if not ((cs and ca) ^ cs ^ ca ^ (est is DiscreteEntropyEstimator)):
                            continue
                        ev = EnvEval(env, est, q, continuous_actions=ca,
                                     continuous_state=cs)
                        res = ev.run(k, gamma, n_traj)
                        d[key] = res
                        with open(OUT_FILE, 'wb') as f:
                            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
