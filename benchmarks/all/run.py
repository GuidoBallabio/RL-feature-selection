from pathlib import Path

import gym
import numpy as np

from src.algorithm.batch_eval import BatchEval
from src.algorithm.info_theory.combo_estimators import (
    DiscreteEntropyEstimator, FastNNEntropyEstimator, NpeetEstimator)
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
    return env, True, False, None


l_env = [lqg, taxi, ll]
l_est = [DiscreteEntropyEstimator, NpeetEstimator]
l_q = [QfunctionFQI]
l_k = [20, 30, 40, 50]
l_gamma = [0.5, 0.9, 0.95]
l_n = [500, 1000, 2000, 5000]

bv = BatchEval(l_env, l_est, l_q, l_n, l_k, l_gamma,
               object_store_memory=int(5*10**10))
bv.run(verbose=2, filename=OUT_FILE)
