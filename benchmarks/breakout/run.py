from pathlib import Path

import gym
import numpy as np
from stable_baselines import A2C
from stable_baselines.common.atari_wrappers import FireResetEnv

from src.algorithm.backward_feature_selection import BackwardFeatureSelector
from src.algorithm.info_theory.combo_estimators import DiscreteEntropyEstimator
from src.algorithm.utils import (dump_list_to_file, episodes_with_len,
                                 load_list_from_file)
from src.envs.breakout import GrayRaveler, PolicyWrapper
from src.wenvs import WrapperEnv

DIR = Path('benchmarks/breakout')
TRAJ_FILE = DIR / 'traj.pkl'
OUT_FILE = DIR / 'backward_errors.pkl'

if not TRAJ_FILE.exists():
    model = A2C.load("agents/a2c-BreakoutNoFrameskip-v4.pkl")

    env_orig = gym.make('Breakout-v0')
    env = FireResetEnv(env_orig)
    env = GrayRaveler(env)
    wenv = WrapperEnv(env, continuous_state=False, continuous_actions=False)

    input_shape = env_orig.observation_space.shape[:-1]
    model_shape = model.observation_space.shape
    policy = PolicyWrapper(
        model, frame_size=model_shape[:-1], frame_stacks=model_shape[-1], reshape=input_shape)

    np.random.seed(0)
    wenv.seed(0)

    k = 20
    num_ep = 1000
    trajectories = episodes_with_len(wenv, num_ep, k, policy=policy)
    dump_list_to_file(trajectories.astype(np.uint8), TRAJ_FILE)
else:
    trajectories = load_list_from_file(TRAJ_FILE)

fs = BackwardFeatureSelector(DiscreteEntropyEstimator(),
                             trajectories.astype(np.uint8), nproc=1, discrete=True)

dump_list_to_file(fs.try_all(20, 0.9), OUT_FILE)
