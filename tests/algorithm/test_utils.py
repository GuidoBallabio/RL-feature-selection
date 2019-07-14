import gym
import numpy as np

from src.algorithm import utils
from src.wenvs.wrapper_env import WrapperEnv

def test_independent_roll():
    arr = np.arange(16).reshape(4,4)
    
    shift = np.zeros(4, dtype=np.int)
    shift[1:3] = -2

    roll = utils.independent_roll(arr, shift)
    
    assert np.all(roll[:, :1] == arr[:, :1])
    assert np.all(roll[:, 3:] == arr[:, 3:])
    assert np.all(roll[2:, 1:3] == arr[:2, 1:3])
    assert np.all(roll[:2, 1:3] == arr[2:, 1:3])

def test_episodes_with_len():
    env = gym.make("CartPole-v1")
    wenv = WrapperEnv(env, continuous_state=True, continuous_actions=False)
    num_ep = 30
    len_ep = 20 
    policy = None
    stop_at_len = True
    
    wenv.seed(0)    
    traj1 = utils.episodes_with_len(wenv, num_ep, len_ep, policy=policy, stop_at_len=stop_at_len)

    wenv.seed(0)
    traj2 = utils.episodes_with_len(wenv, num_ep, len_ep, policy=policy, stop_at_len=stop_at_len)

    assert all([np.all(t1 == t2) for t1, t2 in zip(traj1, traj2)]), "Seed on trajectories creation not working, non-determinism"