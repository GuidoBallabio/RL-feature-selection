import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.envs.toy_text.taxi import TaxiEnv


class TaxiUnraveled(TaxiEnv):
    def decode(self, obs):
        return np.array(np.unravel_index(obs, (5, 5, 5, 4)))

    def step(self, *args, **kwargs):
        res = super().step(*args, **kwargs)
        obs = self.decode(res[0])
        return (obs,) + res[1:]

    def reset(self, *args, **kwargs):
        res = super().reset(*args, **kwargs)
        obs = self.decode(res)
        return obs


class TaxiBinary(TaxiEnv):
    def decode(self, obs):
        bin_obs = np.zeros(500, dtype=np.bool)
        bin_obs[obs] = 1
        return bin_obs

    def step(self, *args, **kwargs):
        res = super().step(*args, **kwargs)
        obs = self.decode(res[0])
        return (obs,) + res[1:]

    def reset(self, *args, **kwargs):
        res = super().reset(*args, **kwargs)
        obs = self.decode(res)
        return obs
