import numpy as np
import gym
from gym import Wrapper
import cv2
from collections import deque


class GrayRaveler(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=self.env.observation_space.low.flat[0], high=self.env.observation_space.high.flat[0], shape=(
            np.prod(self.env.observation_space.shape[:-1]),), dtype=self.env.observation_space.dtype)

    def step(self, *args, **kwargs):
        res = self.env.step(*args, **kwargs)
        obs = self.grayscale(res[0]).ravel()
        return (obs,) + res[1:]

    def reset(self, *args, **kwargs):
        res = self.env.reset(*args, **kwargs)
        obs = self.grayscale(res).ravel()
        return obs

    def grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


class PolicyWrapper():

    def __init__(self, model,  reshape=None, frame_size=None, frame_stacks=1):
        self.model = model
        self.frame_size = frame_size
        self.n_frames = frame_stacks
        self.reshape = reshape
        self.queue = deque([], maxlen=self.n_frames)

    def reset(self, obs):
        obs = self.transform(obs)

        self.queue.clear()
        for _ in range(self.n_frames):
            self.queue.append(obs)

    def transform(self, obs):
        if self.reshape is not None:
            obs = obs.reshape(self.reshape)
        if self.frame_size is not None:
            obs = cv2.resize(obs, self.frame_size,
                             interpolation=cv2.INTER_AREA)
        return obs[:, :, None]

    def __call__(self, obs):
        obs = self.transform(obs)

        self.queue.append(obs)
        return self.model.predict(self._get_ob())[0]

    def _get_ob(self):
        assert len(self.queue) == self.n_frames
        return np.concatenate(list(self.queue), axis=2)
