from wenvs import WrapperEnv
import gym

class Wrap_CartPoleEnv(WrapperEnv):
    
    def __init__(self):
        env = gym.make('CartPole-v1')
        super().__init__(env, n_fake_features=2, n_fake_actions=2, n_combinations=1, continuous_state=True)