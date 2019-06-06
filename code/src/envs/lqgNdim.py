"""classic Linear Quadratic Gaussian Regulator task"""
# import gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

"""
Linear quadratic gaussian regulator task.

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
  - Jan  Peters  and  Stefan  Schaal,
    Reinforcement  learning of motor  skills  with  policy  gradients,
    Neural  Networks, vol. 21, no. 4, pp. 682-697, 2008.

"""

'''
#classic_control
from gym.envs.registration import register
register(
    id='LQG_nD-v0',
    entry_point='src.envs.lqgNdim:LQG_nD',
    timestep_limit=300,
)
'''

class LQG_nD(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, gamma, n_dim=1, discrete_reward=False):        
        self.gamma = gamma
        self.n_dim = n_dim
        self.discrete_reward = discrete_reward

        self.A = np.ones((n_dim, n_dim))
        self.B = np.ones((n_dim, n_dim))
        self.Q = np.full((n_dim, n_dim), 0.9)
        self.R = np.full((n_dim, n_dim), 0.9)

        
        self.max_pos = 10.0
        self.max_action = 8.0
        self.sigma_noise = 0.1

        # gym attributes
        self.viewer = None
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(n_dim,))
        self.observation_space = spaces.Box(low=-self.max_pos,
                                            high=self.max_pos,
                                            shape=(n_dim,))

        # initialize state
        self.seed()
        self.reset()
        
        self.viewer = None

    def get_cost(self, x, u):
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        return np.asscalar(x @ self.Q @ x + u @ self.R @ u)

    def step(self, action, render=False):
        action = np.atleast_1d(action)
        u = np.clip(action, -self.max_action, self.max_action)
        cost = self.get_cost(self.state, u)
        
        noise = self.np_random.randn() * self.sigma_noise
        xn = self.A @ self.state + self.B @ u + noise
        self.state = xn.flatten()

        if self.discrete_reward:
            if np.all(np.abs(self.state) <= 2) and np.all(np.abs(u) <= 2):
                return self.get_state(), 0, False, {}
            return self.get_state(), -1, False, {}
        return self.get_state(), -cost, False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.state = self.np_random.uniform(low=-self.max_pos,
                                                high=self.max_pos, size=self.n_dim)
        else:
            self.state = np.array(state)

        return self.state

    def get_state(self):
        return self.state

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human', close=True):
        assert self.n_dim <= 1, "Visualization is possible only in 1D (or 2D later on)" 

        if close:
            self.close()

        screen_width = 600
        screen_height = 400
        
        world_width = (self.max_pos * 2) * 2
        scale = screen_width / world_width
        bally = 100

        if self.viewer is None:
            ballradius = 3
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _computeP2(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller K * x

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])

        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + K.T @ self.R @ K) / (I - self.gamma * (I + 2*K + K**2))
        else:
            tolerance = 0.0001
            P = I.copy()
            Pnew_base = self.Q + K.T @ self.R @ K
            BK_A = self.B @ K + self.A
            Pnew = Pnew_base + self.gamma * BK_A.T @ P @ BK_A
            while not np.allclose(P, Pnew, atol=tolerance, rtol=0):
                Pnew = Pnew_base + self.gamma * BK_A.T @ P @ BK_A
            return P

            '''
                Pnew = Pnew_base + self.gamma *  (A.T @ P @ A +
                                                  B_K.T @ P @ A +
                                                  A.T @ P @ B_K +
                                                  B_K.T @ P @ B_K)

                Pnew = self.Q + self.gamma * np.dot(self.A.T,
                                                    np.dot(P, self.A)) + \
                       self.gamma * np.dot(K.T, np.dot(self.B.T,
                                                       np.dot(P, self.A))) + \
                       self.gamma * np.dot(self.A.T,
                                           np.dot(P, np.dot(self.B, K))) + \
                       self.gamma * np.dot(K.T,
                                           np.dot(self.B.T,
                                                  np.dot(P, np.dot(self.B,
                                                                   K)))) + \
                       np.dot(K.T, np.dot(self.R, K))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
            '''

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K * x).

        Returns:
            K (matrix): the optimal controller bv

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -self.gamma * np.linalg.inv(self.R + self.gamma * (self.B.T @ P @ self.B)) @ self.B.T @ P @ self.A
            P = self._computeP2(K)
        K = -self.gamma * np.linalg.inv(self.R + self.gamma * (self.B.T @ P @ self.B)) @ self.B.T @ P @ self.A
        return K

    def computeJ(self, K, Sigma, n_random_x0=100):
        """
        This function computes the discounted reward associated to the provided
        linear controller (u = Kx + \epsilon, \epsilon \sim N(0,\Sigma)).
        Args:
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
                            the controller action
            n_random_x0: the number of samples to draw in order to average over
                         the initial state

        Returns:
            J (float): The discounted reward

        """
        K = np.atleast_1d(K)
        Sigma = np.atleast_1d(Sigma)

        P = self._computeP2(K)
        J = 0.0
        for i in range(n_random_x0):
            self._reset()
            x0 = self._getState()
            J -= x0.T @ P @ x0 + (1 / (1 - self.gamma)) * np.trace(Sigma @ (self.R + self.gamma * (self.B.T @ P @ self.B)))
        J /= n_random_x0
        return J

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        K = np.atleast_1d(K)
        Sigma = np.atleast_1d(Sigma)

        P = self._computeP2(K)
        Qfun = 0
        for i in range(n_random_xn):
            noise = np.random.randn() * self.sigma_noise
            action_noise = np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = self.A @ x + self.B @ (u + action_noise) + noise
            Qfun -= x.T @ self.Q @ x + u.T @ self.R @ u + \
                self.gamma * nextstate.T @ P @ nextstate + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(Sigma @ (self.R + self.gamma * (self.B.T @ P @ self.B)))

        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun
    

    #TODO
    def computeVFunction(self, x, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, (int, long, float, complex)):
            x = np.array([x])
        if isinstance(K, (int, long, float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, long, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        Vfun = 0
        for i in range(n_random_xn):
            u = np.random.randn() * Sigma + K * x
            noise = np.random.randn() * self.sigma_noise
            action_noise = np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = self.A @ x + self.B @ (u + action_noise) + noise
            Vfun -= x.T @ self.Q @ x + u.T @ self.R @ u + \
                    self.gamma * nextstate.T @ P @ nextstate + \
                    (self.gamma / (1 - self.gamma)) * \
                    np.trace(Sigma @ (self.R + self.gamma * (self.B.T @ P @ self.B)))

        Qfun = np.asscalar(Vfun) / n_random_xn
        return Qfun

        # TODO check following code

        # def computeM(self, K):
        #     kb = np.dot(K, self.B.T)
        #     size = self.A.shape[1] ** 2;
        #     AT = self.A.T
        #     return np.eye(size) - self.gamma * (np.kron(AT, AT) - np.kron(AT, kb) - np.kron(kb, AT) + np.kron(kb, kb))
        #
        # def computeL(self, K):
        #     return self.Q + np.dot(K, np.dot(self.R, K.T))
        #
        # def to_vec(self, m):
        #     n_dim = self.A.shape[1]
        #     v = m.reshape(n_dim * n_dim, 1)
        #     return v
        #
        # def to_mat(self, v):
        #     n_dim = self.A.shape[1]
        #     M = v.reshape(n_dim, n_dim)
        #     return M
        #
        # def computeJ(self, k, Sigma, n_random_x0=100):
        #     J = 0
        #     K = k
        #     if len(k.shape) == 1:
        #         K = np.diag(k)
        #     P = self.computeP(K)
        #     for i in range(n_random_x0):
        #         self._reset()
        #         x0 = self.state
        #         v = np.asscalar(x0.T * P * x0 + np.trace(
        #             np.dot(Sigma, (self.R + np.dot(self.gamma, np.dot(self.B.T, np.dot(P, self.B)))))) / (1.0 - self.gamma))
        #         J += -v
        #     J /= n_random_x0
        #
        #     return J
        #
        # def solveRiccati(self, k):
        #     K = k
        #     if len(k.shape) == 1:
        #         K = np.diag(k)
        #     return self.computeP(K)
        #
        # def riccatiRHS(self, k, P, r):
        #     K = k
        #     if len(k.shape) == 1:
        #         K = np.diag(k)
        #     return self.Q + self.gamma * (np.dot(self.A.T, np.dot(self.P, self.A))
        #                                   - np.dot(K, np.dot(self.B.T, np.dot(self.P, self.A)))
        #                                   - np.dot(self.A.T, np.dot(self.P, np.dot(self.B, K.T)))
        #                                   + np.dot(K, np.dot(self.B.T, np.dot(self.P, np.dot(self.B, K.T))))) \
        #            + np.dot(K, np.dot(self.R, K.T))
        #
        # def computeP(self, K):
        #     L = self.computeL(K)
        #     M = self.computeM(K)
        #
        #     vecP = np.linalg.solve(M, self.to_vec(L))
        #
        #     P = self.to_mat(vecP)
        #     return P
