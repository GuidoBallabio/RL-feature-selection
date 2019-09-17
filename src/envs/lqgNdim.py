"""classic Linear Quadratic Gaussian Regulator task"""
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


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

    def __init__(self, gamma, n_dim=1, action_dim=None, discrete_reward=False, A=None, B=None, Q=None, R=None):
        self.gamma = gamma
        self.n_dim = n_dim
        self.discrete_reward = discrete_reward

        if action_dim is None:
            self.action_dim = n_dim
        else:
            self.action_dim = action_dim

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        if A is None:
            self.A = np.eye(n_dim, n_dim)
        if B is None:
            self.B = np.eye(n_dim, self.action_dim)
        if Q is None:
            self.Q = np.eye(n_dim, n_dim) * 0.9
        if R is None:
            self.R = np.eye(self.action_dim, self.action_dim) * 0.9

        assert self.A.shape == (n_dim, n_dim)
        assert self.B.shape == (n_dim, self.action_dim)
        assert self.Q.shape == (n_dim, n_dim)
        assert self.R.shape == (self.action_dim, self.action_dim)

        self.max_pos = 10.0
        self.max_action = 8.0
        self.sigma_noise = 0.1

        # gym attributes
        self.viewer = None
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(self.action_dim,))
        self.observation_space = spaces.Box(low=-self.max_pos,
                                            high=self.max_pos,
                                            shape=(n_dim,))

        # initialize state
        self.seed()
        self.reset()

        self.state = None

    def get_cost(self, x, u):
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        return -0.5*np.asscalar(x.T @ self.Q @ x + u.T @ self.R @ u)

    def step(self, action):
        u = np.atleast_1d(action)  # u is a real

        cost = self.get_cost(self.state, u)

        noise = self.np_random.normal(size=self.n_dim, scale=self.sigma_noise)
        xn = self.A @ self.state + self.B @ u + noise
        self.state = np.clip(xn, -self.max_pos, self.max_pos)

        if self.discrete_reward:
            if np.all(np.abs(self.state) <= 2) and np.all(np.abs(u) <= 2):
                return self.get_state(), 0, False, {}
            return self.get_state(), -1, False, {}
        return self.get_state(), cost, False, {}

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

    def __del__(self):
        self.close()

    def render(self, mode='human', close=False):
        assert self.n_dim <= 2, "Visualization is possible only in 1D (or 2D later on)"

        if close:
            self.close()

        screen_width = 600
        screen_height = 400

        scale = np.array([screen_width, screen_height]) / (self.max_pos * 2)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            ballradius = 3
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, 0)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)

            zero_line_x = rendering.Line((0, screen_height / 2),
                                         (screen_width, screen_height / 2))
            zero_line_x.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line_x)

            zero_line_y = rendering.Line((screen_width / 2, 0),
                                         (screen_width / 2, screen_height))
            zero_line_y.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line_y)

        if self.state is None:
            return None

        ballx = self.state[0] * scale[0] + screen_width / 2.0
        if self.n_dim == 2:
            bally = self.state[1] * scale[1] + screen_height / 2.0
        else:
            bally = screen_height / 2

        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def gaussianPolicy(self, K, Sigma=0.0):
        if np.isscalar(Sigma):
            Sigma = np.eye(self.action_dim) * Sigma

        mu = np.zeros(self.action_dim)
        def action_noise(): return np.random.multivariate_normal(mu, Sigma)
        return lambda x: np.clip(K @ x + action_noise(), -self.max_action, self.max_action)

    def optimalPolicy(self, Sigma=0.0):
        return self.gaussianPolicy(self.computeOptimalK(), Sigma)

    def computeP(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller K * x

        Returns:
            P (matrix): the Riccati Matrix

        """
        K = np.atleast_2d(K)
        # K.shape is (1, n_dim) at least (action_dim, n_dim) otherwise
        I = np.eye(self.n_dim)

        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            # makes sense with n_dim not 1?
            P = np.linalg.inv(I - self.gamma * (I + 2*K + K.T @ K)
                              ) @ (self.Q + K.T @ self.R @ K)
        else:
            tolerance = 0.0001
            P_old = I
            P_base = self.Q + K.T @ self.R @ K
            BK_A = self.B @ K + self.A
            P = P_base + self.gamma * BK_A.T @ P_old @ BK_A
            while not np.allclose(P_old, P, atol=tolerance, rtol=0):
                P_old = P
                P = P_base + self.gamma * BK_A.T @ P_old @ BK_A

        return P

    def computeK(self, P):
        return -self.gamma * \
            np.linalg.inv(self.R + self.gamma * self.B.T @
                          P @ self.B) @ self.B.T @ P @ self.A

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K @ x).

        Returns:
            K (matrix): the optimal controller matrix, shape = (n_action_dim, n_dim)

        """
        P = np.eye(self.n_dim)
        K = self.computeK(P)
        equal = False

        while not equal:
            K_old = K
            P = self.computeP(K)
            K = self.computeK(P)
            equal = np.allclose(K_old, K, atol=0.0001, rtol=0)

        return K

    def computeJ(self, K, Sigma, n_random_x0=100):
        """
        This function computes the discounted reward associated to the provided
        linear controller (u = Kx + \epsilon, \epsilon \sim N(0,\Sigma)).
        Args:
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
                            the controller action (shape == action_dim)
            n_random_x0: the number of samples to draw in order to average over
                         the initial state

        Returns:
            J (float): The discounted reward

        """
        K = np.atleast_2d(K)
        Sigma = np.atleast_2d(Sigma)

        P = self.computeP(K)
        J = 0.0
        for i in range(n_random_x0):
            self.reset()
            x0 = self.get_state()
            J -= 0.5 * x0.T @ P @ x0 + 0.5 * (1 / (1 - self.gamma)) * \
                np.trace(Sigma @ (self.R + self.gamma * self.B.T @ P @ self.B))
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
        K = np.atleast_2d(K)
        Sigma = np.atleast_2d(Sigma)

        P = self.computeP(K)
        Qfun = 0
        for _ in range(n_random_xn):
            noise = self.np_random.normal(
                size=self.n_dim, scale=self.sigma_noise)
            action_noise = self.np_random.multivariate_normal(
                np.zeros(self.action_dim), Sigma, 1).reshape(-1)
            nextstate = self.A @ x + self.B @ (u + action_noise) + noise
            delta = x.T @ self.Q @ x + u.T @ self.R @ u + \
                self.gamma * nextstate.T @ P @ nextstate + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(Sigma @ (self.R + self.gamma * self.B.T @ P @ self.B))
            Qfun -= delta/2

        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun

    def computeVFunction(self, x, K, Sigma, n_random_xn=100):
        """
        This function computes the Value of a state x given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Vfun (float): The Value in the given state x under the given
            controller

        """
        x = np.atleast_1d(x)
        K = np.atleast_2d(K)
        Sigma = np.atleast_2d(Sigma)

        P = self.computeP(K)
        Vfun = 0
        for i in range(n_random_xn):
            Vfun -= x.T @ P @ x + \
                (1 / (1 - self.gamma)) * \
                np.trace(Sigma @ (self.R + self.gamma * self.B.T @ P @ self.B))
            Vfun /= 2

        Vfun = np.asscalar(Vfun) / n_random_xn
        return Vfun

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
