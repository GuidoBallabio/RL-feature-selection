"""Various helpers for gym.Space and quick tests of discrete gym.Envs
"""


import numpy as np
from gym import spaces


def atleast_2d(*arrs):
    l = []
    for a in arrs:
        a = np.array(a)
        if len(a.shape) == 1:
            a = a[:, np.newaxis]
        l.append(a)

    return l


def greedy_pi(Q, keepdims=False):
    policy = (Q.max(axis=1, keepdims=True) == Q).astype(np.int)
    if keepdims:
        return policy
    return policy.argmax(axis=1)


def greedy_pi_multidim(Q, state_sizes, act_sizes, keepdims=False):
    if keepdims:
        policy = (Q.max(axis=tuple(len(obs_dim) + np.arange(len(acts_dim))),
                        keepdims=True) == Q).astype(np.int)
    else:
        q_s_max = Q.reshape(*state_sizes, -1).argmax(-1)
        policy = np.stack(np.unravel_index(q_s_max, act_sizes), axis=-1)
    return policy


def eps_greedy(env, s, eps, Q):
    if np.random.rand() < eps:
        return env.encode_act(env.action_space.sample())
    currQ = Q[s, :]
    actions = np.argwhere(currQ == currQ.max()).flatten()
    # actions = currQ.argmax(axis=1) only first
    return np.random.choice(actions)  # if ties on max, random one


def eps_greedy_multidim(env, s_t, eps, Q):
    if np.random.rand() < eps:
        return env.action_space.sample()
    currQ = Q[s_t]
    actions = np.argwhere(currQ == currQ.max()).flatten()
    return np.random.choice(actions)  # if ties on max, random one


def Q_learing(env, control_space, policy=eps_greedy, iterMax=int(1e5), gamma=0.9):

    n_states, n_actions = control_space
    s = env.encode_obs(env.reset())
    Q = np.zeros(control_space)

    for m in range(iterMax):
        α = 1 - m/iterMax
        ϵ = α**2

        a = policy(env, s, ϵ, Q)
        a_d = env.decode_act(a)

        s_next, r, done, _ = env.step(a_d)
        s_next = env.encode_obs(s_next)

        Q[s, a] = Q[s, a] + α * (r + gamma * np.max(Q[s_next, :]) - Q[s, a])

        s = s_next

        if done:
            s = env.encode_obs(env.reset())

    return Q, greedy_pi(Q)


def Q_learing_multidim(env, state_sizes, act_sizes, policy=eps_greedy_multidim, iterMax=int(1e5), gamma=0.9):

    s = env.reset()
    s_t = tuple(s)
    Q = np.zeros(state_sizes + act_sizes)

    for m in range(iterMax):
        α = 1 - m/iterMax
        ϵ = α**2

        a = policy(env, s_t, 1, Q)
        a_t = tuple(a)

        s_next, r, done, _ = env.step(a)
        s_next_t = tuple(s_next)

        Q[s_t][a_t] = Q[s_t][a_t] + α * \
            (r + gamma * np.max(Q[s_next_t]) - Q[s_t][a_t])

        s = s_next
        s_t = s_next_t

        if done:
            s = env.reset()

    return Q, greedy_pi_multidim(Q, state_sizes, act_sizes)


def dim_of_space(space):
    dim = 0
    if isinstance(space, spaces.Tuple):
        for s in space.spaces:
            dim += dim_of_space(s)
    else:
        if isinstance(space, spaces.Box) or isinstance(space, spaces.MultiDiscrete):
            dim += np.prod(space.shape)
        else:
            dim += 1

    return dim


def discrete_space_size(space):
    dims = []
    if isinstance(space, spaces.Tuple):
        for s in space.spaces:
            dims += discrete_space_size(s)
    elif isinstance(space, spaces.MultiDiscrete):
        dims += space.nvec.tolist()
    elif isinstance(space, spaces.Box):
        dims += space.shape + (space.high.flat[0] - space.low.flat[0],)
    else:
        dims += [space.n]

    return dims
