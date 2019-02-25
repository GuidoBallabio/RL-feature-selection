import numpy as np


def greedy_pi(Q):
    policy = (Q.max(axis=1, keepdims=True) == Q).astype(np.int)
    return policy


def eps_greedy(env, s, eps, Q):
    if np.random.rand() < eps:
        return env.encode_act(env.action_space.sample())
    currQ = Q[s,:]
    actions = np.argwhere(currQ == currQ.max()).flatten()
    #actions = currQ.argmax(axis=1) only first
    return np.random.choice(actions) # if ties on max, random one


def eps_greedy_multidim(env, s_t, eps, Q):
    if np.random.rand() < eps:
        return env.action_space.sample()
    currQ = Q[s_t]
    actions = np.argwhere(currQ == currQ.max()).flatten()
    return np.random.choice(actions) # if ties on max, random one


def Q_learing(env, control_space, policy = eps_greedy, iterMax = int(1e5), gamma = 0.9):

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

        Q[s,a] = Q[s,a] + α * (r + gamma *  np.max(Q[s_next,:]) -Q[s,a]) 

        s = s_next
        
        if done:
            s = env.encode_obs(env.reset())
        
    return Q, greedy_pi(Q)


def Q_learing_multidim(env, control_space, policy = eps_greedy_multidim, iterMax = int(1e5), gamma = 0.9):
    
    n_states, n_actions = control_space    
    s = env.reset()
    s_t = tuple(s)
    Q = np.zeros(control_space)

    for m in range(iterMax):
        α = 1 - m/iterMax
        ϵ = α**2


        a = policy(env, s_t, ϵ, Q)
        a_t = tuple(a)

        s_next, r, done, _ = env.step(a)
        s_next_t = tuple(s_next)

        Q[s_t][a_t] = Q[s_t][a_t] + α * (r + gamma *  np.max(Q[s_next_t]) -Q[s_t][a_t]) 

        s = s_next
        s_t = s_next_t
        
        if done:
            s = env.reset()
        
    return Q, greedy_pi(Q)
