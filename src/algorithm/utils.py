import functools

import numpy as np


def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def independent_roll(arr, shifts, axis=0):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

    # Convert to a positive shift
    roll = shifts.copy()
    roll[roll < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - roll[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def episodes_with_len(wenv, num_ep, len_ep, policy=None, stop_at_len=True):
    """Return a list of 'num_ep' episodes of length `len_ep`.

    Parameters
    ----------
    wenv : WrapperEnv
        A wrapped environment, most notably with a run_episode function.

    num_ep: int
        Number of episodes/trajectories to be generates.

    len_ep: int
        Minimum length of the episodes generated.

    stop_at_len: boolean, optional
        If True the trajectories are truncated at the `len_ep` timestep, 
        otherwise it is left arbitrarly long (until it's done). 
        (Default is True)

    policy: fun(obs)->act
        A valid policy for the `wenv`, if None the random policy is
        selected. (Default is None)
    """

    iterMax = len_ep if stop_at_len else 200
    l = []
    while len(l) < num_ep:
        ep = wenv.run_episode(policy=policy, iterMax=iterMax)
        if len(ep[2]) >= len_ep:
            l.append(np.hstack(ep))
    return l
