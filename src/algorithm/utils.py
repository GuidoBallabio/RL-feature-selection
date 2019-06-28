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


def episodes_with_len(wenv, num_ep, len_ep, policy=None):
    """Return a list of 'num_ep' episodes of length `len_ep`.
    """
    l = []
    while len(l) < num_ep:
        ep = wenv.run_episode(policy=policy, iterMax=len_ep)
        if len(ep[2]) >= len_ep:
            l.append(np.hstack(ep))
    return l
