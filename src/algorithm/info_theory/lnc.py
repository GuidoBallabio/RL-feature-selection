import pickle

import numpy as np

from npeet import entropy_estimators as ee


def precompute_alphas(ds, ks, N=500000):
    tab = {}
    for d in ds:
        for dy in range(1, d+1):
            dz = d - dy
            # for k in ks: # sum_cmi=False then True
            #    tab[(k, 1, dy, dz)] = ee.alpha_estimation(k, 1, dy, dz=dz, N=N)
            #    tab[(k, 1, 1, dz)] = ee.alpha_estimation(k, 1, 1, dz=dz, N=N)
            k = np.clip(2 * (1 + 1 + dz), 7, 20)  # heuristic used
            tab[(k, 1, 1, dz)] = ee.alpha_estimation(k, 1, 1, dz=dz, N=N)
    return tab


if __name__ == '__main__':
    tab = precompute_alphas([9], [5, 7, 10, 16, 18])
    with open('alphas_tab.pkl', 'wb') as f:
        pickle.dump(tab, f)
