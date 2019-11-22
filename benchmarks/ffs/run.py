from pathlib import Path
import pickle

import numpy as np

from src.algorithm.backward_feature_selection import BackwardFeatureSelector
from src.algorithm.info_theory.combo_estimators import NpeetEstimator

DIR = Path('benchmarks/ffs')

def CMIFS(trajectories, k, gamma, nproc=None):
    fs = BackwardFeatureSelector(NpeetEstimator(), trajectories, nproc=nproc)
    return [(set(fs.idSet), 0.0)] + list(fs.try_all(k, gamma))

with open(DIR / "100dim-traj.pkl", 'rb') as fp:
        trajectories = pickle.load(fp)

cmifs_hist = CMIFS(trajectories, 20, 0.9, nproc=None)

with open(DIR / "100dim-hist.pkl", 'wb') as fp:
        pickle.dump(cmifs_hist, fp)
