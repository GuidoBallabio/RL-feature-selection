import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from tqdm.autonotebook import tqdm


class QfunctionFQI():
    def __init__(self, gamma, regressor=ExtraTreesRegressor, **regr_kwargs):
        self.gamma = gamma
        self.regressor = regressor(n_estimators=50, **regr_kwargs)

    def _make_db(self, trajectories):
        sar = [t[:, self.features_to_consider + [-1]] for t in trajectories]
        sarsar = [np.hstack([t[:-1, :], t[1:, :]]) for t in sar]

        db = np.vstack(sarsar)

        return db

    def fit(self, trajectories, features_to_consider=None, iter_max=50, show_progress=False, leave=False):
        if features_to_consider is None:
            features_to_consider = range(trajectories[0].shape[1]-1)

        self.features_to_consider = list(features_to_consider)

        n = len(features_to_consider)
        db = self._make_db(trajectories)

        sar = db[:, :n+1]
        sar_next = db[:, n+1:]

        sa = sar[:, :-1]
        r = sar[:, -1].ravel()
        sa_next = sar_next[:, :-1]
        # r_next = sar_next[:, -1].ravel()

        self.regressor.fit(sa, r)

        for _ in tqdm(range(iter_max), leave=leave, disable=not show_progress):
            nextY = r + self.gamma * self.regressor.predict(sa_next)
            self.regressor.fit(sa, nextY)

        return self

    def __call__(self, sa):
        return self.regressor.predict(sa)
