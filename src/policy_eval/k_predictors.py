import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from tqdm.autonotebook import tqdm


class QfunctionAsSum():
    def __init__(self, gamma, regressor=ExtraTreesRegressor, **regr_kwargs):
        self.gamma = gamma
        self.regressor = regressor
        self.regr_kwargs = regr_kwargs
        self.predictors = []
        self.weights = None

    def fit(self, trajectories, features_to_consider=None, iter_max=50):
        if features_to_consider is None:
            features_to_consider = list(range(trajectories[0].shape[1]-1))

        n = len(features_to_consider)
        self.min_len = min([len(t) for t in trajectories])
        self.weights = self.gamma ** np.arange(self.min_len)

        data = np.dstack([t[:self.min_len, features_to_consider + [-1]]
                          for t in trajectories]).transpose(2, 0, 1)

        for i in range(self.min_len):
            regr = self.regressor(n_estimators=50, **self.regr_kwargs)
            regr.fit(data[:, 0, :-1], data[:, i, -1])
            self.predictors.append(regr)

        return self

    def __call__(self, sa, correction_term=True):
        r = np.hstack([p.predict(sa)[:, None] for p in self.predictors])
        value = np.einsum('nk, k -> n', r, self.weights)
        if correction_term:
            value += r[:, -1] * (self.gamma ** self.min_len) / (1 - self.gamma)
        return value
