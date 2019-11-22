import numpy as np
from npeet import entropy_estimators as ee
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity

from src.algorithm.info_theory.entropy import NNEntropyEstimator
from src.algorithm.info_theory.it_estimator import ItEstimator
from src.algorithm.info_theory.mutual_information import MixedRvMiEstimator


class NpeetEstimator(ItEstimator):
    discrete = False

    def __init__(self, k=3):
        self.k = k

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropy(X.copy(order='C'), k=self.k)

    def mi(self, X, Y):
        np.random.seed(0)
        r = ee.mi(X.copy(order='C'), Y.copy(order='C'), k=self.k)
        return r if r >= 0 else 0

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        r = ee.mi(X.copy(order='C'), Y.copy(order='C'), z=Z.copy(order='C'), k=self.k)
        return r if r >= 0 else 0

    def flags(self):
        return True, False, True


class CmiEstimator(ItEstimator):
    discrete = False

    def __init__(self, nproc=1):
        self.h_est = NNEntropyEstimator()
        self.mi_est = MixedRvMiEstimator(3, nproc=nproc)

    def entropy(self, X):
        np.random.seed(0)
        return self.h_est.entropy(X.copy(order='C'))

    def mi(self, X, Y):
        np.random.seed(0)
        return self.mi_est.estimateMI(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return self.mi_est.estimateConditionalMI(X.copy(order='C'), Y.copy(order='C'), Z.copy(order='C'))

    def flags(self):
        return True, False, True


class FastNNEntropyEstimator(ItEstimator):
    EPS = np.finfo(np.float).eps
    discrete = False

    def __init__(self, kfold=10):
        self.kfold = kfold

    def estimateFromData(self, datapoints):
        entropy = 0.0
        n, d = datapoints.shape
        k = 1

        ma = np.ones(n, dtype=np.bool)
        unit = n // self.kfold
        rem = n % self.kfold

        start = 0
        end = unit + rem
        for i in range(self.kfold):
            sel = np.arange(start, end)
            ma[start:end] = False
            curr = datapoints[ma, :]

            nn = NearestNeighbors(n_neighbors=k).fit(curr)
            dist, _ = nn.kneighbors(datapoints[sel, :])
            entropy += np.log(n * dist + self.EPS).sum()

            ma[:] = True
            start = end
            end = min(unit + end, n)

        return entropy / n + np.log(2) + np.euler_gamma

    def entropy(self, X):
        np.random.seed(0)
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False


class KDEntropyEstimator(ItEstimator):
    discrete = False

    def __init__(self, kernel="gaussian",  min_log_proba=-500, bandwith=1.0, kfold=10):
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwith)
        self.min_log_proba = min_log_proba
        self.kfold = kfold

    def estimateFromData(self, datapoints):
        if len(datapoints.shape) == 1:
            datapoints = np.expand_dims(datapoints, 1)

        entropy = 0.0

        n, d = datapoints.shape
        ma = np.ones(n, dtype=np.bool)
        unit = n // self.kfold
        rem = n % self.kfold

        start = 0
        end = unit + rem
        for i in range(self.kfold):
            sel = np.arange(start, end)
            ma[start:end] = False
            curr = datapoints[ma, :]

            self.kde.fit(curr)
            score = self.kde.score(datapoints[sel, :])

            ma[:] = True
            start = end
            end = min(unit + end, n)

            if score < self.min_log_proba:
                continue

            entropy -= score

        return entropy / n

    def entropy(self, X):
        np.random.seed(0)
        return self.estimateFromData(X)

    def flags(self):
        return False, False, False


class DiscreteEntropyEstimator(ItEstimator):
    discrete = True

    def __init__(self):
        pass

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropyd(X.copy(order='C'))

    def mi(self, X, Y):
        np.random.seed(0)
        return ee.midd(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return ee.cmidd(X.copy(order='C'), Y.copy(order='C'), z=Z.copy(order='C'))

    def flags(self):
        return True, False, True


class DJKEntropyEstimator(ItEstimator):
    discrete = True

    def __init__(self):
        pass

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropyd_jk(X.copy(order='C'))

    def mi(self, X, Y):
        np.random.seed(0)
        return ee.midd(X.copy(order='C'), Y.copy(order='C'))

    def cmi(self, X, Y, Z):
        np.random.seed(0)
        return ee.cmidd(X.copy(order='C'), Y.copy(order='C'), z=Z.copy(order='C'))

    def flags(self):
        return False, False, False


alphas_dict = {(5, 1, 1, 7): 0.002787202118368912,
               (5, 1, 2, 6): 0.002831342878940567,
               (5, 1, 3, 5): 0.002222317770520968,
               (5, 1, 4, 4): 0.0003485551828077395,
               (5, 1, 5, 3): 1.2094018231812795e-17,
               (5, 1, 6, 2): 8.245885501682365e-18,
               (5, 1, 7, 1): 6.267564090761928e-18,
               (5, 1, 8, 0): 5.182739936128754e-18,
               (7, 1, 1, 7): 0.007799694008975633,
               (7, 1, 2, 6): 0.0007884635392490138,
               (7, 1, 3, 5): 9.34333614020603e-17,
               (7, 1, 4, 4): 6.558629239569769e-17,
               (7, 1, 5, 3): 5.34595731452511e-17,
               (7, 1, 6, 2): 4.536470369688546e-17,
               (7, 1, 7, 1): 3.9398083165779545e-17,
               (7, 1, 8, 0): 3.4678029648023484e-17,
               (10, 1, 1, 7): 0.006862781536180848,
               (10, 1, 2, 6): 0.004425559114094674,
               (10, 1, 3, 5): 0.003490623423716278,
               (10, 1, 4, 4): 0.0028744496979329027,
               (10, 1, 5, 3): 0.0024607709952948567,
               (10, 1, 6, 2): 0.002140949498116081,
               (10, 1, 7, 1): 0.0018605665056913472,
               (10, 1, 8, 0): 0.0017835439843925003,
               (16, 1, 1, 7): 0.2680946667117242,
               (16, 1, 2, 6): 0.21690089995828854,
               (16, 1, 3, 5): 0.19595611160631077,
               (16, 1, 4, 4): 0.18541894998272948,
               (16, 1, 5, 3): 0.17672568295907848,
               (16, 1, 6, 2): 0.16835377171391372,
               (16, 1, 7, 1): 0.15951556338150538,
               (16, 1, 8, 0): 0.1527612981165255,
               (18, 1, 1, 7): 0.28711729319583673,
               (18, 1, 2, 6): 0.23997308850983703,
               (18, 1, 3, 5): 0.22114872376481529,
               (18, 1, 4, 4): 0.20829107293358715,
               (18, 1, 5, 3): 0.201747318868149,
               (18, 1, 6, 2): 0.19493334498297163,
               (18, 1, 7, 1): 0.18659790639572896,
               (18, 1, 8, 0): 0.1815795466903678,
               (5, 1, 1, 8): 0.0016281682968701918,
               (5, 1, 2, 7): 0.0019240763576986884,
               (5, 1, 3, 6): 0.0018609272130895313,
               (5, 1, 4, 5): 0.0015437746914900342,
               (5, 1, 5, 4): 0.00025892805031350945,
               (5, 1, 6, 3): 7.954765533499874e-18,
               (5, 1, 7, 2): 5.2852518247056115e-18,
               (5, 1, 8, 1): 4.088129440820105e-18,
               (5, 1, 9, 0): 3.319743528798185e-18,
               (7, 1, 1, 8): 0.005588393249594135,
               (7, 1, 2, 7): 0.0040435895931905715,
               (7, 1, 3, 6): 0.0004638802639601499,
               (7, 1, 4, 5): 4.9609233176923894e-17,
               (7, 1, 5, 4): 3.387617826438449e-17,
               (7, 1, 6, 3): 2.7264111260952502e-17,
               (7, 1, 7, 2): 2.3258916231144455e-17,
               (7, 1, 8, 1): 2.0185072379181476e-17,
               (7, 1, 9, 0): 1.776023794686299e-17,
               (10, 1, 1, 8): 3.2107005083402704e-15,
               (10, 1, 2, 7): 2.1078926678993154e-15,
               (10, 1, 3, 6): 1.6517039714193796e-15,
               (10, 1, 4, 5): 1.4200380114341938e-15,
               (10, 1, 5, 4): 1.268192517427911e-15,
               (10, 1, 6, 3): 1.1604421233103674e-15,
               (10, 1, 7, 2): 1.068540374424366e-15,
               (10, 1, 8, 1): 9.761721821829249e-16,
               (10, 1, 9, 0): 9.124439100046511e-16,
               (16, 1, 1, 8): 0.2401609787788258,
               (16, 1, 2, 7): 0.1910614590899333,
               (16, 1, 3, 6): 0.16986267876290015,
               (16, 1, 4, 5): 0.15767328450441156,
               (16, 1, 5, 4): 0.15012992482793036,
               (16, 1, 6, 3): 0.14293689220532507,
               (16, 1, 7, 2): 0.13815199712628345,
               (16, 1, 8, 1): 0.13034429352179006,
               (16, 1, 9, 0): 0.1242278876460557,
               (18, 1, 1, 8): 0.2649268107051951,
               (18, 1, 2, 7): 0.21625016658923263,
               (18, 1, 3, 6): 0.19531268836110677,
               (18, 1, 4, 5): 0.1843968329314726,
               (18, 1, 5, 4): 0.178308135819005,
               (18, 1, 6, 3): 0.1733063301068647,
               (18, 1, 7, 2): 0.16654521278476975,
               (18, 1, 8, 1): 0.15937121842311616,
               (18, 1, 9, 0): 0.15557207264432585,
               (20, 1, 1, 8): 0.278635001695579,
               (16, 1, 1, 6): 0.2926184281101319,
               (14, 1, 1, 5): 0.29683631519933035,
               (12, 1, 1, 4): 0.2941616886814251,
               (10, 1, 1, 3): 0.2770814370711524,
               (8, 1, 1, 2): 0.2401514862672179,
               (7, 1, 1, 1): 0.2484991340296421,
               (7, 1, 1, 0): 0.32025862991515175}


class LNCEstimator(ItEstimator):
    discrete = False

    def __init__(self, alphas=alphas_dict):
        self.alphas = alphas

    def entropy(self, X):
        np.random.seed(0)
        return ee.entropy(X.copy(order='C'))

    def mi(self, X, Y):
        dx, dy, dz = X.shape[-1], Y.shape[-1], 0
        d = dx + dy + dz
        k = np.clip(2 * d, 7, 20)

        if (k, dx, dy, dz) in self.alphas:
            alpha = self.alphas[k, dx, dy, dz]
        else:
            alpha = None

        np.random.seed(0)
        r = ee.mi(X.copy(order='C'), Y.copy(order='C'), k=k, alpha=alpha)
        return r if r >= 0 else 0

    def cmi(self, X, Y, Z):
        dx, dy, dz = X.shape[-1],  Y.shape[-1], Z.shape[-1]
        d = dx + dy + dz
        k = np.clip(2 * d, 7, 20)

        if (k, dx, dy, dz) in self.alphas:
            alpha = self.alphas[k, dx, dy, dz]
        else:
            alpha = None

        np.random.seed(0)
        r = ee.mi(X.copy(order='C'), Y.copy(order='C'), z=Z.copy(order='C'),
                  k=k, alpha=alpha)
        return r if r >= 0 else 0

    def flags(self):
        return True, False, True
