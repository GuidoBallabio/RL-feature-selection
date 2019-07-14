import numpy as np
import pytest

from src.algorithm.info_theory.entropy import NNEntropyEstimator, LeveOneOutEntropyEstimator
from src.algorithm.info_theory.combo_estimators import NpeetEstimator, FastNNEntropyEstimator, KDEntropyEstimator, DiscreteEntropyEstimator, CmiEstimator


@pytest.fixture(scope="module")
def data_gaussians():
    sigma=np.array([[1,0.5],[0.5, 1]])
    mu=np.zeros(2)
    
    entropy = np.log(2 * np.pi * np.e * np.linalg.det(sigma) ** 0.5)
    mi = -np.log(1 - 0.5**2)
    
    np.random.seed(0)
    return np.random.multivariate_normal(mu, sigma, 1000), entropy, mi


@pytest.mark.xfail
def test_nne(data_gaussians):
    data, h, mi = data_gaussians
    x, y = data[:, 0:1], data[:, 1:]
    
    est = NNEntropyEstimator()
    
    h_est = est.entropy(data)
    mi_est = est.entropy(x) + est.entropy(y) - est.entropy(data)
    
    assert np.allclose(h, h_est, rtol=0.3)
    assert np.allclose(mi_est, mi, rtol=0.3)

@pytest.mark.xfail
def test_fnn(data_gaussians):
    data, h, mi = data_gaussians
    x, y = data[:, 0:1], data[:, 1:]
    
    est = FastNNEntropyEstimator()
    
    h_est = est.entropy(data)
    mi_est = est.entropy(x) + est.entropy(y) - est.entropy(data)
    
    assert np.allclose(h, h_est, rtol=0.3)
    assert np.allclose(mi_est, mi, rtol=0.3)

@pytest.mark.xfail
def test_llo(data_gaussians):
    data, h, mi = data_gaussians
    x, y = data[:, 0:1], data[:, 1:]
    
    est = LeveOneOutEntropyEstimator("gaussian", min_log_proba=-200)
    
    h_est = est.entropy(data)
    mi_est = est.entropy(x) + est.entropy(y) - est.entropy(data)
    
    assert np.allclose(h, h_est, rtol=0.3)
    assert np.allclose(mi_est, mi, rtol=0.3)

@pytest.mark.xfail
def test_kde(data_gaussians):
    data, h, mi = data_gaussians
    x, y = data[:, 0:1], data[:, 1:]
    
    est = KDEntropyEstimator()
    
    h_est = est.entropy(data)
    mi_est = est.entropy(x) + est.entropy(y) - est.entropy(data)
    
    assert np.allclose(h, h_est, rtol=0.3)
    assert np.allclose(mi_est, mi, rtol=0.3)

def test_npe(data_gaussians):
    data, h, mi = data_gaussians
    x, y = data[:, 0:1], data[:, 1:]
    
    est = NpeetEstimator()
    
    h_est = est.entropy(data)
    mi_est = est.mi(x,y)
    
    assert np.allclose(h, h_est, rtol=0.3)
    assert np.allclose(mi_est, mi, rtol=0.3)

@pytest.mark.xfail
def test_cmi(data_gaussians):
    data, h, mi = data_gaussians
    x, y = data[:, 0:1], data[:, 1:]
    
    est = CmiEstimator()
    
    h_est = est.entropy(data)
    mi_est = est.mi(x,y)
    
    assert np.allclose(h, h_est, rtol=0.3)
    assert np.allclose(mi_est, mi, rtol=0.3)
