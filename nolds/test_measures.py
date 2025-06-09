# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (
  bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
  oct, open, pow, round, super, filter, map, zip
)
import numpy as np

# import internal module to test helping functions
from nolds import measures as nolds
from nolds import datasets
import unittest
import warnings

# TODO add tests for mfhurst_b and mfhurst_dm

# TODO add more tests using fgn and fbm for hurst_rs and dfa

# TODO split up tests into smaller units => one hypothesis = one test

try:
  from scipy.stats import levy_stable
  SCIPY_AVAILABLE = True
except ImportError:
  SCIPY_AVAILABLE = False


class TestNoldsHelperFunctions(unittest.TestCase):
  """
  Tests for internal helper functions that are not part of the public API
  """
  def assert_array_equals(self, expected, actual, print_arrays=False):
    if print_arrays:
      print(actual)
      print("==")
      print(expected)
      print()
    self.assertTrue(np.all(actual == expected))

  def test_delay_embed_lag2(self):
    data = np.arange(10, dtype="float32")
    embedded = nolds.delay_embedding(data, 4, lag=2)
    expected = np.array([
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [2, 4, 6, 8],
        [3, 5, 7, 9]
    ], dtype="float32")
    self.assert_array_equals(expected, embedded)

  def test_delay_embed(self):
    data = np.arange(6, dtype="float32")
    embedded = nolds.delay_embedding(data, 4)
    expected = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ], dtype="float32")
    self.assert_array_equals(expected, embedded)

  def test_delay_embed_lag3(self):
    data = np.arange(10, dtype="float32")
    embedded = nolds.delay_embedding(data, 4, lag=3)
    expected = np.array([
        [0, 3, 6, 9]
    ], dtype="float32")
    self.assert_array_equals(expected, embedded)

  def test_delay_embed_empty(self):
    data = np.arange(10, dtype="float32")
    try:
      embedded = nolds.delay_embedding(data, 11)
      msg = "embedding array of size 10 with embedding dimension 11 "  \
          + "should fail, got {} instead"
      self.fail(msg.format(embedded))
    except ValueError:
      pass
    data = np.arange(10, dtype="float32")
    try:
      embedded = nolds.delay_embedding(data, 4, lag=4)
      msg = "embedding array of size 10 with embedding dimension 4 and " \
          + "lag 4 should fail, got {} instead"
      self.fail(msg.format(embedded))
    except ValueError:
      pass


class TestNoldsUtility(unittest.TestCase):
  """
  Tests for small utility functions that are part of the public API
  """
  def test_binary_n(self):
    x = nolds.binary_n(1000, min_n=50)
    self.assertSequenceEqual(x, [500, 250, 125, 62])

  def test_binary_n_empty(self):
    x = nolds.binary_n(50, min_n=50)
    self.assertSequenceEqual(x, [])

  def test_logarithmic_n(self):
    x = nolds.logarithmic_n(4, 11, 1.51)
    self.assertSequenceEqual(x, [4, 6, 9])

  def test_logarithmic_r(self):
    x = nolds.logarithmic_r(4, 10, 1.51)
    self.assertSequenceEqual(x, [4, 6.04, 9.1204])


class TestNoldsLyap(unittest.TestCase):
  """
  Tests for lyap_e and lyap_r
  """
  def test_lyap_logistic(self):
    rvals = [2.5, 3.4, 3.7, 4.0]
    sign = [-1, -1, 1, 1]
    x0 = 0.1

    def logistic(x, r):
      return r * x * (1 - x)

    for r, s in zip(rvals, sign):
      log = []
      x = x0
      for _ in range(100):
        x = logistic(x, r)
        log.append(x)
      log = np.array(log, dtype="float32")
      le = np.max(nolds.lyap_e(log, emb_dim=6, matrix_dim=2))
      lr = nolds.lyap_r(log, emb_dim=6, lag=2, min_tsep=10, trajectory_len=20)
      self.assertEqual(s, int(np.sign(le)), "r = {}".format(r))
      self.assertEqual(s, int(np.sign(lr)), "r = {}".format(r))

  def test_lyap_lorenz(self):
      """Test hypothesis: Both lyap_r and lyap_e can reconstruct the largest Lyapunov exponent of the Lorenz system.
      
      The parameters for generating the Lorenz system were chosen to be as close as
      possible to the experiments performed by Leonov and Kuznetsov (see [l_4]_)
      and .

      For performance reasons the size of the input data was reduced and therefore the
      assert conditions needed to be relaxed a bit.

      .. [l_4] G. A. Leonov and N. V. Kuznetsov, “On differences and
        similarities in the analysis of Lorenz, Chen, and Lu systems,”
        Applied Mathematics and Computation, vol. 256, pp. 334–343, 2015,
        doi: 10.1016/j.amc.2014.12.132.
      """
      data = datasets.lorenz_euler(3000, 10, 28, 8/3.0, start=[1,1,1], dt=0.01)[1000:]
      lyap_r_args = dict(min_tsep=10, emb_dim=5, tau=0.01, lag=5, trajectory_len=28, fit_offset=8, fit="poly")
      lyap_rx = nolds.lyap_r(data[:, 0], **lyap_r_args)
      lyap_ry = nolds.lyap_r(data[:, 1], **lyap_r_args)
      lyap_rz = nolds.lyap_r(data[:, 2], **lyap_r_args)
      lyap_e_args = dict(min_tsep=10, emb_dim=5, matrix_dim=5, tau=0.01, min_nb=8)
      lyap_ex = nolds.lyap_e(data[:, 0], **lyap_e_args)
      lyap_ey = nolds.lyap_e(data[:, 1], **lyap_e_args)
      lyap_ez = nolds.lyap_e(data[:, 2], **lyap_e_args)
      self.assertAlmostEqual(2.4, lyap_rx, delta=0.5)
      self.assertAlmostEqual(2.4, lyap_ry, delta=0.5)
      self.assertAlmostEqual(2.4, lyap_rz, delta=0.5)
      self.assertGreater(lyap_ex[0], 1.5)
      self.assertGreater(lyap_ey[0], 1.5)
      self.assertGreater(lyap_ez[0], 1.5)

  def test_lyap_fbm(self):
    data = datasets.fbm(1000, H=0.3)
    le = nolds.lyap_e(data, emb_dim=7, matrix_dim=3)
    self.assertGreater(np.max(le), 0)

  def test_lyap_r_limits(self):
    """
    tests if minimal input size is correctly calculated
    """
    np.random.seed(0)
    for i in range(10):
      kwargs = {
        "emb_dim": np.random.randint(1,10),
        "lag": np.random.randint(1,6),
        "min_tsep": np.random.randint(0,5),
        "trajectory_len": np.random.randint(2,10)
      }
      min_len = nolds.lyap_r_len(**kwargs)
      for i in reversed(range(max(1,min_len-5),min_len+5)):
        data = np.random.random(i)
        if i < min_len:
          ## too few data points => execution should fail
          try:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", RuntimeWarning)
              nolds.lyap_r(data, fit="poly", **kwargs)
            msg = "{} data points should be required for kwargs {}, but " \
                + "{} where enough"
            self.fail(msg.format(
              min_len,
              kwargs,
              i
            ))
          except ValueError as e:
            #print(e)
            pass
        else:
          ## enough data points => execution should succeed
          msg = "{} data points should be enough for kwargs {}, but " \
              + " {} where too few"
          try:
            self.assertTrue(
              np.all(np.isfinite(nolds.lyap_r(data, fit="poly", **kwargs))),
              msg.format(min_len, kwargs, i)
            )
          except ValueError as e:
            self.fail(
              msg.format(min_len, kwargs, i) + ", original error: "+str(e)
            )

  def test_lyap_e_limits(self):
    """
    tests if minimal input size is correctly calculated
    """
    np.random.seed(1)
    for i in range(10):
      kwargs = {
        "matrix_dim": np.random.randint(2,10),
        "min_tsep": np.random.randint(0,10),
        "min_nb": np.random.randint(2,15)
      }
      kwargs["emb_dim"] = np.random.randint(1,4) \
                        * (kwargs["matrix_dim"] - 1) + 1
      min_len = nolds.lyap_e_len(**kwargs)
      for i in reversed(range(max(1,min_len-5),min_len+5)):
        data = np.random.random(i)
        if i < min_len:
          ## too few data points => execution should fail
          try:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", RuntimeWarning)
              nolds.lyap_e(data, **kwargs)
            msg = "{} data points should be required for kwargs {}, but " \
                + "{} where enough"
            self.fail(msg.format(
              min_len,
              kwargs,
              i
            ))
          except ValueError as e:
            #print(e)
            pass
        else:
          ## enough data points => execution should succeed
          msg = "{} data points should be enough for kwargs {}, but " \
              + " {} where too few"
          try:
            self.assertTrue(
              np.all(np.isfinite(nolds.lyap_e(data, **kwargs))),
              msg.format(min_len, kwargs, i)
            )
          except ValueError as e:
            self.fail(
              msg.format(min_len, kwargs, i) + ", original error: "+str(e)
            )


class TestNoldsHurst(unittest.TestCase):
  """
  Tests for hurst_rs
  """
  def test_hurst_basic(self):
    np.random.seed(2)
    # strong negative correlation between successive elements
    seq_neg = []
    x = np.random.random()
    for _ in range(10000):
      x = -x + np.random.random() - 0.5
      seq_neg.append(x)
    h_neg = nolds.hurst_rs(seq_neg)
    #print("h_neg = %.3f" % h_neg)
    # expected h is around 0
    self.assertLess(h_neg, 0.3)

    # no correlation, just random noise
    x = np.random.randn(10000)
    h_rand = nolds.hurst_rs(x)
    #print("h_rand = %.3f" % h_rand)
    # expected h is around 0.5
    self.assertAlmostEqual(h_rand, 0.5, delta=0.1)

    # cumulative sum has strong positive correlation between
    # elements
    walk = np.cumsum(x)
    h_walk = nolds.hurst_rs(walk)
    #print("h_walk = %.3f" % h_walk)
    # expected h is around 1.0
    self.assertGreater(h_walk, 0.9)

  def test_hurst_pracma(self):
    """
    Tests for hurst_rs using the same tests as in the R-package pracma
    """
    np.random.seed(3)
    # This test reproduces the results presented by Ian L. Kaplan on
    # bearcave.com
    h72 = nolds.hurst_rs(
      datasets.brown72, fit="poly", corrected=False, unbiased=False,
      nvals=2**np.arange(3,11))
    #print("h72 = %.3f" % h72)
    self.assertAlmostEqual(h72, 0.72, delta=0.01)

    xgn = np.random.normal(size=10000)
    hgn = nolds.hurst_rs(xgn, fit="poly")
    #print("hgn = %.3f" % hgn)
    self.assertAlmostEqual(hgn, 0.5, delta=0.1)

    xlm = np.fromiter(datasets.logistic_map(0.1,1024),dtype="float32")
    hlm = nolds.hurst_rs(xlm, fit="poly", nvals=2**np.arange(3,11))
    #print("hlm = %.3f" % hlm)
    self.assertAlmostEqual(hlm, 0.43, delta=0.05)
  
  def test_hurst_lorenz(self):
    """Test hypothesis: We get correct values for estimating the hurst exponent of the Lorenz system.

    All parameter values are chosen to replicate the experiment by Suyal et al. (see [l_3]_)
    as closely as possible.

    For performance reasons the size of the input data was reduced and therefore the
    assert conditions needed to be relaxed a bit.

    .. [l_3] V. Suyal, A. Prasad, and H. P. Singh, “Nonlinear Time Series
       Analysis of Sunspot Data,” Sol Phys, vol. 260, no. 2, pp. 441–449,
       2009, doi: 10.1007/s11207-009-9467-x.
    """
    data = datasets.lorenz_euler(3000, 10, 28, 8/3.0, start=[1,1,1], dt=0.01)[1000:]
    hurst_rs_args = dict(fit="poly", nvals=nolds.logarithmic_n(10, 70, 1.1))
    hx = nolds.hurst_rs(data[:, 0], **hurst_rs_args)
    hy = nolds.hurst_rs(data[:, 1], **hurst_rs_args)
    hz = nolds.hurst_rs(data[:, 2], **hurst_rs_args)
    self.assertAlmostEqual(0.9, hx, delta=0.05)
    self.assertAlmostEqual(0.9, hy, delta=0.05)
    self.assertAlmostEqual(0.9, hz, delta=0.05)

class TestNoldsDFA(unittest.TestCase):
  """
  Tests for dfa
  """
  def test_dfa_base(self):
    np.random.seed(4)
    # strong negative correlation between successive elements
    seq_neg = []
    x = np.random.random()
    for _ in range(10000):
      x = -x + np.random.random() - 0.5
      seq_neg.append(x)
    h_neg = nolds.dfa(seq_neg)
    # expected h is around 0
    self.assertLess(h_neg, 0.3)

    # no correlation, just random noise
    x = np.random.randn(10000)
    h_rand = nolds.dfa(x)
    # expected h is around 0.5
    self.assertLess(h_rand, 0.7)
    self.assertGreater(h_rand, 0.3)

    # cumulative sum has strong positive correlation between
    # elements
    walk = np.cumsum(x)
    h_walk = nolds.dfa(walk)
    # expected h is around 1.0
    self.assertGreater(h_walk, 0.7)

  def test_dfa_fbm(self):
    hs = [0.3, 0.5, 0.7]
    for h in hs:
      data = datasets.fbm(1000, H=h)
      he = nolds.dfa(data)
      self.assertAlmostEqual(he, h + 1, delta=0.15)

  def test_dfa_lorenz(self):
    """Test hypothesis: We get correct values for estimating the Hurst parameter of the Lorenz system.

    All parameter values are chosen to replicate the experiment by Wallot et al. (see [l_5]_)
    as closely as possible.

    For performance reasons the size of the input data was reduced and therefore the
    assert conditions needed to be relaxed a bit.

    .. [l_5] S. Wallot, J. P. Irmer, M. Tschense, N. Kuznetsov, A. Højlund,
       and M. Dietz, “A Multivariate Method for Dynamic System Analysis:
       Multivariate Detrended Fluctuation Analysis Using Generalized Variance,”
       Topics in Cognitive Science, p. tops.12688, Sep. 2023,
       doi: 10.1111/tops.12688.
    """
    data = datasets.lorenz_euler(120000, 10, 28, 8/3.0, start=[0.1,0.1,0.1], dt=0.002)[20000:]
    nvals = nolds.logarithmic_n(200, len(data)/8, 2**0.2)
    dfa_args = dict(nvals=nvals, order=2, overlap=False, fit_exp="poly")
    dx = nolds.dfa(data[:, 0], **dfa_args)
    dy = nolds.dfa(data[:, 1], **dfa_args)
    dz = nolds.dfa(data[:, 2], **dfa_args)
    self.assertAlmostEqual(1.008, dx, delta=0.04)
    self.assertAlmostEqual(0.926, dy, delta=0.032)
    self.assertAlmostEqual(0.650, dz, delta=0.44)

  def test_dfa_agreement_with_physionet(self):
    """Test hypothesis: Using the same parameters, the output of nolds is identical to the output of PhysioNet."""
    lorenz_x, physionet_points = datasets.load_lorenz_physionet()
    nvals = [round(x) for x in 10 ** physionet_points[:,0]]
    _, (_, nolds_rs, _) = nolds.dfa(lorenz_x, nvals=nvals, overlap=False, fit_exp="poly", debug_data=True)
    nolds_rs_log10 = nolds_rs / np.log(10)
    # assert that sum of squared errors is less than 1e-9
    self.assertLess(sum((physionet_points[:,1] - nolds_rs_log10)**2), 1e-9)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests using Lévy motion require scipy.")
  def test_dfa_levy(self):
    """Test hypothesis: We get correct values for estimating the Hurst parameter of Lévy motion.

    Reference: https://github.com/CSchoel/nolds/issues/17#issuecomment-1905472813.
    """
    alpha = 1.5
    x = levy_stable.rvs(alpha=alpha, beta=0, size=10000)
    h = nolds.dfa(x, fit_exp="poly")
    self.assertAlmostEqual(0.5, h, delta=0.1)



class TestNoldsCorrDim(unittest.TestCase):
  """
  Tests for corr_dim
  """
  def test_corr_dim(self):
    np.random.seed(5)
    n = 1000
    data = np.arange(n)
    cd = nolds.corr_dim(data, 4)
    self.assertAlmostEqual(cd, 1, delta=0.05)
    # TODO what is the prescribed correlation dimension for random data?
    data = np.random.random(n)
    cd = nolds.corr_dim(data, 4, fit="poly")
    self.assertAlmostEqual(cd, 0.5, delta=0.15)
    # TODO test example for cd > 1

  def test_lorenz(self):
    """Test hypothesis: We get correct values for estimating the correlation dimension of the Lorenz system.

    All parameter values are chosen to replicate the experiment by Grassberger and Procaccia (1983)
    as closely as possible.

    For performance reasons the size of the input data was reduced and therefore the
    assert conditions needed to be relaxed a bit. The settings of n, discard,
    lag, emb_dim, and rvals were determined experimentally to find the smallest
    dataset that yields the results reported.

    .. [l_1] P. Grassberger and I. Procaccia, “Measuring the strangeness
       of strange attractors,” Physica D: Nonlinear Phenomena, vol. 9,
       no. 1, pp. 189–208, 1983.
    """
    discard = 5000
    n = 5000
    lag = 10
    emb_dim = 5
    data = datasets.lorenz_euler(n + discard, 10, 28, 8/3, start=(1,1,1), dt=0.012)
    x = data[discard:,1]
    rvals = nolds.logarithmic_r(1, np.e, 1.1)  # determined experimentally
    cd = nolds.corr_dim(x, emb_dim, fit="poly", rvals=rvals, lag=lag)
    self.assertAlmostEqual(cd, 2.05, delta=0.2)

  def test_logistic(self):
    # TODO replicate tests with logistic map from grassberger-procaccia
    pass


class TestNoldsSampEn(unittest.TestCase):
  """
  Tests for sampen
  """
  def test_sampen_base(self):
    data = [0, 1, 5, 4, 1, 0, 1, 5, 3]
    # matches for m=2: 01-01, 15-15
    # matches for m=3: 015-015
    se = nolds.sampen(data)
    self.assertAlmostEqual(se, -np.log(1.0/2), delta=0.01)
    data = [1, 2, 1, 2.4, 1, 4]
    # matches for m=1: 1-1,1-1,2-2.4,1-1
    # matches for m=2: [1,2]-[1,2.4], [2,1]-[2.4,1]
    se = nolds.sampen(data, emb_dim=1, tolerance=0.5)
    self.assertAlmostEqual(se, -np.log(2.0/4), delta=0.01)
    data = [0, 20, 1, 2, 3, 4, 40, 60, 1.4, 2.4, 3.4, 80, 100, 1.4, 2.4, 3.4,
            4, 120, 140, 180]
    # maches for m=3: [1,2,3]-[1.4,2.4,3.4],[1,2,3]-[1.4,2.4,3.4],
    #                 [2,3,4]-[2.4,3.4,4], [1.4,2.4,3.4]-[1.4,2.4,3.4]
    # matches for m=4: [1,2,3,4]-[1.4,2.4,3.4,4]
    se = nolds.sampen(data, emb_dim=3, tolerance=0.5)
    self.assertAlmostEqual(se, -np.log(1.0/4), delta=0.01)

  def test_sampen_logistic(self):
    # logistic map with r = 2.8 => static value
    data = list(datasets.logistic_map(0.45, 1000, r=2.8))
    self.assertAlmostEqual(0, nolds.sampen(data), delta=0.001)
    self.assertAlmostEqual(0, nolds.sampen(data[100:], emb_dim=5), delta=0.001)
    # logistic map with r = 3.3 => oscillation between two values
    data = list(datasets.logistic_map(0.45, 1000, r=3.3))
    self.assertAlmostEqual(0, nolds.sampen(data), delta=0.001)
    self.assertAlmostEqual(0, nolds.sampen(data[100:], emb_dim=5), delta=0.001)
    # logistic map with r = 3.5 => oscillation between four values
    data = list(datasets.logistic_map(0.45, 1000, r=3.5))
    self.assertAlmostEqual(0, nolds.sampen(data), delta=0.001)
    self.assertAlmostEqual(0, nolds.sampen(data[100:], emb_dim=5), delta=0.001)
    # logistic map with r = 3.9 => chaotic behavior
    data = list(datasets.logistic_map(0.45, 1000, r=3.9))
    self.assertAlmostEqual(0.5, nolds.sampen(data[100:]), delta=0.1)
    self.assertAlmostEqual(0.5, nolds.sampen(data[100:], emb_dim=5), delta=0.1)

  def test_sampen_random(self):
    np.random.seed(6)
    # normally distributed random numbers
    data = np.random.randn(10000)
    self.assertAlmostEqual(2.2, nolds.sampen(data), delta=0.1)
    self.assertAlmostEqual(2.2, nolds.sampen(data, emb_dim=2), delta=0.1)
    # TODO add tests with uniformly distributed random numbers

  def test_sampen_sinus(self):
    # TODO add test with sinus signal
    pass


  def test_sampen_lorenz(self):
    """Test hypothesis: We get correct values for estimating the sample entropy of the Lorenz system.

    All parameter values are chosen to replicate the experiment by Kaffashi et al. (2008)
    as closely as possible.

    For performance reasons the size of the input data was reduced and therefore the
    assert conditions needed to be relaxed a bit.

    .. [l_2] F. Kaffashi, R. Foglyano, C. G. Wilson, and K. A. Loparo,
       “The effect of time delay on Approximate & Sample Entropy
       calculations,” Physica D: Nonlinear Phenomena, vol. 237, no. 23,
       pp. 3069–3074, 2008, doi: 10.1016/j.physd.2008.06.005.
    """
    data = datasets.lorenz_euler(3000, 10, 28, 8/3.0, start=[1,1,1], dt=0.01)[1000:]
    sampen_args = dict(emb_dim=2, lag=1)
    sx = nolds.sampen(data[:, 0], **sampen_args)
    sy = nolds.sampen(data[:, 1], **sampen_args)
    sz = nolds.sampen(data[:, 2], **sampen_args)
    self.assertAlmostEqual(0.15, sx, delta=0.05)
    self.assertAlmostEqual(0.15, sy, delta=0.05)
    self.assertAlmostEqual(0.25, sz, delta=0.05)


class RegressionTests(unittest.TestCase):
  """Regression tests for main algorithms.

  These tests are here to safeguard against accidental algorithmic changes such
  as updates to core dependencies such as numpy or the Python standard library.
  """

  def test_sampen(self):
    """Test hypothesis: The exact output of sampen() on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    se = nolds.sampen(data, emb_dim=2, tolerance=None, lag=1, dist=nolds.rowwise_chebyshev, closed=False)
    self.assertAlmostEqual(2.1876999522832743, se, places=14)

  def test_corr_dim(self):
    """Test hypothesis: The exact output of corr_dim() with `fit=poly` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    cd = nolds.corr_dim(data, emb_dim=5, lag=1, rvals=None, dist=nolds.rowwise_euclidean, fit="poly")
    self.assertAlmostEqual(1.303252839255068, cd, places=14)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
  def test_corr_dim_RANSAC(self):
    """Test hypothesis: The exact output of corr_dim() with `fit=RANSAC` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    sd = np.std(data, ddof=1)
    # fix seed
    np.random.seed(42)
    # usa a too wide range for rvals to give RANSAC something to do ;)
    rvals = nolds.logarithmic_r(0.01 * sd, 2 * sd, 1.03)
    cd = nolds.corr_dim(data, emb_dim=5, lag=1, rvals=rvals, dist=nolds.rowwise_euclidean, fit="RANSAC")
    self.assertAlmostEqual(0.44745494643404665, cd, places=14)

  def test_lyap_e(self):
    """Test hypothesis: The exact output of lyap_e() on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    le = nolds.lyap_e(data, emb_dim=10, matrix_dim=4, min_nb=10, min_tsep=1, tau=1)
    expected = np.array([ 0.03779942603329712,  -0.014314012551504982, -0.08436867977030214, -0.22316730257003717])
    for i in range(le.shape[0]):
      self.assertAlmostEqual(expected[i], le[i], places=14, msg=f"{i+1}th Lyapunov exponent doesn't match")

  def test_lyap_r(self):
    """Test hypothesis: The exact output of lyap_r() with `fit=poly` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    le = nolds.lyap_r(data, emb_dim=10, lag=1, min_tsep=1, tau=1, min_neighbors=10, trajectory_len=10, fit="poly")
    expected = 0.094715945307378
    self.assertAlmostEqual(expected, le, places=14)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
  def test_lyap_r_RANSAC(self):
    """Test hypothesis: The exact output of lyap_r() with `fit=RANSAC` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    np.random.seed(42)
    # set lag to 2 for weird duplicate lines
    # set trajectory_len to 100 to get many datapoints for RANSAC to choose from
    le = nolds.lyap_r(data, emb_dim=10, lag=2, min_tsep=1, tau=1, min_neighbors=10, trajectory_len=100, fit="RANSAC")
    expected = 0.0003401212353253564
    self.assertAlmostEqual(expected, le, places=14)

  def test_hurst_rs(self):
    """Test hypothesis: The exact output of hurst_rs() with `fit=poly` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    rs = nolds.hurst_rs(data, nvals=None, fit="poly", corrected=True, unbiased=True)
    expected = 0.5123887964986258
    self.assertAlmostEqual(expected, rs, places=14)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
  def test_hurst_rs_RANSAC(self):
    """Test hypothesis: The exact output of hurst_rs() with `fit=RANSAC` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    np.random.seed(42)
    # increase nsteps in nvals to have more data points for RANSAC to choose from
    nvals = nolds.logmid_n(data.shape[0], ratio=1/4.0, nsteps=100)
    rs = nolds.hurst_rs(data, nvals=nvals, fit="RANSAC", corrected=True, unbiased=True)
    expected = 0.4805431939943321
    self.assertAlmostEqual(expected, rs, places=14)

  def test_dfa(self):
    """Test hypothesis: The exact output of dfa() with `fit_exp=poly` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    h = nolds.dfa(data, nvals=None, overlap=True, order=1, fit_trend="poly", fit_exp="poly")
    expected = 0.5450874638765073
    self.assertAlmostEqual(expected, h, places=14)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
  def test_dfa_RANSAC(self):
    """Test hypothesis: The exact output of dfa() with `fit_exp=RANSAC` on random data hasn't changed since the last version."""
    # adds trend to data to introduce a less clear line for fitting
    data = datasets.load_qrandom()[:1000] + np.arange(1000) * 100
    np.random.seed(42)
    # adds more steps and higher values to nvals to introduce some scattering for RANSAC to have an effect on
    nvals = nolds.logarithmic_n(10, 0.9 * data.shape[0], 1.1)
    h = nolds.dfa(data, nvals=nvals, overlap=True, order=1, fit_trend="poly", fit_exp="RANSAC")
    expected = 1.1372303125405405
    self.assertAlmostEqual(expected, h, places=14)

  def test_mfhurst_b(self):
    """Test hypothesis: The exact output of mfhurst_b() with `fit=poly` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    h = nolds.mfhurst_b(data, qvals=[1], dists=None, fit="poly")
    expected = [-0.00559398934417339]
    self.assertAlmostEqual(expected[0], h[0], places=14)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
  def test_mfhurst_b_RANSAC(self):
    """Test hypothesis: The exact output of mfhurst_b() with `fit=RANSAC` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    np.random.seed(42)
    h = nolds.mfhurst_b(data, qvals=[1], dists=None, fit="RANSAC")
    expected = [-0.009056463064211057]
    self.assertAlmostEqual(expected[0], h[0], places=14)

  def test_mfhurst_dm(self):
    """Test hypothesis: The exact output of mfhurst_dm() with `fit=poly` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    h, _ = nolds.mfhurst_dm(data, qvals=[1], max_dists=range(5, 20), detrend=True, fit="poly")
    expected = [0.008762803881203145]
    self.assertAlmostEqual(expected[0], h[0], places=14)

  @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
  def test_mfhurst_dm_RANSAC(self):
    """Test hypothesis: The exact output of mfhurst_dm() with `fit=RANSAC` on random data hasn't changed since the last version."""
    data = datasets.load_qrandom()[:1000]
    np.random.seed(42)
    h, _ = nolds.mfhurst_dm(data, qvals=[1], max_dists=range(5, 20), detrend=True, fit="RANSAC")
    expected = [0.005324834328837356]
    self.assertAlmostEqual(expected[0], h[0], places=14)


class PreviousDefectTests(unittest.TestCase):
  """Tests that ensure that a previous bug doesn't come back at some point."""

  def test_lyap_r_complex_min_tsep(self):
    """Test hypothesis: The `min_tsep` parameter can be calculated without creating complex numbers.
    
    Previously, this would lead to an exception in the code. See
    https://github.com/CSchoel/nolds/issues/53 for reference.
    """
    data = np.cos(np.arange(100)*0.01)
    # previously this would fail with the following exception:
    #   TypeError: ufunc 'ceil' not supported for the input types, and the
    #   inputs could not be safely coerced to any supported types according to
    #   the casting rule ''safe''
    nolds.lyap_r(data)

if __name__ == "__main__":
  unittest.main()
