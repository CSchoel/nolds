# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (
  bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
  oct, open, pow, round, super, filter, map, zip
)
import numpy as np

# import internal module to test helping functions
from . import measures as nolds
from . import datasets
import unittest
import warnings

# TODO add tests for mfhurst_b and mfhurst_dm

# TODO add more tests using fgn and fbm for hurst_rs and dfa

# TODO split up tests into smaller units => one hypothesis = one test


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
    self.assertTrue(np.alltrue(actual == expected))

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
  def test_corr_dim_logistic(self):
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

if __name__ == "__main__":
  unittest.main()
