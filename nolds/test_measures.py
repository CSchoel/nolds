import numpy as np
# import internal module to test helping functions
from . import measures as nolds
import unittest


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
      for i in range(100):
        x = logistic(x, r)
        log.append(x)
      log = np.array(log, dtype="float32")
      le = np.max(nolds.lyap_e(log, emb_dim=6, matrix_dim=2))
      lr = nolds.lyap_r(log, emb_dim=6, lag=2, min_tsep=10, trajectory_len=20)
      self.assertEqual(s, int(np.sign(le)), "r = {}".format(r))
      self.assertEqual(s, int(np.sign(lr)), "r = {}".format(r))

  def test_lyap_fbm(self):
    data = nolds.fbm(1000, H=0.3)
    le = nolds.lyap_e(data, emb_dim=7, matrix_dim=3)
    self.assertGreater(np.max(le), 0)


class TestNoldsHurst(unittest.TestCase):
  """
  Tests for hurst_rs
  """
  def test_hurst_basic(self):
    # strong negative correlation between successive elements
    seq_neg = []
    x = np.random.random()
    for i in range(10000):
      x = -x + np.random.random() - 0.5
      seq_neg.append(x)
    h_neg = nolds.hurst_rs(seq_neg)
    # expected h is around 0
    self.assertLess(h_neg, 0.3)

    # no correlation, just random noise
    x = np.random.randn(10000)
    h_rand = nolds.hurst_rs(x)
    # expected h is around 0.5
    self.assertLess(h_rand, 0.7)
    self.assertGreater(h_rand, 0.3)

    # cumulative sum has strong positive correlation between
    # elements
    walk = np.cumsum(x)
    h_walk = nolds.hurst_rs(walk)
    # expected h is around 1.0
    self.assertGreater(h_walk, 0.7)


class TestNoldsDFA(unittest.TestCase):
  """
  Tests for dfa
  """
  def test_dfa_base(self):
    # strong negative correlation between successive elements
    seq_neg = []
    x = np.random.random()
    for i in range(10000):
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
      data = nolds.fbm(1000, H=h)
      he = nolds.dfa(data)
      self.assertAlmostEqual(he, h + 1, delta=0.15)


class TestNoldsCorrDim(unittest.TestCase):
  """
  Tests for corr_dim
  """
  def test_corr_dim(self):
    n = 1000
    data = np.arange(n)
    cd = nolds.corr_dim(data, 4)
    self.assertAlmostEquals(cd, 1, delta=0.05)
    data = np.random.random(n)
    cd = nolds.corr_dim(data, 4)
    self.assertAlmostEquals(cd, 0.5, delta=0.1)
    # TODO test example for cd > 1


class TestNoldsSampEn(unittest.TestCase):
  """
  Tests for sampen
  """
  def test_sampen_base(self):
    data = [0, 1, 5, 4, 1, 0, 1, 5, 3]
    se = nolds.sampen(data)
    self.assertAlmostEqual(se, np.log(2), delta=0.01)

if __name__ == "__main__":
  unittest.main()
