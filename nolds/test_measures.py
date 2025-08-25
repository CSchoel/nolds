"""Unit tests for main measures of measures."""

from __future__ import annotations

import unittest
import warnings

# from numpy.testing import assert_equal as assert_array_equal
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from nolds import datasets, measures

# TODO add tests for mfhurst_b and mfhurst_dm

# TODO add more tests using fgn and fbm for hurst_rs and dfa

# TODO split up tests into smaller units => one hypothesis = one test

try:
    from scipy.stats import levy_stable

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestNoldsHelperFunctions(unittest.TestCase):
    """Tests for internal helper functions that are not part of the public API."""

    def assert_array_equal(
        self,
        expected: ArrayLike,
        actual: ArrayLike,
        dtype: DTypeLike = np.float64,
    ) -> None:
        """Test that two arrays are exactly equal.

        Args:
            expected: The expected result
            actual: The actual result
            dtype: dtype of the arrays to compare
        """
        expected = np.asarray(expected, dtype=dtype)
        actual = np.asarray(actual, dtype=dtype)
        diff_indices = np.array((actual != expected).nonzero()).transpose()
        first_diff = diff_indices[0] if diff_indices.shape[0] > 0 else None
        first_diff_msg = f"{expected[first_diff]} != {actual[first_diff]} at {first_diff}"
        msg = (
            f"Arrays differ!\n\nExpected:\n{expected}\n\nActual:\n{actual}"
            f"\n\nFirst difference:\n{first_diff_msg}"
        )
        assert np.all(actual == expected), msg

    def test_delay_embed_lag2(self) -> None:
        """Hypothesis: Setting a lag of 2 skips every second element in orbit vectors."""
        data = np.arange(10, dtype=np.float64)
        embedded = measures.delay_embedding(data, 4, lag=2)
        expected = np.array(
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [2, 4, 6, 8],
                [3, 5, 7, 9],
            ],
            dtype=np.float64,
        )
        self.assert_array_equal(expected, embedded)

    def test_delay_embed(self) -> None:
        """Hypothesis: Default settings produce consecutive slices of same length."""
        data = np.arange(6, dtype=np.float64)
        embedded = measures.delay_embedding(data, 4)
        expected = np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
            ],
            dtype=np.float64,
        )
        self.assert_array_equal(expected, embedded)

    def test_delay_embed_lag3(self) -> None:
        """Hypothesis: Setting a lag of 3 only takes every third element."""
        data = np.arange(10, dtype=np.float64)
        embedded = measures.delay_embedding(data, 4, lag=3)
        expected = np.array(
            [
                [0, 3, 6, 9],
            ],
            dtype=np.float64,
        )
        self.assert_array_equal(expected, embedded)

    def test_delay_embed_empty(self) -> None:
        """Hpoythesis: An error is raised when settings would lead to an empty orbit vector list."""
        data = np.arange(10, dtype=np.float64)
        try:
            embedded = measures.delay_embedding(data, 11)
            msg = (
                "embedding array of size 10 with embedding dimension 11 should fail, got {} instead"
            )
            self.fail(msg.format(embedded))
        except ValueError:
            pass
        data = np.arange(10, dtype=np.float64)
        try:
            embedded = measures.delay_embedding(data, 4, lag=4)
            msg = (
                "embedding array of size 10 with embedding dimension 4 and "
                "lag 4 should fail, got {} instead"
            )
            self.fail(msg.format(embedded))
        except ValueError:
            pass


class TestNoldsUtility(unittest.TestCase):
    """Tests for small utility functions that are part of the public API."""

    def test_binary_n(self) -> None:
        """Hypothesis: binary_n produces exponentially declining numbers."""
        x = measures.binary_n(1000, min_n=50)
        self.assertSequenceEqual(x, [500, 250, 125, 62])

    def test_binary_n_empty(self) -> None:
        """Hypothesis: binary_n gives empty output if min_n is set too high."""
        x = measures.binary_n(50, min_n=50)
        self.assertSequenceEqual(x, [])

    def test_logarithmic_n(self) -> None:
        """Hypothesis: logarithmic_n outputs integers that follow an exponential series."""
        x = measures.logarithmic_n(4, 11, 1.51)
        self.assertSequenceEqual(x, [4, 6, 9])

    def test_logarithmic_r(self) -> None:
        """Hypothesis: logarithmic_r outputs floats that follow an exponential series."""
        x = measures.logarithmic_r(4, 10, 1.51)
        self.assertSequenceEqual(x, [4, 6.04, 9.1204])


class NoldsMeasure(Protocol):
    """Protocol for typing methods that take a float array as first parameter."""

    def __call__(self, data: measures.FloatArrayLike1D) -> Any:  # noqa: ANN401
        """Call the measure."""


class TestNoldsLyap(unittest.TestCase):
    """Tests for lyap_e and lyap_r."""

    def test_lyap_logistic(self) -> None:
        """Hypothesis: The output of lyap_e and lyap_r on a logistic map has the correct sign."""
        rvals = [2.5, 3.4, 3.7, 4.0]
        sign = [-1, -1, 1, 1]
        x0 = 0.1

        def logistic(x: float, r: float) -> float:
            """Logistic map."""
            return r * x * (1 - x)

        for r, s in zip(rvals, sign):
            log = []
            x = x0
            for _ in range(100):
                x = logistic(x, r)
                log.append(x)
            log = np.array(log, dtype=np.float64)
            with self.subTest(meashure="lyap_e", r=r):
                le = np.max(measures.lyap_e(log, emb_dim=6, matrix_dim=2))
                self.assertEqual(s, np.sign(le))
            with self.subTest(meashure="lyap_r", r=r):
                lr = measures.lyap_r(log, emb_dim=6, lag=2, min_tsep=10, trajectory_len=20)
                self.assertEqual(s, np.sign(lr))

    def test_lyap_lorenz(self) -> None:
        """Hypothesis: lyap_r and lyap_e match expected values for the Lorenz system.

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
        data = datasets.lorenz_euler(3000, 10, 28, 8 / 3.0, start=[1, 1, 1], dt=0.01)[1000:]
        lyap_r_args = {
            "min_tsep": 10,
            "emb_dim": 5,
            "tau": 0.01,
            "lag": 5,
            "trajectory_len": 28,
            "fit_offset": 8,
            "fit": "poly",
        }
        lyap_e_args = {"min_tsep": 10, "emb_dim": 5, "matrix_dim": 5, "tau": 0.01, "min_nb": 8}
        with self.subTest(measure="lyap_r", axis="x"):
            lyap_rx = measures.lyap_r(data[:, 0], **lyap_r_args)
            self.assertAlmostEqual(2.4, lyap_rx, delta=0.5)
        with self.subTest(measure="lyap_r", axis="y"):
            lyap_ry = measures.lyap_r(data[:, 1], **lyap_r_args)
            self.assertAlmostEqual(2.4, lyap_ry, delta=0.5)
        with self.subTest(measure="lyap_r", axis="z"):
            lyap_rz = measures.lyap_r(data[:, 2], **lyap_r_args)
            self.assertAlmostEqual(2.4, lyap_rz, delta=0.5)
        with self.subTest(measure="lyap_e", axis="x"):
            lyap_ex = measures.lyap_e(data[:, 0], **lyap_e_args)
            self.assertGreater(lyap_ex[0], 1.5)
        with self.subTest(measure="lyap_e", axis="y"):
            lyap_ey = measures.lyap_e(data[:, 1], **lyap_e_args)
            self.assertGreater(lyap_ey[0], 1.5)
        with self.subTest(measure="lyap_e", axis="z"):
            lyap_ez = measures.lyap_e(data[:, 2], **lyap_e_args)
            self.assertGreater(lyap_ez[0], 1.5)

    def test_lyap_fbm(self) -> None:
        """Hypothesis: lyap_e produces positive output for fractional brownian motion."""
        data = datasets.fbm(1000, H=0.3)
        le = measures.lyap_e(data, emb_dim=7, matrix_dim=3)
        self.assertGreater(float(np.max(le)), 0)

    def assert_insufficient_length(
        self,
        min_len: int,
        kwargs: dict[str, Any],
        input_data: measures.FloatArray1D,
        measure: NoldsMeasure,
    ) -> None:
        """Ensures that the length of the given data would actually lead to an error.

        Args:
            min_len: reported minimum length
            kwargs: kwargs to be passed to the measure
            input_data: data with length less than `min_len`
            measure: the nolds measure to test (either `lyap_r` or `lyap_e`)
        """
        msg = (
            f"{min_len} data points should be required for kwargs {kwargs}, "
            f"but {input_data.shape[0]} were enough"
        )
        with self.assertRaises(ValueError, msg=msg), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            measure(input_data, **kwargs)  # pyright: ignore reportArgumentType

    def assert_sufficient_length(
        self,
        min_len: int,
        kwargs: dict[str, Any],
        input_data: measures.FloatArray1D,
        measure: NoldsMeasure,
    ) -> None:
        """Ensures that the length of the given data does not lead to an error.

        Args:
            min_len: reported minimum length
            kwargs: kwargs to be passed to the measure
            input_data: data with length at least `min_len`
            measure: the nolds measure to test (either `lyap_r` or `lyap_e`)
        """
        msg = (
            f"{min_len} data points should be enough for kwargs {kwargs}, but "
            f"{input_data.shape[0]} were too few"
        )
        try:
            assert np.all(np.isfinite(measure(input_data, **kwargs))), msg
        except ValueError as e:
            raise ValueError(msg) from e

    def test_lyap_r_limits(self) -> None:
        """Hypothesis: Minimal input size for lyap_r is correctly calculated.

        For each of 10 random parameter settings, we test a range of input sizes around
        the supposed minimum number of inputs. For numbers smaller than the calculated
        minimum we expect the call of lyap_r to fail, for numbers greater or equal, it
        should succeed.
        """
        rng = np.random.default_rng(seed=0)
        for _ in range(10):
            kwargs: dict[str, Any] = {
                "emb_dim": rng.integers(1, 10),
                "lag": rng.integers(1, 6),
                "min_tsep": rng.integers(0, 5),
                "trajectory_len": rng.integers(2, 10),
            }
            min_len = measures.lyap_r_len(**kwargs)  # pyright: ignore reportArgumentType
            kwargs["fit"] = "poly"
            for actual_len in reversed(range(max(1, min_len - 5), min_len + 5)):
                data = rng.random(actual_len)
                with self.subTest(
                    emb_dim=kwargs["emb_dim"],
                    lag=kwargs["lag"],
                    min_tsep=kwargs["min_tsep"],
                    trajectory_len=kwargs["trajectory_len"],
                    min_len=min_len,
                    actual_len=actual_len,
                ):
                    if actual_len < min_len:
                        ## too few data points => execution should fail
                        self.assert_insufficient_length(
                            min_len=min_len, kwargs=kwargs, input_data=data, measure=measures.lyap_r
                        )
                    else:
                        ## enough data points => execution should succeed
                        self.assert_sufficient_length(
                            min_len=min_len, kwargs=kwargs, input_data=data, measure=measures.lyap_r
                        )

    def test_lyap_e_limits(self) -> None:
        """Tests if minimal input size is correctly calculated."""
        rng = np.random.default_rng(seed=1)
        for _ in range(10):
            kwargs = {
                "matrix_dim": rng.integers(2, 10),
                "min_tsep": rng.integers(0, 10),
                "min_nb": rng.integers(2, 15),
            }
            kwargs["emb_dim"] = rng.integers(1, 4) * (kwargs["matrix_dim"] - 1) + 1
            min_len = measures.lyap_e_len(**kwargs)  # pyright: ignore reportArgumentType
            for actual_len in reversed(range(max(1, min_len - 5), min_len + 5)):
                data = rng.random(actual_len)
                with self.subTest(
                    matrix_dim=kwargs["matrix_dim"],
                    min_tsep=kwargs["min_tsep"],
                    min_nb=kwargs["min_nb"],
                    min_len=min_len,
                    actual_len=actual_len,
                ):
                    if actual_len < min_len:
                        ## too few data points => execution should fail
                        self.assert_insufficient_length(
                            min_len=min_len, kwargs=kwargs, input_data=data, measure=measures.lyap_e
                        )
                    else:
                        ## enough data points => execution should succeed
                        self.assert_sufficient_length(
                            min_len=min_len, kwargs=kwargs, input_data=data, measure=measures.lyap_e
                        )


class TestNoldsHurst(unittest.TestCase):
    """Tests for hurst_rs."""

    def test_hurst_basic(self) -> None:
        np.random.seed(2)
        # strong negative correlation between successive elements
        seq_neg = []
        x = np.random.random()
        for _ in range(10000):
            x = -x + np.random.random() - 0.5
            seq_neg.append(x)
        h_neg = measures.hurst_rs(seq_neg)
        # print("h_neg = %.3f" % h_neg)
        # expected h is around 0
        assert h_neg < 0.3

        # no correlation, just random noise
        x = np.random.randn(10000)
        h_rand = measures.hurst_rs(x)
        # print("h_rand = %.3f" % h_rand)
        # expected h is around 0.5
        self.assertAlmostEqual(h_rand, 0.5, delta=0.1)

        # cumulative sum has strong positive correlation between
        # elements
        walk = np.cumsum(x)
        h_walk = measures.hurst_rs(walk)
        # print("h_walk = %.3f" % h_walk)
        # expected h is around 1.0
        assert h_walk > 0.9

    def test_hurst_pracma(self) -> None:
        """Tests for hurst_rs using the same tests as in the R-package pracma."""
        np.random.seed(3)
        # This test reproduces the results presented by Ian L. Kaplan on
        # bearcave.com
        h72 = measures.hurst_rs(
            datasets.brown72,
            fit="poly",
            corrected=False,
            unbiased=False,
            nvals=2 ** np.arange(3, 11),
        )
        # print("h72 = %.3f" % h72)
        self.assertAlmostEqual(h72, 0.72, delta=0.01)

        xgn = np.random.normal(size=10000)
        hgn = measures.hurst_rs(xgn, fit="poly")
        # print("hgn = %.3f" % hgn)
        self.assertAlmostEqual(hgn, 0.5, delta=0.1)

        xlm = np.fromiter(datasets.logistic_map(0.1, 1024), dtype=np.float64)
        hlm = measures.hurst_rs(xlm, fit="poly", nvals=2 ** np.arange(3, 11))
        # print("hlm = %.3f" % hlm)
        self.assertAlmostEqual(hlm, 0.43, delta=0.05)

    def test_hurst_lorenz(self) -> None:
        """Test hypothesis: We get correct values for estimating the hurst exponent of the Lorenz system.

        All parameter values are chosen to replicate the experiment by Suyal et al. (see [l_3]_)
        as closely as possible.

        For performance reasons the size of the input data was reduced and therefore the
        assert conditions needed to be relaxed a bit.

        .. [l_3] V. Suyal, A. Prasad, and H. P. Singh, “Nonlinear Time Series
           Analysis of Sunspot Data,” Sol Phys, vol. 260, no. 2, pp. 441–449,
           2009, doi: 10.1007/s11207-009-9467-x.
        """
        data = datasets.lorenz_euler(3000, 10, 28, 8 / 3.0, start=[1, 1, 1], dt=0.01)[1000:]
        hurst_rs_args = {"fit": "poly", "nvals": measures.logarithmic_n(10, 70, 1.1)}
        hx = measures.hurst_rs(data[:, 0], **hurst_rs_args)
        hy = measures.hurst_rs(data[:, 1], **hurst_rs_args)
        hz = measures.hurst_rs(data[:, 2], **hurst_rs_args)
        self.assertAlmostEqual(0.9, hx, delta=0.05)
        self.assertAlmostEqual(0.9, hy, delta=0.05)
        self.assertAlmostEqual(0.9, hz, delta=0.05)


class TestNoldsDFA(unittest.TestCase):
    """Tests for dfa."""

    def test_dfa_base(self) -> None:
        np.random.seed(4)
        # strong negative correlation between successive elements
        seq_neg = []
        x = np.random.random()
        for _ in range(10000):
            x = -x + np.random.random() - 0.5
            seq_neg.append(x)
        h_neg = measures.dfa(seq_neg)
        # expected h is around 0
        assert h_neg < 0.3

        # no correlation, just random noise
        x = np.random.randn(10000)
        h_rand = measures.dfa(x)
        # expected h is around 0.5
        assert h_rand < 0.7
        assert h_rand > 0.3

        # cumulative sum has strong positive correlation between
        # elements
        walk = np.cumsum(x)
        h_walk = measures.dfa(walk)
        # expected h is around 1.0
        assert h_walk > 0.7

    def test_dfa_fbm(self) -> None:
        hs = [0.3, 0.5, 0.7]
        for h in hs:
            data = datasets.fbm(1000, H=h)
            he = measures.dfa(data)
            self.assertAlmostEqual(he, h + 1, delta=0.15)

    def test_dfa_lorenz(self) -> None:
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
        data = datasets.lorenz_euler(120000, 10, 28, 8 / 3.0, start=[0.1, 0.1, 0.1], dt=0.002)[
            20000:
        ]
        nvals = measures.logarithmic_n(200, len(data) / 8, 2**0.2)
        dfa_args = {"nvals": nvals, "order": 2, "overlap": False, "fit_exp": "poly"}
        dx = measures.dfa(data[:, 0], **dfa_args)
        dy = measures.dfa(data[:, 1], **dfa_args)
        dz = measures.dfa(data[:, 2], **dfa_args)
        self.assertAlmostEqual(1.008, dx, delta=0.04)
        self.assertAlmostEqual(0.926, dy, delta=0.032)
        self.assertAlmostEqual(0.650, dz, delta=0.44)

    def test_dfa_agreement_with_physionet(self) -> None:
        """Test hypothesis: Using the same parameters, the output of nolds is identical to the output of PhysioNet."""
        lorenz_x, physionet_points = datasets.load_lorenz_physionet()
        nvals = [round(x) for x in 10 ** physionet_points[:, 0]]
        _, (_, nolds_rs, _) = measures.dfa(
            lorenz_x, nvals=nvals, overlap=False, fit_exp="poly", debug_data=True
        )
        nolds_rs_log10 = nolds_rs / np.log(10)
        # assert that sum of squared errors is less than 1e-9
        assert sum((physionet_points[:, 1] - nolds_rs_log10) ** 2) < 1e-09

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests using Lévy motion require scipy.")
    def test_dfa_levy(self) -> None:
        """Test hypothesis: We get correct values for estimating the Hurst parameter of Lévy motion.

        Reference: https://github.com/CSchoel/nolds/issues/17#issuecomment-1905472813.
        """
        alpha = 1.5
        x = levy_stable.rvs(alpha=alpha, beta=0, size=10000)
        h = measures.dfa(x, fit_exp="poly")
        self.assertAlmostEqual(0.5, h, delta=0.1)


class TestNoldsCorrDim(unittest.TestCase):
    """Tests for corr_dim."""

    def test_corr_dim(self) -> None:
        np.random.seed(5)
        n = 1000
        data = np.arange(n)
        cd = measures.corr_dim(data, 4)
        self.assertAlmostEqual(cd, 1, delta=0.05)
        # TODO what is the prescribed correlation dimension for random data?
        data = np.random.random(n)
        cd = measures.corr_dim(data, 4, fit="poly")
        self.assertAlmostEqual(cd, 0.5, delta=0.15)
        # TODO test example for cd > 1

    def test_lorenz(self) -> None:
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
        data = datasets.lorenz_euler(n + discard, 10, 28, 8 / 3, start=(1, 1, 1), dt=0.012)
        x = data[discard:, 1]
        rvals = measures.logarithmic_r(1, np.e, 1.1)  # determined experimentally
        cd = measures.corr_dim(x, emb_dim, fit="poly", rvals=rvals, lag=lag)
        self.assertAlmostEqual(cd, 2.05, delta=0.2)

    def test_logistic(self) -> None:
        # TODO replicate tests with logistic map from grassberger-procaccia
        pass


class TestNoldsSampEn(unittest.TestCase):
    """Tests for sampen."""

    def test_sampen_base(self) -> None:
        data = [0, 1, 5, 4, 1, 0, 1, 5, 3]
        # matches for m=2: 01-01, 15-15
        # matches for m=3: 015-015
        se = measures.sampen(data)
        self.assertAlmostEqual(se, -np.log(1.0 / 2), delta=0.01)
        data = [1, 2, 1, 2.4, 1, 4]
        # matches for m=1: 1-1,1-1,2-2.4,1-1
        # matches for m=2: [1,2]-[1,2.4], [2,1]-[2.4,1]
        se = measures.sampen(data, emb_dim=1, tolerance=0.5)
        self.assertAlmostEqual(se, -np.log(2.0 / 4), delta=0.01)
        data = [0, 20, 1, 2, 3, 4, 40, 60, 1.4, 2.4, 3.4, 80, 100, 1.4, 2.4, 3.4, 4, 120, 140, 180]
        # maches for m=3: [1,2,3]-[1.4,2.4,3.4],[1,2,3]-[1.4,2.4,3.4],
        #                 [2,3,4]-[2.4,3.4,4], [1.4,2.4,3.4]-[1.4,2.4,3.4]
        # matches for m=4: [1,2,3,4]-[1.4,2.4,3.4,4]
        se = measures.sampen(data, emb_dim=3, tolerance=0.5)
        self.assertAlmostEqual(se, -np.log(1.0 / 4), delta=0.01)

    def test_sampen_logistic(self) -> None:
        # logistic map with r = 2.8 => static value
        data = list(datasets.logistic_map(0.45, 1000, r=2.8))
        self.assertAlmostEqual(0, measures.sampen(data), delta=0.001)
        self.assertAlmostEqual(0, measures.sampen(data[100:], emb_dim=5), delta=0.001)
        # logistic map with r = 3.3 => oscillation between two values
        data = list(datasets.logistic_map(0.45, 1000, r=3.3))
        self.assertAlmostEqual(0, measures.sampen(data), delta=0.001)
        self.assertAlmostEqual(0, measures.sampen(data[100:], emb_dim=5), delta=0.001)
        # logistic map with r = 3.5 => oscillation between four values
        data = list(datasets.logistic_map(0.45, 1000, r=3.5))
        self.assertAlmostEqual(0, measures.sampen(data), delta=0.001)
        self.assertAlmostEqual(0, measures.sampen(data[100:], emb_dim=5), delta=0.001)
        # logistic map with r = 3.9 => chaotic behavior
        data = list(datasets.logistic_map(0.45, 1000, r=3.9))
        self.assertAlmostEqual(0.5, measures.sampen(data[100:]), delta=0.1)
        self.assertAlmostEqual(0.5, measures.sampen(data[100:], emb_dim=5), delta=0.1)

    def test_sampen_random(self) -> None:
        np.random.seed(6)
        # normally distributed random numbers
        data = np.random.randn(10000)
        self.assertAlmostEqual(2.2, measures.sampen(data), delta=0.1)
        self.assertAlmostEqual(2.2, measures.sampen(data, emb_dim=2), delta=0.1)
        # TODO add tests with uniformly distributed random numbers

    def test_sampen_sinus(self) -> None:
        # TODO add test with sinus signal
        pass

    def test_sampen_lorenz(self) -> None:
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
        data = datasets.lorenz_euler(3000, 10, 28, 8 / 3.0, start=[1, 1, 1], dt=0.01)[1000:]
        sampen_args = {"emb_dim": 2, "lag": 1}
        sx = measures.sampen(data[:, 0], **sampen_args)
        sy = measures.sampen(data[:, 1], **sampen_args)
        sz = measures.sampen(data[:, 2], **sampen_args)
        self.assertAlmostEqual(0.15, sx, delta=0.05)
        self.assertAlmostEqual(0.15, sy, delta=0.05)
        self.assertAlmostEqual(0.25, sz, delta=0.05)


class RegressionTests(unittest.TestCase):
    """Regression tests for main algorithms.

    These tests are here to safeguard against accidental algorithmic changes such
    as updates to core dependencies such as numpy or the Python standard library.
    """

    def test_sampen(self) -> None:
        """Test hypothesis: The exact output of sampen() on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        se = measures.sampen(
            data, emb_dim=2, tolerance=None, lag=1, dist=measures.rowwise_chebyshev, closed=False
        )
        self.assertAlmostEqual(2.1876999522832743, se, places=14)

    def test_corr_dim(self) -> None:
        """Test hypothesis: The exact output of corr_dim() with `fit=poly` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        cd = measures.corr_dim(
            data, emb_dim=5, lag=1, rvals=None, dist=measures.rowwise_euclidean, fit="poly"
        )
        self.assertAlmostEqual(1.303252839255068, cd, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_corr_dim_RANSAC(self) -> None:
        """Test hypothesis: The exact output of corr_dim() with `fit=RANSAC` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        sd = np.std(data, ddof=1)
        # fix seed
        np.random.seed(42)
        # usa a too wide range for rvals to give RANSAC something to do ;)
        rvals = measures.logarithmic_r(0.01 * sd, 2 * sd, 1.03)
        cd = measures.corr_dim(
            data, emb_dim=5, lag=1, rvals=rvals, dist=measures.rowwise_euclidean, fit="RANSAC"
        )
        self.assertAlmostEqual(0.44745494643404665, cd, places=14)

    def test_lyap_e(self) -> None:
        """Test hypothesis: The exact output of lyap_e() on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        le = measures.lyap_e(data, emb_dim=10, matrix_dim=4, min_nb=10, min_tsep=1, tau=1)
        expected = np.array(
            [0.03779942603329712, -0.014314012551504982, -0.08436867977030214, -0.22316730257003717]
        )
        for i in range(le.shape[0]):
            self.assertAlmostEqual(
                expected[i], le[i], places=14, msg=f"{i + 1}th Lyapunov exponent doesn't match"
            )

    def test_lyap_r(self) -> None:
        """Test hypothesis: The exact output of lyap_r() with `fit=poly` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        le = measures.lyap_r(
            data,
            emb_dim=10,
            lag=1,
            min_tsep=1,
            tau=1,
            min_neighbors=10,
            trajectory_len=10,
            fit="poly",
        )
        expected = 0.094715945307378
        self.assertAlmostEqual(expected, le, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_lyap_r_RANSAC(self) -> None:
        """Test hypothesis: The exact output of lyap_r() with `fit=RANSAC` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        np.random.seed(42)
        # set lag to 2 for weird duplicate lines
        # set trajectory_len to 100 to get many datapoints for RANSAC to choose from
        le = measures.lyap_r(
            data,
            emb_dim=10,
            lag=2,
            min_tsep=1,
            tau=1,
            min_neighbors=10,
            trajectory_len=100,
            fit="RANSAC",
        )
        expected = 0.0003401212353253564
        self.assertAlmostEqual(expected, le, places=14)

    def test_hurst_rs(self) -> None:
        """Test hypothesis: The exact output of hurst_rs() with `fit=poly` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        rs = measures.hurst_rs(data, nvals=None, fit="poly", corrected=True, unbiased=True)
        expected = 0.5123887964986258
        self.assertAlmostEqual(expected, rs, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_hurst_rs_RANSAC(self) -> None:
        """Test hypothesis: The exact output of hurst_rs() with `fit=RANSAC` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        np.random.seed(42)
        # increase nsteps in nvals to have more data points for RANSAC to choose from
        nvals = measures.logmid_n(data.shape[0], ratio=1 / 4.0, nsteps=100)
        rs = measures.hurst_rs(data, nvals=nvals, fit="RANSAC", corrected=True, unbiased=True)
        expected = 0.4805431939943321
        self.assertAlmostEqual(expected, rs, places=14)

    def test_dfa(self) -> None:
        """Test hypothesis: The exact output of dfa() with `fit_exp=poly` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        h = measures.dfa(data, nvals=None, overlap=True, order=1, fit_trend="poly", fit_exp="poly")
        expected = 0.5450874638765073
        self.assertAlmostEqual(expected, h, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_dfa_RANSAC(self) -> None:
        """Test hypothesis: The exact output of dfa() with `fit_exp=RANSAC` on random data hasn't changed since the last version."""
        # adds trend to data to introduce a less clear line for fitting
        data = datasets.load_qrandom()[:1000] + np.arange(1000) * 100
        np.random.seed(42)
        # adds more steps and higher values to nvals to introduce some scattering for RANSAC to have an effect on
        nvals = measures.logarithmic_n(10, 0.9 * data.shape[0], 1.1)
        h = measures.dfa(
            data, nvals=nvals, overlap=True, order=1, fit_trend="poly", fit_exp="RANSAC"
        )
        expected = 1.1372303125405405
        self.assertAlmostEqual(expected, h, places=14)

    def test_mfhurst_b(self) -> None:
        """Test hypothesis: The exact output of mfhurst_b() with `fit=poly` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        h = measures.mfhurst_b(data, qvals=[1], dists=None, fit="poly")
        expected = [-0.00559398934417339]
        self.assertAlmostEqual(expected[0], h[0], places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_mfhurst_b_RANSAC(self) -> None:
        """Test hypothesis: The exact output of mfhurst_b() with `fit=RANSAC` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        np.random.seed(42)
        h = measures.mfhurst_b(data, qvals=[1], dists=None, fit="RANSAC")
        expected = [-0.009056463064211057]
        self.assertAlmostEqual(expected[0], h[0], places=14)

    def test_mfhurst_dm(self) -> None:
        """Test hypothesis: The exact output of mfhurst_dm() with `fit=poly` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        h, _ = measures.mfhurst_dm(
            data, qvals=[1], max_dists=range(5, 20), detrend=True, fit="poly"
        )
        expected = [0.008762803881203145]
        self.assertAlmostEqual(expected[0], h[0], places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_mfhurst_dm_RANSAC(self) -> None:
        """Test hypothesis: The exact output of mfhurst_dm() with `fit=RANSAC` on random data hasn't changed since the last version."""
        data = datasets.load_qrandom()[:1000]
        np.random.seed(42)
        h, _ = measures.mfhurst_dm(
            data, qvals=[1], max_dists=range(5, 20), detrend=True, fit="RANSAC"
        )
        expected = [0.005324834328837356]
        self.assertAlmostEqual(expected[0], h[0], places=14)


class PreviousDefectTests(unittest.TestCase):
    """Tests that ensure that a previous bug doesn't come back at some point."""

    def test_lyap_r_complex_min_tsep(self) -> None:
        """Test hypothesis: The `min_tsep` parameter can be calculated without creating complex numbers.

        Previously, this would lead to an exception in the code. See
        https://github.com/CSchoel/nolds/issues/53 for reference.
        """
        data = np.cos(np.arange(100) * 0.01)
        # previously this would fail with the following exception:
        #   TypeError: ufunc 'ceil' not supported for the input types, and the
        #   inputs could not be safely coerced to any supported types according to
        #   the casting rule ''safe''
        measures.lyap_r(data)


if __name__ == "__main__":
    unittest.main()
