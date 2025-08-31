"""Unit tests for main measures of measures."""

from __future__ import annotations

import unittest
import warnings
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.testing import assert_almost_equal

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

from nolds import datasets, measures

# TODO: add tests for mfhurst_b and mfhurst_dm

# TODO: add more tests using fgn and fbm for hurst_rs and dfa

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
        """Hypothesis: An error is raised when settings would lead to an empty orbit vector list."""
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

        for r, s in zip(rvals, sign, strict=True):
            log = []
            x = x0
            for _ in range(100):
                x = logistic(x, r)
                log.append(x)
            log = np.array(log, dtype=np.float64)
            with self.subTest(measure="lyap_e", r=r):
                le = np.max(measures.lyap_e(log, emb_dim=6, matrix_dim=2))
                self.assertEqual(s, np.sign(le))
            with self.subTest(measure="lyap_r", r=r):
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
        lyap_e_args = {
            "min_tsep": 10,
            "emb_dim": 5,
            "matrix_dim": 5,
            "tau": 0.01,
            "min_nb": 8,
        }
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
                            min_len=min_len,
                            kwargs=kwargs,
                            input_data=data,
                            measure=measures.lyap_r,
                        )
                    else:
                        ## enough data points => execution should succeed
                        self.assert_sufficient_length(
                            min_len=min_len,
                            kwargs=kwargs,
                            input_data=data,
                            measure=measures.lyap_r,
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
                            min_len=min_len,
                            kwargs=kwargs,
                            input_data=data,
                            measure=measures.lyap_e,
                        )
                    else:
                        ## enough data points => execution should succeed
                        self.assert_sufficient_length(
                            min_len=min_len,
                            kwargs=kwargs,
                            input_data=data,
                            measure=measures.lyap_e,
                        )


class TestNoldsHurst(unittest.TestCase):
    """Tests for hurst_rs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create data required for test methods."""
        rng = np.random.default_rng(seed=2)
        # strong negative correlation between successive elements
        cls.negative_correlation = []
        x = rng.random()
        for _ in range(10000):
            x = -x + rng.random() - 0.5
            cls.negative_correlation.append(x)
        # no correlation, just gaussian noise
        cls.no_correlation = rng.standard_normal(10000)
        # cumulative sum has strong positive correlation between
        # elements
        cls.positive_correlation = np.cumsum(cls.no_correlation)

    def test_hurst_negative_correlation(self) -> None:
        """Hypothesis: H < 0.5 for data with negative correlation between successive elements."""
        h_neg = measures.hurst_rs(self.negative_correlation)
        # expected h is around 0
        self.assertLess(h_neg, 0.3)

    def test_hurst_gaussian_noise(self) -> None:
        """Hypothesis: H ~= 0.5 for gaussian noise."""
        h_rand = measures.hurst_rs(self.no_correlation)
        # expected h is around 0.5
        self.assertAlmostEqual(h_rand, 0.5, delta=0.1)

    def test_hurst_positive_correlation(self) -> None:
        """Hypothesis: H > 0.5 for data with positive correlation between successive elements."""
        h_walk = measures.hurst_rs(self.positive_correlation)
        # expected h is around 1.0
        self.assertGreater(h_walk, 0.9)

    def test_hurst_pracma_bearcave(self) -> None:
        """Hypothesis: `hurst_rs` passes test from R-package pracma using brown72 dataset."""
        # This test reproduces the results presented by Ian L. Kaplan on
        # http://bearcave.com/misl/misl_tech/wavelets/hurst/index.html
        h72 = measures.hurst_rs(
            datasets.brown72,
            fit="poly",
            corrected=False,
            unbiased=False,
            nvals=2 ** np.arange(3, 11),
        )
        self.assertAlmostEqual(h72, 0.72, delta=0.01)

    def test_hurst_pracma_logistic(self) -> None:
        """Hypothesis: `hurst_rs` passes test from R-package pracma using logistic map."""
        xlm = np.fromiter(datasets.logistic_map(0.1, 1024), dtype=np.float64)
        hlm = measures.hurst_rs(xlm, fit="poly", nvals=2 ** np.arange(3, 11))
        self.assertAlmostEqual(hlm, 0.43, delta=0.05)

    def test_hurst_lorenz(self) -> None:
        """Hypothesis: We get correct values for estimating the hurst exponent of the Lorenz system.

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
        with self.subTest(axis="x"):
            hx = measures.hurst_rs(data[:, 0], **hurst_rs_args)
            self.assertAlmostEqual(0.9, hx, delta=0.05)
        with self.subTest(axis="y"):
            hy = measures.hurst_rs(data[:, 1], **hurst_rs_args)
            self.assertAlmostEqual(0.9, hy, delta=0.05)
        with self.subTest(axis="z"):
            hz = measures.hurst_rs(data[:, 2], **hurst_rs_args)
            self.assertAlmostEqual(0.9, hz, delta=0.05)


class TestNoldsDFA(unittest.TestCase):
    """Tests for dfa."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create data required for test methods."""
        rng = np.random.default_rng(seed=4)
        # strong negative correlation between successive elements
        cls.negative_correlation = []
        x = rng.random()
        for _ in range(10000):
            x = -x + rng.random() - 0.5
            cls.negative_correlation.append(x)
        # no correlation, just gaussian noise
        cls.no_correlation = rng.standard_normal(10000)
        # cumulative sum has strong positive correlation between
        # elements
        cls.positive_correlation = np.cumsum(cls.no_correlation)

    def test_dfa_negative_correlation(self) -> None:
        """Hypothesis: H < 0.5 for data with negative correlation between successive elements."""
        h_neg = measures.dfa(self.negative_correlation)
        # expected h is around 0
        self.assertLess(h_neg, 0.3)

    def test_dfa_no_correlation(self) -> None:
        """Hypothesis: H ~= 0.5 for gaussian noise."""
        h_rand = measures.dfa(self.no_correlation)
        self.assertAlmostEqual(h_rand, 0.5, delta=0.2)

    def test_dfa_positive_correlation(self) -> None:
        """Hypothesis: H > 0.5 for data with positive correlation between successive elements."""
        h_walk = measures.dfa(self.positive_correlation)
        # expected h is around 1.0
        self.assertGreater(h_walk, 0.7)

    def test_dfa_fbm(self) -> None:
        """Hypothesis: H ~= h + 1 for fractional brownian motion with Hurst parameter h."""
        hs = [0.3, 0.5, 0.7]
        for h in hs:
            with self.subTest(h=h):
                data = datasets.fbm(1000, H=h)
                he = measures.dfa(data)
                self.assertAlmostEqual(he, h + 1, delta=0.15)

    def test_dfa_lorenz(self) -> None:
        """Hypothesis: We get correct values for the Lorenz system.

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
        nvals = measures.logarithmic_n(200, np.ceil(len(data) / 8), 2**0.2)
        dfa_args = {"nvals": nvals, "order": 2, "overlap": False, "fit_exp": "poly"}
        with self.subTest(axis="x"):
            dx = measures.dfa(data[:, 0], **dfa_args)
            self.assertAlmostEqual(1.008, dx, delta=0.04)
        with self.subTest(axis="y"):
            dy = measures.dfa(data[:, 1], **dfa_args)
            self.assertAlmostEqual(0.926, dy, delta=0.032)
        with self.subTest(axis="z"):
            dz = measures.dfa(data[:, 2], **dfa_args)
            self.assertAlmostEqual(0.650, dz, delta=0.44)

    def test_dfa_agreement_with_physionet(self) -> None:
        """Hypothesis: The output of nolds is identical to the output of PhysioNet."""
        lorenz_x, physionet_points = datasets.load_lorenz_physionet()
        nvals = [round(x) for x in 10 ** physionet_points[:, 0]]
        _, (_, nolds_rs, _) = measures.dfa(
            lorenz_x, nvals=nvals, overlap=False, fit_exp="poly", debug_data=True
        )
        nolds_rs_log10 = nolds_rs / np.log(10)
        with self.subTest(kind="individual"):
            assert_almost_equal(nolds_rs_log10, physionet_points[:, 1], decimal=5)
        with self.subTest(kind="sse"):
            # assert that sum of squared errors is less than 1e-9
            sse = sum((physionet_points[:, 1] - nolds_rs_log10) ** 2)
            self.assertLess(sse, 1e-09)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests using Lévy motion require scipy.")
    def test_dfa_levy(self) -> None:
        """Hypothesis: We get correct values for estimating the Hurst parameter of Lévy motion.

        Reference: https://github.com/CSchoel/nolds/issues/17#issuecomment-1905472813.
        """
        alpha = 1.5
        x = cast("np.typing.NDArray[Any]", levy_stable.rvs(alpha=alpha, beta=0, size=10000))
        h = measures.dfa(x, fit_exp="poly")
        self.assertAlmostEqual(0.5, h, delta=0.1)


class TestNoldsCorrDim(unittest.TestCase):
    """Tests for corr_dim."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create data required for test methods."""
        rng = np.random.default_rng(seed=5)
        n = 1000
        cls.cd1 = np.arange(n)
        # TODO: what is the prescribed correlation dimension for random data?
        cls.cd0p5 = rng.random(n)

    def test_corr_dim_1(self) -> None:
        """Hypothesis: Correlation dimensions is close to 1 for highly correlated dataset."""
        cd = measures.corr_dim(self.cd1, 4)
        self.assertAlmostEqual(cd, 1, delta=0.05)

    def test_corr_dim_0p5(self) -> None:
        """Hypothesis: Correlation dimension is close to 0.5 for dataset without correlations."""
        cd = measures.corr_dim(self.cd0p5, 4, fit="poly")
        self.assertAlmostEqual(cd, 0.5, delta=0.15)

    # TODO: test example for cd > 1

    def test_lorenz(self) -> None:
        """Hypothesis: We get correct values for the Lorenz system.

        All parameter values are chosen to replicate the experiment by Grassberger and Procaccia
        (1983) as closely as possible.

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
        data = datasets.lorenz_euler(n + discard, 10, 28, 8 / 3, start=[1, 1, 1], dt=0.012)
        x = data[discard:, 1]
        rvals = measures.logarithmic_r(1, np.e, 1.1)  # determined experimentally
        cd = measures.corr_dim(x, emb_dim, fit="poly", rvals=rvals, lag=lag)
        self.assertAlmostEqual(cd, 2.05, delta=0.2)

    def test_logistic(self) -> None:
        """Hypothesis:  We get correct values for the logistic map."""
        # TODO: replicate tests with logistic map from grassberger-procaccia


class TestNoldsSampEn(unittest.TestCase):
    """Tests for sampen."""

    def test_sampen_2(self) -> None:
        """Hypothesis: `sampen` gives expected results for toy dataset with `emb_dim=2`."""
        data = [0, 1, 5, 4, 1, 0, 1, 5, 3]
        # matches for m=2: 01-01, 15-15
        # matches for m=3: 015-015
        se = measures.sampen(data)
        self.assertAlmostEqual(se, -np.log(1.0 / 2), delta=0.01)

    def test_sampen_1(self) -> None:
        """Hypothesis: `sampen` gives expected results for toy dataset with `emb_dim=1`."""
        data = [1, 2, 1, 2.4, 1, 4]
        # matches for m=1: 1-1,1-1,2-2.4,1-1
        # matches for m=2: [1,2]-[1,2.4], [2,1]-[2.4,1]
        se = measures.sampen(data, emb_dim=1, tolerance=0.5)
        self.assertAlmostEqual(se, -np.log(2.0 / 4), delta=0.01)

    def test_sampen_3(self) -> None:
        """Hypothesis: `sampen` gives expected results for toy dataset with `emb_dim=3`."""
        data = [
            0,
            20,
            1,
            2,
            3,
            4,
            40,
            60,
            1.4,
            2.4,
            3.4,
            80,
            100,
            1.4,
            2.4,
            3.4,
            4,
            120,
            140,
            180,
        ]
        # maches for m=3: [1,2,3]-[1.4,2.4,3.4],[1,2,3]-[1.4,2.4,3.4],
        #                 [2,3,4]-[2.4,3.4,4], [1.4,2.4,3.4]-[1.4,2.4,3.4]  # noqa: ERA001
        # matches for m=4: [1,2,3,4]-[1.4,2.4,3.4,4]
        se = measures.sampen(data, emb_dim=3, tolerance=0.5)
        self.assertAlmostEqual(se, -np.log(1.0 / 4), delta=0.01)

    def test_sampen_logistic_static(self) -> None:
        """Hypothesis: `sampen` gives correct outputs for logistic map with static value."""
        # logistic map with r = 2.8 => static value
        data = list(datasets.logistic_map(0.45, 1000, r=2.8))
        with self.subTest(emb_dim=2):
            self.assertAlmostEqual(0, measures.sampen(data), delta=0.001)
        with self.subTest(emb_dim=5):
            self.assertAlmostEqual(0, measures.sampen(data[100:], emb_dim=5), delta=0.001)

    def test_sampen_logistic_oscillation_2(self) -> None:
        """Hypothesis: `sampen` is correct for logistic map oscillating between 2 values."""
        # logistic map with r = 3.3 => oscillation between two values
        data = list(datasets.logistic_map(0.45, 1000, r=3.3))
        with self.subTest(emb_dim=2):
            self.assertAlmostEqual(0, measures.sampen(data), delta=0.001)
        with self.subTest(emb_dim=5):
            self.assertAlmostEqual(0, measures.sampen(data[100:], emb_dim=5), delta=0.001)

    def test_sampen_logistic_oscillation_4(self) -> None:
        """Hypothesis: `sampen` is correct for logistic map oscillating between 4 values."""
        # logistic map with r = 3.5 => oscillation between four values
        data = list(datasets.logistic_map(0.45, 1000, r=3.5))
        with self.subTest(emb_dim=2):
            self.assertAlmostEqual(0, measures.sampen(data), delta=0.001)
        with self.subTest(emb_dim=5):
            self.assertAlmostEqual(0, measures.sampen(data[100:], emb_dim=5), delta=0.001)

    def test_sampen_logistic_chaotic(self) -> None:
        """Hypothesis: `sampen` is correct for logistic map with chaotic behavior."""
        # logistic map with r = 3.9 => chaotic behavior
        data = list(datasets.logistic_map(0.45, 1000, r=3.9))
        with self.subTest(emb_dim=2):
            self.assertAlmostEqual(0.5, measures.sampen(data[100:]), delta=0.1)
        with self.subTest(emb_dim=5):
            self.assertAlmostEqual(0.5, measures.sampen(data[100:], emb_dim=5), delta=0.1)

    def test_sampen_gaussian(self) -> None:
        """Hypothesis: `sampen` is correct for gaussian noise."""
        rng = np.random.default_rng(seed=6)
        # normally distributed random numbers
        data = rng.standard_normal(10000)
        with self.subTest(emb_dim=2):
            self.assertAlmostEqual(2.2, measures.sampen(data), delta=0.1)
        with self.subTest(emb_dim=5):
            self.assertAlmostEqual(2, measures.sampen(data[100:], emb_dim=5), delta=0.1)

    def test_sampen_sinus(self) -> None:
        """Hypothesis: `sampen` is correct for a sinus signal."""
        # TODO: add test with sinus signal

    def test_sampen_lorenz(self) -> None:
        """Hypothesis: We get correct values for estimating the sample entropy of the Lorenz system.

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
        with self.subTest(axis="x"):
            sx = measures.sampen(data[:, 0], **sampen_args)  # pyright: ignore reportCallIssue
            self.assertAlmostEqual(0.15, sx, delta=0.05)
        with self.subTest(axis="y"):
            sy = measures.sampen(data[:, 1], **sampen_args)  # pyright: ignore reportCallIssue
            self.assertAlmostEqual(0.15, sy, delta=0.05)
        with self.subTest(axis="z"):
            sz = measures.sampen(data[:, 2], **sampen_args)  # pyright: ignore reportCallIssue
            self.assertAlmostEqual(0.25, sz, delta=0.05)


class RegressionTests(unittest.TestCase):
    """Regression tests for main algorithms.

    These tests are here to safeguard against accidental algorithmic changes such
    as updates to core dependencies such as numpy or the Python standard library.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Loads random data for tests."""
        cls.random_data = datasets.load_qrandom()[:1000]

    def test_sampen(self) -> None:
        """Hypothesis: The exact output of sampen remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        se = measures.sampen(
            self.random_data,
            emb_dim=2,
            tolerance=None,
            lag=1,
            dist=measures.rowwise_chebyshev,
            closed=False,
        )
        self.assertAlmostEqual(2.1876999522832743, se, places=14)

    def test_corr_dim(self) -> None:
        """Hypothesis: The exact output of corr_dim with `fit=poly` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        cd = measures.corr_dim(
            self.random_data,
            emb_dim=5,
            lag=1,
            rvals=None,
            dist=measures.rowwise_euclidean,
            fit="poly",
        )
        self.assertAlmostEqual(0.0810185360746645, cd, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_corr_dim_RANSAC(self) -> None:  # noqa: N802
        """Hypothesis: The exact output of corr_dim with `fit=RANSAC` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        sd = float(np.std(self.random_data, ddof=1))
        # usa a too wide range for rvals to give RANSAC something to do ;)
        rvals = measures.logarithmic_r(0.01 * sd, 2 * sd, 1.03)
        cd = measures.corr_dim(
            self.random_data,
            emb_dim=5,
            lag=1,
            rvals=rvals,
            dist=measures.rowwise_euclidean,
            fit="RANSAC",
            random_state=42,
        )
        self.assertAlmostEqual(0.0008971209283844629, cd, places=14)

    def test_lyap_e(self) -> None:
        """Hypothesis: The exact output of lyap_e remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        le = measures.lyap_e(
            self.random_data, emb_dim=10, matrix_dim=4, min_nb=10, min_tsep=1, tau=1
        )
        expected = np.array(
            [
                0.03779942603329712,
                -0.014314012551504982,
                -0.08436867977030214,
                -0.22316730257003717,
            ]
        )
        assert_almost_equal(le, expected, decimal=14)

    def test_lyap_r(self) -> None:
        """Hypothesis: The exact output of lyap_r with `fit=poly` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        le = measures.lyap_r(
            self.random_data,
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
    def test_lyap_r_RANSAC(self) -> None:  # noqa: N802
        """Hypothesis: The exact output of lyap_r with `fit=RANSAC` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        # set lag to 2 for weird duplicate lines
        # set trajectory_len to 100 to get many datapoints for RANSAC to choose from
        le = measures.lyap_r(
            self.random_data,
            emb_dim=10,
            lag=2,
            min_tsep=1,
            tau=1,
            min_neighbors=10,
            trajectory_len=100,
            fit="RANSAC",
            random_state=42,
        )
        expected = 0.0003401212353253564
        self.assertAlmostEqual(expected, le, places=14)

    def test_hurst_rs(self) -> None:
        """Hypothesis: The exact output of hurst_rs with `fit=poly` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        rs = measures.hurst_rs(
            self.random_data, nvals=None, fit="poly", corrected=True, unbiased=True
        )
        expected = 0.5123887964986258
        self.assertAlmostEqual(expected, rs, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_hurst_rs_RANSAC(self) -> None:  # noqa: N802
        """Hypothesis: The exact output of hurst_rs with `fit=RANSAC` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        # increase nsteps in nvals to have more data points for RANSAC to choose from
        nvals = measures.logmid_n(self.random_data.shape[0], ratio=1 / 4.0, nsteps=100)
        rs = measures.hurst_rs(
            self.random_data,
            nvals=nvals,
            fit="RANSAC",
            corrected=True,
            unbiased=True,
            random_state=42,
        )
        expected = 0.4805431939943321
        self.assertAlmostEqual(expected, rs, places=14)

    def test_dfa(self) -> None:
        """Hypothesis: The exact output of dfa with `fit_exp=poly` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        h = measures.dfa(
            self.random_data,
            nvals=None,
            overlap=True,
            order=1,
            fit_trend="poly",
            fit_exp="poly",
        )
        expected = 0.5450874638765073
        self.assertAlmostEqual(expected, h, places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_dfa_RANSAC(self) -> None:  # noqa: N802
        """Hypothesis: The exact output of dfa with `fit_exp=RANSAC` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        # adds trend to data to introduce a less clear line for fitting
        random_data = self.random_data + np.arange(1000) * 100
        # adds more steps and higher values to nvals to introduce some scattering
        # for RANSAC to have an effect on
        nvals = measures.logarithmic_n(10, 0.9 * random_data.shape[0], 1.1)
        h = measures.dfa(
            random_data,
            nvals=nvals,
            overlap=True,
            order=1,
            fit_trend="poly",
            fit_exp="RANSAC",
            random_state=42,
        )
        expected = 1.1372303125405405
        self.assertAlmostEqual(expected, h, places=14)

    def test_mfhurst_b(self) -> None:
        """Hypothesis: The exact output of mfhurst_b with `fit=poly` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        h = measures.mfhurst_b(self.random_data, qvals=[1], dists=None, fit="poly")
        expected = [-0.00559398934417339]
        self.assertAlmostEqual(expected[0], h[0], places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_mfhurst_b_RANSAC(self) -> None:  # noqa: N802
        """Hypothesis: The exact output of mfhurst_b with `fit=RANSAC` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        h = measures.mfhurst_b(
            self.random_data, qvals=[1], dists=None, fit="RANSAC", random_state=42
        )
        expected = [-0.009056463064211057]
        self.assertAlmostEqual(expected[0], h[0], places=14)

    def test_mfhurst_dm(self) -> None:
        """Hypothesis: The exact output of mfhurst_dm with `fit=poly` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        h, _ = measures.mfhurst_dm(
            self.random_data,
            qvals=[1],
            max_dists=range(5, 20),
            detrend=True,
            fit="poly",
        )
        expected = [0.008762803881203145]
        self.assertAlmostEqual(expected[0], h[0], places=14)

    @unittest.skipUnless(SCIPY_AVAILABLE, "Tests with RANSAC require scipy.")
    def test_mfhurst_dm_RANSAC(self) -> None:  # noqa: N802
        """Hypothesis: The exact output of mfhurst_dm with `fit=RANSAC` remains unchanged.

        The test uses random data as input and compares outputs to the previous version.
        """
        h, _ = measures.mfhurst_dm(
            self.random_data,
            qvals=[1],
            max_dists=range(5, 20),
            detrend=True,
            fit="RANSAC",
            random_state=42,
        )
        expected = [0.0068840609945006685]
        self.assertAlmostEqual(expected[0], h[0], places=14)


class PreviousDefectTests(unittest.TestCase):
    """Tests that ensure that a previous bug doesn't come back at some point."""

    def test_lyap_r_complex_min_tsep(self) -> None:
        """Hypothesis: The `min_tsep` parameter can be calculated without creating complex numbers.

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
