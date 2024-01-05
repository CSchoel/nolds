# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- new parameter `closed` for `sampen` to determine if `<` or `<=` should be used for checking if the distance between a vector pair is within the tolerance
- new example `sampen-tol` that compares old and new default tolerance values for `sampen`
- more tests for `sampen` based on the logistic map and random data
- new measures
  - `mfhurst_b` calculates the multifractal "generalized" Hurst exponent according to Barabasi et al.
  - `mfhurst_dm` calculates the multifractal "generalized" Hurst exponent according to Di Matteo et al.
- new datasets
  - `load_financial` loads three financial datasets from finance.yahoo.com that can be used to recreate the results from Di Matteo et al. 2003.
  - `barabasi1991_fractal` generates the fractal data used by Barabasi et al. in their 1991 paper
- new examples
  - `hurst_mf_stock` example function recreates a plot from Di Matteo 2003.
  - `barabasi_1991_figure2` and `barabasi_1991_figure3` recreate the respective plots from Barabasi et al. 1991
  - `lorenz` calculates all main measures of `nolds` for x, y, and z coordinates of a Lorenz plot and compares them to prescribed values from the literature
- tests for all `nolds` existing measures based on the Lorenz system

### Changed

- `debug_data` for `sampen` now also contains counts of similar vectors
- `sampen` now issues a warning if one or both count variables are zero
- the parameter `tolerance` in `sampen` now has a more sophisticated default value that takes into account that the chebyshev distance rises logarithmically with increasing dimension
- uses `np.float64` as standard `dtype` instead of `"float32"`
- input data of `lyap_e` is now converted to `np.float64` to avoid errors with `inf` values for integer inputs (see https://github.com/CSchoel/nolds/issues/21)

### Fixed

- the test `test_measures.TestNoldsCorrDim.test_corr_dim` would fail if `sklearn` was not installed, because the standard "RANSAC" fitting method produces quite different results compared to the fallback "poly" method
- uses `ddof=1` in `np.std` when creating debug plot for `sampen` and when computing default `rvals` for `corr_dim`

## [0.5.2]

### Fixed - 2019-06-16

- Issue #13: corr_dim ignored the fit argument

## [0.5.1] - 2018-10-24

### Added

- documentation for `lyap_r_len`, `lyap_e_len` and the `hurst-nvals` example

### Changed

- `hurst_compare_nvals` now also uses `np.asarray`

### Fixed

- some formatting problems in the documentation

## [0.5.0] - 2018-10-24

### Added

- test function `hurst_compare_nvals` that compares different choices for the `nvals` parameter for `hurst_rs`
- example for `hurst_compare_nvals` (can be called using `python -m nolds.examples hurst-nvals`)
- helper functions `lyap_r_len` and `lyap_e_len` to calculate minimum data length required for `lyap_r` and `lyap_e`
- test cases `test_lyap_r_limits` and `test_lyap_e_limits` to ensure that `lyap_r_len` and `lyap_e_len` are calculated correctly
- description of parameter `min_nb` for `lyap_e`
- uses `np.asarray` wherever possible. The following functions should now also work with pandas objects and other "array-like" structures:
  - `lyap_r`
  - `lyap_e`
  - `sampen`
  - `hurst_rs`
  - `corr_dim`
  - `dfa `
- nolds documentation can now also be found [on readthedocs.org](http://nolds.readthedocs.io/)

### Changed

- the previously internal helper function `expected_rs` is now available from the main module
- calculates minimum data length for lyap_r to provide better error messages
- uses `rcond=-1` in lstseq to keep behavior consistent between numpy versions
- mutes `ImportWarning`s from `sklearn` in unit tests
- disables an ugly hack when using `RANSAC` as fitting method and instead requires `sklearn>=0.19` that fixes the underlying issue
- makes test case for correlation dimension less strict
- added hint when `nolds.examples` is called with an unknown example name

### Fixed

- note in the description of the parameter `tau` in `lyap_r` was misleading/wrong (probably a copy-pase error)

### Removed

- distance values `"euler"`, `"chebychev"`, `rowwise_euler` and `rowwise_chebychev` for `sampen` and `corr_dim` (was deprecated)
- keyword parameter `min_vectors` for `lyap_r` (was deprecated)

## [0.4.1] - 2017-11-30

### Added

- function `logmid_n` that allows for a better choice of `nvals` parameter in `hurst_rs`

### Changed

- adds more descriptions and instructions for comparing `hurst_rs` with other implementations

## [0.4.0] - 2017-11-21

### Added

- module `datasets`
  - dataset `brown72` that has a prescribed hurst exponent of 0.72
  - generators for the logistic and the tent map
  - true random numbers using the package `quantumrandom`
- test `test_hurst_pracma` that uses the same testing sequences for `hurst_rs` as the R-package `pracma`
- example function `plot_hurst_hist` that plots a histogram of hurst exponent values for random data
- example function `weron_2002_figure2`
- `fgn` for fractional gaussian noise in the `datasets` module
- documentation for unittests and examples
- parameter `unbiased` for `hurst_rs` that allows to choose between the new (fixed) behavior and the old one (using the wrong version of the standard deviation)
- parameter `corrected` for `hurst_rs` that applies the Anis-Lloyd correction factor to the example by default

### Changed

- default choice for the parameter `nvals` in `hurst_rs` now favors higher n values and always uses 16 n values
- `fbm` is now moved to the `datasets` module

### Fixed

- using fitting method `'ransac'` when sklearn was not installed resulted in an exception instead of a warning
- NaNs in `hurst_rs` where filtered from the set of (R/S)_n values, but the filtered values for n would remain in the calculation and fitting
- `hurst_rs` used the wrong standard deviation, since we estimate the mean of the samples from the data we need to set the parameter `ddof` to `1`

## [0.3.4] - 2017-08-10

### Added
- `lyap_r` now has a new parameter `fit_offset` that allows to ignore the first steps of the plot in the fitting process.

### Changed
- The parameter `min_vectors` is now called `min_neighbors` in `lyap_r` and refers to the number of vectors that are candidates for the closest neighbor.

### Fixed
- The algorithm for choosing the `lag` would always choose 0 in `lyap_r`.
- There was an error in the calculation of the number of vectors used for `min_vectors` in `lyap_r`.

## [0.3.3] - 2017-06-26

### Added
- more test cases for `sampen`
- `debug_data` parameter for most measures that allows to retrieve the data used for debug plots for logging and creation of custom plots

### Changed
- `sampen` now takes functions for the `dist` parameter and not strings
- using something else than `rowwise_chebychev` for `dist` in `sampen` is now officially discouraged

### Fixed
- naming confusion: "Euler" distance should be "Euclidean" distance
- typo in the name "Chebyshev"

### Notes
- all changes mentioned above are backwards-compatible, but this compatibility will be dropped in the next version (since these are really stupid errors that I want to sweep under the rug :wink:)

## [0.3.2] - 2016-11-19
### Added
- `LICENSE.txt` is now part of the distribution
- specifies platform (any) and license (MIT) in `setup.py`
- loads `long_description` from `README.rst`

## [0.3.1] - 2016-11-18
### Fixed
- typo in `setup.py` regarding `extras_require`

## [0.3.0] - 2016-11-18
### Added
- Allows to use RANSAC as line fitting algorithm
- Uses classifiers in `setup.py`
- Adds requirements for the packages `future` and `setuptools`
- Adds custom clean command to `setup.py`

### Changed
- Made support of both Python 3 and Python 2 official using the `future` package (Previous versions also supported Python 2 but did not state this and may have small performance issues.)

### Fixed
- deprecation warning about `assertAlmostEquals` in test cases

## [0.2.1] - 2016-10-17
### Fixed
- Description on PyPI was broken due to formatting error in README.rst

## [0.2.0] - 2016-10-14
### Added
- exportable documentation with Sphinx
- this change log
- unit tests (`python -m unittest nolds.test_measures`)
- example code can be run with `python -m nolds.examples all`

### Changed
- code formatted according to PEP8 (but with 2 spaces indent instead of 4)

### Fixed
- wrong default plotting parameters for function `sampen`

## [0.1.1] - 2016-08-03
### Added
- nolds now lists numpy as dependency (it had the dependency before, but did not tell the user, because who the hell uses python without numpy ;P)

## [0.1.0] - 2016-08-05
### Added
- initial release including the following algorithms:
  - sample entropy (`sampen`)
  - correlation dimension (`corr_dim`)
  - Lyapunov exponent (`lyap_r`, `lyap_e`)
  - Hurst exponent (`hurst_rs`)
  - detrended fluctuation analysis (DFA) (`dfa`)

[Unreleased]: https://github.com/CSchoel/nolds/compare/0.5.2..HEAD
[0.5.2]: https://github.com/CSchoel/nolds/compare/0.5.1..0.5.2
[0.5.1]: https://github.com/CSchoel/nolds/compare/0.5.0..0.5.1
[0.5.0]: https://github.com/CSchoel/nolds/compare/0.4.1..0.5.0
[0.4.1]: https://github.com/CSchoel/nolds/compare/0.4.0..0.4.1
[0.4.0]: https://github.com/CSchoel/nolds/compare/0.3.4..0.4.0
[0.3.4]: https://github.com/CSchoel/nolds/compare/0.3.3..0.3.4
[0.3.3]: https://github.com/CSchoel/nolds/compare/0.3.2..0.3.3
[0.3.2]: https://github.com/CSchoel/nolds/compare/0.3.1..0.3.2
[0.3.1]: https://github.com/CSchoel/nolds/compare/0.3.0..0.3.1
[0.3.0]: https://github.com/CSchoel/nolds/compare/0.2.1..0.3.0
[0.2.1]: https://github.com/CSchoel/nolds/compare/0.2.0..0.2.1
[0.2.0]: https://github.com/CSchoel/nolds/compare/0.1.1..0.2.0
[0.1.1]: https://github.com/CSchoel/nolds/compare/0.1.0..0.1.1
[0.1.0]: https://github.com/CSchoel/nolds/releases/tag/0.1.0
