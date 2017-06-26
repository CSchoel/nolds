# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added
### Changed
### Fixed

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
