# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [0.3.0] - 2016-11-17
### Added
- Allows to use RANSAC as line fitting algorithm
- Uses classifiers in setup.py

### Changed
- Made support of both Python 3 and Python 2 official using the `future` package (Previous versions also worked on Python 2 but did not state this and may have small performance issues.)

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
