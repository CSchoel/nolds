NOnLinear measures for Dynamical Systems (nolds)
================================================

Nolds is a small numpy-based library that provides an implementation and a learning resource for nonlinear measures for dynamical systems based on one-dimensional time series.
Currently the following measures are implemented:

**sample entropy** (``sampen``)
  Measures the complexity of a time-series, based on approximate entropy
**correlation dimension** (``corr_dim``)
  A measure of the *fractal dimension* of a time series which is also related to complexity.
**Lyapunov exponent** (``lyap_r``, ``lyap_e``)
  Positive Lyapunov exponents indicate chaos and unpredictability.
  Nolds provides the algorithm of Rosenstein et al. (``lyap_r``) to estimate the largest Lyapunov exponent and the algorithm of Eckmann et al. (``lyap_e``) to estimate the whole spectrum of Lyapunov exponents.
**Hurst exponent** (``hurst_rs``)
	The hurst exponent is a measure of the "long-term memory" of a time series.
	It can be used to determine whether the time series is more, less, or equally likely to increase if it has increased in previous steps.
	This property makes the Hurst exponent especially interesting for the analysis of stock data.
**detrended fluctuation analysis (DFA)** (``dfa``)
	DFA measures the Hurst parameter *H*, which is very similar to the Hurst exponent.
	The main difference is that DFA can be used for non-stationary processes (whose mean and/or variance change over time).

Example
-------

::

	import nolds
	import numpy as np

	rwalk = np.cumsum(np.random.random(1000))
	h = nolds.dfa(rwalk)

Requirements
------------
Nolds is build for Python 3 and requires the package numpy_.

.. _numpy: http://numpy.scipy.org/

Installation
------------
Nolds is available through PyPI and can be installed using pip:

``pip install nolds``

You can test your installation by running some sample code with:

``python -m nolds.examples all``

Documentation
-------------

Nolds is designed as a learning resource for the measures mentioned above.
Therefore the corresponding functions feature extensive documentation that not only explains the interface but also the algorithm used and points the user to additional reference code and papers.
The documentation can be found in the code, but it is also available as `HTML-Version <https://cschoel.github.io/nolds/>`_.

All relevant code can be found in the file ``nolds/measures.py``.
