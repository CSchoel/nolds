``nolds`` module
================

Nolds only consists of a single module called ``nolds`` which contains all relevant algorithms and helper functions.

Algorithms
----------
Lyapunov exponent (Rosenstein et al.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.lyap_r

Lyapunov exponent (Eckmann et al.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.lyap_e

Sample entropy
~~~~~~~~~~~~~~
.. autofunction:: nolds.sampen

Hurst exponent
~~~~~~~~~~~~~~
.. autofunction:: nolds.hurst_rs

Correlation dimension
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.corr_dim

Detrended fluctuation analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.dfa


Helper functions
-----------------
.. autofunction:: nolds.fbm
.. autofunction:: nolds.binary_n
.. autofunction:: nolds.logarithmic_n
.. autofunction:: nolds.logarithmic_r
