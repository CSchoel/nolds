``nolds`` module
================

Nolds only consists of to single module called ``nolds`` which contains all
relevant algorithms and helper functions.

Internally these functions are subdivided into different modules such as
``measures`` and ``datasets``, but you should not need to import these modules
directly unless you want access to some internal helper functions.


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
.. autofunction:: nolds.binary_n
.. autofunction:: nolds.logarithmic_n
.. autofunction:: nolds.logarithmic_r
.. autofunction:: nolds.expected_h
.. autofunction:: nolds.expected_rs
.. autofunction:: nolds.logmid_n
.. autofunction:: nolds.lyap_r_len
.. autofunction:: nolds.lyap_e_len

Datasets
--------

Benchmark dataset for hurst exponent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autodata:: nolds.brown72

   The brown72 dataset has a prescribed (uncorrected) Hurst exponent of 0.7270.
   It is a synthetic dataset from the book "Chaos and Order in the Capital
   markets"[b7_a]_.

   It is included here, because the dataset can be found online [b7_b]_ and is
   used by other software packages such as the R-package ``pracma`` [b7_c]_.
   As such it can be used to compare different implementations.

   However, it should be noted that the idea that the "true" Hurst exponent of
   this series is indeed 0.7270 is problematic for several reasons:

   1. This value does not take into account the Anis-Lloyd-Peters correction
      factor.
   2. It was obtained using the biased version of the standard deviation.
   3. It depends (as always for the Hurst exponent) on the choice of the length
      of the subsequences.

   If you want to reproduce the prescribed value, you can use the following
   code:

   .. code-block:: python

      nolds.hurst_rs(
         nolds.brown72,
         nvals=2**np.arange(3,11),
         fit="poly", corrected=False, unbiased=False
      )

   References:

   .. [b7_a] Edgar Peters, “Chaos and Order in the Capital Markets: A New
      View of Cycles, Prices, and Market Volatility”, Wiley: Hoboken, 
      2nd Edition, 1996.
   .. [b7_b] Ian L. Kaplan, "Estimating the Hurst Exponent", 
      url: http://www.bearcave.com/misl/misl_tech/wavelets/hurst/
   .. [b7_c] HwB, "Pracma: brown72",
      url: https://www.rdocumentation.org/packages/pracma/versions/1.9.9/topics/brown72

Tent map
~~~~~~~~
.. autofunction:: nolds.tent_map

Logistic map
~~~~~~~~~~~~
.. autofunction:: nolds.logistic_map

Fractional brownian motion
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.fbm

Fractional gaussian noise
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.fgn

Quantum random numbers
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nolds.qrandom
.. autofunction:: nolds.load_qrandom
