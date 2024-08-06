Nolds examples
==============

You can run some examples for the functions in nolds with the command
``python -m nolds.examples <key>`` where ``<key>`` can be one of the following:

* ``lyapunov-logistic`` shows a bifurcation plot of the logistic map and compares
  the true lyapunov exponent to the estimates obtained with ``lyap_e`` and
  ``lyap_r``.
* ``lyapunov-tent`` shows the same plot as ``lyapunov-logistic``, but for the tent
  map.
* ``profiling`` runs a profiling test with the package ``cProfile``.
* ``hurst-weron2`` plots a reconstruction of figure 2 of the weron 2002 paper
  about the hurst exponent.
* ``hurst-hist`` plots a histogram of hurst exponents obtained for random noise.
* ``hurst-nvals`` creates a plot that compares the results of different choices for nvals
  for the function ``hurst_rs``.
* ``sampen-tol`` compares old and new default tolerance values for ``sampen``.
* ``hurst_mf_stock`` example function recreates a plot from Di Matteo (2003).
* ``barabasi_1991_figure2`` and ``barabasi_1991_figure3`` recreate the respective plots from Barabasi et al. (1991)
* ``lorenz`` calculates all main measures of ``nolds`` for x, y, and z coordinates of a Lorenz plot and compares them to prescribed values from the literature.

These tests are also available as functions inside the module ``nolds.examples``.

Functions in ``nolds.examples``
-------------------------------

.. autofunction:: nolds.examples.plot_lyap
.. autofunction:: nolds.examples.profiling
.. autofunction:: nolds.examples.weron_2002_figure2
.. autofunction:: nolds.examples.plot_hurst_hist
.. autofunction:: nolds.examples.hurst_compare_nvals
