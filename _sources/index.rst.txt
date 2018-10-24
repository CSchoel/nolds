Welcome to Nolds' documentation!
=================================

The acronym Nolds stands for 'NOnLinear measures for Dynamical Systems'. It is a small numpy-based library that provides an implementation and a learning resource for nonlinear measures for dynamical systems based on one-dimensional time series.

Nolds is hosted `on GitHub <https://github.com/CSchoel/nolds>`_.
This documentation describes the latest version. A `change log <https://github.com/CSchoel/nolds/blob/master/CHANGELOG.md>`_ of the different versions can be found on GitHub.

For the impatient, here is a small example how you can calculate the lyapunov exponent of the logistic map with Nolds:

.. code-block:: python

   import nolds
   import numpy as np
   lm = nolds.logistic_map(0.1, 1000, r=4)
   x = np.fromiter(lm, dtype="float32")
   l = max(nolds.lyap_e(x))

Contents:

.. toctree::
   :maxdepth: 3

   nolds
   examples
   tests


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

