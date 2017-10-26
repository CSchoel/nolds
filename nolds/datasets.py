# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
import numpy as np
import pkg_resources

def brown72():
  """
  Loads the dataset brown72 with a prescribed Hurst exponent of 0.72

  Source: http://www.bearcave.com/misl/misl_tech/wavelets/hurst/

  Returns:
    float array:
      the dataset
  """
  fname = "datasets/brown72.npy"
  with pkg_resources.resource_stream(__name__, fname) as f:
    return np.load(f)

def tent_map(x, steps, mu=2):
  """
  Generates a time series of the tent map.

  Characteristics and Background:
    The name of the tent map is derived from the fact that the plot of x_i vs
    x_i+1 looks like a tent. For mu > 1 one application of the mapping function
    can be viewed as stretching the surface on which the value is located and
    then folding the area that is greater than one back towards the zero. This
    corresponds nicely to the definition of chaos as expansion in one dimension
    which is counteracted by a compression in another dimension.

  Calculating the Lyapunov exponent:
    The lyapunov exponent of the tent map can be easily calculated as due to
    this stretching behavior a small difference delta between two neighboring
    points will indeed grow exponentially by a factor of mu in each iteration.
    We thus can assume that:

    delta_n = delta_0 * mu^n

    We now only have to change the basis to e to obtain the exact formula that
    is used for the definition of the lyapunov exponent:

    delta_n = delta_0 * e^(ln(mu) * n)

    Therefore the lyapunov exponent of the tent map is:

    lambda = ln(mu)

  References:
    .. [tm-1] https://en.wikipedia.org/wiki/Tent_map

  Args:
    x (float):
      starting point
    steps (int):
      number of steps for which the generator should run

  Kwargs:
    mu (int):
      parameter mu that controls the behavior of the map

  Returns:
    generator object:
      the generator that creates the time series
  """
  for _ in range(steps):
    x = mu * x if x < 0.5 else mu * (1 - x)
    yield x

def logistic_map(x, steps, r=4):
  """
  Generates a time series of the logistic map.

  Characteristics and Background:
    The logistic map is among the simplest examples for a time series that can
    exhibit chaotic behavior depending on the parameter r. For r between 2 and 
    3, the series quickly becomes static. At r=3 the first bifurcation point is
    reached after which the series starts to oscillate. Beginning with r = 3.6
    it shows chaotic behavior with a few islands of stability until perfect
    chaos is achieved at r = 4.

  Calculating the Lyapunov exponent:
    To calculate the "true" Lyapunov exponent of the logistic map, we first
    have to make a few observations for maps in general that are repeated
    applications of a function to a starting value.

    If we have two starting values that differ by some infinitesimal delta_0
    then according to the definition of the lyapunov exponent we will have
    an exponential divergence:

    |delta_n| = |delta_0| e^(lambda n)

    We can now write that:

    e^(lambda n) = lim delta_0 -> 0: |delta_n / delta_0|

    This is the definition of the derivative dx_n/dx_0 of a point x_n in the
    time series with respect to the starting point x_0 (or rather the absolute
    value of that derivative). Now we can use the fact that due to the
    definition of our map as repetitive application of some f we have:

    f^n'(x) = f(f(f(...f(x_0)...))) = f'(x_n-1) * f'(x_n-2) * ... * f'(x_0)

    with

    e^(lambda n) = |f^n'(x)|

    we now have

    e^(lambda n) = |f'(x_n-1) * f'(x_n-2) * ... * f'(x_0)|
    <=>
    lambda n = ln |f'(x_n-1) * f'(x_n-2) * ... * f'(x_0)|
    <=>
    lambda = 1/n ln |f'(x_n-1) * f'(x_n-2) * ... * f'(x_0)|
           = 1/n sum_{k=0}^{n-1} ln |f'(x_k)|

    With this sum we can now calculate the lyapunov exponent for any map.
    For the logistic map we simply have to calculate f'(x) and as we have

    f(x) = r * x * (1-x) = rx - rx²

    we now get

    f'(x) = r - 2 rx



  References:
    .. [lm-1] https://en.wikipedia.org/wiki/Tent_map
    .. [lm-2] https://blog.abhranil.net/2015/05/15/lyapunov-exponent-of-the-logistic-map-mathematica-code/

  Args:
    x (float):
      starting point
    steps (int):
      number of steps for which the generator should run

  Kwargs:
    r (int):
      parameter r that controls the behavior of the map

  Returns:
    generator object:
      the generator that creates the time series
  """
  for _ in range(steps):
    x = r * x * (1 - x)
    yield x

brown72 = brown72()

