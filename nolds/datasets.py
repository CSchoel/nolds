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

brown72 = brown72()

