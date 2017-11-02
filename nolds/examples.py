# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from . import measures as nolds
from . import datasets
import numpy as np


def plot_hurst_hist():
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  hs = [nolds.hurst_rs(np.random.normal(size=1000), corrected=True) for _ in range(100)]
  plt.hist(hs, bins=20)
  plt.show()

def plot_lyap(maptype="logistic"):
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt

  x_start = 0.1
  n = 140
  nbifur = 40
  if maptype == "logistic":
    param_name = "r"
    param_range = np.arange(2, 4, 0.01)
    full_data = np.array([
      np.fromiter(datasets.logistic_map(x_start, n, r),dtype="float32")
      for r in param_range
    ])
    # It can be proven that the lyapunov exponent of the logistic map
    # (or any map that is an iterative application of a function) can be
    # calculated as the mean of the logarithm of the absolute of the
    # derivative at the individual data points.
    # For a proof see for example: 
    # https://blog.abhranil.net/2015/05/15/lyapunov-exponent-of-the-logistic-map-mathematica-code/
    # Derivative of logistic map: f(x) = r * x * (1 - x) = r * x - r * x²
    # => f'(x) = r - 2 * r * x
    lambdas = [
      np.mean(np.log(abs(r - 2 * r * x[np.where(x != 0.5)])))
      for x,r in zip(full_data, param_range)
    ]
  elif maptype == "tent":
    param_name = "$\mu$"
    param_range = np.arange(0, 2, 0.01)
    full_data = np.array([
      np.fromiter(datasets.tent_map(x_start, n, mu),dtype="float32")
      for mu in param_range
    ])
    # for the tent map the lyapunov exponent is much easier to calculate
    # since the values are multiplied by mu in each step, two trajectories
    # starting in x and x + delta will have a distance of delta * mu^n after n
    # steps. Therefore the lyapunov exponent should be log(mu).
    lambdas = np.log(param_range, where=param_range > 0)
    lambdas[np.where(param_range <= 0)] = np.nan
  else:
    raise Error("maptype %s not recognized" % maptype)

  kwargs_e = { "emb_dim": 6, "matrix_dim": 2 }
  kwargs_r = { "emb_dim": 6, "lag": 2, "min_tsep": 20, "trajectory_len": 20}
  lambdas_e = [max(nolds.lyap_e(d, **kwargs_e)) for d in full_data]
  lambdas_r = [nolds.lyap_r(d, **kwargs_r) for d in full_data]
  bifur_x = np.repeat(param_range, nbifur)
  bifur = np.reshape(full_data[:,-nbifur:], nbifur * param_range.shape[0])

  plt.title("Lyapunov exponent of the %s map" % maptype)
  plt.plot(param_range, lambdas, "b-", label="true lyap. exponent")
  elab = "estimation using lyap_e"
  rlab = "estimation using lyap_r"
  plt.plot(param_range, lambdas_e, color="#00AAAA", label=elab)
  plt.plot(param_range, lambdas_r, color="#AA00AA", label=rlab)
  plt.plot(param_range, np.zeros(len(param_range)), "g--")
  plt.plot(bifur_x, bifur, "ro", alpha=0.1, label="bifurcation plot")
  plt.ylim((-2, 2))
  plt.xlabel(param_name)
  plt.ylabel("lyap. exp / %s(x, %s)" % (maptype, param_name))
  plt.legend(loc="best")
  plt.show()

def profiling():
  import cProfile
  n = 10000
  data = np.cumsum(np.random.random(n) - 0.5)
  cProfile.runctx('lyap_e(data)', {'lyap_e': nolds.lyap_e}, {'data': data})

if __name__ == "__main__":
  # run this with the following command:
  # python -m nolds.examples all
  import sys
  if len(sys.argv) < 2:
    print("please tell me which tests you want to run")
    print("options are:")
    print("  lyapunov-all")
    print("  lyapunov-logistic")
    print("  lyapunov-tent")
    print("  profiling")
  elif sys.argv[1] == "lyapunov-all" or len(sys.argv) == 1:
    plot_lyap()
    plot_lyap("tent")
  elif sys.argv[1] == "lyapunov-logistic":
    plot_lyap()
  elif sys.argv[1] == "lyapunov-tent":
    plot_lyap("tent")
  elif sys.argv[1] == "profiling":
    profiling()
