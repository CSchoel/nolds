# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (
  bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
  oct, open, pow, round, super, filter, map, zip
)
from . import measures as nolds
from . import datasets
import numpy as np


def weron_2002_figure2(n=10000):
  """
  Recreates figure 2 of [w]_ comparing the reported values by Weron to the
  values obtained by the functions in this package.

  The experiment consists of n iterations where the hurst exponent of randomly
  generated gaussian noise is calculated. This is done with differing sequence
  lengths of 256, 512, 1024, ...., 65536. The average estimated hurst exponent
  over all iterations is plotted for the following configurations:

  * ``weron`` is the Anis-Lloyd-corrected Hurst exponent calculated by Weron
  * ``rs50`` is the Anis-Lloyd-corrected Hurst exponent calculated by Nolds
    with the same parameters as used by Weron
  * ``weron_raw`` is the uncorrected Hurst exponent calculated by Weron
  * ``rs50_raw`` is the uncorrected Hurst exponent calculated by Nolds with the
    same parameters as used by Weron
  * ``rsn`` is the Anis-Lloyd-corrected Hurst exponent calculated by Nolds with
    the default settings of Nolds

  The values reported by Weron are only measured from the plot in the PDF
  version of the paper and can therefore have some small inaccuracies.

  This function requires the package ``matplotlib``.

  References:

  .. [w] R. Weron, “Estimating long-range dependence: finite sample
     properties and confidence intervals,” Physica A: Statistical Mechanics
     and its Applications, vol. 312, no. 1, pp. 285–299, 2002.

  Kwargs:
    n (int):
      number of iterations of the experiment (Weron used 10000, but this takes
      a while)
  """
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  # note: these values are calculated by measurements in inkscape of the plot
  # from the paper
  reported = [6.708, 13.103, 20.240, 21.924, 22.256, 24.112, 24.054, 26.299,
              26.897]
  reported_raw = [160.599, 141.663, 128.454, 115.617, 103.651, 95.481, 86.810,
                  81.799, 76.270]

  def height_to_h(height):
    return 0.49 + height / 29.894 * 0.01
  reported = height_to_h(np.array(reported))
  reported_raw = height_to_h(np.array(reported_raw))
  data = []
  for e in range(8, 17):
    l = 2**e
    nvals = 2**np.arange(6, e)
    rsn = np.mean([
      nolds.hurst_rs(np.random.normal(size=l), fit="poly")
      for _ in range(n)
    ])
    rs50 = np.mean([
      nolds.hurst_rs(np.random.normal(size=l), fit="poly", nvals=nvals)
      for _ in range(n)
    ])
    rs50_raw = np.mean([
      nolds.hurst_rs(
        np.random.normal(size=l), fit="poly", nvals=nvals, corrected=False
      )
      for _ in range(n)
    ])
    data.append((rsn, rs50, rs50_raw))
  lines = plt.plot(np.arange(8, 17), data)
  r = plt.plot(np.arange(8, 17), reported)
  rr = plt.plot(np.arange(8, 17), reported_raw)
  plt.legend(r + rr + lines, ("weron", "weron_raw", "rsn", "rs50", "rs50_raw"))
  plt.xticks(np.arange(8, 17), 2**np.arange(8, 17))
  plt.xlabel("sequence length")
  plt.ylabel("estimated hurst exponent")
  plt.show()


def plot_hurst_hist():
  """
  Plots a histogram of values obtained for the hurst exponent of uniformly
  distributed white noise.

  This function requires the package ``matplotlib``.
  """
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  hs = [
    nolds.hurst_rs(np.random.random(size=10000), corrected=True)
    for _ in range(100)
  ]
  plt.hist(hs, bins=20)
  plt.xlabel("esimated value of hurst exponent")
  plt.ylabel("number of experiments")
  plt.show()


def plot_lyap(maptype="logistic"):
  """
  Plots a bifurcation plot of the given map and superimposes the true
  lyapunov exponent as well as the estimates of the largest lyapunov exponent
  obtained by ``lyap_r`` and ``lyap_e``. The idea for this plot is taken
  from [ll]_.

  This function requires the package ``matplotlib``.

  References:

  .. [ll] Manfred Füllsack, "Lyapunov exponent",
     url: http://systems-sciences.uni-graz.at/etextbook/sw2/lyapunov.html

  Kwargs:
    maptype (str):
      can be either ``"logistic"`` for the logistic map or ``"tent"`` for the
      tent map.
  """
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt

  x_start = 0.1
  n = 140
  nbifur = 40
  if maptype == "logistic":
    param_name = "r"
    param_range = np.arange(2, 4, 0.01)
    full_data = np.array([
      np.fromiter(datasets.logistic_map(x_start, n, r), dtype="float32")
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
      for x, r in zip(full_data, param_range)
    ]
  elif maptype == "tent":
    param_name = "$\\mu$"
    param_range = np.arange(0, 2, 0.01)
    full_data = np.array([
      np.fromiter(datasets.tent_map(x_start, n, mu), dtype="float32")
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

  kwargs_e = {"emb_dim": 6, "matrix_dim": 2}
  kwargs_r = {"emb_dim": 6, "lag": 2, "min_tsep": 20, "trajectory_len": 20}
  lambdas_e = [max(nolds.lyap_e(d, **kwargs_e)) for d in full_data]
  lambdas_r = [nolds.lyap_r(d, **kwargs_r) for d in full_data]
  bifur_x = np.repeat(param_range, nbifur)
  bifur = np.reshape(full_data[:, -nbifur:], nbifur * param_range.shape[0])

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
  """
  Runs a profiling test for the function ``lyap_e`` (mainly used for
  development)

  This function requires the package ``cProfile``.
  """
  import cProfile
  n = 10000
  data = np.cumsum(np.random.random(n) - 0.5)
  cProfile.runctx('lyap_e(data)', {'lyap_e': nolds.lyap_e}, {'data': data})


def hurst_compare_nvals(data, nvals=None):
  """
  Creates a plot that compares the results of different choices for nvals
  for the function hurst_rs.

  Args:
    data (array-like of float):
      the input data from which the hurst exponent should be estimated

  Kwargs:
    nvals (array of int):
      a manually selected value for the nvals parameter that should be plotted
      in comparison to the default choices
  """
  import matplotlib.pyplot as plt
  data = np.asarray(data)
  n_all = np.arange(2, len(data)+1)
  dd_all = nolds.hurst_rs(data, nvals=n_all, debug_data=True, fit="poly")
  dd_def = nolds.hurst_rs(data, debug_data=True, fit="poly")
  n_def = np.round(np.exp(dd_def[1][0])).astype("int32")
  n_div = n_all[np.where(len(data) % n_all[:-1] == 0)]
  dd_div = nolds.hurst_rs(data, nvals=n_div, debug_data=True, fit="poly")

  def corr(nvals):
    return [np.log(nolds.expected_rs(n)) for n in nvals]

  l_all = plt.plot(dd_all[1][0], dd_all[1][1] - corr(n_all), "o")
  l_def = plt.plot(dd_def[1][0], dd_def[1][1] - corr(n_def), "o")
  l_div = plt.plot(dd_div[1][0], dd_div[1][1] - corr(n_div), "o")
  l_cst = []
  t_cst = []

  if nvals is not None:
    dd_cst = nolds.hurst_rs(data, nvals=nvals, debug_data=True, fit="poly")
    l_cst = plt.plot(dd_cst[1][0], dd_cst[1][1] - corr(nvals), "o")
    t_cst = ["custom"]
  plt.xlabel("log(n)")
  plt.ylabel("log((R/S)_n - E[(R/S)_n])")
  plt.legend(
    l_all + l_def + l_div + l_cst, ["all", "default", "divisors"] + t_cst
  )
  labeled_data = zip([dd_all[0], dd_def[0], dd_div[0]], ["all", "def", "div"])
  for data, label in labeled_data:
    print("%s: %.3f" % (label, data))
  if nvals is not None:
    print("custom: %.3f" % dd_cst[0])
  plt.show()

def sampen_default_tolerance():
  data = list(datasets.logistic_map(0.34, 1000, r=3.9))
  oldtol = 0.2 * np.std(data)
  old_res = [
    nolds.sampen(data, emb_dim=i, tolerance=oldtol)
    for i in range(1, 25)
  ]
  new_res = [
    nolds.sampen(data, emb_dim=i)
    for i in range(1, 25)
  ]
  for i, old, new in zip(range(1, 25), old_res, new_res):
    print("emb_dim={} old={:.3f} corrected={:.3f}".format(i, old, new))
  print("      old variance: {:.3f}".format(np.var(old_res)))
  print("corrected variance: {:.3f}".format(np.var(new_res)))

def aste_line_fitting(N=100):
  """
  Shows plot that proves that the line fitting in T. Astes original MATLAB code
  provides the same results as `np.polyfit`.
  """
  slope = np.random.random() * 10 - 5
  intercept = np.random.random() * 100 - 50
  xvals = np.arange(N)
  yvals = xvals * slope + intercept + np.random.randn(N)*100
  import matplotlib.pyplot as plt
  plt.plot(xvals, yvals, "rx", label="data")
  plt.plot(
    [0, N-1], [intercept, intercept + slope * (N-1)],
    "r-", label="true ({:.3f} x + {:.3f})".format(slope, intercept), alpha=0.5
  )
  i_aste, s_aste = nolds._aste_line_fit(xvals, yvals)
  s_np, i_np = np.polyfit(xvals, yvals, 1)
  plt.plot(
    [0, N-1], [i_aste, i_aste + s_aste * (N-1)],
    "b-", label="aste ({:.3f} x + {:.3f})".format(s_aste, i_aste), alpha=0.5
  )
  plt.plot(
    [0, N-1], [i_np, i_np + s_np * (N-1)],
    "g-", label="numpy ({:.3f} x + {:.3f})".format(s_np, i_np), alpha=0.5
  )
  plt.legend()
  plt.show()


def hurst_mf_stock(debug=False):
  """
  Recreates results from [mfs_1]_ (table at start of section 4) as print
  output.

  Unfortunately as a layman in finance, I could not determine the exact data
  that Di Matteo et al. used. Instead I use the data from
  `nolds.datasets.load_financial()`.

  Plots H(2) for the following datasets and algorithms.

  Datasets (opening values from `load_financial()`):

  - jkse: Jakarta Composite Index
  - n225: Nikkei 225
  - ndx: NASDAQ 100

  Algorithms:

  - mfhurst_b: GHE according to Barabási et al.
  - mfhurst_b + dt: like mfhurst_b, but with linear detrending performed first
  - mfhurst_dm: GHE according to Di Matteo et al. (should be identical to
                _genhurst)
  - _genhurst: GHE according to translated MATLAB code by T. Aste (one of the
               co-authors of Di Matteo).

  References:

    .. [mfs_1] T. Di Matteo, T. Aste, and M. M. Dacorogna, “Scaling behaviors
       in differently developed markets,” Physica A: Statistical Mechanics
       and its Applications, vol. 324, no. 1–2, pp. 183–188, 2003.

  Kwargs:
    debug (boolean):
      if `True`, a debug plot will be shown for each calculated GHE value
      except for the ones generated by `_genhurst`.
  """
  print("Dataset  mfhurst_b  mfhurst_b + dt  mfhurst_dm  _genhurst")
  financial = [
    (datasets.jkse, "jkse"), (datasets.n225, "n225"), (datasets.ndx, "ndx")
  ]
  for data, lab in financial:
    data = data[1][:, 0]
    data = np.log(data)
    dists = range(1, 20)
    mfh_b = nolds.mfhurst_b(data, qvals=[2], dists=dists, debug_plot=debug)[0]
    mfh_b_dt = nolds.mfhurst_b(
      nolds.detrend_data(data, order=1),
      qvals=[2], dists=dists, debug_plot=debug
    )[0]
    mfh_dm = nolds.mfhurst_dm(data, qvals=[2], debug_plot=debug)[0][0]
    gh = nolds._genhurst(data, 2)
    print("{:10s}   {:5.3f}           {:5.3f}       {:5.3f}      {:5.3f}".format(lab, mfh_b, mfh_b_dt, mfh_dm, gh))


def barabasi_1991_figure2():
  """
  Recreates figure 2 from [bf2]_.

  This figure compares calculated and estimated values for H(q) for
  a fractal generated by 9 iterations of the `barabasi1991_fractal` function
  with b1 = 0.8 and b2 = 0.5.

  References:
    .. [bf2] A.-L. Barabási and T. Vicsek, “Multifractality of self-affine
       fractals,” Physical Review A, vol. 44, no. 4, pp. 2730–2733, 1991.
  """
  import matplotlib.pyplot as plt
  b1991 = datasets.barabasi1991_fractal(10000000, 9)
  qvals = range(1, 11)
  qvals_t = range(-10, 11)
  b1 = 0.8
  b2 = 0.5
  dists = [4 ** i for i in range(6, 11)]
  # dists = nolds.logarithmic_n(100, 0.01 * len(b1991), 2)
  Hq = nolds.mfhurst_b(b1991, qvals=qvals, dists=dists)
  Hq_t = [np.log((b1 ** q + b2 ** q) / 2) / np.log(0.25) / q for q in qvals_t]
  plt.plot(qvals, Hq, "r+", label="mfhurst_b")
  plt.plot(qvals_t, Hq_t, label="calculated value")
  plt.legend(loc="best")
  plt.xlabel("q")
  plt.ylabel("H(q)")
  plt.show()


def barabasi_1991_figure3():
  """
  Recreates figure 3 from [bf3]_.

  This figure compares calculated and estimated values for H(q) for a simple
  Brownian motion that moves in unit steps (-1 or +1) in each time step.

  References:
    .. [bf3] A.-L. Barabási and T. Vicsek, “Multifractality of self-affine
       fractals,” Physical Review A, vol. 44, no. 4, pp. 2730–2733, 1991.
  """
  import matplotlib.pyplot as plt
  brown = np.cumsum(np.random.randint(0, 2, size=10000000)*2-1)
  qvals = [-5, -4, -3, -2, -1.1, 0.1, 1, 2, 3, 4, 5]
  Hq_t = [0.5 if q > -1 else -0.5/q for q in qvals]
  dists = [2 ** i for i in range(6, 15)]
  # dists = nolds.logarithmic_n(100, 0.01 * len(brown), 1.5)
  Hq = nolds.mfhurst_b(brown, qvals=qvals, dists=dists, debug_plot=False)
  plt.plot(qvals, Hq, "r+", label="mfhurst_b")
  plt.plot(qvals, Hq_t, label="calculated value")
  plt.ylim(0, 1)
  plt.legend(loc="best")
  plt.xlabel("q")
  plt.ylabel("H(q)")
  plt.show()


if __name__ == "__main__":
  # run this with the following command:
  # python -m nolds.examples lyapunov-logistic
  import sys

  def print_options():
    print("options are:")
    print("  lyapunov-logistic")
    print("  lyapunov-tent")
    print("  profiling")
    print("  hurst-weron2")
    print("  hurst-hist")
    print("  hurst-nvals")
    print("  sampen-tol")
    print("  aste-line")
    print("  hurst-mf-stock")
  if len(sys.argv) < 2:
    print("please tell me which tests you want to run")
    print_options()
  elif sys.argv[1] == "lyapunov-logistic":
    plot_lyap()
  elif sys.argv[1] == "lyapunov-tent":
    plot_lyap("tent")
  elif sys.argv[1] == "profiling":
    profiling()
  elif sys.argv[1] == "hurst-weron2":
    n = 1000 if len(sys.argv) < 3 else int(sys.argv[2])
    weron_2002_figure2(n)
  elif sys.argv[1] == "hurst-hist":
    plot_hurst_hist()
  elif sys.argv[1] == "hurst-nvals":
    hurst_compare_nvals(datasets.brown72)
  elif sys.argv[1] == "sampen-tol":
    sampen_default_tolerance()
  elif sys.argv[1] == "aste-line":
    aste_line_fitting()
  elif sys.argv[1] == "hurst-mf-stock":
    hurst_mf_stock()
  elif sys.argv[1] == "hurst-mf-barabasi2":
    barabasi_1991_figure2()
  elif sys.argv[1] == "hurst-mf-barabasi3":
    barabasi_1991_figure3()
  else:
    print("i do not know any test of that name")
    print_options()
