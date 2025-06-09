# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (
  bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next,
  oct, open, pow, round, super, filter, map, zip
)
import numpy as np
import warnings
import math


def rowwise_chebyshev(x, y):
  return np.max(np.abs(x - y), axis=1)


def rowwise_euclidean(x, y):
  return np.sqrt(np.sum((x - y)**2, axis=1))


def poly_fit(x, y, degree, fit="RANSAC"):
  # check if we can use RANSAC
  if fit == "RANSAC":
    try:
      # ignore ImportWarnings in sklearn
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        import sklearn.linear_model as sklin
        import sklearn.preprocessing as skpre
    except ImportError:
      warnings.warn(
        "fitting mode 'RANSAC' requires the package sklearn, using"
        + " 'poly' instead",
        RuntimeWarning)
      fit = "poly"

  if fit == "poly":
    return np.polyfit(x, y, degree)
  elif fit == "RANSAC":
    model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
    xdat = np.asarray(x)
    if len(xdat.shape) == 1:
      # interpret 1d-array as list of len(x) samples instead of
      # one sample of length len(x)
      xdat = xdat.reshape(-1, 1)
    polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
    try:
      model.fit(polydat, y)
      coef = model.estimator_.coef_[::-1]
    except ValueError:
      warnings.warn(
        "RANSAC did not reach consensus, "
        + "using numpy's polyfit",
        RuntimeWarning)
      coef = np.polyfit(x, y, degree)
    return coef
  else:
    raise ValueError("invalid fitting mode ({})".format(fit))


def delay_embedding(data, emb_dim, lag=1):
  """
  Perform a time-delay embedding of a time series

  Args:
    data (array-like):
      the data that should be embedded
    emb_dim (int):
      the embedding dimension
  Kwargs:
    lag (int):
      the lag between elements in the embedded vectors

  Returns:
    emb_dim x m array:
      matrix of embedded vectors of the form
      [data[i], data[i+lag], data[i+2*lag], ... data[i+(emb_dim-1)*lag]]
      for i in 0 to m-1 (m = len(data)-(emb_dim-1)*lag)
  """
  data = np.asarray(data)
  min_len = (emb_dim - 1) * lag + 1
  if len(data) < min_len:
    msg = "cannot embed data of length {} with embedding dimension {} " \
        + "and lag {}, minimum required length is {}"
    raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
  m = len(data) - min_len + 1
  indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
  indices += np.arange(m).reshape((m, 1))
  return data[indices]


def lyap_r_len(**kwargs):
  """
  Helper function that calculates the minimum number of data points required
  to use lyap_r.

  Note that none of the required parameters may be set to None.

  Kwargs:
    kwargs(dict):
      arguments used for lyap_r (required: emb_dim, lag, trajectory_len and
      min_tsep)

  Returns:
    minimum number of data points required to call lyap_r with the given
    parameters
  """
  # minimum length required to find single orbit vector
  min_len = (kwargs['emb_dim'] - 1) * kwargs['lag'] + 1
  # we need trajectory_len orbit vectors to follow a complete trajectory
  min_len += kwargs['trajectory_len'] - 1
  # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
  min_len += kwargs['min_tsep'] * 2 + 1
  return min_len


def lyap_r(data, emb_dim=10, lag=None, min_tsep=None, tau=1, min_neighbors=20,
           trajectory_len=20, fit="RANSAC", debug_plot=False, debug_data=False,
           plot_file=None, fit_offset=0):
  """
  Estimates the largest Lyapunov exponent using the algorithm of Rosenstein
  et al. [lr_1]_.

  Explanation of Lyapunov exponents:
    See lyap_e.

  Explanation of the algorithm:
    The algorithm of Rosenstein et al. is only able to recover the largest
    Lyapunov exponent, but behaves rather robust to parameter choices.

    The idea for the algorithm relates closely to the definition of Lyapunov
    exponents. First, the dynamics of the data are reconstructed using a delay
    embedding method with a lag, such that each value x_i of the data is mapped
    to the vector

    X_i = [x_i, x_(i+lag), x_(i+2*lag), ..., x_(i+(emb_dim-1) * lag)]

    For each such vector X_i, we find the closest neighbor X_j using the
    euclidean distance. We know that as we follow the trajectories from X_i and
    X_j in time in a chaotic system the distances between X_(i+k) and X_(j+k)
    denoted as d_i(k) will increase according to a power law
    d_i(k) = c * e^(lambda * k) where lambda is a good approximation of the
    highest Lyapunov exponent, because the exponential expansion along the axis
    associated with this exponent will quickly dominate the expansion or
    contraction along other axes.

    To calculate lambda, we look at the logarithm of the distance trajectory,
    because log(d_i(k)) = log(c) + lambda * k. This gives a set of lines
    (one for each index i) whose slope is an approximation of lambda. We
    therefore extract the mean log trajectory d'(k) by taking the mean of
    log(d_i(k)) over all orbit vectors X_i. We then fit a straight line to
    the plot of d'(k) versus k. The slope of the line gives the desired
    parameter lambda.

  Method for choosing min_tsep:
    Usually we want to find neighbors between points that are close in phase
    space but not too close in time, because we want to avoid spurious
    correlations between the obtained trajectories that originate from temporal
    dependencies rather than the dynamic properties of the system. Therefore it
    is critical to find a good value for min_tsep. One rather plausible
    estimate for this value is to set min_tsep to the mean period of the
    signal, which can be obtained by calculating the mean frequency using the
    fast fourier transform. This procedure is used by default if the user sets
    min_tsep = None. Note that this default procedure uses a naive approach
    for estimating the power spectral density, which just takes the FFT of the
    whole signal without applying any windowing function to avoid biases. If
    you have a non-stationary input and want more than a rough estimate,
    consider calculating min_tsep manually using a sliding window approach
    like Welch's method (implemented in `scipy.signal.welch`).

  Method for choosing lag:
    Another parameter that can be hard to choose by instinct alone is the lag
    between individual values in a vector of the embedded orbit. Here,
    Rosenstein et al. suggest to set the lag to the distance where the
    autocorrelation function drops below 1 - 1/e times its original (maximal)
    value. This procedure is used by default if the user sets lag = None.

  References:
    .. [lr_1] M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
       “A practical method for calculating largest Lyapunov exponents from
       small data sets,” Physica D: Nonlinear Phenomena, vol. 65, no. 1,
       pp. 117–134, 1993.

  Reference Code:
    .. [lr_a] mirwais, "Largest Lyapunov Exponent with Rosenstein's Algorithm",
       url: http://www.mathworks.com/matlabcentral/fileexchange/38424-largest-lyapunov-exponent-with-rosenstein-s-algorithm
    .. [lr_b] Shapour Mohammadi, "LYAPROSEN: MATLAB function to calculate
       Lyapunov exponent",
       url: https://ideas.repec.org/c/boc/bocode/t741502.html
    .. [lr_c] Rainer Hegger, Holger Kantz, and Thomas Schreiber, "TISEAN 3.0.0 - Nonlinear Time Series Analysis",
       url: https://www.pks.mpg.de/tisean/Tisean_3.0.0/docs/docs_c/lyap_r.html

  Args:
    data (iterable of float):
      (one-dimensional) time series
  Kwargs:
    emb_dim (int):
      embedding dimension for delay embedding
    lag (float):
      lag for delay embedding
    min_tsep (float):
      minimal temporal separation between two "neighbors" (default:
      find a suitable value by calculating the mean period of the data)
    tau (float):
      step size between data points in the time series in seconds
      (normalization scaling factor for exponents)
    min_neighbors (int):
      if lag=None, the search for a suitable lag will be stopped when the
      number of potential neighbors for a vector drops below min_neighbors
    trajectory_len (int):
      the time (in number of data points) to follow the distance
      trajectories between two neighboring points
    fit (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will
      be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``
    fit_offset (int):
      neglect the first fit_offset steps when fitting

  Returns:
    float:
      an estimate of the largest Lyapunov exponent (a positive exponent is
      a strong indicator for chaos)
    (1d-vector, 1d-vector, list):
      only present if debug_data is True: debug data of the form
      ``(ks, div_traj, poly)`` where ``ks`` are the x-values of the line fit,
      ``div_traj`` are the y-values and ``poly`` are the line coefficients
      (``[slope, intercept]``).

  """
  # convert data to float to avoid overflow errors in rowwise_euclidean
  data = np.asarray(data, dtype=np.float64)
  n = len(data)
  max_tsep_factor = 0.25
  if lag is None or min_tsep is None:
    # both the algorithm for lag and min_tsep need the fft
    f = np.fft.rfft(data, n * 2 - 1)
  if min_tsep is None:
    # calculate min_tsep as mean period (= 1 / mean frequency)
    # to get the mean frequency, we weight the frequency buckets in the
    # fft result by the absolute power in that bucket and then divide
    # by the total power across all buckets to get a weighted mean.
    # This can be inaccurate for non-stationary inputs. A better approach would
    # be to use scipy.signal.welch, but this requires making some other
    # parameter choices like the size of the sliding window that require some
    # knowledge about the input data, which we don't have at this point.
    freqs = np.fft.rfftfreq(n * 2 - 1)
    psd = np.abs(f)**2
    mf = np.sum(freqs[1:] * psd[1:]) / np.sum(psd[1:])
    min_tsep = int(np.ceil(1.0 / mf))
    if min_tsep > max_tsep_factor * n:
      min_tsep = int(max_tsep_factor * n)
      msg = "signal has very low mean frequency, setting min_tsep = {:d}"
      warnings.warn(msg.format(min_tsep), RuntimeWarning)
  if lag is None:
    # calculate the lag as point where the autocorrelation drops to (1 - 1/e)
    # times its maximum value
    # note: the Wiener–Khinchin theorem states that the spectral
    # decomposition of the autocorrelation function of a process is the power
    # spectrum of that process
    # => we can use fft to calculate the autocorrelation
    acorr = np.fft.irfft(f * np.conj(f))
    acorr = np.roll(acorr, n - 1)
    eps = acorr[n - 1] * (1 - 1.0 / np.e)
    lag = 1

    # small helper function to calculate resulting number of vectors for a
    # given lag value
    def nb_neighbors(lag_value):
      min_len = lyap_r_len(
        emb_dim=emb_dim, lag=lag_value, trajectory_len=trajectory_len,
        min_tsep=min_tsep
      )
      return max(0, n - min_len)
    # find lag
    for i in range(1, n):
      lag = i
      if acorr[n - 1 + i] < eps or acorr[n - 1 - i] < eps:
        break
      if nb_neighbors(i) < min_neighbors:
        msg = "autocorrelation declined too slowly to find suitable lag" \
          + ", setting lag to {}"
        warnings.warn(msg.format(lag), RuntimeWarning)
        break
  min_len = lyap_r_len(
    emb_dim=emb_dim, lag=lag, trajectory_len=trajectory_len,
    min_tsep=min_tsep
  )
  if len(data) < min_len:
    msg = "for emb_dim = {}, lag = {}, min_tsep = {} and trajectory_len = {}" \
      + " you need at least {} datapoints in your time series"
    warnings.warn(
      msg.format(emb_dim, lag, min_tsep, trajectory_len, min_len),
      RuntimeWarning
    )
  # delay embedding
  orbit = delay_embedding(data, emb_dim, lag)
  m = len(orbit)
  # construct matrix with pairwise distances between vectors in orbit
  dists = np.array([rowwise_euclidean(orbit, orbit[i]) for i in range(m)])
  # we do not want to consider vectors as neighbor that are less than min_tsep
  # time steps together => mask the distances min_tsep to the right and left of
  # each index by setting them to infinity (will never be considered as nearest
  # neighbors)
  for i in range(m):
    dists[i, max(0, i - min_tsep):i + min_tsep + 1] = float("inf")
  # check that we have enough data points to continue
  ntraj = m - trajectory_len + 1
  min_traj = min_tsep * 2 + 2  # in each row min_tsep + 1 disances are inf
  if ntraj <= 0:
    msg = "Not enough data points. Need {} additional data points to follow " \
        + "a complete trajectory."
    raise ValueError(msg.format(-ntraj+1))
  if ntraj < min_traj:
    # not enough data points => there are rows where all values are inf
    assert np.any(np.all(np.isinf(dists[:ntraj, :ntraj]), axis=1))
    msg = "Not enough data points. At least {} trajectories are required " \
        + "to find a valid neighbor for each orbit vector with min_tsep={} " \
        + "but only {} could be created."
    raise ValueError(msg.format(min_traj, min_tsep, ntraj))
  assert np.all(np.any(np.isfinite(dists[:ntraj, :ntraj]), axis=1))
  # find nearest neighbors (exclude last columns, because these vectors cannot
  # be followed in time for trajectory_len steps)
  nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)

  # build divergence trajectory by averaging distances along the trajectory
  # over all neighbor pairs
  div_traj = np.zeros(trajectory_len, dtype=float)
  for k in range(trajectory_len):
    # calculate mean trajectory distance at step k
    indices = (np.arange(ntraj) + k, nb_idx + k)
    div_traj_k = dists[indices]
    # filter entries where distance is zero (would lead to -inf after log)
    nonzero = np.where(div_traj_k != 0)
    if len(nonzero[0]) == 0:
      # if all entries where zero, we have to use -inf
      div_traj[k] = -np.inf
    else:
      div_traj[k] = np.mean(np.log(div_traj_k[nonzero]))
  # filter -inf entries from mean trajectory
  ks = np.arange(trajectory_len)
  finite = np.where(np.isfinite(div_traj))
  ks = ks[finite]
  div_traj = div_traj[finite]
  if len(ks) < 1:
    # if all points or all but one point in the trajectory is -inf, we cannot
    # fit a line through the remaining points => return -inf as exponent
    poly = [-np.inf, 0]
  else:
    # normal line fitting
    poly = poly_fit(ks[fit_offset:], div_traj[fit_offset:], 1, fit=fit)
  if debug_plot:
    plot_reg(
      ks[fit_offset:], div_traj[fit_offset:],
      poly, "k", "log(d(k))", fname=plot_file)
  le = poly[0] / tau
  if debug_data:
    return (le, (ks, div_traj, poly))
  else:
    return le


def lyap_e_len(**kwargs):
  """
  Helper function that calculates the minimum number of data points required
  to use lyap_e.

  Note that none of the required parameters may be set to None.

  Kwargs:
    kwargs(dict):
      arguments used for lyap_e (required: emb_dim, matrix_dim, min_nb
      and min_tsep)

  Returns:
    minimum number of data points required to call lyap_e with the given
    parameters
  """
  m = (kwargs['emb_dim'] - 1) // (kwargs['matrix_dim'] - 1)
  # minimum length required to find single orbit vector
  min_len = kwargs['emb_dim']
  # we need to follow each starting point of an orbit vector for m more steps
  min_len += m
  # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
  min_len += kwargs['min_tsep'] * 2
  # we need at least min_nb neighbors for each orbit vector
  min_len += kwargs['min_nb']
  return min_len


def lyap_e(data, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1,
           debug_plot=False, debug_data=False, plot_file=None):
  """
  Estimates the Lyapunov exponents for the given data using the algorithm of
  Eckmann et al. [le_1]_.

  Recommendations for parameter settings by Eckmann et al.:
    * long recording time improves accuracy, small tau does not
    * use large values for emb_dim
    * matrix_dim should be 'somewhat larger than the expected number of
      positive Lyapunov exponents'
    * min_nb = min(2 * matrix_dim, matrix_dim + 4)

  Explanation of Lyapunov exponents:
    The Lyapunov exponent describes the rate of separation of two
    infinitesimally close trajectories of a dynamical system in phase space.
    In a chaotic system, these trajectories diverge exponentially following
    the equation:

    \|X(t, X_0) - X(t, X_0 + eps)| = e^(lambda * t) * \|eps|

    In this equation X(t, X_0) is the trajectory of the system X starting at
    the point X_0 in phase space at time t. eps is the (infinitesimal)
    difference vector and lambda is called the Lyapunov exponent. If the
    system has more than one free variable, the phase space is
    multidimensional and each dimension has its own Lyapunov exponent. The
    existence of at least one positive Lyapunov exponent is generally seen as
    a strong indicator for chaos.

  Explanation of the Algorithm:
    To calculate the Lyapunov exponents analytically, the Jacobian of the
    system is required. The algorithm of Eckmann et al. therefore tries to
    estimate this Jacobian by reconstructing the dynamics of the system from
    which the time series was obtained. For this, several steps are required:

    * Embed the time series [x_1, x_2, ..., x_(N-1)] in an orbit of emb_dim
      dimensions (map each point x_i of the time series to a vector
      [x_i, x_(i+1), x_(i+2), ... x_(i+emb_dim-1)]).
    * For each vector X_i in this orbit find a radius r_i so that at least
      min_nb other vectors lie within (chebyshev-)distance r_i around X_i.
      These vectors will be called "neighbors" of X_i.
    * Find the Matrix T_i that sends points from the neighborhood of X_i to
      the neighborhood of X_(i+1). To avoid undetermined values in T_i, we
      construct T_i not with size (emb_dim x emb_dim) but with size
      (matrix_dim x matrix_dim), so that we have a larger "step size" m in the
      X_i, which are now defined as X'_i = [x_i, x_(i+m), x_(i+2m),
      ... x_(i+(matrix_dim-1)*m)]. This means that emb_dim-1 must be divisible
      by matrix_dim-1. The T_i are then found by a linear least squares fit,
      assuring that T_i (X_j - X_i) ~= X_(j+m) - X_(i+m) for any X_j in the
      neighborhood of X_i.
    * Starting with i = 1 and Q_0 = identity successively decompose the matrix
      T_i * Q_(i-1) into the matrices Q_i and R_i by a QR-decomposition.
    * Calculate the Lyapunov exponents from the mean of the logarithm of the
      diagonal elements of the matrices R_i. To normalize the Lyapunov
      exponents, they have to be divided by m and by the step size tau of the
      original time series.

  References:
    .. [le_1] J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto,
       “Liapunov exponents from time series,” Physical Review A,
       vol. 34, no. 6, pp. 4971–4979, 1986.

  Reference code:
    .. [le_a] Manfred Füllsack, "Lyapunov exponent",
       url: http://systems-sciences.uni-graz.at/etextbook/sw2/lyapunov.html
    .. [le_b] Steve SIU, Lyapunov Exponents Toolbox (LET),
       url: http://www.mathworks.com/matlabcentral/fileexchange/233-let/content/LET/findlyap.m
    .. [le_c] Rainer Hegger, Holger Kantz, and Thomas Schreiber, TISEAN,
       url: http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html

  Args:
    data (array-like of float):
      (scalar) data points

  Kwargs:
    emb_dim (int):
      embedding dimension
    matrix_dim (int):
      matrix dimension (emb_dim - 1 must be divisible by matrix_dim - 1)
    min_nb (int):
      minimal number of neighbors
      (default: min(2 * matrix_dim, matrix_dim + 4))
    min_tsep (int):
      minimal temporal separation between two "neighbors"
    tau (float):
      step size of the data in seconds
      (normalization scaling factor for exponents)
    debug_plot (boolean):
      if True, a histogram matrix of the individual estimates will be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``

  Returns:
    float array:
      array of matrix_dim Lyapunov exponents (positive exponents are indicators
      for chaos)
    2d-array of floats:
      only present if debug_data is True: all estimates for the matrix_dim
      Lyapunov exponents from the x iterations of R_i. The shape of this debug
      data is (x, matrix_dim).
  """
  # convert to float to avoid errors when using 'inf' as distance
  data = np.asarray(data, dtype=np.float64)
  n = len(data)
  if (emb_dim - 1) % (matrix_dim - 1) != 0:
    raise ValueError("emb_dim - 1 must be divisible by matrix_dim - 1!")
  m = (emb_dim - 1) // (matrix_dim - 1)
  if min_nb is None:
    # minimal number of neighbors as suggested by Eckmann et al.
    min_nb = min(2 * matrix_dim, matrix_dim + 4)

  min_len = lyap_e_len(
    emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb, min_tsep=min_tsep
  )
  if n < min_len:
    msg = "{} data points are not enough! For emb_dim = {}, matrix_dim = {}" \
      + ", min_tsep = {} and min_nb = {} you need at least {} data points " \
      + "in your time series"
    warnings.warn(
      msg.format(n, emb_dim, matrix_dim, min_tsep, min_nb, min_len),
      RuntimeWarning
    )

  # construct orbit as matrix (e = emb_dim)
  # x0 x1 x2 ... xe-1
  # x1 x2 x3 ... xe
  # x2 x3 x4 ... xe+1
  # ...

  # note: we need to be able to step m points further for the beta vector
  #       => maximum start index is n - emb_dim - m
  orbit = delay_embedding(data[:-m], emb_dim, lag=1)
  if len(orbit) < min_nb:
    assert len(data) < min_len
    msg = "Not enough data points. Need at least {} additional data points " \
        + "to have min_nb = {} neighbor candidates"
    raise ValueError(msg.format(min_nb-len(orbit), min_nb))
  old_Q = np.identity(matrix_dim)
  lexp = np.zeros(matrix_dim, dtype=np.float64)
  lexp_counts = np.zeros(lexp.shape)
  debug_values = []
  # TODO reduce number of points to visit?
  # TODO performance test!
  for i in range(len(orbit)):
    # find neighbors for each vector in the orbit using the chebyshev distance
    diffs = rowwise_chebyshev(orbit, orbit[i])
    # ensure that we do not count the difference of the vector to itself
    diffs[i] = float('inf')
    # mask all neighbors that are too close in time to the vector itself
    mask_from = max(0, i - min_tsep)
    mask_to = min(len(diffs), i + min_tsep + 1)
    diffs[mask_from:mask_to] = np.inf
    indices = np.argsort(diffs)
    idx = indices[min_nb - 1]  # index of the min_nb-nearest neighbor
    r = diffs[idx]  # corresponding distance
    if np.isinf(r):
      assert len(data) < min_len
      msg = "Not enough data points. Orbit vector {} has less than min_nb = " \
          + "{} valid neighbors that are at least min_tsep = {} time steps " \
          + "away. Input must have at least length {}."
      raise ValueError(msg.format(i, min_nb, min_tsep, min_len))
    # there may be more than min_nb vectors at distance r (if multiple vectors
    # have a distance of exactly r)
    # => update index accordingly
    indices = np.where(diffs <= r)[0]

    # find the matrix T_i that satisifies
    # T_i (orbit'[j] - orbit'[i]) = (orbit'[j+m] - orbit'[i+m])
    # for all neighbors j where orbit'[i] = [x[i], x[i+m],
    # ... x[i + (matrix_dim-1)*m]]

    # note that T_i has the following form:
    # 0  1  0  ... 0
    # 0  0  1  ... 0
    # ...
    # a0 a1 a2 ... a(matrix_dim-1)

    # This is because for all rows except the last one the aforementioned
    # equation has a clear solution since orbit'[j+m] - orbit'[i+m] =
    # [x[j+m]-x[i+m], x[j+2*m]-x[i+2*m], ... x[j+d_M*m]-x[i+d_M*m]]
    # and
    # orbit'[j] - orbit'[i] =
    # [x[j]-x[i], x[j+m]-x[i+m], ... x[j+(d_M-1)*m]-x[i+(d_M-1)*m]]
    # therefore x[j+k*m] - x[i+k*m] is already contained in
    # orbit'[j] - orbit'[x] for all k from 1 to matrix_dim-1. Only for
    # k = matrix_dim there is an actual problem to solve.

    # We can therefore find a = [a0, a1, a2, ... a(matrix_dim-1)] by
    # formulating a linear least squares problem (mat_X * a = vec_beta)
    # as follows.

    # build matrix X for linear least squares (d_M = matrix_dim)
    # x_j1 - x_i   x_j1+m - x_i+m   ...   x_j1+(d_M-1)m - x_i+(d_M-1)m
    # x_j2 - x_i   x_j2+m - x_i+m   ...   x_j2+(d_M-1)m - x_i+(d_M-1)m
    # ...

    # note: emb_dim = (d_M - 1) * m + 1
    mat_X = np.array([data[j:j + emb_dim:m] for j in indices])
    mat_X -= data[i:i + emb_dim:m]

    # build vector beta for linear least squares
    # x_j1+(d_M)m - x_i+(d_M)m
    # x_j2+(d_M)m - x_i+(d_M)m
    # ...
    if max(np.max(indices), i) + matrix_dim * m >= len(data):
      assert len(data) < min_len
      msg = "Not enough data points. Cannot follow orbit vector {} for " \
          + "{} (matrix_dim * m) time steps. Input must have at least " \
          + "length {}."
      raise ValueError(msg.format(i, matrix_dim * m, min_len))
    vec_beta = data[indices + matrix_dim * m] - data[i + matrix_dim * m]

    # perform linear least squares
    a, _, _, _ = np.linalg.lstsq(mat_X, vec_beta, rcond=-1)
    # build matrix T
    # 0  1  0  ... 0
    # 0  0  1  ... 0
    # ...
    # 0  0  0  ... 1
    # a1 a2 a3 ... a_(d_M)
    mat_T = np.zeros((matrix_dim, matrix_dim))
    mat_T[:-1, 1:] = np.identity(matrix_dim - 1)
    mat_T[-1] = a

    # QR-decomposition of T * old_Q
    mat_Q, mat_R = np.linalg.qr(np.dot(mat_T, old_Q))
    # force diagonal of R to be positive
    # (if QR = A then also QLL'R = A with L' = L^-1)
    sign_diag = np.sign(np.diag(mat_R))
    sign_diag[np.where(sign_diag == 0)] = 1
    sign_diag = np.diag(sign_diag)
    mat_Q = np.dot(mat_Q, sign_diag)
    mat_R = np.dot(sign_diag, mat_R)

    old_Q = mat_Q
    # successively build sum for Lyapunov exponents
    diag_R = np.diag(mat_R)
    # filter zeros in mat_R (would lead to -infs)
    idx = np.where(diag_R > 0)
    lexp_i = np.zeros(diag_R.shape, dtype=np.float64)
    lexp_i[idx] = np.log(diag_R[idx])
    lexp_i[np.where(diag_R == 0)] = np.inf
    if debug_plot or debug_data:
      debug_values.append(lexp_i / tau / m)
    lexp[idx] += lexp_i[idx]
    lexp_counts[idx] += 1
  # end of loop over orbit vectors
  # it may happen that all R-matrices contained zeros => exponent really has
  # to be -inf
  if debug_plot:
    plot_histogram_matrix(np.array(debug_values), "layp_e", fname=plot_file)
  # normalize exponents over number of individual mat_Rs
  idx = np.where(lexp_counts > 0)
  lexp[idx] /= lexp_counts[idx]
  lexp[np.where(lexp_counts == 0)] = np.inf
  # normalize with respect to tau
  lexp /= tau
  # take m into account
  lexp /= m
  if debug_data:
    return (lexp, np.array(debug_values))
  return lexp


def plot_dists(dists, tolerance, m, title=None, fname=None):
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  nstd = 3
  nbins = 50
  dists_full = np.concatenate(dists)
  ymax = len(dists_full) * 0.05
  mean = np.mean(dists_full)
  std = np.std(dists_full, ddof=1)
  rng = (0, mean + std * nstd)
  i = 0
  colors = ["green", "blue"]
  for h, bins in [np.histogram(dat, nbins, rng) for dat in dists]:
    bw = bins[1] - bins[0]
    plt.bar(bins[:-1], h, bw, label="m={:d}".format(m + i),
            color=colors[i], alpha=0.5)
    i += 1
  plt.axvline(tolerance, color="red")
  plt.legend(loc="best")
  plt.xlabel("distance")
  plt.ylabel("count")
  plt.ylim(0, ymax)
  if title is not None:
    plt.title(title)
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()


def sampen(data, emb_dim=2, tolerance=None, lag=1, dist=rowwise_chebyshev,
           closed=False, debug_plot=False, debug_data=False, plot_file=None):
  """
  Computes the sample entropy of the given data.

  Explanation of the sample entropy:
    The sample entropy of a time series is defined as the negative natural
    logarithm of the conditional probability that two sequences similar for
    emb_dim points remain similar at the next point, excluding self-matches.

    A lower value for the sample entropy therefore corresponds to a higher
    probability indicating more self-similarity.

  Explanation of the algorithm:
    The algorithm constructs all subsequences of length emb_dim
    [s_1, s_1+lag, s_1+2*lag, ...] and then counts each pair (s_i, s_j) with i != j
    where dist(s_i, s_j) < tolerance. The same process is repeated for all
    subsequences of length emb_dim + 1. The sum of similar sequence pairs
    with length emb_dim + 1 is divided by the sum of similar sequence pairs
    with length emb_dim. The result of the algorithm is the negative logarithm
    of this ratio/probability.

  References:
    .. [se_1] J. S. Richman and J. R. Moorman, “Physiological time-series
       analysis using approximate entropy and sample entropy,”
       American Journal of Physiology-Heart and Circulatory Physiology,
       vol. 278, no. 6, pp. H2039–H2049, 2000.

  Reference code:
    .. [se_a] "sample_entropy" function in R-package "pracma",
        url: https://cran.r-project.org/web/packages/pracma/pracma.pdf

  Args:
    data (array-like of float):
      input data

  Kwargs:
    emb_dim (int):
      the embedding dimension (length of vectors to compare)
    tolerance (float):
      distance threshold for two template vectors to be considered equal
      (default: 0.2 * std(data) at emb_dim = 2, corrected for dimension effect
      for other values of emb_dim)
    lag (int):
      delay for the delay embedding
    dist (function (2d-array, 1d-array) -> 1d-array):
      distance function used to calculate the distance between template
      vectors. Sampen is defined using ``rowwise_chebyshev``. You should only
      use something else, if you are sure that you need it.
    closed (boolean):
      if True, will check for vector pairs whose distance is in the closed
      interval [0, r] (less or equal to r), otherwise the open interval
      [0, r) (less than r) will be used
    debug_plot (boolean):
      if True, a histogram of the individual distances for m and m+1
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``

  Returns:
    float:
      the sample entropy of the data (negative logarithm of ratio between
      similar template vectors of length emb_dim + 1 and emb_dim)
    [c_m, c_m1]:
      list of two floats: count of similar template vectors of length emb_dim
      (c_m) and of length emb_dim + 1 (c_m1)
    [float list, float list]:
      Lists of lists of the form ``[dists_m, dists_m1]`` containing the
      distances between template vectors for m (dists_m)
      and for m + 1 (dists_m1).
  """
  data = np.asarray(data)

  if tolerance is None:
    # the reasoning behind this default value is the following:
    # 1. physionet uses the default values emb_dim = 2, tolerance = 0.2
    # 2. the chebyshev distance rises logarithmically with increasing dimension
    # 3. 0.5627 * np.log(emb_dim) + 1.3334 is the logarithmic trend line for
    #    the chebyshev distance of vectors sampled from a univariate normal
    #    distribution
    # 4. 0.1164 is used as a factor to ensure that tolerance == std * 0.2 for
    #    emb_dim == 2
    tolerance = np.std(data, ddof=1) * 0.1164 * (0.5627 * np.log(emb_dim) + 1.3334)
  n = len(data)

  # build matrix of "template vectors"
  # (all consecutive subsequences of length m)
  # x0 x1 x2 x3 ... xm-1
  # x1 x2 x3 x4 ... xm
  # x2 x3 x4 x5 ... xm+1
  # ...
  # x_n-m-1     ... xn-1

  # since we need two of these matrices for m = emb_dim and m = emb_dim +1,
  # we build one that is large enough => shape (emb_dim+1, n-emb_dim)

  # note that we ignore the last possible template vector with length emb_dim,
  # because this vector has no corresponding vector of length m+1 and thus does
  # not count towards the conditional probability
  # (otherwise first dimension would be n-emb_dim+1 and not n-emb_dim)
  tVecs = delay_embedding(np.asarray(data), emb_dim+1, lag=lag)
  plot_data = []
  counts = []
  for m in [emb_dim, emb_dim + 1]:
    counts.append(0)
    plot_data.append([])
    # get the matrix that we need for the current m
    tVecsM = tVecs[:n - m + 1, :m]
    # successively calculate distances between each pair of template vectors
    for i in range(len(tVecsM) - 1):
      dsts = dist(tVecsM[i + 1:], tVecsM[i])
      if debug_plot or debug_data:
        plot_data[-1].extend(dsts)
      # count how many distances are smaller than the tolerance
      if closed:
        counts[-1] += np.sum(dsts <= tolerance)
      else:
        counts[-1] += np.sum(dsts < tolerance)
  if counts[0] > 0 and counts[1] > 0:
    saen = -np.log(1.0 * counts[1] / counts[0])
  else:
    # log would be infinite or undefined => cannot determine saen
    zcounts = []
    if counts[0] == 0:
      zcounts.append("emb_dim")
    if counts[1] == 0:
      zcounts.append("emb_dim + 1")
    warnings.warn(
      (
        "Zero vectors are within tolerance for %s. " \
        + "Consider raising the tolerance parameter to avoid %s result."
      ) % (" and ".join(zcounts), "NaN" if len(zcounts) == 2 else "inf"),
      RuntimeWarning
    )
    if counts[0] == 0 and counts[1] == 0:
      saen = np.nan
    elif counts[0] == 0:
      saen = -np.inf
    else:
      saen = np.inf
  if debug_plot:
    plot_dists(plot_data, tolerance, m, title="sampEn = {:.3f}".format(saen),
               fname=plot_file)
  if debug_data:
    return (saen, counts, plot_data)
  else:
    return saen


def binary_n(total_N, min_n=50):
  """
  Creates a list of values by successively halving the total length total_N
  until the resulting value is less than min_n.

  Non-integer results are rounded down.

  Args:
    total_N (int):
      total length
  Kwargs:
    min_n (int):
      minimal length after division

  Returns:
    list of integers:
      total_N/2, total_N/4, total_N/8, ... until total_N/2^i < min_n
  """
  max_exp = np.log2(1.0 * total_N / min_n)
  max_exp = int(np.floor(max_exp))
  return [int(np.floor(1.0 * total_N / (2**i))) for i in range(1, max_exp + 1)]


def logarithmic_n(min_n, max_n, factor):
  """
  Creates a list of values by successively multiplying a minimum value min_n by
  a factor > 1 until a maximum value max_n is reached.

  Non-integer results are rounded down.

  Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)

  Returns:
    list of integers:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
      without duplicates
  """
  assert max_n > min_n
  assert factor > 1
  # stop condition: min * f^x = max
  # => f^x = max/min
  # => x = log(max/min) / log(f)
  max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
  ns = [min_n]
  for i in range(max_i + 1):
    n = int(np.floor(min_n * (factor ** i)))
    if n > ns[-1]:
      ns.append(n)
  return ns


def logmid_n(max_n, ratio=1/4.0, nsteps=15):
  """
  Creates an array of integers that lie evenly spaced in the "middle" of the
  logarithmic scale from 0 to log(max_n).

  If max_n is very small and/or nsteps is very large, this may lead to
  duplicate values which will be removed from the output.

  This function has benefits in hurst_rs, because it cuts away both very small
  and very large n, which both can cause problems, and still produces a
  logarithmically spaced sequence.

  Args:
    max_n (int):
      largest possible output value (should be the sequence length when used in
      hurst_rs)

  Kwargs:
    ratio (float):
      width of the "middle" of the logarithmic interval relative to log(max_n).
      For example, for ratio=1/2.0 the logarithm of the resulting values will
      lie between 0.25 * log(max_n) and 0.75 * log(max_n).
    nsteps (float):
      (maximum) number of values to take from the specified range

  Returns:
    array of int:
      a logarithmically spaced sequence of at most nsteps values (may be less,
      because only unique values are returned)
  """
  l = np.log(max_n)
  span = l * ratio
  start = l * (1 - ratio) * 0.5
  midrange = start + 1.0*np.arange(nsteps)/nsteps*span
  nvals = np.round(np.exp(midrange)).astype("int32")
  return np.unique(nvals)


def logarithmic_r(min_n, max_n, factor):
  """
  Creates a list of values by successively multiplying a minimum value min_n by
  a factor > 1 until a maximum value max_n is reached.

  Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)

  Returns:
    list of floats:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
  """
  assert max_n > min_n
  assert factor > 1
  max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
  return [min_n * (factor ** i) for i in range(max_i + 1)]


def expected_rs(n):
  """
  Calculates the expected (R/S)_n for white noise for a given n.

  This is used as a correction factor in the function hurst_rs. It uses the
  formula of Anis-Lloyd-Peters (see [h_3]_).

  Args:
    n (int):
      the value of n for which the expected (R/S)_n should be calculated

  Returns:
    float:
      expected (R/S)_n for white noise
  """
  front = (n - 0.5) / n
  i = np.arange(1, n)
  back = np.sum(np.sqrt((n - i) / i))
  if n <= 340:
    middle = math.gamma((n-1) * 0.5) / math.sqrt(math.pi) / math.gamma(n * 0.5)
  else:
    middle = 1.0 / math.sqrt(n * math.pi * 0.5)
  return front * middle * back


def expected_h(nvals, fit="RANSAC"):
  """
  Uses expected_rs to calculate the expected value for the Hurst exponent h
  based on the values of n used for the calculation.

  Args:
    nvals (iterable of int):
      the values of n used to calculate the individual (R/S)_n

  KWargs:
    fit (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers

  Returns:
    float:
      expected h for white noise
  """
  rsvals = [expected_rs(n) for n in nvals]
  poly = poly_fit(np.log(nvals), np.log(rsvals), 1, fit=fit)
  return poly[0]


def rs(data, n, unbiased=True):
  """
  Calculates an individual R/S value in the rescaled range approach for
  a given n.

  Note: This is just a helper function for hurst_rs and should not be called
  directly.

  Args:
    data (array-like of float):
      time series
    n (float):
      size of the subseries in which data should be split

  Kwargs:
    unbiased (boolean):
      if True, the standard deviation based on the unbiased variance
      (1/(N-1) instead of 1/N) will be used. This should be the default choice,
      since the true mean of the sequences is not known. This parameter should
      only be changed to recreate results of other implementations.

  Returns:
    float:
      (R/S)_n
  """
  data = np.asarray(data)
  total_N = len(data)
  m = total_N // n  # number of sequences
  # cut values at the end of data to make the array divisible by n
  data = data[:total_N - (total_N % n)]
  # split remaining data into subsequences of length n
  seqs = np.reshape(data, (m, n))
  # calculate means of subsequences
  means = np.mean(seqs, axis=1)
  # normalize subsequences by substracting mean
  y = seqs - means.reshape((m, 1))
  # build cumulative sum of subsequences
  y = np.cumsum(y, axis=1)
  # find ranges
  r = np.max(y, axis=1) - np.min(y, axis=1)
  # find standard deviation
  # we should use the unbiased estimator, since we do not know the true mean
  s = np.std(seqs, axis=1, ddof=1 if unbiased else 0)
  # some ranges may be zero and have to be excluded from the analysis
  idx = np.where(r != 0)
  r = r[idx]
  s = s[idx]
  # it may happen that all ranges are zero (if all values in data are equal)
  if len(r) == 0:
    return np.nan
  else:
    # return mean of r/s along subsequence index
    return np.mean(r / s)


def plot_histogram_matrix(data, name, bin_range="3sigma", fname=None):
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  nhists = len(data[0])
  nbins = 25
  ylim = (0, 0.5)
  nrows = int(np.ceil(np.sqrt(nhists)))
  plt.figure(figsize=(nrows * 4, nrows * 4))
  for i in range(nhists):
    plt.subplot(nrows, nrows, i + 1)
    absmax = max(abs(np.max(data[:, i])), abs(np.min(data[:, i])))
    if bin_range == "absmax":
      rng = (-absmax, absmax)
    elif bin_range.endswith("sigma"):
      n = int(bin_range[:-len("sigma")])
      mu = np.mean(data[:,i])
      sigma = np.std(data[:, i], ddof=1)
      rng = (mu - n * sigma, mu + n * sigma)
    h, bins = np.histogram(data[:, i], nbins, rng)
    bin_width = bins[1] - bins[0]
    h = h.astype(np.float64) / np.sum(h)
    plt.bar(bins[:-1], h, bin_width)
    plt.axvline(np.mean(data[:, i]), color="red")
    plt.ylim(ylim)
    plt.title("{:s}[{:d}]".format(name, i))
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()


def plot_reg(xvals, yvals, poly, x_label="x", y_label="y", data_label="data",
             reg_label="regression line", fname=None):
  """
  Helper function to plot trend lines for line-fitting approaches. This
  function will show a plot through ``plt.show()`` and close it after the
  window has been closed by the user.

  Args:
    xvals (list/array of float):
      list of x-values
    yvals (list/array of float):
      list of y-values
    poly (list/array of float):
      polynomial parameters as accepted by ``np.polyval``
  Kwargs:
    x_label (str):
      label of the x-axis
    y_label (str):
      label of the y-axis
    data_label (str):
      label of the data
    reg_label(str):
      label of the regression line
    fname (str):
      file name (if not None, the plot will be saved to disc instead of
      showing it though ``plt.show()``)
  """
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  plt.plot(xvals, yvals, "bo", label=data_label)
  if not (poly is None):
    plt.plot(xvals, np.polyval(poly, xvals), "r-", label=reg_label)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc="best")
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()


def plot_reg_tiled(xvals, yvals, polys, x_label="x", y_label="y",
                   data_labels=None, reg_labels=None, fname=None,
                   columns=None):
  """
  TODO
  """
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  max_span = max([np.max(y) - np.min(y) for y in yvals])
  means = [np.mean(y) for y in yvals]
  if columns is None:
    columns = min(4, int(np.ceil(np.sqrt(len(xvals)))))
  if data_labels is None:
    data_labels = ["data"] * len(xvals)
  if reg_labels is None:
    reg_labels = ["regression line"] * len(xvals)
  for i in range(len(xvals)):
    plt.subplot(int(np.ceil(len(xvals) / columns)), columns, i + 1)
    plt.plot(xvals[i], yvals[i], "bo", label=data_labels[i])
    if not (polys is None):
      plt.plot(xvals[i], np.polyval(polys[i], xvals[i]), "r-", label=reg_labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(means[i] - max_span / 2, means[i] + max_span / 2)
    plt.legend(loc="best")
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()


def plot_reg_multiple(xvals, yvals, polys, x_label="x", y_label="y",
                      data_labels=None, reg_labels=None, fname=None):
  """
  TODO
  """
  import matplotlib.pyplot as plt
  if data_labels is None:
    data_labels = ["data"] * len(xvals)
  if reg_labels is None:
    reg_labels = ["regression line"] * len(xvals)
  for i in range(len(xvals)):
    plt.plot(xvals[i], yvals[i], "+", label=data_labels[i])
    if not (polys is None):
      plt.plot(xvals[i], np.polyval(polys[i], xvals[i]), label=reg_labels[i])
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc="best")
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()


def hurst_rs(data, nvals=None, fit="RANSAC", debug_plot=False,
             debug_data=False, plot_file=None, corrected=True, unbiased=True):
  """
  Calculates the Hurst exponent by a standard rescaled range (R/S) approach.

  Explanation of Hurst exponent:
    The Hurst exponent is a measure for the "long-term memory" of a
    time series, meaning the long statistical dependencies in the data that do
    not originate from cycles.

    It originates from H.E. Hursts observations of the problem of long-term
    storage in water reservoirs. If x_i is the discharge of a river in year i
    and we observe this discharge for N years, we can calculate the storage
    capacity that would be required to keep the discharge steady at its mean
    value.

    To do so, we first subtract the mean over all x_i from the individual
    x_i to obtain the departures x'_i from the mean for each year i. As the
    excess or deficit in discharge always carries over from year i to year i+1,
    we need to examine the cumulative sum of x'_i, denoted by y_i. This
    cumulative sum represents the filling of our hypothetical storage. If the
    sum is above 0, we are storing excess discharge from the river, if it is
    below zero we have compensated a deficit in discharge by releasing
    water from the storage. The range (maximum - minimum) R of y_i therefore
    represents the total capacity required for the storage.

    Hurst showed that this value follows a steady trend for varying N if it
    is normalized by the standard deviation sigma over the x_i. Namely he
    obtained the following formula:

    R/sigma = (N/2)^K

    In this equation, K is called the Hurst exponent. Its value is 0.5 for
    white noise, but becomes greater for time series that exhibit some positive
    dependency on previous values. For negative dependencies it becomes less
    than 0.5.

  Explanation of the algorithm:
    The rescaled range (R/S) approach is directly derived from Hurst's
    definition. The time series of length N is split into non-overlapping
    subseries of length n. Then, R and S (S = sigma) are calculated for each
    subseries and the mean is taken over all subseries yielding (R/S)_n. This
    process is repeated for several lengths n. Finally, the exponent K is
    obtained by fitting a straight line to the plot of log((R/S)_n) vs log(n).

    There seems to be no consensus how to chose the subseries lenghts n.
    This function therefore leaves the choice to the user. The module provides
    some utility functions for "typical" values:

      * binary_n: N/2, N/4, N/8, ...
      * logarithmic_n: min_n, min_n * f, min_n * f^2, ...

  References:
    .. [h_1] H. E. Hurst, “The problem of long-term storage in reservoirs,”
       International Association of Scientific Hydrology. Bulletin, vol. 1,
       no. 3, pp. 13–27, 1956.
    .. [h_2] H. E. Hurst, “A suggested statistical model of some time series
       which occur in nature,” Nature, vol. 180, p. 494, 1957.
    .. [h_3] R. Weron, “Estimating long-range dependence: finite sample
       properties and confidence intervals,” Physica A: Statistical Mechanics
       and its Applications, vol. 312, no. 1, pp. 285–299, 2002.

  Reference Code:
    .. [h_a] "hurst" function in R-package "pracma",
             url: https://cran.r-project.org/web/packages/pracma/pracma.pdf

             Note: Pracma yields several estimates of the Hurst exponent, which
             are listed below. Unless otherwise stated they use the divisors
             of the length of the sequence as n. The length is reduced by at
             most 1% to find the value that has the most divisors.

             * The "Simple R/S" estimate is just log((R/S)_n) / log(n) for
               n = N.
             * The "theoretical Hurst exponent" is the value that would be
               expected of an uncorrected rescaled range approach for random
               noise of the size of the input data.
             * The "empirical Hurst exponent" is the uncorrected Hurst exponent
               obtained by the rescaled range approach.
             * The "corrected empirical Hurst exponent" is the
               Anis-Lloyd-Peters corrected Hurst exponent, but with
               sqrt(1/2 * pi * n) added to the (R/S)_n before the log.
             * The "corrected R over S Hurst exponent" uses the R-function "lm"
               instead of pracmas own "polyfit" and uses n = N/2, N/4, N/8, ...
               by successively halving the subsequences (which means that some
               subsequences may be one element longer than others). In contrast
               to its name it does not use the Anis-Lloyd-Peters correction
               factor.

             If you want to compare the output of pracma to the output of
             nolds, the "empirical hurst exponent" is the only measure that
             exactly corresponds to the Hurst measure implemented in nolds
             (by choosing corrected=False, fit="poly" and employing the same
             strategy for choosing n as the divisors of the (reduced)
             sequence length).
    .. [h_b] Rafael Weron, "HURST: MATLAB function to compute the Hurst
             exponent using R/S Analysis",
             url: https://ideas.repec.org/c/wuu/hscode/m11003.html

             Note: When the same values for nvals are used and fit is set to
             "poly", nolds yields exactly the same results as this
             implementation.
    .. [h_c] Bill Davidson, "Hurst exponent",
             url: http://www.mathworks.com/matlabcentral/fileexchange/9842-hurst-exponent

  Args:
    data (array-like of float):
      time series
  Kwargs:
    nvals (iterable of int):
      sizes of subseries to use
      (default: logmid_n(total_N, ratio=1/4.0, nsteps=15) , that is 15
      logarithmically spaced values in the medium 25% of the logarithmic range)

      Generally, the choice for n is a trade-off between the length and the
      number of the subsequences that are used for the calculation of the
      (R/S)_n. Very low values of n lead to high variance in the ``r`` and
      ``s`` while very high values may leave too few subsequences that the mean
      along them is still meaningful. Logarithmic spacing makes sense, because
      it translates to even spacing in the log-log-plot.
    fit (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``
    corrected (boolean):
      if True, the Anis-Lloyd-Peters correction factor will be applied to the
      output according to the expected value for the individual (R/S)_n
      (see [h_3]_)
    unbiased (boolean):
      if True, the standard deviation based on the unbiased variance
      (1/(N-1) instead of 1/N) will be used. This should be the default choice,
      since the true mean of the sequences is not known. This parameter should
      only be changed to recreate results of other implementations.

  Returns:
    float:
      estimated Hurst exponent K using a rescaled range approach (if K = 0.5
      there are no long-range correlations in the data, if K < 0.5 there are
      negative long-range correlations, if K > 0.5 there are positive
      long-range correlations)
    (1d-vector, 1d-vector, list):
      only present if debug_data is True: debug data of the form
      ``(nvals, rsvals, poly)`` where ``nvals`` are the values used for log(n),
      ``rsvals`` are the corresponding log((R/S)_n) and ``poly`` are the line
      coefficients (``[slope, intercept]``)
  """
  data = np.asarray(data)
  total_N = len(data)
  if nvals is None:
    # chooses a default value for nvals that will give 15 logarithmically
    # spaced datapoints leaning towards the middle of the logarithmic range
    # (since both too small and too large n introduce too much variance)
    nvals = logmid_n(total_N, ratio=1/4.0, nsteps=15)
  # get individual values for (R/S)_n
  rsvals = np.array([rs(data, n, unbiased=unbiased) for n in nvals])
  # filter NaNs (zeros should not be possible, because if R is 0 then
  # S is also zero)
  not_nan = np.logical_not(np.isnan(rsvals))
  rsvals = rsvals[not_nan]
  nvals = np.asarray(nvals)[not_nan]
  # it may happen that no rsvals are left (if all values of data are the same)
  if len(rsvals) == 0:
    poly = [np.nan, np.nan]
    if debug_plot:
      warnings.warn(
        "Cannot display debug plot, all (R/S)_n are NaN",
        RuntimeWarning
      )
  else:
    # fit a line to the logarithm of the obtained (R/S)_n
    xvals = np.log(nvals)
    yvals = np.log(rsvals)
    if corrected:
      yvals -= np.log([expected_rs(n) for n in nvals])
    poly = poly_fit(xvals, yvals, 1, fit=fit)
    if debug_plot:
      plot_reg(xvals, yvals, poly, "log(n)", "log((R/S)_n)",
               fname=plot_file)
  # account for correction if necessary
  h = poly[0] + 0.5 if corrected else poly[0]
  # return line slope (+ correction) as hurst exponent
  if debug_data:
    return (h, (np.log(nvals), np.log(rsvals), poly))
  else:
    return h

# TODO implement MFDFA as second (more reliable) measure for multifractality
# NOTE: probably not needed, since mfhurst_b is already pretty reliable


def mfhurst_b(data, qvals=None, dists=None, fit='poly',
              debug_plot=False, debug_data=False, plot_file=None):
  """
  Calculates the Generalized Hurst Exponent H_q for different q according to
  A.-L. Barabási and T. Vicsek.

  Explanation of the Generalized Hurst Exponent:
    The Generalized Hurst Exponent (GHE, H_q or H(q)) can (as the name implies)
    be seen as a generalization of the Hurst exponent for data series with
    multifractal properties. It's origins are however not directly related
    to Hurst's rescaled range approach, but to the definition of self-affine
    functions.

    A single-valued self-affine function h by definition satisfies the relation

      h(x) ~= lambda^(-H) h(lambda x)

    for any positive real valued lambda and some positive real valued exponent
    H, which is called the Hurst, Hölder, Hurst-Hölder or roughness exponent
    in the literature. In other words you can view lambda as a scaling factor
    or "step size". With lambda < 1 we decrease the step size and zoom into our
    function. In this case lambda^(-H) becomes greater than one, meaning that
    h(lambda x) looks similar to a smaller version of h(x). With lambda > 1 we
    zoom out and get lambda^(-H) < 1.

    To calculate H, you can use the height-height correlation function (also
    called autocorrelation) c(d) = <(h(x) - h(x + d))^2>_x where <...>_x
    denotes the expected value over x. Here, the aforementioned self-affine
    property is equivalent to c(d) ~ d^(2H). You can also think of d as a step
    size. Increasing or decreasing d from 1 to some y is the same as setting
    lambda = y: It increases or decreases the scale of the function by a factor
    of 1/y^(-H) = y^H. Therefore the squared differences will be proportional
    to y^2H.

    A.-L. Barabási and T. Vicsek extended this notion to an infinite hierarchy
    of exponents H_q for the qth-order correlation function with

      c_q(d) = <(h(x) - h(x + d))^q>_x ~ d^(q H_q)

    With q = 1 you get a value H_1 that is closely related to the normal Hurst
    exponent, but with different q you either get a constant value H_q = H_0
    independent of q, which indicates that the function has no multifractal
    properties, or different H_q, which is a sign for multifractal behavior.

    T. Di Matteo, T. Aste and M. M. Dacorogna applied this technique to
    financial data series and gave it the name "Generalized Hurst Exponent".

  Explanation of the Algorithm:
    Curiously, I could not find any algorithmic description how to calculate
    H_q in the literature. Researchers seem to just imply that you can obtain
    the exponent by a line fitting algorithm in a log-log plot, but they do not
    talk about the actual procedure or the required parameters.

    Essentially, we can calculate c_q(d) of a discrete evenly sampled time
    series Y = [y_0, y_1, y_2, ... y_(N-1)] by taking the absolute differences
    [\|y_0 - y_d\|, \|y_1 - y_(d+1)\|, ... , \|y_(N-d-1) - y_(N-1)\|] raising them to
    the qth power and taking the mean.

    Now we take the logarithm on both sides of our relation c_q(d) ~ d^(q H_q)
    and get

    log(c_q(d)) ~ log(d) * q H_q

    So in other words if we plot log(c_q(d)) against log(d) for several d we
    should get a straight line with slope q H_q. This enables us to use a
    linear least squares algorithm to obtain H_q.

    Note that we consider x as a discrete variable in the range 0 <= x < N.
    We can do this, because the actual sampling rate of our data series does
    not alter the result. After taking the logarithm any scaling factor delta_x
    would only result in an additive term since
    log(delta_x * x) = log(x) + log(delta_x) and we only care about the slope
    of the line and not the intercept.

  References:
    .. [mh_1] A.-L. Barabási and T. Vicsek, “Multifractality of self-affine
       fractals,” Physical Review A, vol. 44, no. 4, pp. 2730–2733, 1991.

  Args:
    data (array-like of float):
      time series of data points (should be evenly sampled)

  Kwargs:
    qvals (iterable of float or int):
      values of q for which H_q should be calculated (default: [1])
    dists (iterable of int):
      distances for which the height-height correlation should be calculated
      (determines the x-coordinates in the log-log plot)
      default: logarithmic_n(1, max(20, 0.02 * len(data)), 1.5) to ensure
      even spacing on the logarithmic axis
    fit (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``

  Returns:
    array of float:
      list of H_q for every q given in ``qvals``
    (1d-vector, 2d-vector, 2d-vector):
      only present if debug_data is True: debug data of the form
      ``(xvals, yvals, poly)`` where ``xvals`` is the logarithm of ``dists``,
      ``yvals`` are the logarithms of the corresponding height-height-
      correlations for each distance (first dimension) and each q
      (second dimension) in the shape len(dists) x len(qvals) and ``poly`` are
      the line coefficients (``[slope, intercept]``) for each q in the shape
      len(qvals) x 2.
  """
  # transform to array if necessary
  data = np.asarray(data, dtype=np.float64)
  if qvals is None:
    # actual default parameter would introduce shared list
    # see: http://pylint-messages.wikidot.com/messages:w0102
    qvals = [1]
  if dists is None:
    dists = logarithmic_n(1, max(20, 0.02 * len(data)), 1.5)
  dists = np.asarray(dists)
  if len(data) < 60:
    warnings.warn(
      "H(q) is not reliable for small time series ({} < 60)".format(len(data))
    )

  def hhcorr(d, q):
    diffs = np.abs(data[:-d] - data[d:])
    diffs = diffs[np.where(diffs > 0)]
    return np.mean(diffs ** q)

  # calculate height-height correlations
  corrvals = [hhcorr(d, q) for d in dists for q in qvals]
  corrvals = np.array(corrvals, dtype=np.float64)
  corrvals = corrvals.reshape(len(dists), len(qvals))

  # line fitting
  xvals = np.log(dists)
  yvals = np.log(corrvals)
  polys = [
    poly_fit(xvals, yvals[:, qi], 1, fit=fit)
    for qi in range(len(qvals))
  ]
  H = np.array(polys)[:, 0] / qvals
  if debug_plot:
    plot_reg_multiple(
      [xvals] * len(qvals),
      [yvals[:, qi] / qvals[qi] for qi in range(len(qvals))],
      [p / q for p, q in zip(polys, qvals)],
      x_label="log(x)", y_label="$\\log(c_q(x)) / q$",
      data_labels=["q = %d" % q for q in qvals],
      reg_labels=["reg. line (H = {:.3f})".format(h) for h in H],
      fname=plot_file
    )
  if debug_data:
    return H, (xvals, yvals, polys)
  else:
    return H


def _genhurst(S, q):
    """
    Computes the generalized hurst exponent H_q for time series S.

    This function should not be used. It is only kept here to demonstrate that
    ``mfhurst_dm`` is implemented correctly. You can use the following call to
    get the exact same result:

    ``mfhurst_dm(S, [q])``

    Reference code:
      .. [gh_a] Tomaso Aste, "Generalized Hurst exponent",
         url: http://de.mathworks.com/matlabcentral/fileexchange/30076-generalized-hurst-exponent
      .. [gh_b] Peter Rupprecht, "GenHurst",
         url: https://github.com/PTRRupprecht/GenHurst

    Below you can find the original documentation by T. Aste:

    ####################################
    # Calculates the generalized Hurst exponent H(q) from the scaling
    # of the renormalized q-moments of the distribution
    #
    #       <|x(t+r)-x(t)|^q>/<x(t)^q> ~ r^[qH(q)]
    #
    ####################################
    # H = genhurst(S,q)
    # S is 1xT data series (T>50 recommended)
    # calculates H, specifies the exponent q
    #
    # example:
    #   generalized Hurst exponent for a random vector
    #   H=genhurst(np.random.rand(10000,1),3)
    #
    ####################################
    # for the generalized Hurst exponent method please refer to:
    #
    #   T. Di Matteo et al. Physica A 324 (2003) 183-188
    #   T. Di Matteo et al. Journal of Banking & Finance 29 (2005) 827-851
    #   T. Di Matteo Quantitative Finance, 7 (2007) 21-36
    #
    ####################################
    ##   written in Matlab : Tomaso Aste, 30/01/2013 ##
    ##   translated to Python (3.6) : Peter Rupprecht, p.t.r.rupprecht (AT) gmail.com, 25/05/2017 ##
    ##   formatting and datatype fixes : Christopher Schölzel, 17/02/2019 ##
    """
    L = len(S)
    if L < 100:
        warnings.warn('Data series very short!')
    H = np.zeros((len(range(5, 20)), 1))
    k = 0

    for Tmax in range(5, 20):

        x = np.arange(1, Tmax+1, 1)
        mcord = np.zeros((Tmax, 1))

        for tt in range(1, Tmax+1):
            dV = S[np.arange(tt, L, tt)] - S[np.arange(tt, L, tt)-tt]
            VV = S[np.arange(tt, L+tt, tt)-tt]
            N = len(dV) + 1
            X = np.arange(1, N+1, dtype=np.float64)
            Y = VV
            mx = np.sum(X)/N
            SSxx = np.sum(X**2) - N*mx**2
            my = np.sum(Y)/N
            SSxy = np.sum(np.multiply(X, Y)) - N*mx*my
            cc1 = SSxy/SSxx
            cc2 = my - cc1*mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1, np.arange(1, N+1, dtype=np.float64)) \
                      - cc2
            mcord[tt-1] = np.mean(np.abs(ddVd)**q)/np.mean(np.abs(VVVd)**q)
        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x)**2) - Tmax*mx**2
        my = np.mean(np.log10(mcord))
        SSxy = np.sum(
          np.multiply(
            np.log10(x), np.transpose(np.log10(mcord))
          )
        ) - Tmax*mx*my
        H[k] = SSxy/SSxx
        k = k + 1
    mH = np.mean(H)/q

    return mH


def _aste_line_fit(x, y):
  """
  Simple linear regression with ordinary least squares
  https://en.wikipedia.org/wiki/Simple_linear_regression

  NOTE: this function is left here to demonstrate the correctness of
  T. Aste's MATLAB code for hurst_multifractal_dm. You can get the same
  results with a call to ``np.polyfit(x, y, 1)[::-1]``.
  """
  # convert to float to avoid integer overflow problems
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  N = len(x)
  mx = np.mean(x)
  my = np.mean(y)
  # calculate the variance in x
  # sum((x - mx) ^ 2) = sum(x ^ 2) - 2 * sum(x * mx) + N * mx ^ 2
  #                   = sum(x ^ 2) - 2 * mx * sum(x) + N * mx ^ 2
  #                   = sum(x ^ 2) - 2 * mx * N * mx + N * mx ^ 2
  #                   = sum(x ^ 2) - N * mx ^ 2
  var = np.sum(x ** 2) - N * mx * mx
  # corvariance of x and y
  # sum((x - mx) * (y - my))
  #   = sum(xy) - sum(mx * y) - sum(my * x) + N * mx * my
  #   = sum(xy) - mx * sum(y) - my * sum(x) + N * mx * my
  #   = sum(xy) - mx * my * N - my * mx * N + N * mx * my
  #   = sum(xy) - N * mx * my
  # NOTE: T. Aste's code is a little confusing here
  #    X = 1:N;
  #    Y = S(((tt+1):tt:(L+tt))-tt)';
  #    ...
  #    SSxy = sum(X.*Y) - N*mx*my;
  # Here, Y is transposed and the multiplication for SSxy uses .* instead of *.
  # This suggests that we have a matrix multiplication with (possible)
  # broadcasting. If X was an array and not a range, we would have a NxN array
  # as a result since size(X) = [1, N] and size(Y) = [N, 1]. Ranges behave
  # differently in MATLAB and this is the only reason why we get the correct
  # result here.
  cov = np.sum(x * y) - N * mx * my
  # calculate slope and intercept (this is correct again)
  slope = cov / var
  intercept = my - slope * mx
  return [intercept, slope]


def mfhurst_dm(data, qvals=None, max_dists=range(5, 20), detrend=True,
               fit="poly", debug_plot=False, debug_data=False, plot_file=None):
  """
  Calculates the Generalized Hurst Exponent H_q for different q according to
  the MATLAB code of Tomaso Aste - one of the authors that introduced this
  measure.

  Explanation of the General Hurst Exponent:
    See mfhurst_b.

  Warning: I do not recommend to use this function unless you want to reproduce
  examples from Di Matteo et al.. From my experiments and a critical code
  analysis it seems that mfhurst_b should provide more robust results.

  The design choices that make mfhurst_dm different than mfhurst_d are the
  following:

  - By default, a linear trend is removed from the data. This can be sensible
      in some application areas (such as stock market analysis), but I think
      this should be an additional preprocessing step and not part of this
      algorithm.
  - In the calculation of the height-height correlations, the differences
      (h(x) - h(x + d) are not calculated for every possible x from 0 to N-d-1,
      but instead d is used as a step size for x. I see no justification for
      this choice. It makes the algorithm run faster, but it also takes away
      a lot of statistical robustness, especially for large values of d.
      This effect can be clearly seen when setting `debug_plot` to `True`.
  - The algorithm uses a linear scale for the distance values d = 1, 2, 3,
      ..., tau_max. This is counter intuitive, since we later plot log(d)
      against log(c_q(d)). A linear scale will have a bias towards larger
      values in the logarithmic scale. A logarithmic scale for d seems to be
      a more natural fit. If low values of d yield statistically unstable
      results, they should simply be omitted.
  - The algorithm tests multiple values for tau_max, which is the maximum
      distance that will be calculated. In [mhd_1]_ the authors state that this
      is done to test the robustness of the approach. However, taking the
      mean of several runs with different tau_max will not produce any more
      information than performing one run with the largest tau_max. Instead
      it will only introduce a bias towards low values for d.

  References:
    .. [mhd_1] T. Di Matteo, T. Aste, and M. M. Dacorogna, “Scaling behaviors
       in differently developed markets,” Physica A: Statistical Mechanics
       and its Applications, vol. 324, no. 1–2, pp. 183–188, 2003.

  Reference code:
    .. [mhd_a] Tomaso Aste, "Generalized Hurst exponent",
       url: http://de.mathworks.com/matlabcentral/fileexchange/30076-generalized-hurst-exponent

  Args:
    data (1d-vector of float):
      input data (should be evenly sampled)
    qvals (1d-vector of float)
      values of q for which H_q should be calculated (default: [1])

  Kwargs:
    max_dists (1d-vector of int):
      different values to test for tau_max, the maximum value for the distance
      d. The resulting H_q will be a mean of all H_q calculated with tau_max
      = max_dists[0], max_dists[1], ... .
    detrend (boolean):
      if True, a linear trend will be removed from the data before H_q will
      be calculated
    fit (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``

  Returns:
    array of float:
      array of mH_q for every q given in ``qvals`` where mH_q is the mean of
      all H_q calculated for different max distances in max_dists.
    array of float:
      array of standard deviations sH_q for each mH_q returned
    (1d-vector, 2d-vector, 2d-vector):
      only present if debug_data is True: debug data of the form
      ``(xvals, yvals, poly)`` where ``xvals`` is the logarithm of ``dists``,
      ``yvals`` are the logarithms of the corresponding height-height-
      correlations for each distance (first dimension) and each q
      (second dimension) in the shape len(dists) x len(qvals) and ``poly`` are
      the line coefficients (``[slope, intercept]``) for each q in the shape
      len(qvals) x 2.
  """
  # transform to array if necessary
  data = np.asarray(data)
  if qvals is None:
    # actual default parameter would introduce shared list
    # see: http://pylint-messages.wikidot.com/messages:w0102
    qvals = [1]
  if len(data) < 60:
    warnings.warn(
      "H(q) is not reliable for small time series ({} < 60)".format(len(data))
    )
  max_max_dist = np.max(max_dists)
  hhcorr = []
  # NOTE: I don't think it's a good idea to use a linear scale for the distance
  # values. Our fit is in logarithmic space, so this will place more weight on
  # the higher distance. This is not bad per se, but if you think that the
  # first values are unreliable, it would be better to skip them alltogether.
  for dist in range(1, max_max_dist+1):
    # NOTE: I don't think applying a step size to the input data is reasonable.
    # I cannot find any justification for this in the papers and reduces the
    # number of points that we can use to make our mean statistically stable.
    step_size = dist
    stepdata = data[::step_size]
    if detrend:
      stepdata = detrend_data(stepdata, order=1)
    diffs = stepdata[1:] - stepdata[:-1]
    hhcorr.append([
      np.mean(np.abs(diffs) ** q) / np.mean(np.abs(stepdata) ** q)
      for q in qvals
    ])
  hhcorr = np.array(hhcorr, dtype=np.float64)
  xvals = np.log(np.arange(1, max_max_dist+1))
  yvals = np.log(hhcorr)
  # NOTE: Using several maximum distances seems to be a strange way to
  # introduce stability, since it only places emphasis on the lower distance
  # ranges and does not introduce any new information.
  H = np.array([
    poly_fit(xvals[:md], yvals[:md, qi], 1, fit=fit)[0]
    for qi in range(len(qvals))
    for md in max_dists
  ], dtype=np.float64).reshape(len(qvals), len(max_dists))
  if debug_plot:
    polys = [
      np.array(poly_fit(xvals, yvals[:, qi], 1)) / qvals[qi]
      for qi in range(len(qvals))
    ]
    plot_reg_multiple(
      [xvals] * len(qvals),
      [yvals[:, qi] / qvals[qi] for qi in range(len(qvals))],
      polys,
      x_label="log(x)", y_label="$\\log(c_q(x)) / q$",
      data_labels=["q = %d" % q for q in qvals],
      reg_labels=["reg. line (H = {:.3f})".format(h) for h in H[:, -1] / qvals],
      fname=plot_file
    )
  mH = np.mean(H, axis=1) / qvals
  sH = np.std(H, axis=1) / qvals
  if debug_data:
    return [mH, sH, (xvals, yvals, polys)]
  else:
    return [mH, sH]


def corr_dim(data, emb_dim, lag=1, rvals=None, dist=rowwise_euclidean,
             fit="RANSAC", debug_plot=False, debug_data=False, plot_file=None):
  """
  Calculates the correlation dimension with the Grassberger-Procaccia algorithm

  Explanation of correlation dimension:
    The correlation dimension is a characteristic measure that can be used
    to describe the geometry of chaotic attractors. It is defined using the
    correlation sum C(r) which is the fraction of pairs of points X_i in the
    phase space whose distance is smaller than r.

    If the relation between C(r) and r can be described by the power law

    C(r) ~ r^D

    then D is called the correlation dimension of the system.

    In a d-dimensional system, the maximum value for D is d. This value is
    obtained for systems that expand uniformly in each dimension with time.
    The lowest possible value is 0 for a system with constant C(r) (i.e. a
    system that visits just one point in the phase space). Generally if D is
    lower than d and the system has an attractor, this attractor is called
    "strange" and D is a measure of this "strangeness".

  Explanation of the algorithm:
    The Grassberger-Procaccia algorithm calculates C(r) for a range of
    different r and then fits a straight line into the plot of log(C(r))
    versus log(r).

    This version of the algorithm is created for one-dimensional (scalar) time
    series. Therefore, before calculating C(r), a delay embedding of the time
    series is performed to yield emb_dim dimensional vectors
    Y_i = [X_i, X_(i+1*lag), X_(i+2*lag), ... X_(i+(embd_dim-1)*lag)]. Choosing
    a higher value for emb_dim allows to reconstruct higher dimensional dynamics
    and avoids "systematic errors due to corrections to scaling". Choosing a
    higher value for lag allows to avoid overestimating correlation because
    X_i ~= X_i+1, but it should also not be set too high to not underestimate
    correlation due to exponential divergence of trajectories in chaotic systems.

  References:
    .. [cd_1] P. Grassberger and I. Procaccia, “Characterization of strange
              attractors,” Physical review letters, vol. 50, no. 5, p. 346,
              1983.
    .. [cd_2] P. Grassberger and I. Procaccia, “Measuring the strangeness of
              strange attractors,” Physica D: Nonlinear Phenomena, vol. 9,
              no. 1, pp. 189–208, 1983.
    .. [cd_3] P. Grassberger, “Grassberger-Procaccia algorithm,”
              Scholarpedia, vol. 2, no. 5, p. 3043.
              urL: http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm

  Reference Code:
    .. [cd_a] "corrDim" function in R package "fractal",
              url: https://cran.r-project.org/web/packages/fractal/fractal.pdf
    .. [cd_b] Peng Yuehua, "Correlation dimension",
              url: http://de.mathworks.com/matlabcentral/fileexchange/24089-correlation-dimension

  Args:
    data (array-like of float):
      time series of data points
    emb_dim (int):
      embedding dimension
  Kwargs:
    rvals (iterable of float):
      list of values for to use for r
      (default: logarithmic_r(0.1 * std, 0.5 * std, 1.03))
    dist (function (2d-array, 1d-array) -> 1d-array):
      row-wise difference function
    fit (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``

  Returns:
    float:
      correlation dimension as slope of the line fitted to log(r) vs log(C(r))
    (1d-vector, 1d-vector, list):
      only present if debug_data is True: debug data of the form
      ``(rvals, csums, poly)`` where ``rvals`` are the values used for log(r),
      ``csums`` are the corresponding log(C(r)) and ``poly`` are the line
      coefficients (``[slope, intercept]``)
  """
  # TODO determine lag in units of time instead of number of datapoints
  data = np.asarray(data)

  # TODO what are good values for r?
  # TODO do this for multiple values of emb_dim?
  if rvals is None:
    sd = np.std(data, ddof=1)
    rvals = logarithmic_r(0.1 * sd, 0.5 * sd, 1.03)
  orbit = delay_embedding(data, emb_dim, lag=lag)
  n = len(orbit)
  dists = np.zeros((len(orbit), len(orbit)), dtype=np.float64)
  for i in range(len(orbit)):
    # calculate distances between X_i and X_i+1, X_i+2, ... , X_n-1
    # NOTE: strictly speaking, [cd_1] does not specify to exclude self-matches
    # however, since both [cd_2] and [cd_3] specify to only compare i with j != i
    # or j > i respectively, it is safe to assume that this was an oversight in
    # [cd_1]
    d = dist(orbit[i+1:], orbit[i])
    dists[i+1:,i] = d  # fill column i
    dists[i,i+1:] = d  # fill row i
  csums = []
  for r in rvals:
    # NOTE: The [cd_1] and [cd_2] both use the factor 1/N^2 here.
    # However, since we only use these values to fit a line in a log-log plot
    # any multiplicative constant doesn't change the result since it will
    # only result in an offset on the y-axis. Also, [cd_3] has a point here
    # in that if we exclude self-matches in the numerator, it makes sense to
    # also exclude self-matches from the denominator.
    s = 1.0 / (n * (n - 1)) * np.sum(dists <= r)
    csums.append(s)
  csums = np.array(csums)
  # filter zeros from csums
  nonzero = np.where(csums != 0)
  rvals = np.array(rvals)[nonzero]
  csums = csums[nonzero]
  if len(csums) == 0:
    # all sums are zero => we cannot fit a line
    poly = [np.nan, np.nan]
  else:
    poly = poly_fit(np.log(rvals), np.log(csums), 1, fit=fit)
  if debug_plot:
    plot_reg(np.log(rvals), np.log(csums), poly, "log(r)", "log(C(r))",
             fname=plot_file)
  if debug_data:
    return (poly[0], (np.log(rvals), np.log(csums), poly))
  else:
    return poly[0]


def detrend_data(data, order=1, fit="poly"):
  """
  Removes a trend of given order from the data.
  """
  # TODO also use this function in dfa
  xvals = np.arange(len(data))
  trend = poly_fit(xvals, data, order, fit=fit)
  detrended = data - np.polyval(trend, xvals)
  return detrended


def dfa(data, nvals=None, overlap=True, order=1, fit_trend="poly",
        fit_exp="RANSAC", debug_plot=False, debug_data=False, plot_file=None):
  """
  Performs a detrended fluctuation analysis (DFA) on the given data

  Recommendations for parameter settings by Hardstone et al.:
    * nvals should be equally spaced on a logarithmic scale so that each window
      scale hase the same weight
    * min(nvals) < 4 does not make much sense as fitting a polynomial (even if
      it is only of order 1) to 3 or less data points is very prone to errors.
    * max(nvals) > len(data) / 10 does not make much sense as we will then have
      less than 10 windows to calculate the average fluctuation
    * use overlap=True to obtain more windows and therefore better statistics
      (at an increased computational cost)

  Explanation of DFA:
    Detrended fluctuation analysis, much like the Hurst exponent, is used to
    find long-term statistical dependencies in time series. However, while the
    Hurst exponent will indicate long-term correlations for any non-stationary
    process (i.e. a stochastic process whose probability distribution changes
    when shifted in time, such as a random walk whose mean changes over time),
    DFA was designed to distinguish between correlations that are purely an
    artifact of non-stationarity and those that show inherent long-term
    behavior of the studied system.

    Mathematically, the long-term correlations that we are interested in can
    be characterized using the autocorrelation function C(s). For a time series
    (x_i) with i = 1, ..., N it is defined as follows:

    C(s) = 1/(N-s) * (y_1 * y_1+s + y_2 * y_2+s + ... y_(N-s) * y_N)

    with y_i = x_i - mean(x). If there are no correlations at all, C(s) would
    be zero for s > 0. For short-range correlations, C(s) will decline
    exponentially, but for long-term correlations the decline follows a power
    law of the form C(s) ~ s^(-gamma) instead with 0 < gamma < 1.

    Due to noise and underlying trends, calculating C(s) directly is usually not
    feasible. The main idea of DFA is therefore to remove trends up to a given
    order from the input data and analyze the remaining fluctuations. Trends
    in this sense are smooth signals with monotonous or slowly oscillating
    behavior that are caused by external effects and not the dynamical system
    under study.
  
    To get a hold of these trends, the first step is to calculate the "profile"
    of our time series as the cumulative sum of deviations from the mean,
    effectively integrating our data. This both smoothes out measurement noise
    and makes it easier to distinguish the fractal properties of bounded time
    series (i.e. time series whose values cannot grow or shrink beyond certain
    bounds such as most biological or physical signals) by applying random walk
    theory (see [dfa_3]_ and [dfa_4]_).

    y_i = x_1 - mean(x) + x_2 - mean(x) + ... + x_i - mean(x).

    After that, we split Y(i) into (usually non-overlapping) windows of length
    n to calculate local trends at this given scale. The ith window of this
    size has the form

    W_(n,i) = [y_i, y_(i+1), y_(i+2), ... y_(i+n-1)]
    
    The local trends are then removed for each window separately by fitting a
    polynomial p_(n,i) to the window W_(n,i) and then calculating
    W'_(n,i) = W_(n,i) - p_(n,i) (element-wise subtraction).

    This leaves us with the deviations from the trend - the "fluctuations" -
    that we are interested in. To quantify them, we take the root mean square
    of these fluctuations. It is important to note that we have to sum up all
    individual fluctuations across all windows and divide by the total number
    of fluctuations here before finally taking the root as last step. Some
    implementations apply another root per window, which skews the result.

    The resulting fluctuation F(n) is then only dependent on the window size n,
    the scale at which we observe our data. It behaves similar to the
    autocorrelation function in that it follows a power-law for long-term
    correlations:

    F(n) ~ n^alpha

    Where alpha is the Hurst parameter, which we can obtain from fitting a line
    into the plot of log(n) versus log(F(n)) and taking the slope.

    The result can be interpreted as follows: For alpha < 1 the underlying
    process is stationary and can be modelled as fractional Gaussian noise with
    H = alpha. This means for alpha = 0.5 we have no long-term correlation or
    "memory", for 0.5 < alpha < 1 we have positive long-term correlations and
    for alpha < 0.5 the long-term correlations are negative.

    For alpha > 1 the underlying process is non-stationary and can be modeled
    as fractional Brownian motion with H = alpha - 1.

  References:
    .. [dfa_1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
               H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
               DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.
    .. [dfa_2] J. W. Kantelhardt, E. Koscielny-Bunde, H. H. A. Rego, S.
               Havlin, and A. Bunde, “Detecting long-range correlations with
               detrended fluctuation analysis,” Physica A: Statistical
               Mechanics and its Applications, vol. 295, no. 3–4, pp. 441–454,
               Jun. 2001, doi: 10.1016/S0378-4371(01)00144-3.
    .. [dfa_3] C. Peng, J. M. Hausdorff, and A. L. Goldberger, “Fractal
               mechanisms in neuronal control: human heartbeat and gait
               dynamics in health and disease,” in Self-Organized Biological
               Dynamics and Nonlinear Control, 1st ed., J. Walleczek, Ed.,
               Cambridge University Press, 2000, pp. 66–96.
               doi: 10.1017/CBO9780511535338.006.
    .. [dfa_4] A. Bashan, R. Bartsch, J. W. Kantelhardt, and S. Havlin,
               “Comparison of detrending methods for fluctuation analysis,”
               Physica A: Statistical Mechanics and its Applications, vol. 387,
               no. 21, pp. 5080–5090, Sep. 2008,
               doi: 10.1016/j.physa.2008.04.023.
    .. [dfa_5] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
               V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
               “Detrended fluctuation analysis: A scale-free view on neuronal
               oscillations,” Frontiers in Physiology, vol. 30, 2012.

  Reference code:
    .. [dfa_a] Peter Jurica, "Introduction to MDFA in Python",
       url: http://bsp.brain.riken.jp/~juricap/mdfa/mdfaintro.html
    .. [dfa_b] JE Mietus, "dfa",
       url: https://www.physionet.org/physiotools/dfa/dfa-1.htm
    .. [dfa_c] "DFA" function in R package "fractal"

  Args:
    data (array-like of float):
      time series
  Kwargs:
    nvals (iterable of int):
      subseries sizes at which to calculate fluctuation
      (default: logarithmic_n(4, 0.1*len(data), 1.2))
    overlap (boolean):
      if True, the windows W_(n,i) will have a 50% overlap,
      otherwise non-overlapping windows will be used
    order (int):
      (polynomial) order of trend to remove
    fit_trend (str):
      the fitting method to use for fitting the trends, either 'poly'
      for normal least squares polynomial fitting or 'RANSAC' for
      RANSAC-fitting which is more robust to outliers but also tends to
      lead to unstable results
    fit_exp (str):
      the fitting method to use for the line fit, either 'poly' for normal
      least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
      is more robust to outliers
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    debug_data (boolean):
      if True, debugging data will be returned alongside the result
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      ``plt.show()``
  Returns:
    float:
      the estimate alpha for the Hurst parameter (alpha < 1: stationary
      process similar to fractional Gaussian noise with H = alpha,
      alpha > 1: non-stationary process similar to fractional Brownian
      motion with H = alpha - 1)
    (1d-vector, 1d-vector, list):
      only present if debug_data is True: debug data of the form
      ``(nvals, fluctuations, poly)`` where ``nvals`` are the values used for
      log(n), ``fluctuations`` are the corresponding log(std(X,n)) and ``poly``
      are the line coefficients (``[slope, intercept]``)
  """
  data = np.asarray(data)
  total_N = len(data)
  if nvals is None:
    if total_N > 70:
      nvals = logarithmic_n(4, 0.1 * total_N, 1.2)
    elif total_N > 10:
      nvals = [4, 5, 6, 7, 8, 9]
    else:
      nvals = [total_N-2, total_N-1]
      msg = "choosing nvals = {} , DFA with less than ten data points is " \
          + "extremely unreliable"
      warnings.warn(msg.format(nvals), RuntimeWarning)
  if len(nvals) < 2:
    raise ValueError("at least two nvals are needed")
  if np.min(nvals) < 2:
    raise ValueError("nvals must be at least two")
  if np.max(nvals) >= total_N:
    raise ValueError("nvals cannot be larger than the input size")
  # create the signal profile
  # (cumulative sum of deviations from the mean => "walk")
  walk = np.cumsum(data - np.mean(data))
  fluctuations = []
  for n in nvals:
    assert n >= 2
    # subdivide data into chunks of size n
    if overlap:
      # step size n/2 instead of n
      d = np.array([walk[i:i + n] for i in range(0, len(walk) - n, n // 2)])
    else:
      # non-overlapping windows => we can simply do a reshape
      d = walk[:total_N - (total_N % n)]
      d = d.reshape((total_N // n, n))
    # calculate local trends as polynomes
    x = np.arange(n)
    tpoly = [poly_fit(x, d[i], order, fit=fit_trend)
             for i in range(len(d))]
    tpoly = np.array(tpoly)
    trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
    # calculate mean-square differences for each walk in d around trend
    flucs = np.sum((d - trend) ** 2, axis=1) / n
    # take another mean across all walks and finally take the square root of that
    # NOTE: To map this to the formula in Peng1995, observe that this simplifies
    # to np.sqrt(np.sum((d - trend) ** 2) / total_N) if we have non-overlapping
    # windows and the last window matches the end of the data perfectly.
    f_n = np.sqrt(np.sum(flucs) / len(flucs))
    fluctuations.append(f_n)
  fluctuations = np.array(fluctuations)
  # filter zeros from fluctuations
  nonzero = np.where(fluctuations != 0)
  nvals = np.array(nvals)[nonzero]
  fluctuations = fluctuations[nonzero]
  if len(fluctuations) == 0:
    # all fluctuations are zero => we cannot fit a line
    poly = [np.nan, np.nan]
  else:
    poly = poly_fit(np.log(nvals), np.log(fluctuations), 1,
                    fit=fit_exp)
  if debug_plot:
    plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)",
             fname=plot_file)
  if debug_data:
    return (poly[0], (np.log(nvals), np.log(fluctuations), poly))
  else:
    return poly[0]
