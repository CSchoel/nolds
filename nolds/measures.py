# -*- coding: utf-8 -*-
import numpy as np
import warnings

# TODO: use RANSAC instead of simple polyfit?
# TODO: is description of 0.5 for brownian motion really correct for hurst_rs?
# FIXME: dfa fails for very small input sequences


def fbm(n, H=0.75):
  """
  Generates fractional brownian motions of desired length.

  Author:
    Christian Thomae

  References:
    .. [fbm-1] https://en.wikipedia.org/wiki/Fractional_Brownian_motion#Method_1_of_simulation

  Args:
    n (int):
      length of sequence to generate
  Kwargs:
    H (float):
      hurst parameter
  """
  # TODO more detailed description of fbm
  assert H > 0 and H < 1

  def R(t, s):
    twoH = 2 * H
    return 0.5 * (s**twoH + t**twoH - np.abs(t - s)**twoH)
  # form the matrix tau
  gamma = R(*np.mgrid[0:n, 0:n])  # apply R to every element in matrix
  w, P = np.linalg.eigh(gamma)
  L = np.diag(w)
  sigma = np.dot(np.dot(P, np.sqrt(L)), np.linalg.inv(P))
  v = np.random.randn(n)
  return np.dot(sigma, v)

# TODO maybe we can use this function also in other algorithms than lyap_r?


def delay_embedding(data, emb_dim, lag=1):
  """
  Perform a time-delay embedding of a time series

  Args:
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
  if len(data) < (emb_dim - 1) * lag + 1:
    msg = "cannot embed data of length {} with embedding dimension {} " \
        + "and lag {}"
    raise ValueError(msg.format(len(data), emb_dim, lag))
  m = len(data) - (emb_dim - 1) * lag
  indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
  indices += np.arange(m).reshape((m, 1))
  return data[indices]


def lyap_r(data, emb_dim=10, lag=None, min_tsep=None, tau=1, min_vectors=20,
           trajectory_len=20, debug_plot=False, plot_file=None):
  """
  Estimates the largest Lyapunov exponent using the algorithm of Rosenstein
  et al. [lr-1]_.

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
    min_tsep = None.

  Method for choosing lag:
    Another parameter that can be hard to choose by instinct alone is the lag
    between individual values in a vector of the embedded orbit. Here,
    Rosenstein et al. suggest to set the lag to the distance where the
    autocorrelation function drops below 1 - 1/e times its original (maximal)
    value. This procedure is used by default if the user sets lag = None.

  References:
    .. [lr-1] M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
       “A practical method for calculating largest Lyapunov exponents from
       small data sets,” Physica D: Nonlinear Phenomena, vol. 65, no. 1,
       pp. 117–134, 1993.

  Reference Code:
    .. [lr-a] mirwais, "Largest Lyapunov Exponent with Rosenstein's Algorithm",
       url: http://www.mathworks.com/matlabcentral/fileexchange/38424-largest-lyapunov-exponent-with-rosenstein-s-algorithm
    .. [lr-b] Shapour Mohammadi, "LYAPROSEN: MATLAB function to calculate
       Lyapunov exponent",
       url: https://ideas.repec.org/c/boc/bocode/t741502.html

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
      step size between data points in the time series in seconds (default:
      find a suitable value using the autocorrelation function)
    min_vectors (int):
      if lag=None, the search for a suitable lag will be stopped
      when the number of resulting vectors drops below min_vectors
    trajectory_len (int):
      the time (in number of data points) to follow the distance
      trajectories between two neighboring points
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will
      be shown
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      `plt.show()`
  Returns:
    float:
      an estimate of the largest Lyapunov exponent (a positive exponent is
      a strong indicator for chaos)
  """
  # convert data to float to avoid overflow errors in rowwise_euler
  data = data.astype("float32")
  n = len(data)
  max_tsep_factor = 0.25
  if lag is None or min_tsep is None:
    # calculate min_tsep as mean period (= 1 / mean frequency)
    f = np.fft.rfft(data, n * 2 - 1)
    mf = np.fft.rfftfreq(n * 2 - 1) * np.abs(f)
    mf = np.mean(mf[1:]) / np.sum(np.abs(f[1:]))
    min_tsep = int(np.ceil(1.0 / mf))
    if min_tsep > max_tsep_factor * n:
      min_tsep = int(max_tsep_factor * n)
      msg = "signal has very low mean frequency, setting min_tsep = {:d}"
      warnings.warn(msg.format(min_tsep), RuntimeWarning)
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
    for i in range(n):
      if acorr[n - 1 + i] < eps \
          or acorr[n - 1 - i] < eps \
          or 1.0 * n / emb_dim * i < min_vectors:
        lag = i
        break
    if 1.0 * n / emb_dim * lag < min_vectors:
      msg = "autocorrelation declined too slowly to find suitable lag"
      warnings.warn(msg, RuntimeWarning)
  # delay embedding
  orbit = delay_embedding(data, emb_dim, lag)
  m = len(orbit)
  # construct matrix with pairwise distances between vectors in orbit
  dists = np.array([rowwise_euler(orbit, orbit[i]) for i in range(m)])
  # we do not want to consider vectors as neighbor that are less than min_tsep
  # time steps together => mask the distances min_tsep to the right and left of
  # each index by setting them to infinity (will never be considered as nearest
  # neighbors)
  for i in range(m):
    dists[i, max(0, i - min_tsep):i + min_tsep + 1] = float("inf")
  # find nearest neighbors (exclude last columns, because these vectors cannot
  # be followed in time for trajectory_len steps)
  ntraj = m - trajectory_len + 1
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
    poly = np.polyfit(ks, div_traj, 1)
  if debug_plot:
    plot_reg(ks, div_traj, poly, "k", "log(d(k))", fname=plot_file)
  return poly[0] / tau


def lyap_e(data, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1,
           debug_plot=False, plot_file=None):
  """
  Estimates the Lyapunov exponents for the given data using the algorithm of
  Eckmann et al. [le-1]_.

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
      min_nb other vectors lie within (chebychev-)distance r_i around X_i.
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
    .. [le-1] J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto,
       “Liapunov exponents from time series,” Physical Review A,
       vol. 34, no. 6, pp. 4971–4979, 1986.

  Reference code:
    .. [le-a] Manfred Füllsack, "Lyapunov exponent",
       url: http://systems-sciences.uni-graz.at/etextbook/sw2/lyapunov.html
    .. [le-b] Steve SIU, Lyapunov Exponents Toolbox (LET),
       url: http://www.mathworks.com/matlabcentral/fileexchange/233-let/content/LET/findlyap.m
    .. [le-c] Rainer Hegger, Holger Kantz, and Thomas Schreiber, TISEAN,
       url: http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html

  Args:
    data (iterable):
      list/array of (scalar) data points

  Kwargs:
    emb_dim (int):
      embedding dimension
    matrix_dim (int):
      matrix dimension (emb_dim - 1 must be divisible by matrix_dim - 1)
    min_nb (int):
      minimal number of neighbors
      (default: min(2 * matrix_dim, matrix_dim + 4))
    tau (float):
      step size of the data in seconds
      (normalization scaling factor for exponents)
    debug_plot (boolean):
      if True, a histogram matrix of the individual estimates will be shown
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      `plt.show()`

  Returns:
    float array:
      array of matrix_dim Lyapunov exponents (positive exponents are indicators
      for chaos)
  """
  n = len(data)
  if (emb_dim - 1) % (matrix_dim - 1) != 0:
    raise ValueError("emb_dim - 1 must be divisible by matrix_dim - 1!")
  m = (emb_dim - 1) // (matrix_dim - 1)
  if min_nb is None:
    # minimal number of neighbors as suggested by Eckmann et al.
    min_nb = min(2 * matrix_dim, matrix_dim + 4)

  # construct orbit as matrix (e = emb_dim)
  # x0 x1 x2 ... xe-1
  # x1 x2 x3 ... xe
  # x2 x3 x4 ... xe+1
  # ...

  # note: we need to be able to step m points further for the beta vector
  #       => maximum start index is n - emb_dim - m
  orbit_l = [data[i:i + emb_dim] for i in range(n - emb_dim + 1 - m)]
  orbit = np.array(orbit_l, dtype=float)
  old_Q = np.identity(matrix_dim)
  lexp = np.zeros(matrix_dim, dtype="float32")
  lexp_counts = np.zeros(lexp.shape)
  debug_data = []
  # TODO reduce number of points to visit?
  # TODO performance test!
  for i in range(len(orbit)):
    # find neighbors for each vector in the orbit using the chebychev distance
    diffs = np.max(np.abs(orbit - orbit[i]), axis=1)
    # ensure that we do not count the difference of the vector to itself
    diffs[i] = float('inf')
    # mask all neighbors that are too close in time to the vector itself
    diffs[max(0, i - min_tsep):min(len(diffs), i + min_tsep + 1)] = np.inf
    indices = np.argsort(diffs)
    idx = indices[min_nb - 1]  # index of the min_nb-nearest neighbor
    r = diffs[idx]  # corresponding distance
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
    vec_beta = data[indices + matrix_dim * m] - data[i + matrix_dim * m]

    # perform linear least squares
    a, _, _, _ = np.linalg.lstsq(mat_X, vec_beta)
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
    lexp_i = np.zeros(diag_R.shape, dtype="float32")
    lexp_i[idx] = np.log(diag_R[idx])
    lexp_i[np.where(diag_R == 0)] = np.inf
    if debug_plot:
      debug_data.append(lexp_i / tau / m)
    lexp[idx] += lexp_i[idx]
    lexp_counts[idx] += 1
  # end of loop over orbit vectors
  # it may happen that all R-matrices contained zeros => exponent really has
  # to be -inf
  if debug_plot:
    plot_histogram_matrix(np.array(debug_data), "layp_e", fname=plot_file)
  # normalize exponents over number of individual mat_Rs
  idx = np.where(lexp_counts > 0)
  lexp[idx] /= lexp_counts[idx]
  lexp[np.where(lexp_counts == 0)] = np.inf
  # normalize with respect to tau
  lexp /= tau
  # take m into account
  lexp /= m
  return lexp


def plot_dists(dists, tolerance, m, title=None, fname=None):
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  nstd = 3
  nbins = 50
  dists_full = np.concatenate(dists)
  ymax = len(dists_full) * 0.05
  mean = np.mean(dists_full)
  std = np.std(dists_full)
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


def sampen(data, emb_dim=2, tolerance=None, dist="chebychev",
           debug_plot=False, plot_file=None):
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
    [s_1, s_2, s_3, ...] and then counts each pair (s_i, s_j) with i != j
    where dist(s_i, s_j) < tolerance. The same process is repeated for all
    subsequences of length emb_dim + 1. The sum of similar sequence pairs
    with length emb_dim + 1 is divided by the sum of similar sequence pairs
    with length emb_dim. The result of the algorithm is the negative logarithm
    of this ratio/probability.

  References:
    .. [se-1] J. S. Richman and J. R. Moorman, “Physiological time-series
       analysis using approximate entropy and sample entropy,”
       American Journal of Physiology-Heart and Circulatory Physiology,
       vol. 278, no. 6, pp. H2039–H2049, 2000.

  Reference code:
    .. [se-a] "sample_entropy" function in R-package "pracma",
        url: https://cran.r-project.org/web/packages/pracma/pracma.pdf

  Args:
    data (iterable):
      the list/array of data points

  Kwargs:
    emb_dim (int):
      the embedding dimension (length of vectors to compare)
    tolerance (float):
      distance threshold for two template vectors to be considered equal
      (default: 0.2 * std(data))
    dist (string):
      distance function used to calculate the distance between template
      vectors, can be 'euler' or 'chebychev'
    debug_plot (boolean):
      if True, a histogram of the individual distances for m and m+1
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      `plt.show()`

  Returns:
    float:
      the sample entropy of the data (negative logarithm of ratio between
      similar template vectors of length emb_dim + 1 and emb_dim)
  """
  if tolerance is None:
    tolerance = 0.2 * np.std(data)
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
  tVecs = np.zeros((n - emb_dim, emb_dim + 1))
  for i in range(tVecs.shape[0]):
    tVecs[i, :] = data[i:i + tVecs.shape[1]]
  plot_data = []
  counts = []
  for m in [emb_dim, emb_dim + 1]:
    counts.append(0)
    plot_data.append([])
    # get the matrix that we need for the current m
    tVecsM = tVecs[:n - m + 1, :m]
    # successively calculate distances between each pair of template vectors
    for i in range(len(tVecsM) - 1):
      diff = tVecsM[i + 1:] - tVecsM[i]
      if dist == "chebychev":
        dsts = np.max(np.abs(diff), axis=1)
      elif dist == "euler":
        dsts = np.norm(diff, axis=1)
      else:
        raise "unknown distance function: %s" % dist
      if debug_plot:
        plot_data[-1].extend(dsts)
      # count how many distances are smaller than the tolerance
      counts[-1] += np.sum(dsts < tolerance)
  if counts[1] == 0:
    # log would be infinite => cannot determine saen
    saen = np.inf
  else:
    saen = -np.log(1.0 * counts[1] / counts[0])
  if debug_plot:
    plot_dists(plot_data, tolerance, m, title="sampEn = {:.3f}".format(saen),
               fname=plot_file)
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


def rs(data, n):
  """
  Calculates an individual R/S value in the rescaled range approach for
  a given n.

  Note: This is just a helper function for hurs_rs and should not be called
  directly.

  Args:
    data (array of float):
      time series
    n (float):
      size of the subseries in which data should be split

  Returns:
    float:
      (R/S)_n
  """
  total_N = len(data)
  # cut values at the end of data to make the array divisible by n
  data = data[:total_N - (total_N % n)]
  # split remaining data into subsequences of length n
  seqs = np.reshape(data, (total_N // n, n))
  # calculate means of subsequences
  means = np.mean(seqs, axis=1)
  # normalize subsequences by substracting mean
  y = seqs - means.reshape((total_N // n, 1))
  # build cumulative sum of subsequences
  y = np.cumsum(y, axis=1)
  # find ranges
  r = np.max(y, axis=1) - np.min(y, axis=1)
  # find standard deviation
  s = np.std(seqs, axis=1)
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


def plot_histogram_matrix(data, name, fname=None):
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
    rng = (-absmax, absmax)
    h, bins = np.histogram(data[:, i], nbins, rng)
    bin_width = bins[1] - bins[0]
    h = h.astype("float32") / np.sum(h)
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
  function will show a plot through `plt.show()` and close it after the window
  has been closed by the user.

  Args:
    xvals (list/array of float):
      list of x-values
    yvals (list/array of float):
      list of y-values
    poly (list/array of float):
      polynomial parameters as accepted by `np.polyval`
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
      showing it though `plt.show()`)
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


def hurst_rs(data, nvals=None, debug_plot=False, plot_file=None):
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

    To do so, we first substract the mean over all x_i from the individual
    x_i to obtain the departures x'_i from the mean for each year i. As the
    excess or deficit in discharge always carrys over from year i to year i+1,
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

    In this equation, K is called the Hurst exponent. Its value is 0.5 for a
    purely brownian motion, but becomes greater for time series that exhibit
    a bias in one direction.

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
    .. [h-1] H. E. Hurst, “The problem of long-term storage in reservoirs,”
       International Association of Scientific Hydrology. Bulletin, vol. 1,
       no. 3, pp. 13–27, 1956.
    .. [h-2] H. E. Hurst, “A suggested statistical model of some time series
       which occur in nature,” Nature, vol. 180, p. 494, 1957.
    .. [h-3] R. Weron, “Estimating long-range dependence: finite sample
       properties and confidence intervals,” Physica A: Statistical Mechanics
       and its Applications, vol. 312, no. 1, pp. 285–299, 2002.

  Reference Code:
    .. [h-a] "hurst" function in R-package "pracma",
             url: https://cran.r-project.org/web/packages/pracma/pracma.pdf
    .. [h-b] Rafael Weron, "HURST: MATLAB function to compute the Hurst
             exponent using R/S Analysis",
             url: https://ideas.repec.org/c/wuu/hscode/m11003.html
    .. [h-c] Bill Davidson, "Hurst exponent",
             url: http://www.mathworks.com/matlabcentral/fileexchange/9842-hurst-exponent
    .. [h-d] Tomaso Aste, "Generalized Hurst exponent",
             url: http://de.mathworks.com/matlabcentral/fileexchange/30076-generalized-hurst-exponent

  Args:
    data (array of float):
      time series
  Kwargs:
    nvals (iterable of int):
      sizes of subseries to use
      (default: `logarithmic_n(4, 0.1*len(data), 1.2)`)
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      `plt.show()`

  Returns:
    float:
      estimated Hurst exponent K using a rescaled range approach (if K = 0.5
      there are no long-range correlations in the data, if K < 0.5 there are
      negative long-range correlations, if K > 0.5 there are positive
      long-range correlations)
  """
  total_N = len(data)
  if nvals is None:
    nvals = logarithmic_n(4, 0.1 * total_N, 1.2)
  # get individual values for (R/S)_n
  rsvals = np.array([rs(data, n) for n in nvals])
  # filter NaNs (zeros should not be possible, because if R is 0 then
  # S is also zero)
  rsvals = rsvals[np.logical_not(np.isnan(rsvals))]
  # it may happen that no rsvals are left (if all values of data are the same)
  if len(rsvals) == 0:
    poly = [np.nan, np.nan]
  else:
    # fit a line to the logarithm of the obtained (R/S)_n
    poly = np.polyfit(np.log(nvals), np.log(rsvals), 1)
  if debug_plot:
    plot_reg(np.log(nvals), np.log(rsvals), poly, "log(n)", "log((R/S)_n)",
             fname=plot_file)
  # return line slope
  return poly[0]


def rowwise_chebychev(x, y):
  return np.max(np.abs(x - y), axis=1)


def rowwise_euler(x, y):
  return np.sqrt(np.sum((x - y)**2, axis=1))


def corr_dim(data, emb_dim, rvals=None, dist=rowwise_euler, debug_plot=False,
             plot_file=None):
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
    Y_i = [X_i, X_(i+1), X_(i+2), ... X_(i+embd_dim-1)]. Choosing a higher
    value for emb_dim allows to reconstruct higher dimensional dynamics and
    avoids "systematic errors due to corrections to scaling".

  References:
    .. [cd-1] P. Grassberger and I. Procaccia, “Characterization of strange
              attractors,” Physical review letters, vol. 50, no. 5, p. 346,
              1983.
    .. [cd-2] P. Grassberger and I. Procaccia, “Measuring the strangeness of
              strange attractors,” Physica D: Nonlinear Phenomena, vol. 9,
              no. 1, pp. 189–208, 1983.
    .. [cd-3] P. Grassberger, “Grassberger-Procaccia algorithm,”
              Scholarpedia, vol. 2, no. 5, p. 3043.
              urL: http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm

  Reference Code:
    .. [cd-a] "corrDim" function in R package "fractal",
              url: https://cran.r-project.org/web/packages/fractal/fractal.pdf
    .. [cd-b] Peng Yuehua, "Correlation dimension",
              url: http://de.mathworks.com/matlabcentral/fileexchange/24089-correlation-dimension

  Args:
    data (array of float):
      time series of data points
    emb_dim (int):
      embedding dimension
  Kwargs:
    rvals (iterable of float):
      list of values for to use for r
      (default: logarithmic_r(0.1 * std, 0.5 * std, 1.03))
    dist (function (2d-array, 1d-array) -> 1d-array):
      row-wise difference function
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      `plt.show()`

  Returns:
    float:
      correlation dimension as slope of the line fitted to log(r) vs log(C(r))
  """
  # TODO what are good values for r?
  # TODO do this for multiple values of emb_dim?
  if rvals is None:
    sd = np.std(data)
    rvals = logarithmic_r(0.1 * sd, 0.5 * sd, 1.03)
  n = len(data)
  orbit = np.array([data[i:i + emb_dim] for i in range(n - emb_dim + 1)])
  dists = np.array([dist(orbit, orbit[i]) for i in range(len(orbit))])
  csums = []
  for r in rvals:
    s = 1.0 / (n * (n - 1)) * np.sum(dists < r)
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
    poly = np.polyfit(np.log(rvals), np.log(csums), 1)
  if debug_plot:
    plot_reg(np.log(rvals), np.log(csums), poly, "log(r)", "log(C(r))",
             fname=plot_file)
  return poly[0]


def dfa(data, nvals=None, overlap=True, order=1,
        debug_plot=False, plot_file=None):
  """
  Performs a detrended fluctuation analysis (DFA) on the given data

  Recommendations for parameter settings by Hardstone et al.:
    * nvals should be equally spaced on a logarithmic scale so that each window
      scale hase the same weight
    * min(nvals) < 4 does not make much sense as fitting a polynomial (even if
      it is only of order 1) to 3 or less data points is very prone.
    * max(nvals) > len(data) / 10 does not make much sense as we will then have
      less than 10 windows to calculate the average fluctuation
    * use overlap=True to obtain more windows and therefore better statistics
      (at an increased computational cost)

  Explanation of DFA:
    Detrended fluctuation analysis, much like the Hurst exponent, is used to
    find long-term statistical dependencies in time series.

    The idea behind DFA originates from the definition of self-affine
    processes. A process X is said to be self-affine if the standard deviation
    of the values within a window of length n changes with the window length
    factor L in a power law:

    std(X,L * n) = L^H * std(X, n)

    where std(X, k) is the standard deviation of the process X calculated over
    windows of size k. In this equation, H is called the Hurst parameter, which
    behaves indeed very similar to the Hurst exponent.

    Like the Hurst exponent, H can be obtained from a time series by
    calculating std(X,n) for different n and fitting a straight line to the
    plot of log(std(X,n)) versus log(n).

    To calculate a single std(X,n), the time series is split into windows of
    equal length n, so that the ith window of this size has the form

    W_(n,i) = [x_i, x_(i+1), x_(i+2), ... x_(i+n-1)]

    The value std(X,n) is then obtained by calculating std(W_(n,i)) for each i
    and averaging the obtained values over i.

    The aforementioned definition of self-affinity, however, assumes that the
    process is  non-stationary (i.e. that the standard deviation changes over
    time) and it is highly influenced by local and global trends of the time
    series.

    To overcome these problems, an estimate alpha of H is calculated by using a
    "walk" or "signal profile" instead of the raw time series. This walk is
    obtained by substracting the mean and then taking the cumulative sum of the
    original time series. The local trends are removed for each window
    separately by fitting a polynomial p_(n,i) to the window W_(n,i) and then
    calculating W'_(n,i) = W_(n,i) - p_(n,i) (element-wise substraction).

    We then calculate std(X,n) as before only using the "detrended" window
    W'_(n,i) instead of W_(n,i). Instead of H we obtain the parameter alpha
    from the line fitting.

    For alpha < 1 the underlying process is stationary and can be modelled as
    fractional Gaussian noise with H = alpha. This means for alpha = 0.5 we
    have no correlation or "memory", for 0.5 < alpha < 1 we have a memory with
    positive correlation and for alpha < 0.5 the correlation is negative.

    For alpha > 1 the underlying process is non-stationary and can be modeled
    as fractional Brownian motion with H = alpha - 1.

  References:
    .. [dfa-1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
               H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
               DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.
    .. [dfa-2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
               V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
               “Detrended fluctuation analysis: A scale-free view on neuronal
               oscillations,” Frontiers in Physiology, vol. 30, 2012.

  Reference code:
    .. [dfa-a] Peter Jurica, "Introduction to MDFA in Python",
       url: http://bsp.brain.riken.jp/~juricap/mdfa/mdfaintro.html
    .. [dfa-b] JE Mietus, "dfa",
       url: https://www.physionet.org/physiotools/dfa/dfa-1.htm
    .. [dfa-c] "DFA" function in R package "fractal"

  Args:
    data (array of float):
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
    debug_plot (boolean):
      if True, a simple plot of the final line-fitting step will be shown
    plot_file (str):
      if debug_plot is True and plot_file is not None, the plot will be saved
      under the given file name instead of directly showing it through
      `plt.show()`
  Returns:
    float:
      the estimate alpha for the Hurst parameter (alpha < 1: stationary
      process similar to fractional Gaussian noise with H = alpha,
      alpha > 1: non-stationary process similar to fractional Brownian
      motion with H = alpha - 1)
  """
  total_N = len(data)
  if nvals is None:
    nvals = logarithmic_n(4, 0.1 * total_N, 1.2)
  # create the signal profile
  # (cumulative sum of deviations from the mean => "walk")
  walk = np.cumsum(data - np.mean(data))
  fluctuations = []
  for n in nvals:
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
    tpoly = np.array([np.polyfit(x, d[i], order) for i in range(len(d))])
    trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
    # calculate standard deviation ("fluctuation") of walks in d around trend
    flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
    # calculate mean fluctuation over all subsequences
    f_n = np.sum(flucs) / len(flucs)
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
    poly = np.polyfit(np.log(nvals), np.log(fluctuations), 1)
  if debug_plot:
    plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)",
             fname=plot_file)
  return poly[0]
