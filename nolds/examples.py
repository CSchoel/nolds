from . import measures as nolds
import numpy as np


def plot_lyap():
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  rvalues = np.arange(2, 4, 0.01)
  lambdas = []
  lambdas_est = []
  lambdas_est2 = []
  maps = []

  def logistic(x):
    return r * x * (1 - x)

  for r in rvalues:
    x = 0.1
    result = []
    full_data = [x]
    # ignore first 100 values for bifurcation plot
    for t in range(100):
      x = logistic(x)
      tmp = abs(r - 2 * r * x)
      dx = np.nan if tmp <= 0 else np.log(tmp)
      result.append(dx)
      full_data.append(x)
    lambdas.append(np.mean(result))
    for t in range(20):
      x = logistic(x)
      maps.append(x)
      full_data.append(x)
    le = nolds.lyap_e(np.array(full_data), emb_dim=6, matrix_dim=2)
    lambdas_est.append(np.max(le))
    lr = nolds.lyap_r(np.array(full_data), emb_dim=6, lag=2, min_tsep=10,
                      trajectory_len=20)
    lambdas_est2.append(lr)
  plt.title("Lyapunov exponent of the logistic map")
  plt.plot(rvalues, lambdas, "b-", label="true lyap. exponent")
  elab = "estimation using lyap_e"
  rlab = "estimation using lyap_r"
  plt.plot(rvalues, lambdas_est, color="#00AAAA", label=elab)
  plt.plot(rvalues, lambdas_est2, color="#AA00AA", label=rlab)
  plt.plot(rvalues, np.zeros(len(rvalues)), "g--")
  xvals = np.repeat(rvalues, 20)
  plt.plot(xvals, maps, "ro", alpha=0.1, label="bifurcation plot")
  plt.ylim((-2, 2))
  plt.xlabel("r")
  plt.ylabel("lyap. exp / logistic(x, r)")
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
    print("  all")
    print("  lyapunov")
    print("  profiling")
  elif sys.argv[1] == "all" or len(sys.argv) == 1:
    plot_lyap()
  elif sys.argv[1] == "lyapunov":
    plot_lyap()
  elif sys.argv[1] == "profiling":
    profiling()
