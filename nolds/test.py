import numpy as np
import nolds.measures as nolds # import internal module to test helping functions

def test_lyap():
	import matplotlib.pyplot as plt # local import to avoid dependency for non-debug use
	rvalues = np.arange(2, 4, 0.01)
	lambdas = []
	lambdas_est = []
	lambdas_est2 = []
	maps = []
	logistic = lambda x : r * x * (1 - x)
	for r in rvalues:
		x = 0.1
		result = []
		full_data = [x]
		# ignore first 100 values for bifurcation plot
		for t in range(100):
			x = logistic(x)
			result.append(np.log(abs(r-2*r*x)))
			full_data.append(x)
		lambdas.append(np.mean(result))
		for t in range(20):
			x = logistic(x)
			maps.append(x)
			full_data.append(x)
		le = nolds.lyap_e(np.array(full_data), emb_dim=6, matrix_dim=2)
		#print(full_data)
		#print(le)
		lambdas_est.append(np.max(le))
		lambdas_est2.append(nolds.lyap_r(np.array(full_data), emb_dim=6, lag=2, min_tsep=10, trajectory_len=20))
	#print(lambdas_est)
	print(lambdas_est2)
	plt.plot(rvalues, lambdas, "b-")
	plt.plot(rvalues, lambdas_est, color="#00AAAA")
	plt.plot(rvalues, lambdas_est2, color="#AA00AA")
	plt.plot(rvalues, np.zeros(len(rvalues)), "g--")
	xvals = np.repeat(rvalues, 20)
	plt.plot(xvals, maps, "ro", alpha=0.1)
	plt.ylim((-2,2))
	plt.show()

def test_lyap2():
	#test_lyap()
	data = [1,2,4,5,6,6,1,5,1,2,4,5,6,6,1,5,1,2,4,5,6,6,1,5]
	data = np.random.random((100,)) * 10
	data = np.concatenate([np.arange(100)] * 3)
	# TODO random numbers should give positive exponents, what is happening here?
	l = nolds.lyap_e(np.array(data), emb_dim=7, matrix_dim=3)
	print(l)

def test_hurst():
	# TODO why does this not work for the brownian motion?
	n = 10000
	data = np.arange(n) # should give result 1
	#data = np.cumsum(np.random.randn(n)) # brownian motion, should give result 0.5
	#data = np.random.randn(n) # should give result 0
	data = np.sin(np.arange(n,dtype=float) / (n-1) * np.pi * 100)
	print(nolds.hurst_rs(data, debug_plot=True))

def test_corr():
	n = 1000
	data = np.arange(n)
	print(nolds.corr_dim(data, 4))

def test_dfa():
	import matplotlib.pyplot as plt # local import to avoid dependency for non-debug use
	n = 10000
	data = np.arange(n)
	data = np.random.randn(n)
	data = np.cumsum(data)
	plt.plot(data)
	plt.show()
	print(nolds.dfa(data))

def test_logarithmic_n():
	import matplotlib.pyplot as plt # local import to avoid dependency for non-debug use
	print(nolds.binary_n(1000))
	print(nolds.logarithmic_n(4,100,1.1))
	x = nolds.logarithmic_n(4,100,1.1)
	x = np.log(list(x))
	plt.plot(x,np.arange(len(x)))
	plt.show()

def test_delay_embed():
	data = np.arange(57)
	print(nolds.delay_embedding(data,4,lag=2))

def profiling():
	import cProfile
	n = 100000
	data = np.cumsum(np.random.random(n)-0.5)
	cProfile.runctx('lyap_e(data)',{'lyap_e': lyap_e},{'data': data})

if __name__ == "__main__":
	#test_hurst()
	#test_lyap()
	#test_corr()
	#test_dfa()
	test_delay_embed()
	# r = 3.9
	# logistic = lambda x : r * x * (1 - x)
	# x = [0.1]
	# for i in range(100):
	# 	x.append(logistic(x[-1]))
	# plt.plot(x)
	# plt.show()
	#profiling()

