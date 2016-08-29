import numpy as np
from . import measures as nolds # import internal module to test helping functions
import unittest

class TestNoldsHelperFunctions(unittest.TestCase):
	"""
	Tests for internal helper functions that are not part of the public API
	"""
	def assert_array_equals(self, expected, actual):
		print(actual)
		print("==")
		print(expected)
		print()
		self.assertTrue(np.alltrue(actual == expected))
	def test_delay_embed_lag2(self):
		data = np.arange(10, dtype="float32")
		embedded = nolds.delay_embedding(data,4,lag=2)
		expected = np.array([
			[0,2,4,6],
			[1,3,5,7],
			[2,4,6,8],
			[3,5,7,9]
		], dtype="float32")
		self.assert_array_equals(expected, embedded)
	def test_delay_embed(self):
		data = np.arange(6, dtype="float32")
		embedded = nolds.delay_embedding(data,4)
		expected = np.array([
			[0,1,2,3],
			[1,2,3,4],
			[2,3,4,5]
		], dtype="float32")
		self.assert_array_equals(expected, embedded)
	def test_delay_embed_lag3(self):
		data = np.arange(10, dtype="float32")
		embedded = nolds.delay_embedding(data,4,lag=3)
		expected = np.array([
			[0,3,6,9]
		], dtype="float32")
		self.assert_array_equals(expected, embedded)
	def test_delay_embed_empty(self):
		data = np.arange(10, dtype="float32")
		try:
			embedded = nolds.delay_embedding(data,11)
			self.fail("embedding array of size 10 with embedding dimension 11 should fail, got {} instead".format(embedded))
		except ValueError:
			pass
		data = np.arange(10, dtype="float32")
		try:
			embedded = nolds.delay_embedding(data,4,lag=4)
			self.fail("embedding array of size 10 with embedding dimension 4 and lag 4 should fail, got {} instead".format(embedded))
		except ValueError:
			pass

class TestNoldsUtility(unittest.TestCase):
	"""
	Tests for small utility functions that are part of the public API
	"""
	def test_binary_n(self):
		x = nolds.binary_n(1000, min_n=50)
		self.assertSequenceEqual(x, [500, 250, 125, 62])
	def test_binary_n_empty(self):
		x = nolds.binary_n(50, min_n=50)
		self.assertSequenceEqual(x, [])
	def test_logarithmic_n(self):
		x = nolds.logarithmic_n(4,11,1.51)
		self.assertSequenceEqual(x, [4,6,9])
	def test_logarithmic_r(self):
		x = nolds.logarithmic_r(4,10,1.51)
		self.assertSequenceEqual(x, [4, 6.04, 9.1204])

class TestNoldsLyap(unittest.TestCase):
	def test_lyap_fbm(self):
		# TODO use logistic function as example
		data = nolds.fbm(1000, H=0.3)
		le = nolds.lyap_e(data, emb_dim=7, matrix_dim=3)
		self.assertGreater(np.max(le), 0)

def plot_lyap():
	# TODO this should be moved into an example file
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
			tmp = abs(r-2*r*x)
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
		lambdas_est2.append(nolds.lyap_r(np.array(full_data), emb_dim=6, lag=2, min_tsep=10, trajectory_len=20))
	plt.plot(rvalues, lambdas, "b-", label="true lyap. exponent")
	plt.plot(rvalues, lambdas_est, color="#00AAAA", label="estimation using lyap_e")
	plt.plot(rvalues, lambdas_est2, color="#AA00AA", label="estimation using lyap_r")
	plt.plot(rvalues, np.zeros(len(rvalues)), "g--")
	xvals = np.repeat(rvalues, 20)
	plt.plot(xvals, maps, "ro", alpha=0.1, label="bifurcation plot")
	plt.ylim((-2,2))
	plt.xlabel("r")
	plt.ylabel("lyap. exp / logistic(x, r)")
	plt.legend(loc="best")
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

def profiling():
	import cProfile
	n = 100000
	data = np.cumsum(np.random.random(n)-0.5)
	cProfile.runctx('lyap_e(data)',{'lyap_e': lyap_e},{'data': data})


if __name__ == "__main__":
	unittest.main()
	#test_hurst()
	#test_lyap()
	#test_corr()
	#test_dfa()
	#test_delay_embed()
	# r = 3.9
	# logistic = lambda x : r * x * (1 - x)
	# x = [0.1]
	# for i in range(100):
	# 	x.append(logistic(x[-1]))
	# plt.plot(x)
	# plt.show()
	#profiling()

