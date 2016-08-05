# -*- coding: utf-8 -*-
from setuptools import setup
setup(
  name = 'nolds',
  packages = ['nolds'],
  version = '0.1.1',
  description = 'Nonlinear measures for dynamical systems (based on one-dimensional time series)',
  author = 'Christopher Schölzel',
  author_email = 'christopher.schoelzel@gmx.net',
  url = 'https://github.com/CSchoel/nolds',
  download_url = 'https://github.com/CSchoel/nolds/tarball/0.1.1',
  keywords = [
    'nonlinear', 'dynamical system', 'chaos', 
    'lyapunov', 'hurst', 'hurst exponent', 'rescaled range', 'DFA', 
    'detrended fluctuation analysis', 'sample entropy', 'correlation dimension'],
  classifiers = [],
  install_requires = ['numpy'],
)
