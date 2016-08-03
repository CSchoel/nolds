# -*- coding: utf-8 -*-
from distutils.core import setup
setup(
  name = 'nolds',
  packages = ['nolds'],
  version = '0.1.0',
  description = 'Nonlinear measures for dynamical systems (based on one-dimensional time series)',
  author = 'Christopher Schölzel',
  author_email = 'christopher.schoelzel@gmx.net',
  #url = 'https://github.com/peterldowns/mypackage', # use the URL to the github repo
  #download_url = 'https://github.com/peterldowns/mypackage/tarball/0.1',
  keywords = [
    'nonlinear', 'dynamical system', 'chaos', 
    'lyapunov', 'hurst', 'hurst exponent', 'rescaled range', 'DFA', 
    'detrended fluctuation analysis', 'sample entropy', 'correlation dimension'],
  classifiers = [],
)
