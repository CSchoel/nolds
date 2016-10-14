# -*- coding: utf-8 -*-
from setuptools import setup
version = '0.2.0'
setup(
    name='nolds',
    packages=['nolds'],
    version=version,
    description='Nonlinear measures for dynamical systems '
                + '(based on one-dimensional time series)',
    author='Christopher Schölzel',
    author_email='christopher.schoelzel@gmx.net',
    url='https://github.com/CSchoel/nolds',
    download_url='https://github.com/CSchoel/nolds/tarball/' + version,
    keywords=[
        'nonlinear', 'dynamical system', 'chaos',
        'lyapunov', 'hurst', 'hurst exponent', 'rescaled range', 'DFA',
        'detrended fluctuation analysis', 'sample entropy',
        'correlation dimension'],
    classifiers=[],
    install_requires=['numpy'],
)
