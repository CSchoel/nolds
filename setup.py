# -*- coding: utf-8 -*-
from setuptools import setup, Command
import glob
import shutil


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree("build")
        shutil.rmtree("dist")
        shutil.rmtree("nolds.egg-info")


version = '0.2.1'
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4'
    ],
    test_suite='nolds.test_measures',
    install_requires=['numpy'],
    cmdclass={
        'clean': CleanCommand
    }
)
