#!/usr/bin/env python

from setuptools import setup, find_packages

long_description = """
Compyrtment is a simple Python library to work with compartment models. It's 
a simple tool to aid with fast prototyping with models such as the SIR, SIS 
and SEIR epidemic models or the Lotka-Volterra model of predation. It supports:

- Designing models with arbitrary states and constant, linear and quadratic
  couplings between them;
- Numerical integration of the ODE system of a model;
- Numerical integration of 'sensitivity curves' - the gradient of each curve
  with respect to parameters as well as initial conditions;
- Fitting a model's parameters to existing data;
- Stochastic simulation using Gillespie's algorithm
"""

setup(name='Compyrtment',
      version='0.5.3',
      description='A simple Python library for compartment models',
      long_description=long_description,
      url='https://github.com/stur86/compyrtment',
      author='Simone Sturniolo',
      author_email='simonesturniolo@gmail.com',
      keywords=['differential equations', 'compartment models', 'epidemiology',
                'ecology', 'reactions', 'gillespie'],
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 4 - Beta',

          # Intended audience
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',

          # Topics
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Life',
          'Topic :: Scientific/Engineering :: Chemistry',

          # License
          'License :: OSI Approved :: MIT License',
          # Specify the Python versions you support here. In particular,
          # ensure that you indicate whether you support Python 2, Python 3
          # or both.
          'Programming Language :: Python :: 3',
      ],
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'numba'
      ],
      python_requires='>=3.1.*'
      )
