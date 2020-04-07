#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='Compyrtment',
      version='0.1',
      description='A simple Python library for compartment models',
      author='Simone Sturniolo',
      author_email='simonesturniolo@gmail.com',
      url='https://github.com/stur86/compyrtment',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'ase'
      ],
      python_requires='>=3.1.*'
      )

