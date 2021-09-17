# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='hdc2021',
      version='0.0.1',
      description='Learned Gradient Descent for Deblurring',
      url='https://github.com/alexdenker/hdc2021_LGD',
      author='Alexander Denker, Maximilian Schmidt, Johannes Leuschner, et. al.',
      author_email='adenker@uni-bremen.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pytorch-lightning>=1.3.8',
          'torch>=1.9.0'
	   ],
      include_package_data=True,
      zip_safe=False)
