try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize

'''
@author: MAR
Builds the c files for decoder.pyx

Can be compiled by calling "python setup.py build_ext --inplace" on the
command line.

Warning, it may compile the .so file to a nested folder in which case you
need to move it.
'''

setup(
    ext_modules=cythonize("decoder.pyx")
)
