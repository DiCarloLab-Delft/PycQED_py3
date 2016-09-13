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

Can be compiled by calling

"python setup.py build_ext --inplace"

on the command line.

Warning, it may compile the .so file to a nested folder in which case you
need to move it.

Most likely you will get a vcvarshall not found error when compiling on
windows the first time. This is because cython needs MS visual studio.
see also: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

The following gives a clear explanation on how to fix versions redirection
of the visual studio for compiling c++
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_Cython_On_Anaconda_On_Windows?lang=en
MS visual studio 2015 is free for academic research and can be downloaded here.
https://www.visualstudio.com/en-us/products/vs-2015-product-editions.aspx
make sure you check c++ drivers when installing.
'''

setup(
    ext_modules=cythonize("decoder.pyx")
)
