from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module
import re


def get_version(verbose=1):
    """ Extract version information from source code """

    try:
        with open('pycqed/version.py', 'r') as f:
            ln = f.readline()
            # print(ln)
            m = re.search('.* ''(.*)''', ln)
            version = (m.group(1)).strip('\'')
    except Exception as E:
        print(E)
        version = 'none'
    if verbose:
        print('get_version: %s' % version)
    return version


def readme():
    with open('README.md') as f:
        return f.read()


def license():
    with open('LICENSE.md') as f:
        return f.read()

setup(name='PycQED',
      version=get_version(),
      use_2to3=False,
      author='Adriaan Rol',
      author_email='adriaan.rol@gmail.com',
      maintainer='Adriaan Rol',
      maintainer_email='adriaan.rol@gmail.com',
      description='Python based Circuit QED data acquisition framework '
                  'developed by members of the DiCarlo-lab at '
                   'QuTech, Delft University of Technology',
      long_description=readme(),
      url='https://github.com/DiCarloLab-Delft/PycQED_py3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering'
      ],
      license=license(),
      # if we want to install without tests:
      # packages=find_packages(exclude=["*.tests", "tests"]),
      packages=find_packages(),
      install_requires=[
          'numpy>=1.10',
          'pyvisa>=1.8',
          'IPython>=4.0',
          'ipywidgets>=4.1',
          'lmfit>=0.9.5',
          'qcodes>=0.1.0',
          'scipy>=0.17',
          'h5py>=2.6',
      ],
      test_suite='pycqed.tests',
      zip_safe=False)



