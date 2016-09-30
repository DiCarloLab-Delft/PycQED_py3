# Installing PycQED
1. Install python
2. Install dependencies
3. Clone PycQED_py3 git repository

## Install python
We recommend using [anaconda python](https://www.continuum.io/downloads) (3.5+) as it covers most dependencies

## Dependencies
Besides the standard scientific python packages we require the following packages.

* qcodes
* numpy
* scipy
* matplotlib
* lmfit
* uncertainties
* seaborn
* h5py
* pyvisa
* pyqtgraph (live-plotting)
* cython (only for specific drivers) [cython on windows](compiling_with_cython.md)
* flake8 (to ensure proper linting when editing code)

These can be installed by typing (on the command line)
``` pip install 'package name' ```
or in some cases
``` conda install 'package name' ```

Note that qcodes is not yet (Jul 29 2016) publicly released so has to be installed by hand. See the qcodes repository for instructions.

## Setting up PycQED
After cloning PycQED go to the directory where you cloned PycQED and enter: `python setup.py develop`.
You can start a session by `import pycqed as pq`.


