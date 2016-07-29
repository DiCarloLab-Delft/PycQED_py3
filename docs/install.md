# Install using Anaconda python 
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
* cython (only for specific drivers)
* flake8 (to ensure proper linting when editing code)

These can be installed by typing (on the command line) 
``` pip install 'package name' ``` 
or in some cases 
``` conda install 'package name' ``` 

Note that qcodes is not yet (Jul 29 2016) publicly released so has to be installed by hand. See the qcodes repository for instructions. 

## Adding PycQED to the path 
At the start of your python session add the PycQED_py3 folder to the path using 

```python 
import sys
if PyCQEDpath not in sys.path:
    sys.path.append(PyCQEDpath)
```

