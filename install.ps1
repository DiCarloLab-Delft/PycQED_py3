# This is a PowerShell based script used to install PycQED on your Windows OS computer.

# Prerequisite:
#  OS: Windows 7 and later versions.
#  - Anaconda based on Python 3.5+
#    - Download: https://www.continuum.io/downloads
#  - The PyqCED source code
#    - Download: https://github.com/DiCarloLab-Delft/PycQED_py3

# install packages.
conda install h5py
conda install matplotlib
pip install pyqtgraph
conda install numpy
conda install scipy
pip install lmfit
pip install uncertainties
conda install seaborn
pip install pyvisa
pip install qcodes

# install PyqCED
python setup.py develop