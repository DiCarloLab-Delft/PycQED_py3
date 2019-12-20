# PycQED 
[![Build Status](https://github.com/DiCarloLab-Delft/pycqed_py3/workflows/Build%20Status/badge.svg)](https://github.com/DiCarloLab-Delft/pycqed_py3/actions)
[![DOI](https://zenodo.org/badge/49057179.svg)](https://zenodo.org/badge/latestdoi/49057179)
[![Codacy](https://api.codacy.com/project/badge/Grade/1266308dd9b84d7b933c2b46804aeb12)](https://www.codacy.com/app/AdriaanOrganization/PycQED_py3?utm_source=github.com&utm_medium=referral&utm_content=DiCarloLab-Delft/PycQED_py3&utm_campaign=badger)
[![codecov](https://codecov.io/gh/DiCarloLab-Delft/PycQED_py3/branch/master/graph/badge.svg)](https://codecov.io/gh/DiCarloLab-Delft/PycQED_py3)

A Python-based measurement environment for circuit-QED experiments by the 
[DiCarlo group](http://dicarlolab.tudelft.nl/) at [QuTech](http://qutech.nl/),
Delft University of Technology.
This module is build on top of [QCoDeS](http://qcodes.github.io/Qcodes/) and 
is not intended as a stand-alone 
package.

## License
This software is released under the [MIT License](LICENSE.md)


## Contributing
Please see [Contributing.md](.github/CONTRIBUTING.md)

## Installation

To use PycQED, clone this repository and add the directory to your path (no pip-install yet) and make sure you have a working python3 environment with the required dependencies.
Go to the directory where you cloned the repository (in the shell) and run
* `python setup.py develop`.
For more details see the [installation instructions](docs/install.md).

Test your installation using
* `py.test pycqed/tests -v`
(be sure that you are in the PycQED_py3 root folder)

Or run a specific test using e.g.
* `py.test pycqed/tests/test_cliffords.py -v`
(be sure that you are in the PycQED_py3 root folder)

## Usage

+ start up QCoDeS
+ load corresponding [config file](init\config\)
+ set folder locations
+ create instruments
+ load values set in the init

```python
import sys
import qcodes as qc
import pycqed as pq
from init.your_initscript import *
```

You are now ready to start your experiment.

If you use this software in any of your publications we would appreciate it if you cite this repository using the [![DOI](https://zenodo.org/badge/49057179.svg)](https://zenodo.org/badge/latestdoi/49057179).

## Overview of the main modules
Below follows an overview of the main structure of the code. It makes sense to take a look around here if your are new to get a feeling where to find things.
Also take a look at [this presentation](docs/160714_qcodes_meetup.pdf), where the relation to qcodes and the core concepts in the package are explained.
Mind however that the code is continuously under development so if you think something should be in a different location feel free to tap me (Adriaan) on the shoulder or create an issue to discuss it.

### Folder Structure
+ [docs](docs/)
+ [init](init/)
+ [analysis](analysis/)
+ [measurement](measurement/)
+ [utilities](utilities/)
+ [instrument_drivers](instrument_drivers/)
+ [scripts](scripts/)
    + [testing](scripts/testing/)
    + [personal_folders](scripts/personal_folders/)


### The init folder
Contains script that are to be used when setting up an experiment. Used to store configuration info and create instruments.

###The instruments folder

PycQED makes extensive use of instruments. Instruments are children of the qcodes instrument class and can be used as drivers for physical instruments,
but can also provide a layer of abstraction in the form of meta-instruments, which contain other instruments.

We use these qcodes instruments for several reasons; the class provides logging of variables, provides a standardized format for getting and setting parameters, protects the underlying instruments against setting 'bad' values. Additionally qcodes itself comes with drivers for most instruments we use in the lab.

We split the instrument folder up in several subfolders. The physical instruments are drivers that control physical instruments. Meta-instruments are higher level instruments that control other lower level instruments. The main point here is that commands only flow down and information flows up.

#### Measurement Control
The **Measurement Control** is a special object that is used to run experiments. It takes care of preparing an experiment, giving instructions to the instruments involved and saving the data.

Below is an example of running a homodyne experiment using the measurement control.

```python
MC.set_sweep_function(Source.frequency)
MC.set_sweep_points(np.arange(freq_start,freq_stop,freq_step))
MC.set_detector_function(det.HomodyneDetector())
MC.run()
```

A sweep_function determines what parameter is varied, a qcodes parameter that contains a .set method can also be inserted here.
A deterector_function determines what parameter is measrued, a qcodes parameter that has a .get method can also be inserted here.

#### The qubit object
The qubit object is a (meta) instrument but it defies the general categorization of the other instruments.

It is the object that one is actively manipulating during an experiment and as such contains functions such as qubit.measure_Rabi() and qubit.find_frequency_spec(). It is also used to store the known parameters of the physical qubit object.

Because the qubit object is also used to start experiments and update itself it makes active use of the measurement control.
It's functions are (generally) split into the following types, these correspond to prefixes in the function names.
* measure
* calibrate
* find
* tune

### The modules folder
The modules folder contains core modules of PyCQED. It is split into measurement and analysis. It also contains a utilities module which contains some general functions (such as send_email) which do not really fit anywhere else.

#### Measurement

The measurement module contains mostly modules with sweep and detector functions. These are used to define what instruments have to do when performing a measurement.
The sweep function defines what variable(s) are sweeped and the detector function defines how it is to be detected. Both sweep and detector functions are split into soft(ware) and hard(ware) controlled sweep and detector functions.

In a soft measurement the measurement loop is controlled completely by PyCQED which sets a sweep point on an instrument and then uses another instrument to 'probe' the measured quantity. An example of such a measurement is a Homodyne measurement.

In a hard measurement the measurement is prepared by the software and then triggered and controlled completely by the hardware which returns all the data after completing an iteration of the experiment. An example of such a measurement would be running an AWG-sequence.

#### Analysis
The measurement analysis currently (april 2015) contains three modules.

##### Measurement analysis
The measurement analysis module contains the main analysis objects. By instantiating one of these objects a dataset can be analyzed. It contains default methods for the most common experiments and makes extensive use of object oriented hereditary relations. This means that for example the Rabi_Analysis is a child of the TD_Analysis which is a child of the MeasurementAnalysis and has all the functions of it's parent classes.

When instantiating such an object you can pass it a timestamp and/or a label to determine what datafile to load. By giving it the argument auto=True the default analysis script is run.
##### Analysis Toolbox
This toolbox contains tools for analysis such as file-handling tools, plotting tools and some data analysis tools such as a peak finder.

##### Fitting models
This module contains the lmfit model definitions and fitting functions used for curve fitting.

### The scripts folder
The scripts that run an experiment. Dig around a bit here to get a feel of what a measurement looks like.
It is split into personal folders for messing around with your personal files and project folders where people working on the same project have their experimental scripts.

## Other useful stuff

A little document containing some handy git commands:
[Git tips & tricks](docs/git_tips_and_tricks.md).

Lecture series on scientific Python: 
[Scientific Computing with Python](https://github.com/jrjohansson/scientific-python-lectures)
