FIXME: WIP

Original architecture
---------------------

Classes
- CCLight_Transmon
- Base_LutMan and its descendants
- DeviceCCL
- MeasurementControl
- Sweep_function and its descendants
- Detector_Function and its descendants

Problems:
- hardware support scattered over many files
- duplicate functionality in single vs. multi qubit support (CCLight_Transmon vs. DeviceCCL)
- manual control of waveform set used
- manual control of waveform uploading
- not scalable in terms of number of qubits (hardware instruments)
- flaky use of classes in some places
- lots of unused code and code not covered by tests

HAL_Transmon
------------

HAL_ShimSQ
----------

Class HAL_ShimSQ implements a shim between the HAL_Transmon and the instrument hardware for
single qubit operations. It contains hardware dependent functions extracted from CCL_Transmon.py, extended with
functions that abstract the instrument hardware that used to be directly accessed by the end user methods.

FIXME: the latter is Work In Progress, so old style code is still present in HAL_Transmon

QCoDeS parameters referring to instrument hardware are added here, and not in child class HAL_Transmon where they were
originally added. These parameters should only accessed here (although nothing really stops you from violating this
design). Note that we try to find a balance between compatibility with exiting code and proper design here.

The following hardware related attributes are managed here:
- physical instruments of the signal chain, and their
    - connectivity
    - settings
    - modes (if any)
    - signal chain related properties (e.g. modulation, mixer calibration)
    - FIXME: etc


A future improvement is to merge this HAL functions for Single Qubits with those for Multi Qubits.

HAL_Device
----------


HAL_ShimMQ
----------



Ideas
-----

- allow sweeping of LutMan parameters without manually uploading and stop/starting


