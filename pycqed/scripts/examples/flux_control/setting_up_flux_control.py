"""
This example will contain all the steps required to set up a decoupled flux.
These steps are
    - measure the crosstalk between dacs using spectroscopy
    - perform analysis to extract decoupling matrix
    - set up flux control instrument and integrate it in qubit objects
"""
import numpy as np
######################################
# This part will contain some MC functions most likely?
# see decoupling.py in this same folder
######################################
# This part will contain a single function call?
# Ramiro will work on this.

dac_offsets= np.array([75, -16.59])

dac_mapping = np.array([2, 1])
A = np.array([[-1.71408478e-13,   1.00000000e+00],
              [1.00000000e+00,  -1.32867524e-14]])
######################################
#


from pycqed.instrument_drivers.meta_instruments.flux_control import Flux_Control
FC = Flux_Control('FC', 2, 'IVVI')
station.add_component(FC)


FC.transfer_matrix(A)
FC.dac_mapping(dac_mapping)
FC.dac_offsets(dac_offsets)

# Do not forget to add this to the init as arrays do not get reloaded properly

