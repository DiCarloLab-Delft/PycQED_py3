import numpy as np
import time
import matplotlib.pyplot as plt
from importlib import reload # Useful for reloading during testin
# load the qcodes path, until we have this installed as a package

import qcodes as qc

# Globally defined config
qc_config = {'datadir':'D:\data',
             'PycQEDdir': 'D:\GitHub\PycQED_py3'}

from modules.measurement import mc_parameter_wrapper as pw
reload(pw)

from instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
from instrument_drivers.physical_instruments import cbox_v3_driver as qcb3
from instrument_drivers.physical_instruments._controlbox import test_suite_v3
from instrument_drivers.physical_instruments._controlbox import defHeaders_CBox_v3 as header

reload(qcb3)
reload(test_suite_v3)
