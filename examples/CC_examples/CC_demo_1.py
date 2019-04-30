#!/usr/bin/python

import os

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC_core import QuTechCC_core

from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo



ip = '192.168.0.241'
qubit_idx = 10

curdir = os.path.dirname(__file__)
cfg_openql_platform_fn = os.path.join(curdir, 'test_cfg_cc.json')

p = sqo.AllXY(qubit_idx=qubit_idx, double_points=True, platf_cfg=cfg_openql_platform_fn)

print(p.filename)

