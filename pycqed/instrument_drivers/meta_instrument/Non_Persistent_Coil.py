
import time
import logging
import numpy as np
from qcodes.utils import validators as vals
from pycqed.analysis import analysis_toolbox as atools

from pycqed.measurement import detector_functions as det

from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import analysis_toolbox as a_tools
import logging
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter




class Non_Persistent_Coil(Instrument):

    def __init__(self, name,
                 Current_source_magnet,
                 field_to_current,
                 min_field,
                 max_field,
                 field_step,
                 ramp_rate,
                 MC_inst,**kw):
        super().__init__(name, **kw)

        # Set current source
        self.add_parameter('i_magnet', parameter_class=InstrumentParameter)
        self.i_magnet = Current_source_magnet
        self.MC = MC_inst
        self.add_parameter('field',
                           get_cmd=self.get_field,
                           set_cmd=self.set_field,
                           label='Magnetic Field',
                           unit='T',
                           vals=vals.Numbers(min_value=min_field,
                                             max_value=max_field),
                           docstring='Magnetic Field sourced by Coil')
        self.field_to_current = field_to_current # units of T/A
        self.field_step = field_step
        self.ramp_rate = ramp_rate # units of A/s

        self.persistent = False

        self.get_field()

    def set_field(self,field):
        self.step_field_to_value(field)
        return 0.


    def get_field(self):
        I = self.i_magnet.measurei()
        B = self.curr_to_field(I)
        return B

    def step_field_to_value(self,field):
        MagCurrRatio = self.field_to_current # Tesla/Ampere
        step_time = 0.01 # in seconds
        if self.ramp_rate is not None:
            Ramp_rate_I = self.ramp_rate
            current_step = Ramp_rate_I  * step_time
        elif self.field_step is not None:
            current_step = self.field_to_curr(self.field_step)
        else:
            raise ValueError('Specify either ramp rate or field step!')
        I_now = self.field_to_curr(self.get_field())
        current_target = self.field_to_curr(field)
        if current_target >= I_now:
            current_step *= +1
        if current_target < I_now:
            current_step *= -1
        num_steps = int(1.*(current_target-I_now)/(current_step))
        sweep_time = step_time*num_steps
        print('Sweep time is '+str(np.abs(sweep_time))+' seconds')
        for tt in range(num_steps):
            time.sleep(step_time)
            self.i_magnet.seti(I_now)
            I_now += current_step
        self.i_magnet.seti(self.field_to_curr(field))


    def field_to_curr(self,field):
        return field/self.field_to_current


    def curr_to_field(self,current):
        return self.field_to_current*current



