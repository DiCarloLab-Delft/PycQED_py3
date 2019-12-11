# American Magnetics, Inc. (AMI) One Axis magnet with PCS_SN14768

import time
import logging
import numpy as np
# from scipy.optimize import brent
# from math import gcd
# from qcodes import Instrument
from qcodes.utils import validators as vals
# from qcodes.instrument.parameter import ManualParameter

# from pycqed.utilities.general import add_suffix_to_dict_keys

# from pycqed.measurement import detector_functions as det
# from pycqed.measurement import composite_detector_functions as cdet
# from pycqed.measurement import mc_parameter_wrapper as pw

# from pycqed.measurement import sweep_functions as swf
# from pycqed.measurement import awg_sweep_functions as awg_swf
# from pycqed.analysis import measurement_analysis as ma
# from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_5014
# from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_UHFQC
# from pycqed.measurement.calibration_toolbox import mixer_skewness_calibration_5014
# from pycqed.measurement.optimization import nelder_mead

# import pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts as sq
import logging
import numpy as np
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter

class AMI_Magnet_with_PCS_SN14768(Instrument):
    '''
    Instrument used for translating Fields into current settings
    and controlling the persistent current switch.

    '''

    def __init__(self, name,
                 Current_source_magnet_name,
                 Current_source_heater_name,**kw): # IVVI.dac1
        super().__init__(name, **kw)

        self.protection_state=False
        # Set instrumentss
        self.add_parameter('I_magnet', parameter_class=InstrumentParameter)
        self.I_magnet = Current_source_magnet_name
        self.add_parameter('I_heater', parameter_class=InstrumentParameter)
        self.I_heater = Current_source_heater_name

        # Specifications of One Axis AMI magnet with PCS_14768

        self.Max_Current = 5.0 # Amperes
        self.Field_to_Current = 5e-2 # Telsa/Ampere
                                    #   (Spec sheet says 0.500 kG/A = 50 mT/A)
        self.Max_Field = self.Max_Current*self.Field_to_Current  # Tesla
                            #(Spec sheet says 20 kG = 2.0 T)
        # self.Ramp_Rate = 0.05 # Ampere/Second
        self.Max_Ramp_Rate = 0.2 # Max ramp rate: 1.32 # Ampere/Second
        self.Init_Ramp_Rate = 0.025
        self.Charging_Voltage = 1.0 # Volt
        self.Inductance = 0.7 # Henry
        self.Persistent_Switch_Heater_Current = 21e-3 # Ampere (21 mA)
        self.Heater_mVolt_to_Current = 0.020/1e3 # A/mV (20 mA per 1000 mV)
        self.Persistent_Switch_Heater_Nominal_Resistance = 82 # Ohms
                    #(Measured at room temperature, thus normal conducing.)
        self.Magnet_Resistanc_in_Parallel_with_Switch = 35 # Ohms
                    #(Measured at room temperature, thus normal conducting.)

        self.add_parameter('Source_Current',
                           get_cmd=self.get_source_current,
                           set_cmd=self.set_source_current,
                           label='Source Current',
                           unit='A',
                           vals=vals.Numbers(min_value=0.,max_value=self.Max_Current),
                           docstring='Current supplied to the magnet')
        self.add_parameter('Ramp_Rate',
                           label='Ramp Rate',
                           unit='A/s',
                           initial_value=self.Init_Ramp_Rate,
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.,max_value=self.Max_Ramp_Rate),
                           parameter_class=ManualParameter,
                           docstring='Ramp Rate of the magnet current source')
        self.add_parameter('Persistent_Field',
                           label='Persistent Field',
                           unit='T',
                           initial_value=0.,
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.,max_value=self.Max_Field),
                           parameter_class=ManualParameter,
                           docstring='Ramp Rate of the magnet current source')
        # It would be great if the variable Persistent_Field could only be
        # set by the program, not by the used. It should only serve as a memory
        # of the previous persistent field.
        self.add_parameter('Switch_State',
                           get_cmd=self.get_switch_state,
                           set_cmd=self.set_switch_state,
                           label='Switch State',
                           unit='',
                           vals=vals.Enum('SuperConducting','NormalConducting'),
                           docstring='Indicating whether the persistent current\
                               switch is superconducting or normal conducting')
        self.add_parameter('Field',
                           get_cmd=self.get_field,
                           set_cmd=self.set_field,
                           label='Field',
                           unit='T',
                           initial_value=0.,
                           get_parser=float,
                           # initial_value=0,
                           vals=vals.Numbers(min_value=0.,max_value=self.Max_Field),
                           docstring='Magnetic field')
        self.protection_state=True

        '''
        You need to heat the persistent current switch to turn it from
        a superconductor into a normal conductor. When the persistent \
        current switch is superconducting there is a persistent current,
        when it is normal conducting there is no persistent current and
        you can controll the current with the current source.
        !! Thus it is important to heat the persistent current switch if
        you want to change the field !!
        !! Also important that when you want to switch off the persistent
        current that you provide the same current with the current source
        on the leads !! BEFORE !! you heat the persistent current switch !!
        '''



    def BtoI(self, magfield):
        MagCurrRatio = self.Field_to_Current # Tesla/Ampere
        I = magfield/MagCurrRatio
        return I

    def ItoB(self, current):
        MagCurrRatio = self.Field_to_Current # Tesla/Ampere
        B = current*MagCurrRatio
        return B

    def get_switch_state(self):
        heater_current = self.get_heater_current()
        if 1.05*self.Persistent_Switch_Heater_Current>heater_current\
                            >0.95*self.Persistent_Switch_Heater_Current:
            return 'NormalConducting'
        elif 0.05*self.Persistent_Switch_Heater_Current>heater_current\
                        >-0.05*self.Persistent_Switch_Heater_Current:
            return 'SuperConducting'
        else:
            raise ValueError('Switch is not in a well defined state!')

    def set_switch_state(self,desired_state):
        if desired_state == 'SuperConducting' and\
                self.get_switch_state() == 'SuperConducting':
            print('Already SuperConducting')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state() == 'NormalConducting':
            print('Already NormalConducting')
            return 'NormalConducting'
        elif desired_state == 'SuperConducting' and\
                self.get_switch_state() == 'NormalConducting':
            print('Ramping current down...')
            self.I_heater(0)
            print('Wait 2 minutes to cool the switch.')
            time.sleep(12) # 120
            print('Switch is now SuperConducting')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state() == 'SuperConducting':
            if self.I_magnet.measureR() > 1.:
                raise ValueError('Magnet leads not connected!')
            else:
                supplied_current = self.get_source_current()
                if self.Field()==None:
                    print('Sourcing current...')
                    self.I_heater(self.Persistent_Switch_Heater_Current\
                                  /self.Heater_mVolt_to_Current)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(3) # 30
                    print('Switch is now NormalConducting')
                    return 'NormalConducting'
                elif supplied_current<2e-3 and np.abs(self.Field())<1e-4:
                    print('Sourcing current...')
                    self.I_heater(self.Persistent_Switch_Heater_Current\
                                  /self.Heater_mVolt_to_Current)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(3) # 30
                    print('Switch is now NormalConducting')
                    return 'NormalConducting'
                elif not 0.98*supplied_current<self.BtoI(self.Field())\
                    <1.02*supplied_current:
                    raise ValueError('Current is not \
                                        according to the field value! Use \
                                        bring_source_to_field function to \
                                        bring it to the correct value.')
                else:
                    print('Sourcing current...')
                    self.I_heater(self.Persistent_Switch_Heater_Current\
                                  /self.Heater_mVolt_to_Current)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(3) # 30
                    print('Switch is now NormalConducting')
                    return 'NormalConducting'
        else:
            return 'Input SuperConducting or NormalConducting as desired state.'

    def get_source_current(self):
        print('Shit be workin yo')
        return self.I_magnet.measurei()

    def set_source_current(self,current):
        self.I_magnet.seti(current)
        return 'Current set to '+str(current)+' A'


    def get_heater_current(self):
        return self.Heater_mVolt_to_Current*self.I_heater()

    def measure_field(self):
        if self.I_magnet is not None:
            I = self.get_source_current()
            B = self.ItoB(I)
            return B
        else:
            print('no I_magnet')

    def set_field(self,field):
        if not self.protection_state:
            return 0.
        if self.Switch_State() == 'SuperConducting':
            raise ValueError('Switch is SuperConducting. Can not change the field.')
        elif self.Switch_State() =='NormalConducting':
            if self.I_magnet.measureR()>1:
                raise ValueError('Magnet leads are not connected \
                                 or manget quenched!')
            else:
                self.step_magfield_to_value(field)
                self.Persistent_Field(field)
                return 'Field at ' +str(field)+' T'

    def get_field(self):
        if self.Switch_State()=='SuperConducting':
            return self.Persistent_Field()
        else:
            meas_field = self.measure_field()
            return meas_field


    def disconnect_source(self):
        if self.Switch_State() == 'SuperConducting':
            self.step_magfield_to_value(0)
        else:
            raise ValueError('Switch is not superconducting!')

    def bring_source_to_field(self):
        if not self.Switch_State() == 'SuperConducting':
            raise ValueError('Switch is not superconducting!')
        if self.I_magnet.measureR() > 1.:
            raise ValueError('Magnet leads not connected!')
        target_field = self.Persistent_Field()
        self.step_magfield_to_value(target_field)


    def step_magfield_to_value(self, field):
        MagCurrRatio = self.Field_to_Current # Tesla/Ampere
        Ramp_rate_I = self.Ramp_Rate()
        step_time = 0.01 # in seconds
        current_step = Ramp_rate_I  * step_time
        I_now = self.get_source_current()
        current_target = self.BtoI(field)
        if current_target >= I_now:
            current_step *= +1
        if current_target < I_now:
            current_step *= -1
        num_steps = int(1.*(current_target-I_now)/(current_step))
        sweep_time = step_time*num_steps
        print('Sweep time is '+str(np.abs(sweep_time))+' seconds')

        for tt in range(num_steps):

            time.sleep(step_time)
            self.I_magnet.seti(I_now)
            I_now += current_step

        if self.I_magnet.measureR() > 1:
            self.I_magnet.seti(0)
            raise ValueError('Switch is not in a well defined state!')

        self.I_magnet.seti(self.BtoI(field))



















































