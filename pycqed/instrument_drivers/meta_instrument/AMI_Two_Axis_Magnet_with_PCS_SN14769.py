# American Magnetics, Inc. (AMI) Two Axis magnet with two PCS, SN14769

import time
import logging
import numpy as np
# from scipy.optimize import brent
# from math import gcd
# from qcodes import Instrument
from qcodes.utils import validators as vals
# from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis import analysis_toolbox as atools
# from pycqed.utilities.general import add_suffix_to_dict_keys

from pycqed.measurement import detector_functions as det
# from pycqed.measurement import composite_detector_functions as cdet
# from pycqed.measurement import mc_parameter_wrapper as pw

from pycqed.measurement import sweep_functions as swf
# from pycqed.measurement import awg_sweep_functions as awg_swf
# from pycqed.analysis import measurement_analysis as ma
# from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_5014
# from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_UHFQC
# from pycqed.measurement.calibration_toolbox import mixer_skewness_calibration_5014
# from pycqed.measurement.optimization import nelder_mead
from pycqed.analysis import analysis_toolbox as a_tools
# import pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts as sq
import logging
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter



class AMI_Two_Axis_Magnet_PCS_SN14769(Instrument):
    '''
    Instrument used for translating Fields into current settings
    and controlling the persistent current switch.

    Initialization when the previous measurement did not have the magnet is
    a bit awkward. The driver checks the last measurement for the value and if
    this does not exists it fails. To do the first initialization it is necessary
    to start everything up while having the 'get_field' function return 0 always.
    Make a two fake folders with the name
    'Switch_Z_is_changed_to_SuperConducting_state' and
    'Switch_Y_is_changed_to_SuperConducting_state'.
    Then exit and reinitialize with the
    'get_field' function returning what it is supposed to.

    Coordinate system: Righthanded, Z is the strong axis, Y the weak.
    Idea is that 'field' and 'angle' are the two variables that are controlled
    by the user. The two underlying variables are field_z and field_y, with the
    corresponding currents and switch states.

    '''

    def __init__(self, name,
                 Current_source_magnet_Z_name,
                 Current_source_heater_Z_name,
                 Current_source_magnet_Y_name,
                 Current_source_heater_Y_name,
                 MC_inst,**kw): # IVVI.dac1
        super().__init__(name, **kw)

        self.protection_state=False
        # Set instrumentss
        self.add_parameter('i_magnet_z', parameter_class=InstrumentParameter)
        self.i_magnet_z = Current_source_magnet_Z_name
        self.add_parameter('i_heater_z', parameter_class=InstrumentParameter)
        self.i_heater_z = Current_source_heater_Z_name
        self.add_parameter('i_magnet_y', parameter_class=InstrumentParameter)
        self.i_magnet_y = Current_source_magnet_Y_name
        self.add_parameter('i_heater_y', parameter_class=InstrumentParameter)
        self.i_heater_y = Current_source_heater_Y_name
        self.MC = MC_inst
        # Specifications of One Axis AMI magnet with PCS_14768

        self.max_current_z = 10.0 # 5.0 # Amperes
        self.max_current_y = 10.0 # Amperes
        self.field_to_current_z = 0.0496 # Telsa/Ampere
                                    #   (Spec sheet says 0.496 kG/A = 49.6 mT/A)
        self.field_to_current_y = 0.0132 # Telsa/Ampere
                                    #   (Spec sheet says 0.132 kG/A = 13.2 mT/A)
        self.max_field_z = self.max_current_z*self.field_to_current_z  # Tesla
                            #(Spec sheet says 20 kG = 2.0 T)
        self.max_field_y = self.max_current_y*self.field_to_current_y  # Tesla
                            #(Spec sheet says 5 kG = 0.5 T)
        # self.Ramp_Rate = 0.05 # Ampere/Second
        self.step_grid_points_mag = 1e-3 # zig-zag step size in mT
        self.max_ramp_rate_z = 0.1 # Max ramp rate: 0.677 # Ampere/Second
        self.max_ramp_rate_y = 0.1 # Max ramp rate: 0.054 # Ampere/Second
        self.init_ramp_rate_z = 0.025
        self.init_ramp_rate_y = 0.1
        self.charging_voltage_z = 0.5 # Volt
        self.charging_voltage_y = 0.1 # Volt
        self.inductance_z = 0.7 # Henry
        self.inductance_y = 1.8 # Henry
        self.persistent_switch_heater_current_z = 19e-3 # Ampere (19 mA)
        self.persistent_switch_heater_current_y = 19.7e-3 # Ampere (19.7 mA)
        self.heater_mvolt_to_current_z = 0.020*1e-3 # A/mV (20 mA per 1000 mV)
        self.heater_mvolt_to_current_y = 0.020*1e-3 # A/mV (20 mA per 1000 mV)
        self.persistent_switch_heater_nominal_resistance_z = 78 # Ohms
                    #(Measured at room temperature, thus normal conducing.)
        self.persistent_switch_heater_nominal_resistance_y = 81 # Ohms
                    #(Measured at room temperature, thus normal conducing.)
        self.magnet_resistanc_in_parallel_with_switch_z = 36 # Ohms
                    #(Measured at room temperature, thus normal conducting.)
        self.magnet_resistanc_in_parallel_with_switch_y = 37 # Ohms
                    #(Measured at room temperature, thus normal conducting.)

        self.add_parameter('source_current_z',
                           get_cmd=self.get_source_current_z,
                           set_cmd=self.set_source_current_z,
                           label='Source Current Z',
                           unit='A',
                           vals=vals.Numbers(min_value=0.,max_value=self.max_current_z),
                           docstring='Current supplied to the Z-axis of the magnet')
        self.add_parameter('source_current_y',
                           get_cmd=self.get_source_current_y,
                           set_cmd=self.set_source_current_y,
                           label='Source Current Y',
                           unit='A',
                           vals=vals.Numbers(min_value=0.,max_value=self.max_current_y),
                           docstring='Current supplied to the Y-axis of the magnet')
        #ramp rate should not be a parameter
        self.add_parameter('ramp_rate_z',
                           label='Ramp Rate Z',
                           unit='A/s',
                           initial_value=self.init_ramp_rate_z,
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.,max_value=self.max_ramp_rate_z),
                           parameter_class=ManualParameter,
                           docstring='Ramp Rate of the Z-axis magnet current source')
        self.add_parameter('ramp_rate_y',
                           label='Ramp Rate Y',
                           unit='A/s',
                           initial_value=self.init_ramp_rate_y,
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.,max_value=self.max_ramp_rate_y),
                           parameter_class=ManualParameter,
                           docstring='Ramp Rate of the Y-axis magnet current source')
        self.add_parameter('field',
                           get_cmd=self.get_field,
                           set_cmd=self.set_field,
                           label='Persistent Field',
                           unit='T',
                           # vals=vals.Numbers(min_value=0.,max_value=min(self.max_field_y,self.max_field_z)),
                           vals=vals.Numbers(min_value=0.,max_value=max(self.max_field_y,self.max_field_z)),
                           docstring='Persistent absolute magnetic field')
        self.add_parameter('angle',
                           get_cmd=self.get_angle,
                           set_cmd=self.set_angle,
                           label='Field angle',
                           unit='deg',
                           vals=vals.Numbers(min_value=-180.,max_value=180.),
                           docstring='Angle of the field wrt. Z-axis')
        self.add_parameter('field_z',
                           get_cmd=self.get_field_z,
                           set_cmd=self.set_field_z,
                           label='Z-Field',
                           unit='T',
                           vals=vals.Numbers(min_value=0.,max_value=self.max_field_z),
                           docstring='Persistent Z-magnetic field')
        self.add_parameter('field_y',
                           get_cmd=self.get_field_y,
                           set_cmd=self.set_field_y,
                           label='Y-Field',
                           unit='T',
                           vals=vals.Numbers(min_value=-self.max_field_y,max_value=self.max_field_y),
                           docstring='Persistent Y-magnetic field')
        # It would be great if the variable field could only be
        # set by the program, not by the user. It should only serve as a memory
        # of the previous persistent field.
        self.add_parameter('switch_state_z',
                           get_cmd=self.get_switch_state_z,
                           set_cmd=self.set_switch_state_z,
                           label='Switch State Z',
                           unit='',
                           vals=vals.Enum('SuperConducting','NormalConducting'),
                           docstring='Indicating whether the Z-Axis persistent current\
                               switch is superconducting or normal conducting')
        self.add_parameter('switch_state_y',
                           get_cmd=self.get_switch_state_y,
                           set_cmd=self.set_switch_state_y,
                           label='Switch State Y',
                           unit='',
                           vals=vals.Enum('SuperConducting','NormalConducting'),
                           docstring='Indicating whether the Y-Axis persistent current\
                               switch is superconducting or normal conducting')

        self.protection_state=True

        self.get_all()

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



    def get_all(self):
        self.get_source_current_z()
        self.switch_state_z()
        self.get_source_current_y()
        self.switch_state_y()
        self.get_field()
        self.get_angle()
        self.get_field_z()
        self.get_field_y()
        return


    def get_source_current_z(self):
        return self.i_magnet_z.measurei()

    def get_source_current_y(self):
        return self.i_magnet_y.measurei()

    def set_source_current_z(self,current):
        self.i_magnet_z.seti(current)
        return 'Z-current set to '+str(current)+' A'

    def set_source_current_y(self,current):
        self.i_magnet_y.seti(current)
        return 'Y-current set to '+str(current)+' A'

    def get_heater_current_z(self):
        return self.heater_mvolt_to_current_z*self.i_heater_z()

    def get_heater_current_y(self):
        return self.heater_mvolt_to_current_y*self.i_heater_y()


    def get_switch_state_z(self):
        heater_current = self.get_heater_current_z()
        if 1.05*self.persistent_switch_heater_current_z>heater_current\
                            >0.95*self.persistent_switch_heater_current_z:
            return 'NormalConducting'
        elif 0.05*self.persistent_switch_heater_current_z>heater_current\
                        >-0.05*self.persistent_switch_heater_current_z:
            return 'SuperConducting'
        else:
            raise ValueError('Switch is not in a well defined state!')

    def get_switch_state_y(self):
        heater_current = self.get_heater_current_y()
        if 1.05*self.persistent_switch_heater_current_y>heater_current\
                            >0.95*self.persistent_switch_heater_current_y:
            return 'NormalConducting'
        elif 0.05*self.persistent_switch_heater_current_y>heater_current\
                        >-0.05*self.persistent_switch_heater_current_y:
            return 'SuperConducting'
        else:
            raise ValueError('Switch is not in a well defined state!')

    def set_switch_state_z(self,desired_state):
        if desired_state == 'SuperConducting' and\
                self.get_switch_state_z() == 'SuperConducting':
            print('Already SuperConducting')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state_z() == 'NormalConducting':
            print('Already NormalConducting')
            return 'NormalConducting'
        elif desired_state == 'SuperConducting' and\
                self.get_switch_state_z() == 'NormalConducting':
            print('Ramping current down...')
            self.i_heater_z(0)
            print('Wait 2 minutes to cool the switch.')
            time.sleep(120) # 120
            print('Switch is now SuperConducting')
            self.fake_folder(folder_name='Switch_Z_is_changed_to_SuperConducting_state')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state_z() == 'SuperConducting':
            if self.i_magnet_z.measureR() > 1.:
                raise ValueError('Magnet leads not connected!')
            else:
                supplied_current_z = self.get_source_current_z()
                if self.get_field_z()==None:
                    print('Sourcing current...')
                    self.i_heater_z(self.persistent_switch_heater_current_z\
                                  /self.heater_mvolt_to_current_z)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_Z_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif supplied_current_z<2e-3 and np.abs(self.get_field_z())<1e-4:
                    print('Sourcing current...')
                    self.i_heater_z(self.persistent_switch_heater_current_z\
                                  /self.heater_mvolt_to_current_z)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_Z_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif not 0.98*supplied_current_z<self.BtoI_z(self.get_field_z())\
                    <1.02*supplied_current_z:
                    raise ValueError('Current is not \
                                        according to the Z-field value! Use \
                                        bring_source_to_field function to \
                                        bring it to the correct value.')
                else:
                    print('Sourcing current...')
                    self.i_heater_z(self.persistent_switch_heater_current_z\
                                  /self.heater_mvolt_to_current_z)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_Z_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
        else:
            return 'Input SuperConducting or NormalConducting as desired state.'


    def set_switch_state_y(self,desired_state):
        if desired_state == 'SuperConducting' and\
                self.get_switch_state_y() == 'SuperConducting':
            print('Already SuperConducting')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state_y() == 'NormalConducting':
            print('Already NormalConducting')
            return 'NormalConducting'
        elif desired_state == 'SuperConducting' and\
                self.get_switch_state_y() == 'NormalConducting':
            print('Ramping current down...')
            self.i_heater_y(0)
            print('Wait 2 minutes to cool the switch.')
            time.sleep(120) # 120
            print('Switch is now SuperConducting')
            self.fake_folder(folder_name='Switch_Y_is_changed_to_SuperConducting_state')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state_y() == 'SuperConducting':
            if self.i_magnet_y.measureR() > 1.:
                raise ValueError('Magnet leads not connected!')
            else:
                supplied_current_y = self.get_source_current_y()
                if self.get_field_y()==None:
                    print('Sourcing current...')
                    self.i_heater_y(self.persistent_switch_heater_current_y\
                                  /self.heater_mvolt_to_current_y)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_Y_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif supplied_current_y<2e-3 and np.abs(self.get_field_y())<1e-4:
                    print('Sourcing current...')
                    self.i_heater_y(self.persistent_switch_heater_current_y\
                                  /self.heater_mvolt_to_current_y)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_Y_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif not 0.98*supplied_current_y<self.BtoI_y(self.get_field_y())\
                    <1.02*supplied_current_y:
                    raise ValueError('Current is not \
                                        according to the Y-field value! Use \
                                        bring_source_to_field function to \
                                        bring it to the correct value.')
                else:
                    print('Sourcing current...')
                    self.i_heater_y(self.persistent_switch_heater_current_y\
                                  /self.heater_mvolt_to_current_y)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_Y_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
        else:
            return 'Input SuperConducting or NormalConducting as desired state.'

    def set_field(self,field):
        if not self.protection_state:
            return 0.
        angle = self.angle()
        desired_z_field_total = field*np.cos(angle*2.*np.pi/360)
        desired_y_field_total = field*np.sin(angle*2.*np.pi/360)

        current_field = self.field()
        current_z_field = self.field_z()
        current_y_field = self.field_y()
        step_mag_z = self.step_grid_points_mag*np.cos(angle*2.*np.pi/360)
        step_mag_y = self.step_grid_points_mag*np.sin(angle*2.*np.pi/360)
        if field>=current_field:
            step_z = +1*step_mag_z
            step_y = +1*step_mag_y
        if field<current_field :
            step_z = -1*step_mag_z
            step_y = -1*step_mag_y
        num_steps = int(np.ceil(np.abs(current_field-field)/self.step_grid_points_mag))
        for tt in range(num_steps):
            if tt == num_steps-1:
                current_z_field = desired_z_field_total
                current_y_field = desired_y_field_total
            else:
                current_z_field += step_z
                current_y_field += step_y

            # if current_y_field == 0. and self.switch_state_z() == 'SuperConducting':
            #     raise ValueError('Switch_Z is SuperConducting. Can not change the field.')
            # elif current_z_field == 0. and self.switch_state_y() == 'SuperConducting':
            #     raise ValueError('Switch_Y is SuperConducting. Can not change the field.')
            if self.switch_state_z() == 'SuperConducting' or self.switch_state_y() == 'SuperConducting':
                raise ValueError('Switch_Y and/or Switch_Z are SuperConducting. Can not change the field.')
            elif self.switch_state_z() =='NormalConducting' and self.switch_state_y() =='NormalConducting':
                if self.i_magnet_z.measurei()<1e-3:
                    self.step_z_magfield_to_value(current_z_field)
                    # self.field(field)
                    # field_folder_name = 'Changed_Z_field_to_' + str(current_z_field) + '_T'
                    # self.fake_folder(folder_name=field_folder_name)
                    # return 'field at ' +str(field)+' T'
                elif self.i_magnet_z.measureR()>1:
                    raise ValueError('Magnet Z leads are not connected \
                                     or manget quenched!')
                else:
                    self.step_z_magfield_to_value(current_z_field)
                    # self.field(field)
                    # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                    # self.fake_folder(folder_name=field_folder_name)
                    # return 'field at ' +str(field)+' T'
                if np.abs(self.i_magnet_y.measurei())<1e-3:
                    self.step_y_magfield_to_value(current_y_field)
                    # self.field(field)
                    # field_folder_name = 'Changed_y_field_to_' + str(current_y_field) + '_T'
                    # self.fake_folder(folder_name=field_folder_name)
                    # return 'field at ' +str(field)+' T'
                elif self.i_magnet_y.measureR()>1:
                    raise ValueError('Magnet Y leads are not connected \
                                     or manget quenched!')
                else:
                    self.step_y_magfield_to_value(current_y_field)
                    # self.field(field)
                    # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                    # self.fake_folder(folder_name=field_folder_name)
                    # return 'field at ' +str(field)+' T'
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                print('Field at ' +str(self.field())+' T, angle at '+str(self.angle())+' deg.')
        return 0.


    def get_field(self):
        # return 0.0 # Only add this line when doing the first initialization!
        if self.switch_state_z()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_Z_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field_z':'Magnet.field_z'}
            numeric_params = ['field_z']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            field_val_z = data['field_z'][0]
        else: ## Normal conducting
            field_val_z = self.measure_field_z()

        if self.switch_state_y()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_Y_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field_y':'Magnet.field_y'}
            numeric_params = ['field_y']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            field_val_y = data['field_y'][0]
        else: ## Normal conducting
            field_val_y = self.measure_field_y()
        return np.sqrt(field_val_z**2+field_val_y**2)

    def set_angle(self,angle):
        if not self.protection_state:
            return 0.
        field = self.field()
        desired_z_field = field*np.cos(angle*2.*np.pi/360)
        desired_y_field = field*np.sin(angle*2.*np.pi/360)
        if self.switch_state_z() == 'SuperConducting' or self.switch_state_y() == 'SuperConducting':
            raise ValueError('Switch is SuperConducting. Can not change the field.')
        elif self.switch_state_z() =='NormalConducting' and self.switch_state_y() =='NormalConducting':
            if self.i_magnet_z.measurei()<1e-3:
                self.field_z(desired_z_field)
                # self.field(field)
                # field_folder_name = 'Changed_Z_field_to_' + str(desired_z_field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                # return 'field at ' +str(field)+' T'
            elif self.i_magnet_z.measureR()>1:
                raise ValueError('Magnet Z leads are not connected \
                                 or manget quenched!')
            else:
                self.field_z(desired_z_field)
                # self.field(field)
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                # return 'field at ' +str(field)+' T'
            if np.abs(self.i_magnet_y.measurei())<1e-3:
                self.field_y(desired_y_field)
                # self.field(field)
                # field_folder_name = 'Changed_y_field_to_' + str(desired_y_field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                # return 'field at ' +str(field)+' T'
            elif self.i_magnet_y.measureR()>1:
                raise ValueError('Magnet Y leads are not connected \
                                 or manget quenched!')
            else:
                self.field_y(desired_y_field)
                # self.field(field)
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                # return 'field at ' +str(field)+' T'
            # field_folder_name = 'Changed_angle_to_' + str(angle) + '_deg'
            # self.fake_folder(folder_name=field_folder_name)
            return 'Field at ' +str(field)+' T, angle at '+str(angle)+' deg.'

    def get_angle(self):
        # return 0.0 # Only add this line when doing the first initialization!
        if self.switch_state_z()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_Z_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field_z':'Magnet.field_z'}
            numeric_params = ['field_z']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            field_val_z = data['field_z'][0]
        else: ## Normal conducting
            field_val_z = self.measure_field_z()

        if self.switch_state_y()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_Y_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field_y':'Magnet.field_y'}
            numeric_params = ['field_y']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            field_val_y = data['field_y'][0]
        else: ## Normal conducting
            field_val_y = self.measure_field_y()
        field = np.sqrt(field_val_z**2+field_val_y**2)
        if field==0:
            return 0
        elif field_val_z>=0:
            return np.arcsin(field_val_y/field)*360./(2*np.pi)
        else:
            return -np.arcsin(field_val_y/field)*360./(2*np.pi)

    def set_field_z(self,field):
        if not self.protection_state:
            return 0.
        if self.switch_state_z() == 'SuperConducting':
            raise ValueError('Z-Switch is SuperConducting. Can not change the field.')
        elif self.switch_state_z() =='NormalConducting':
            if self.i_magnet_z.measurei()<1e-3:
                self.step_z_magfield_to_value(field)
                # self.field(field)
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                return 'Z-field at ' +str(field)+' T'
            elif self.i_magnet_z.measureR()>1:
                raise ValueError('Magnet leads are not connected \
                                 or manget quenched!')
            else:
                self.step_z_magfield_to_value(field)
                # self.field(field)
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                return 'Z-field at ' +str(field)+' T'

    def set_field_y(self,field):
        if not self.protection_state:
            return 0.
        if self.switch_state_y() == 'SuperConducting':
            raise ValueError('Y-Switch is SuperConducting. Can not change the field.')
        elif self.switch_state_y() =='NormalConducting':
            if np.abs(self.i_magnet_y.measurei())<1e-3:
                self.step_y_magfield_to_value(field)
                # self.field(field)
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                return 'Y-field at ' +str(field)+' T'
            elif self.i_magnet_y.measureR()>1:
                raise ValueError('Magnet leads are not connected \
                                 or manget quenched!')
            else:
                self.step_y_magfield_to_value(field)
                # self.field(field)
                # field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                # self.fake_folder(folder_name=field_folder_name)
                return 'Y-field at ' +str(field)+' T'


    def get_field_z(self):
        # return 0.0 # Only add this line when doing the first initialization!
        if self.switch_state_z()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_Z_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field':'Magnet.field_z'}
            numeric_params = ['field']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            return data['field'][0]
        else: ## Normal conducting
            meas_field = self.measure_field_z()
            return meas_field

    def get_field_y(self):
        # return 0.0 # Only add this line when doing the first initialization!
        if self.switch_state_y()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_Y_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field':'Magnet.field_y'}
            numeric_params = ['field']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            return data['field'][0]
        else: ## Normal conducting
            meas_field = self.measure_field_y()
            return meas_field

    def step_z_magfield_to_value(self, field):
        MagCurrRatio = self.field_to_current_z # Tesla/Ampere
        Ramp_rate_I = self.ramp_rate_z()
        step_time = 0.01 # in seconds
        current_step = Ramp_rate_I  * step_time
        I_now = self.get_source_current_z()
        current_target = self.BtoI_z(field)
        if current_target >= I_now:
            current_step *= +1
        if current_target < I_now:
            current_step *= -1
        num_steps = int(1.*(current_target-I_now)/(current_step))
        sweep_time = step_time*num_steps
        print('Sweep time is '+str(np.abs(sweep_time))+' seconds')

        for tt in range(num_steps):

            time.sleep(step_time)
            self.i_magnet_z.seti(I_now)
            I_now += current_step

        if self.i_magnet_z.measureR() > 1:
            if not self.field_z()<1e-3:
                self.i_magnet_z.seti(0)
                raise ValueError('Switch is not in a well defined state!')

        self.i_magnet_z.seti(self.BtoI_z(field))
        self.source_current_z()

    def step_y_magfield_to_value(self, field):
        MagCurrRatio = self.field_to_current_y # Tesla/Ampere
        Ramp_rate_I = self.ramp_rate_y()
        step_time = 0.05 # in seconds
        current_step = Ramp_rate_I  * step_time
        I_now = self.get_source_current_y()
        current_target = self.BtoI_y(field)
        if current_target >= I_now:
            current_step *= +1
        if current_target < I_now:
            current_step *= -1
        num_steps = int(1.*(current_target-I_now)/(current_step))
        sweep_time = step_time*num_steps
        print('Sweep time is '+str(np.abs(sweep_time)*3)+' seconds')

        for tt in range(num_steps):

            time.sleep(step_time)
            self.i_magnet_y.seti(I_now)
            I_now += current_step

        if self.i_magnet_y.measureR() > 1:
            if not self.field_y()<1e-3:
                self.i_magnet_y.seti(0)
                raise ValueError('Switch is not in a well defined state!')

        self.i_magnet_y.seti(self.BtoI_y(field))
        self.source_current_y()


    def disconnect_z_source(self):
        if self.switch_state_z() == 'SuperConducting':
            self.step_z_magfield_to_value(0)
            self.fake_folder(folder_name='Ramped_down_current_you_are_able_to_disconnect_the_Z_source_now')
        else:
            raise ValueError('Switch is not superconducting!')

    def disconnect_y_source(self):
        if self.switch_state_y() == 'SuperConducting':
            self.step_y_magfield_to_value(0)
            self.fake_folder(folder_name='Ramped_down_current_you_are_able_to_disconnect_the_Y_source_now')
        else:
            raise ValueError('Switch is not superconducting!')

    def bring_z_source_to_field(self):
        if not self.switch_state_z() == 'SuperConducting':
            print('Switch Z is normal conducting.')
            return
        if self.i_magnet_z.measureR() > 1.:
            raise ValueError('Magnet leads not connected!')
        target_field = self.field_z()
        self.step_z_magfield_to_value(target_field)
        self.fake_folder(folder_name='Ramped_Z_current_up_to_match_persistent_current')

    def bring_y_source_to_field(self):
        if not self.switch_state_y() == 'SuperConducting':
            print('Switch Y is normal conducting.')
            return
        if self.i_magnet_y.measureR() > 1.:
            raise ValueError('Magnet leads not connected!')
        target_field = self.field_y()
        self.step_y_magfield_to_value(target_field)
        self.fake_folder(folder_name='Ramped_Y_current_up_to_match_persistent_current')

    def measure_field_z(self):
        if self.i_magnet_z is not None:
            I = self.get_source_current_z()
            B = self.ItoB_z(I)
            return B
        else:
            print('no i_magnet_z')

    def measure_field_y(self):
        if self.i_magnet_y is not None:
            I = self.get_source_current_y()
            B = self.ItoB_y(I)
            return B
        else:
            print('no i_magnet_y')

    def BtoI_z(self, magfield):
        MagCurrRatio = self.field_to_current_z # Tesla/Ampere
        I = magfield/MagCurrRatio
        return I

    def BtoI_y(self, magfield):
        MagCurrRatio = self.field_to_current_y # Tesla/Ampere
        I = magfield/MagCurrRatio
        return I

    def ItoB_z(self, current):
        MagCurrRatio = self.field_to_current_z # Tesla/Ampere
        B = current*MagCurrRatio
        return B

    def ItoB_y(self, current):
        MagCurrRatio = self.field_to_current_y # Tesla/Ampere
        B = current*MagCurrRatio
        return B

    def fake_folder(self,folder_name):
        '''
        Give folder_name in the form of a string.
        Create a fake folder in the nanowire experiments folder with the desired folder name.
        This is usefull to use when magnetic fields change or when you start a new measurement cycle.
        Such that you can distinguish the different measurement sets.
        '''
        if isinstance(folder_name,str):
            sweep_pts = np.linspace(0, 10, 3)
            self.MC.set_sweep_function(swf.None_Sweep())
            self.MC.set_sweep_points(sweep_pts)
            self.MC.set_detector_function(det.Dummy_Detector_Soft())
            dat = self.MC.run(folder_name)
        else:
            raise ValueError('Please enter a string as the folder name!')









