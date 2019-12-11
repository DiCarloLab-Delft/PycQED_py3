# American Magnetics, Inc. (AMI) One Axis magnet with PCS_SN14768

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
import numpy as np
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter

class AMI_Magnet_PCS_SN14768(Instrument):
    '''
    Instrument used for translating Fields into current settings
    and controlling the persistent current switch.

    Initialization when the previous measurement did not have the magnet is
    a bit awkward. The driver checks the last measurement for the value and if
    this does not exists it fails. To do the first initialization it is necessary
    to start everything up while having the 'get_field' function return 0 always.
    Make a fake folder with the name
    'Switch_is_changed_to_SuperConducting_state'.
    Then exit and reinitialize with the
    'get_field' function returning what it is supposed to.

    '''

    def __init__(self, name,
                 Current_source_magnet_name,
                 Current_source_heater_name,MC_inst,**kw): # IVVI.dac1
        super().__init__(name, **kw)

        self.protection_state=False
        # Set instrumentss
        self.add_parameter('i_magnet', parameter_class=InstrumentParameter)
        self.i_magnet = Current_source_magnet_name
        self.add_parameter('i_heater', parameter_class=InstrumentParameter)
        self.i_heater = Current_source_heater_name
        self.MC = MC_inst
        # Specifications of One Axis AMI magnet with PCS_14768

        self.max_current = 5.0 # Amperes
        #### Dirty hack to get the z-axis of the new magnet running
        self.field_to_current = 0.0496 # Telsa/Ampere
                                    #   (Spec sheet says 0.500 kG/A = 50 mT/A)
        self.max_field = self.max_current*self.field_to_current  # Tesla
                            #(Spec sheet says 20 kG = 2.0 T)
        # self.Ramp_Rate = 0.05 # Ampere/Second
        self.max_ramp_rate = 0.2 # Max ramp rate: 1.32 # Ampere/Second
        self.init_ramp_rate = 0.025
        self.charging_voltage = 1.0 # Volt
        self.inductance = 0.7 # Henry
        self.persistent_switch_heater_current = 21e-3 # Ampere (21 mA)
        self.heater_mvolt_to_current = 0.020*1e-3 # A/mV (20 mA per 1000 mV)
        self.persistent_switch_heater_nominal_resistance = 82 # Ohms
                    #(Measured at room temperature, thus normal conducing.)
        self.magnet_resistanc_in_parallel_with_switch = 35 # Ohms
                    #(Measured at room temperature, thus normal conducting.)

        self.add_parameter('source_current',
                           get_cmd=self.get_source_current,
                           set_cmd=self.set_source_current,
                           label='Source Current',
                           unit='A',
                           vals=vals.Numbers(min_value=0.,max_value=self.max_current),
                           docstring='Current supplied to the magnet')
        #ramp rate should not be a parameter
        self.add_parameter('ramp_rate',
                           label='Ramp Rate',
                           unit='A/s',
                           initial_value=self.init_ramp_rate,
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.,max_value=self.max_ramp_rate),
                           parameter_class=ManualParameter,
                           docstring='Ramp Rate of the magnet current source')
        self.add_parameter('field',
                           get_cmd=self.get_field,
                           set_cmd=self.set_field,
                           label='Persistent Field',
                           unit='T',
                           vals=vals.Numbers(min_value=0.,max_value=self.max_field),
                           docstring='Persistent magnetic field')
        # It would be great if the variable field could only be
        # set by the program, not by the used. It should only serve as a memory
        # of the previous persistent field.
        self.add_parameter('switch_state',
                           get_cmd=self.get_switch_state,
                           set_cmd=self.set_switch_state,
                           label='Switch State',
                           unit='',
                           vals=vals.Enum('SuperConducting','NormalConducting'),
                           docstring='Indicating whether the persistent current\
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
        self.get_source_current()
        self.get_field()
        self.switch_state()
        return


    def get_source_current(self):
        return self.i_magnet.measurei()

    def set_source_current(self,current):
        self.i_magnet.seti(current)
        return 'Current set to '+str(current)+' A'

    def get_heater_current(self):
        return self.heater_mvolt_to_current*self.i_heater()


    def get_switch_state(self):
        heater_current = self.get_heater_current()
        if 1.05*self.persistent_switch_heater_current>heater_current\
                            >0.95*self.persistent_switch_heater_current:
            return 'NormalConducting'
        elif 0.05*self.persistent_switch_heater_current>heater_current\
                        >-0.05*self.persistent_switch_heater_current:
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
            self.i_heater(0)
            print('Wait 2 minutes to cool the switch.')
            time.sleep(120) # 120
            print('Switch is now SuperConducting')
            self.fake_folder(folder_name='Switch_is_changed_to_SuperConducting_state')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state() == 'SuperConducting':
            if self.i_magnet.measureR() > 1.:
                raise ValueError('Magnet leads not connected!')
            else:
                supplied_current = self.get_source_current()
                if self.field()==None:
                    print('Sourcing current...')
                    self.i_heater(self.persistent_switch_heater_current\
                                  /self.heater_mvolt_to_current)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif supplied_current<2e-3 and np.abs(self.field())<1e-4:
                    print('Sourcing current...')
                    self.i_heater(self.persistent_switch_heater_current\
                                  /self.heater_mvolt_to_current)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif not 0.98*supplied_current<self.BtoI(self.field())\
                    <1.02*supplied_current:
                    raise ValueError('Current is not \
                                        according to the field value! Use \
                                        bring_source_to_field function to \
                                        bring it to the correct value.')
                else:
                    print('Sourcing current...')
                    self.i_heater(self.persistent_switch_heater_current\
                                  /self.heater_mvolt_to_current)
                    print('Wait 30 seconds to heat up the switch.')
                    time.sleep(30) # 30
                    print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_is_changed_to_NormalConducting_state')
                    return 'NormalConducting'
        else:
            return 'Input SuperConducting or NormalConducting as desired state.'




    def set_field(self,field):
        if not self.protection_state:
            return 0.
        if self.switch_state() == 'SuperConducting':
            raise ValueError('Switch is SuperConducting. Can not change the field.')
        elif self.switch_state() =='NormalConducting':
            if self.i_magnet.measurei()==0:
                self.step_magfield_to_value(field)
                # self.field(field)
                field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                self.fake_folder(folder_name=field_folder_name)
                return 'field at ' +str(field)+' T'
            elif self.i_magnet.measureR()>1:
                raise ValueError('Magnet leads are not connected \
                                 or manget quenched!')
            else:
                self.step_magfield_to_value(field)
                # self.field(field)
                field_folder_name = 'Changed_field_to_' + str(field) + '_T'
                self.fake_folder(folder_name=field_folder_name)
                return 'field at ' +str(field)+' T'

    def get_field(self):
        return 0.0 # Only add this line when doing the first initialization!
        if self.switch_state()=='SuperConducting':
            ## get the persistent field from the HDF5 file

            timestamp = atools.latest_data(contains='Switch_is_changed_to_SuperConducting_state',
                                           return_timestamp=True)[0]
            params_dict = {'field':'Magnet.field'}
            numeric_params = ['field']
            data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
                                               numeric_params=numeric_params, filter_no_analysis=False)
            return data['field'][0]
        else: ## Normal conducting
            meas_field = self.measure_field()
            return meas_field





    def step_magfield_to_value(self, field):
        MagCurrRatio = self.field_to_current # Tesla/Ampere
        Ramp_rate_I = self.ramp_rate()
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
            self.i_magnet.seti(I_now)
            I_now += current_step

        if self.i_magnet.measureR() > 1:
            self.i_magnet.seti(0)
            raise ValueError('Switch is not in a well defined state!')

        self.i_magnet.seti(self.BtoI(field))
        self.source_current()

    def disconnect_source(self):
        if self.switch_state() == 'SuperConducting':
            self.step_magfield_to_value(0)
            self.fake_folder(folder_name='Ramped_down_current_you_are_able_to_disconnect_the_source_now')
        else:
            raise ValueError('Switch is not superconducting!')

    def bring_source_to_field(self):
        if not self.switch_state() == 'SuperConducting':
            raise ValueError('Switch is not superconducting!')
        if self.i_magnet.measureR() > 1.:
            raise ValueError('Magnet leads not connected!')
        target_field = self.field()
        self.step_magfield_to_value(target_field)
        self.fake_folder(folder_name='Ramped_current_up_to_match_persistent_current')


    def measure_field(self):
        if self.i_magnet is not None:
            I = self.get_source_current()
            B = self.ItoB(I)
            return B
        else:
            print('no i_magnet')

    def BtoI(self, magfield):
        MagCurrRatio = self.field_to_current # Tesla/Ampere
        I = magfield/MagCurrRatio
        return I

    def ItoB(self, current):
        MagCurrRatio = self.field_to_current # Tesla/Ampere
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



























































####################################################
### This is a working version that contains bugs ###
####################################################



# # American Magnetics, Inc. (AMI) One Axis magnet with PCS_SN14768

# import time
# import logging
# import numpy as np
# # from scipy.optimize import brent
# # from math import gcd
# # from qcodes import Instrument
# from qcodes.utils import validators as vals
# # from qcodes.instrument.parameter import ManualParameter

# # from pycqed.utilities.general import add_suffix_to_dict_keys

# # from pycqed.measurement import detector_functions as det
# # from pycqed.measurement import composite_detector_functions as cdet
# # from pycqed.measurement import mc_parameter_wrapper as pw

# # from pycqed.measurement import sweep_functions as swf
# # from pycqed.measurement import awg_sweep_functions as awg_swf
# # from pycqed.analysis import measurement_analysis as ma
# # from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_5014
# # from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_UHFQC
# # from pycqed.measurement.calibration_toolbox import mixer_skewness_calibration_5014
# # from pycqed.measurement.optimization import nelder_mead

# # import pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts as sq
# import logging
# import numpy as np
# from copy import deepcopy,copy

# import qcodes as qc
# from qcodes.instrument.base import Instrument
# from qcodes.utils import validators as vals
# from qcodes.instrument.parameter import ManualParameter
# from pycqed.instrument_drivers.pq_parameters import InstrumentParameter

# class AMI_Magnet_with_PCS_SN14768(Instrument):
#     '''
#     Instrument used for translating fields into current settings
#     and controlling the persistent current switch.

#     '''

#     def __init__(self, name,
#                  Current_source_magnet_name,
#                  Current_source_heater_name,**kw): # IVVI.dac1
#         super().__init__(name, **kw)

#         self.protection_state=False
#         # Set instrumentss
#         self.add_parameter('I_magnet', parameter_class=InstrumentParameter)
#         self.I_magnet = Current_source_magnet_name
#         self.add_parameter('I_heater', parameter_class=InstrumentParameter)
#         self.I_heater = Current_source_heater_name

#         # Specifications of One Axis AMI magnet with PCS_14768

#         self.Max_Current = 5.0 # Amperes
#         self.Field_to_Current = 5e-2 # Telsa/Ampere
#                                     #   (Spec sheet says 0.500 kG/A = 50 mT/A)
#         self.Max_Field = self.Max_Current*self.Field_to_Current  # Tesla
#                             #(Spec sheet says 20 kG = 2.0 T)
#         # self.Ramp_Rate = 0.05 # Ampere/Second
#         self.Max_Ramp_Rate = 0.2 # Max ramp rate: 1.32 # Ampere/Second
#         self.Init_Ramp_Rate = 0.025
#         self.Charging_Voltage = 1.0 # Volt
#         self.Inductance = 0.7 # Henry
#         self.Persistent_Switch_Heater_Current = 21e-3 # Ampere (21 mA)
#         self.Heater_mVolt_to_Current = 0.020/1e3 # A/mV (20 mA per 1000 mV)
#         self.Persistent_Switch_Heater_Nominal_Resistance = 82 # Ohms
#                     #(Measured at room temperature, thus normal conducing.)
#         self.Magnet_Resistanc_in_Parallel_with_Switch = 35 # Ohms
#                     #(Measured at room temperature, thus normal conducting.)

#         self.add_parameter('Source_Current',
#                            get_cmd=self.get_source_current,
#                            set_cmd=self.set_source_current,
#                            label='Source Current',
#                            unit='A',
#                            vals=vals.Numbers(min_value=0.,max_value=self.Max_Current),
#                            docstring='Current supplied to the magnet')
#         self.add_parameter('Ramp_Rate',
#                            label='Ramp Rate',
#                            unit='A/s',
#                            initial_value=self.Init_Ramp_Rate,
#                            get_parser=float,
#                            vals=vals.Numbers(min_value=0.,max_value=self.Max_Ramp_Rate),
#                            parameter_class=ManualParameter,
#                            docstring='Ramp Rate of the magnet current source')
#         self.add_parameter('Persistent_Field',
#                            label='Persistent Field',
#                            unit='T',
#                            initial_value=0.,
#                            get_parser=float,
#                            vals=vals.Numbers(min_value=0.,max_value=self.Max_Field),
#                            parameter_class=ManualParameter,
#                            docstring='Ramp Rate of the magnet current source')
#         # It would be great if the variable Persistent_Field could only be
#         # set by the program, not by the used. It should only serve as a memory
#         # of the previous persistent field.
#         self.add_parameter('Switch_State',
#                            get_cmd=self.get_switch_state,
#                            set_cmd=self.set_switch_state,
#                            label='Switch State',
#                            unit='',
#                            vals=vals.Enum('SuperConducting','NormalConducting'),
#                            docstring='Indicating whether the persistent current\
#                                switch is superconducting or normal conducting')
#         self.add_parameter('Field',
#                            get_cmd=self.get_field,
#                            set_cmd=self.set_field,
#                            label='Field',
#                            unit='T',
#                            initial_value=0.,
#                            get_parser=float,
#                            # initial_value=0,
#                            vals=vals.Numbers(min_value=0.,max_value=self.Max_Field),
#                            docstring='Magnetic field')
#         self.protection_state=True

#         '''
#         You need to heat the persistent current switch to turn it from
#         a superconductor into a normal conductor. When the persistent \
#         current switch is superconducting there is a persistent current,
#         when it is normal conducting there is no persistent current and
#         you can controll the current with the current source.
#         !! Thus it is important to heat the persistent current switch if
#         you want to change the field !!
#         !! Also important that when you want to switch off the persistent
#         current that you provide the same current with the current source
#         on the leads !! BEFORE !! you heat the persistent current switch !!
#         '''



#     def BtoI(self, magfield):
#         MagCurrRatio = self.Field_to_Current # Tesla/Ampere
#         I = magfield/MagCurrRatio
#         return I

#     def ItoB(self, current):
#         MagCurrRatio = self.Field_to_Current # Tesla/Ampere
#         B = current*MagCurrRatio
#         return B

#     def get_switch_state(self):
#         heater_current = self.get_heater_current()
#         if 1.05*self.Persistent_Switch_Heater_Current>heater_current\
#                             >0.95*self.Persistent_Switch_Heater_Current:
#             return 'NormalConducting'
#         elif 0.05*self.Persistent_Switch_Heater_Current>heater_current\
#                         >-0.05*self.Persistent_Switch_Heater_Current:
#             return 'SuperConducting'
#         else:
#             raise ValueError('Switch is not in a well defined state!')

#     def set_switch_state(self,desired_state):
#         if desired_state == 'SuperConducting' and\
#                 self.get_switch_state() == 'SuperConducting':
#             print('Already SuperConducting')
#             return 'SuperConducting'
#         elif desired_state == 'NormalConducting' and\
#                 self.get_switch_state() == 'NormalConducting':
#             print('Already NormalConducting')
#             return 'NormalConducting'
#         elif desired_state == 'SuperConducting' and\
#                 self.get_switch_state() == 'NormalConducting':
#             print('Ramping current down...')
#             self.I_heater(0)
#             print('Wait 2 minutes to cool the switch.')
#             time.sleep(120) # 120
#             print('Switch is now SuperConducting')
#             return 'SuperConducting'
#         elif desired_state == 'NormalConducting' and\
#                 self.get_switch_state() == 'SuperConducting':
#             if self.I_magnet.measureR() > 1.:
#                 raise ValueError('Magnet leads not connected!')
#             else:
#                 supplied_current = self.get_source_current()
#                 if self.Field()==None:
#                     print('Sourcing current...')
#                     self.I_heater(self.Persistent_Switch_Heater_Current\
#                                   /self.Heater_mVolt_to_Current)
#                     print('Wait 30 seconds to heat up the switch.')
#                     time.sleep(30) # 30
#                     print('Switch is now NormalConducting')
#                     return 'NormalConducting'
#                 elif supplied_current<2e-3 and np.abs(self.Field())<1e-4:
#                     print('Sourcing current...')
#                     self.I_heater(self.Persistent_Switch_Heater_Current\
#                                   /self.Heater_mVolt_to_Current)
#                     print('Wait 30 seconds to heat up the switch.')
#                     time.sleep(30) # 30
#                     print('Switch is now NormalConducting')
#                     return 'NormalConducting'
#                 elif not 0.98*supplied_current<self.BtoI(self.Field())\
#                     <1.02*supplied_current:
#                     raise ValueError('Current is not \
#                                         according to the field value! Use \
#                                         bring_source_to_field function to \
#                                         bring it to the correct value.')
#                 else:
#                     print('Sourcing current...')
#                     self.I_heater(self.Persistent_Switch_Heater_Current\
#                                   /self.Heater_mVolt_to_Current)
#                     print('Wait 30 seconds to heat up the switch.')
#                     time.sleep(30) # 30
#                     print('Switch is now NormalConducting')
#                     return 'NormalConducting'
#         else:
#             return 'Input SuperConducting or NormalConducting as desired state.'

#     def get_source_current(self):
#         print('Shit be workin yo')
#         return self.I_magnet.measurei()

#     def set_source_current(self,current):
#         self.I_magnet.seti(current)
#         return 'Current set to '+str(current)+' A'


#     def get_heater_current(self):
#         return self.Heater_mVolt_to_Current*self.I_heater()

#     def measure_field(self):
#         if self.I_magnet is not None:
#             I = self.get_source_current()
#             B = self.ItoB(I)
#             return B
#         else:
#             print('no I_magnet')

#     def set_field(self,field):
#         if not self.protection_state:
#             return 0.
#         if self.Switch_State() == 'SuperConducting':
#             raise ValueError('Switch is SuperConducting. Can not change the field.')
#         elif self.Switch_State() =='NormalConducting':
#             if self.I_magnet.measureR()>1:
#                 raise ValueError('Magnet leads are not connected \
#                                  or manget quenched!')
#             else:
#                 self.step_magfield_to_value(field)
#                 self.Persistent_Field(field)
#                 return 'Field at ' +str(field)+' T'

#     def get_field(self):
#         if self.Switch_State()=='SuperConducting':
#             return self.Persistent_Field()
#         else:
#             meas_field = self.measure_field()
#             return meas_field


#     def disconnect_source(self):
#         if self.Switch_State() == 'SuperConducting':
#             self.step_magfield_to_value(0)
#         else:
#             raise ValueError('Switch is not superconducting!')

#     def bring_source_to_field(self):
#         if not self.Switch_State() == 'SuperConducting':
#             raise ValueError('Switch is not superconducting!')
#         if self.I_magnet.measureR() > 1.:
#             raise ValueError('Magnet leads not connected!')
#         target_field = self.Persistent_Field()
#         self.step_magfield_to_value(target_field)


#     def step_magfield_to_value(self, field):
#         MagCurrRatio = self.Field_to_Current # Tesla/Ampere
#         Ramp_rate_I = self.Ramp_Rate()
#         step_time = 0.01 # in seconds
#         current_step = Ramp_rate_I  * step_time
#         I_now = self.get_source_current()
#         current_target = self.BtoI(field)
#         if current_target >= I_now:
#             current_step *= +1
#         if current_target < I_now:
#             current_step *= -1
#         num_steps = int(1.*(current_target-I_now)/(current_step))
#         sweep_time = step_time*num_steps
#         print('Sweep time is '+str(np.abs(sweep_time))+' seconds')

#         for tt in range(num_steps):

#             time.sleep(step_time)
#             self.I_magnet.seti(I_now)
#             I_now += current_step

#         if self.I_magnet.measureR() > 1:
#             self.I_magnet.seti(0)
#             raise ValueError('Switch is not in a well defined state!')

#         self.I_magnet.seti(self.BtoI(field))


