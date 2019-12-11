
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

from pycqed.instrument_drivers.meta_instrument.Persistent_Coil import Persistent_Coil

class Dummy_Coil(Instrument):

    def __init__(self, name,
                 field_to_current,
                 min_field,
                 max_field,
                 ramp_rate,
                 switch_heater_current,
                 MC_inst,**kw):
        super().__init__(name, **kw)

        #Needs from the 3-axis magnet
        #method switch_state()
        #method bring source to field
        #methods set/get_field()
        #methods compensate for drift

        self.protection_state=False
        # Set current source
        self.add_parameter('i_magnet', parameter_class=InstrumentParameter)
        self.MC = MC_inst
        self.name = name
        self.add_parameter('field',
                           get_cmd=self.get_field,
                           set_cmd=self.set_field,
                           label='Magnetic Field',
                           unit='T',
                           vals=vals.Numbers(min_value=min_field,
                                             max_value=max_field),
                           docstring='Magnetic Field sourced by Coil')
        self.add_parameter('source_field',
                           get_cmd=self.get_source_field,
                           set_cmd=self.set_source_field,
                           label='Source Field',
                           unit='T',
                           vals=vals.Numbers(min_value=min_field,
                                             max_value=max_field),
                           docstring='Field maintained as seen by the AMI')
        self.add_parameter('switch_state',
                           get_cmd=self.get_switch_state,
                           set_cmd=self.set_switch_state,
                           label='Switch State',
                           unit='',
                           vals=vals.Enum('SuperConducting','NormalConducting'),
                           docstring='Indicating whether the coil is in\
                               persistenet current mode')
        # self.add_parameter("ramp_rate",
        #                    get_cmd=self.i_source.ramp_rate,
        #                    set_cmd=lambda d: self.i_source.ramp_rate(d),
        #                    label='Ramp Rate',
        #                    unit='T/s',
        #                    vals=vals.Numbers(min_value=0,
        #                                      max_value=ramp_rate*field_to_current),
        #                    docstring='Indicating whether the coil is in\
        #                        persistenet current mode')

        self.field_to_current = field_to_current # units of T/A
        self.ramp_rate = ramp_rate # units of A/s
        self.switch_heater_current = switch_heater_current #Not used for the AMI
        self.persistent = True
        self.protection_state=True

        #Dummy coil parameters
        self._field=0
        self._is_quenched=False
        self._heater_state=True


        self.get_all()

    def get_all(self):
        self.get_source_field()
        self.switch_state()
        self.get_field()
        return

    ##########
    ####The only functions encapsulating of heater and current source
    ### Aka the wrapper functions
    ##########
    def check_for_quench(self):
        '''
        Checks if there is a quench present
        '''
        return self._is_quenched

    def set_quenched(self,value):
        self._is_quenched=value
    def set_source_field(self,field):
        ##set_field of the AMI is blocking. Use ramp_to to have a non-blocking call
        self._field = field

        return self.name+' field set to '+str(field)+' T'

    def get_source_field(self):
        return self._field

    def heater_enabled(self):
        # return self.i_source.switch_heater_enabled()
        return self._heater_state

    def turn_on_heater(self,turn_on):
        self._heater_state=turn_on

    ############
    ### Functions only using the wrapper functions
    ############
    def get_switch_state(self):
        if self.heater_enabled():
            return 'NormalConducting'
        else:
            return 'SuperConducting'


    def set_switch_state(self,desired_state):
        if desired_state == 'SuperConducting' and\
                self.get_switch_state() == 'SuperConducting':
            # print(self.name+' coil already SuperConducting')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state() == 'NormalConducting':
            # print(self.name+' coil already NormalConducting')
            return 'NormalConducting'
        elif desired_state == 'SuperConducting' and\
                self.get_switch_state() == 'NormalConducting':
            # print('Ramping current down...')
            self.turn_on_heater(False)
            # print('Wait 2 minutes to cool the switch for coil '+self.name)
            # time.sleep(120) # 120
            # print('Switch of '+self.name+' coil is now SuperConducting')
            self.fake_folder(folder_name='Switch_'+self.name+'_Coil_changed_to_SuperConducting_state')
            return 'SuperConducting'
        elif desired_state == 'NormalConducting' and\
                self.get_switch_state() == 'SuperConducting':
            if self.check_for_quench():
                raise ValueError('Magnet leads not connected!')
            else:
                supplied_field = self.get_source_field()
                # supplied_current = self.get_source_current()
                if self.get_field()==None:
                    # print('Sourcing current...')
                    self.turn_on_heater(True)
                    # print('Wait 30 seconds to heat up the switch.')
                    # time.sleep(30) # 30
                    # print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_'+self.name+'_Coil_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                # elif supplied_current<2e-3 and np.abs(self.get_field())<1e-4:
                elif supplied_field<1e-4 :

                    # print('Sourcing current...')
                    self.turn_on_heater(True)
                    # print('Wait 30 seconds to heat up the switch.')
                    # time.sleep(30) # 30
                    # print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_'+self.name+'_Coil_changed_to_NormalConducting_state')
                    return 'NormalConducting'
                elif not 0.98*supplied_field<self.get_field()\
                    <1.02*supplied_field:
                    raise ValueError('Current of '+self.name+' coil is not \
                                        according to the field value! Use \
                                        bring_source_to_field function to \
                                        bring it to the correct value.')
                else:
                    # print('Sourcing current...')
                    self.turn_on_heater(True)
                    # print('Wait 30 seconds to heat up the switch.')
                    # time.sleep(30) # 30
                    # print('Switch is now NormalConducting')
                    self.fake_folder(folder_name='Switch_'+self.name+'_Coil_changed_to_NormalConducting_state')
                    return 'NormalConducting'
        else:
            return 'Input SuperConducting or NormalConducting as desired state.'


    def set_field(self,field):
        if not self.protection_state:
            return 0.
        if self.switch_state() == 'SuperConducting':
            raise ValueError('Switch of '+self.name+' coil is SuperConducting. Can not change the field.')
        elif self.switch_state() =='NormalConducting':
            if self.get_source_field()<1e-4:
                self.step_field_to_value(field)
                return 'Field of '+self.name+' coil at ' +str(field)+' T'
            elif self.check_for_quench():
                raise ValueError('Magnet leads are not connected \
                                 or manget quenched!')
            else:
                self.step_field_to_value(field)
                return 'Field of '+self.name+' coil at ' +str(field)+' T'

    def get_field(self):
        # return 0.0 # Only add this line when doing the first initialization!
        if self.switch_state()=='SuperConducting':
            #We do not save the actual field in a file
            return self.get_source_field()


            # ## get the persistent field from the HDF5 file
            # timestamp = atools.latest_data(contains='Switch_'+self.name+'_Coil_changed_to_SuperConducting_state',
            #                                return_timestamp=True)[0]
            # params_dict = {'field':self.name+'.field'}
            # numeric_params = ['field']
            # data = a_tools.get_data_from_timestamp_list([timestamp], params_dict,
            #                                    numeric_params=numeric_params, filter_no_analysis=False)
            # return data['field'][0]
        else: ## Normal conducting
            meas_field = self.measure_field()
            return meas_field


    def field_to_curr(self,field):
        return field/self.field_to_current

    def curr_to_field(self,current):
        return self.field_to_current*current

    def fake_folder(self,folder_name):
        '''
        Give folder_name in the form of a string.
        Creates a folder with the desired folder name containing a dummy measurement.
        '''
        if isinstance(folder_name,str):
            sweep_pts = np.linspace(0, 10, 3)
            self.MC.set_sweep_function(swf.None_Sweep())
            self.MC.set_sweep_points(sweep_pts)
            self.MC.set_detector_function(det.Dummy_Detector_Soft())
            dat = self.MC.run(folder_name)
        else:
            raise ValueError('Please enter a string as the folder name!')

    def compensate_for_drift(self, compensation_factor=1.05):
        '''
        Use this function to supply more current when the switch is
        superconducting in order to compensate drift.
        Can also be used to ramp the current to 0 and disconnect the source.
        '''
        if not self.switch_state() == 'SuperConducting':
            print('Switch is normal conducting.')
            return
        if self.check_for_quench():
            raise ValueError('Quench!')
        target_field = self.field()*compensation_factor
        self.step_field_to_value(target_field)


    def bring_source_to_field(self):
        '''
        Use this function to source the appropriate current when the
        switch is superconducting and a current not corresponding to the field
        is supplied. After that, the switch can be turned normal again.
        '''
        self.compensate_for_drift(compensation_factor=1.)

########
###Below are the 2nd level of interface
########
    def step_field_to_value(self,field):
        '''
        Don't use this function, this function is the internal ramping
        that actually communicates with the current/magnet source
        to ramp the field to a certain value
        '''



        step_time = 0.01 # in seconds
        if self.ramp_rate is not None:
            Ramp_rate_I = self.ramp_rate
            # current_step = Ramp_rate_I  * step_time
            field_step = self.curr_to_field(Ramp_rate_I*step_time)
        else:
            raise ValueError('Specify either ramp rate or field step!')
        # I_now = self.field_to_curr(self.get_field())
        B_now = self.get_field()
        # current_target = self.field_to_curr(field)
        B_target = field
        # if current_target >= I_now:
        #     current_step *= +1
        # if current_target < I_now:
        #     current_step *= -1
        if B_target >= B_now:
            field_step *= +1
        elif B_target <=B_now:
            field_step *= -1
        num_steps = int(1.*(B_target-B_now)/(field_step))
        sweep_time = step_time*num_steps
        print('Sweep time is '+str(np.abs(sweep_time))+' seconds')
        for tt in range(num_steps):
            time.sleep(step_time)
            self.set_source_field(B_now)
            B_now += field_step

        self.set_source_field(field)


    def measure_field(self):
        if True:
            B = self.get_source_field()
            return B
        else:
            print(self.name+' coil has no magnet current source!')
