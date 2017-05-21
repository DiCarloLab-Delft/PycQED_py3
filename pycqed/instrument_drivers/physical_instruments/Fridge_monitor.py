import logging

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from time import time

from urllib.request import urlopen
import re  # used for string parsing

# Dictionary containing fridge addresses
dcl = 'http://dicarlolab.tudelft.nl//wp-content/uploads/'
address_dict = {'LaMaserati': dcl + 'LaMaseratiMonitor/',
                'LaDucati': dcl + 'LaDucatiMonitor/',
                'LaFerrari': dcl + 'LaFerrariMonitor/',
                'LaAprilia': dcl + 'LaApriliaMonitor/'}

monitored_pars_dict = \
    {'LaMaserati': {'temp': ['T_CP', 'T_CP (P)', 'T_3K', 'T_3K (P)',
                             'T_Still', 'T_Still (P)', 'T_MClo',  'T_MClo (P)',
                             'T_MChi', 'T_MChi (P)', 'T_50K (P)'],
                    'press': ['P_5', 'P_Still', 'P_IVC', 'P_Probe',
                              'P_OVC', 'P_4He', 'P_3He']},

     'LaDucati': {'temp': ['T_Sorb', 'T_Still', 'T_MClo', 'T_MChi'],
                  'press': ['P_5', 'P_Still', 'P_IVC', 'P_OVC',
                            'P_4He', 'P_3He']},

     'LaAprilia': {'temp': ['T_3K', 'T_Still', 'T_CP',
                            'T_MChi', 'T_MClo', 'T_MCloCMN '],
                   'press': ['P_5', 'P_Still', 'P_IVC', 'P_OVC',
                             'P_4He', 'P_3He']},

     'LaFerrari': {'temp': ['T_Sorb', 'T_Still', 'T_MChi', 'T_MCmid',
                            'T_MClo', 'T_MCStage'],
                   'press': ['P_5', 'P_Still', 'P_IVC', 'P_OVC', 'P_4He',
                             'P_3He']}}


class Fridge_Monitor(Instrument):

    def __init__(self, name, fridge_name, update_interval=60, **kw):
        super().__init__(name, **kw)
        self.add_parameter('update_interval', unit='s',
                           initial_value=update_interval,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(min_value=0))

        self.add_parameter(
            'fridge_name', initial_value=fridge_name,
            vals=vals.Enum('LaMaserati', 'LaDucati', 'LaFerrari', 'LaAprilia'),
            parameter_class=ManualParameter)

        self.add_parameter('last_mon_update',
                           label='Last website update',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.url = address_dict[self.fridge_name()]
        # These parameters could also be extracted by reading the website.
        # might be nicer :)
        self.monitored_temps = monitored_pars_dict[self.fridge_name()]['temp']
        self.monitored_press = monitored_pars_dict[self.fridge_name()]['press']

        for par_name in self.monitored_temps:
            self.add_parameter(par_name, unit='K',
                               get_cmd=self._gen_temp_get(par_name))
        for par_name in self.monitored_press:
            self.add_parameter(par_name, unit='Bar',
                               get_cmd=self._gen_press_get(par_name))

        self._last_temp_update = 0
        self._update_monitor()

    def get_idn(self):
        return {'vendor': 'QuTech',
                'model': 'FridgeMon',
                'serial': None, 'firmware': '0.2'}

    def _gen_temp_get(self, par_name):
        def get_cmd():
            self._update_monitor()
            try:
                return float(self.temp_dict[par_name]) * 1e-3  # mK -> K
            except Exception as e:
                logging.info('Could not extract {} from {}'.format(
                    par_name, self.url))
                logging.info(e)
        return get_cmd

    def _gen_press_get(self, par_name):
        def get_cmd():
            self._update_monitor()
            try:
                return float(self.press_dict[par_name]) * 1e-3  # mBar -> Bar
            except Exception as e:
                logging.info('Could not extract {} from {}'.format(
                    par_name, self.url))
                logging.info(e)
        return get_cmd

    def snapshot(self, update=False):
        # Overwrites the snapshot to make it update the parameters if
        # time is larger than update interval
        time_since_update = time() - self._last_temp_update
        if time_since_update > (self.update_interval()):
            return super().snapshot(update=True)
        else:
            return super().snapshot(update=update)

    def _update_monitor(self):
        time_since_update = time() - self._last_temp_update
        if time_since_update > (self.update_interval()):
            self.temp_dict = {}
            try:
                s = urlopen(self.url, timeout=5)
                source = s.read()
                s.close()
                # Extract the last update time of the fridge website
                upd_time = re.findall(
                    'Current time: (.*)<br>L', str(source))[0]
                self.last_mon_update(upd_time)
                # Extract temperature names with temperature from source code
                temperaturegroups = re.findall(
                    r'<br>(T_[\w_]+(?: \(P\))?) = ([\d\.]+)', str(source))

                self.temp_dict = {elem[0]: float(
                    elem[1]) for elem in temperaturegroups}
                pressuregroups = re.findall(
                    r'<br>(P_[\w_]+(?: \(P\))?) = ([\d\.]+)', str(source))
                self.press_dict = {elem[0]: float(
                    elem[1]) for elem in pressuregroups}

            except Exception as e:
                logging.warning(e)

                for temperature_name in self.monitored_temps:
                    self.temp_dict[temperature_name] = 0
                for press_name in self.monitored_press:
                    self.press_dict[press_name] = 0
