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
                'LaFerrari': dcl + 'LaFerrariMonitor/'}


class Fridge_Monitor(Instrument):

    def __init__(self, name, fridge_name, update_interval=60, **kw):
        super().__init__(name, **kw)
        self.add_parameter('update_interval', units='s',
                           initial_value=update_interval,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(min_value=0))

        self.add_parameter(
            'fridge_name', initial_value=fridge_name,
            vals=vals.Enum('LaMaserati', 'LaDucati', 'LaFerrari'),
            parameter_class=ManualParameter)

        self.url = address_dict[self.fridge_name()]
        # These parameters could also be extracted by reading the website.
        # might be nicer :)
        if self.fridge_name() == 'LaMaserati':
            self.monitored_pars = ['T_CP', 'T_CP (P)',
                                   'T_3K', 'T_3K (P)',
                                   'T_Still', 'T_Still (P)',
                                   'T_MClo',  'T_MClo (P)',
                                   'T_MChi', 'T_MChi (P)',
                                   'T_50K (P)']

        elif self.fridge_name() == 'LaDucati':
            self.monitored_pars = ['T_Sorb', 'T_Still', 'T_MClo', 'T_MChi']

        elif self.fridge_name() == 'LaFerrari':
            self.monitored_pars = ['T_Sorb', 'T_Still', 'T_MChi', 'T_MCmid',
                                   'T_MClo', 'T_MCStage']

        for par_name in self.monitored_pars:
            self.add_parameter(par_name, units='mK',
                               get_cmd=self._gen_temp_get(par_name))
        self._last_temp_update = 0
        self._update_temperatures()

    def get_idn(self):
        return {'vendor': 'QuTech',
                'model': 'FridgeMon',
                'serial': None, 'firmware': '0.2'}

    def _gen_temp_get(self, par_name):
        def get_cmd():
            self._update_temperatures()
            try:

                return float(self.temp_dict[par_name])
            except:
                logging.warning('Could not extract {} from {}'.format(
                    par_name, self.url))
        return get_cmd

    def snapshot(self, update=False):
        # Overwrites the snapshot to make it update the parameters if
        # time is larger than update interval
        time_since_update = time() - self._last_temp_update
        if time_since_update > (self.update_interval()):
            return super().snapshot(update=True)
        else:
            return super().snapshot(update=update)

    def _update_temperatures(self):
        time_since_update = time() - self._last_temp_update
        if time_since_update > (self.update_interval()):
            self.temp_dict = {}
            try:
                s = urlopen(self.url, timeout=5)
                source = s.read()
                s.close()
                # Extract temperature names with temperature from source code
                temperaturegroups = re.findall(
                    r'<br>(T_[\w_]+(?: \(P\))?) = ([\d\.]+)', str(source))

<<<<<<< HEAD
                woerterbuch = {elem[0]: float(elem[1])
                               for elem in temperaturegroups}
            except:
                print('\nTemperatures could not be extracted from website\n')
                for temperature_name in self.temperature_names:
                    woerterbuch[temperature_name] = 0

            return woerterbuch

=======
                self.temp_dict = {elem[0]: float(elem[1])
                                  for elem in temperaturegroups}
            except Exception:
                logging.warning(
                    '\nTemperatures could not be extracted from website\n')
                for temperature_name in self.monitored_pars:
                    self.temp_dict[temperature_name] = 0
>>>>>>> origin/fridge_monitor
