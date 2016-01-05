from instrument import Instrument
import numpy as np
import types
import logging
import qt
from time import time
import sys
import urllib.request, urllib.error, urllib.parse
import re


class Fridge_Monitor(Instrument):

    def __init__(self, name, fridge_name, update_interval=60):

        self._automatic_update_interval = update_interval
        self._temperature_updated = time()

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])
        self.add_parameter('fridge_name', type=bytes,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('temperatures', type=dict,
                           flags=Instrument.FLAG_GETSET,
                           probe_interval=self._automatic_update_interval*1000)

        self.set_fridge_name(fridge_name)

        if self._fridge_name == 'LaMaserati':
            self.url = r'http://dicarlolab.tudelft.nl//' + \
                'wp-content/uploads/LaMaseratiMonitor/'
            self.temperature_names = ['T_MClo', 'T_CMN', 'T_MChi', 'T_50mK',
                                      'T_Still', 'T_3K']

        elif self._fridge_name == 'LaDucati':
            self.url = r'http://dicarlolab.tudelft.nl//' + \
                'wp-content/uploads/LaDucatiMonitor/'
            self.temperature_names = ['T_MClo', 'T_MChi', 'T_50mK', 'T_Still',
                                      'T_3K']

        elif self._fridge_name == 'LaFerrari':
            print('Initialized the Scuderia Ferrari.')
            self.url = r'http://dicarlolab.tudelft.nl//' + \
                'wp-content/uploads/LaFerrariMonitor/'
            self.temperature_names = ['T_Sorb', 'T_Still', 'T_MChi', 'T_MCmid',
                                      'T_MClo', 'T_MCStage']

        self.do_get_temperatures()

    def do_get_fridge_name(self):
        return self._fridge_name

    def do_set_fridge_name(self, name):
        self._fridge_name = name

    def do_get_temperatures(self, doupdate=False):
        last_updated = time() - self._temperature_updated
        if last_updated > (self._automatic_update_interval - 2) \
            or doupdate is True:
            woerterbuch = {}

            try:
                s = urllib.request.urlopen(self.url, timeout=5)
                source = s.read()
                s.close()

                # Extract temperature names with temperature from source code
                temperaturegroups = re.findall(
                    r'<br>(T_[\w_]+(?: \(P\))?) = ([\d\.]+)', source)

                woerterbuch = {elem[0]: float(elem[1])
                               for elem in temperaturegroups}
            except:
                print('\nTemperatures could not be extracted from website\n')
                for temperature_name in self.temperature_names:
                    woerterbuch[temperature_name] = 0

            self.do_set_temperatures(woerterbuch)
            return self.Temperatures

    def do_set_temperatures(self, temp_dict):
        self.Temperatures = temp_dict
