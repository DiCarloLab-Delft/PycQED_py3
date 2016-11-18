import numpy as np
import types
import logging

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from time import time
import sys
from urllib.request import urlopen
# import urllib2
import re


class Fridge_Monitor(Instrument):

    def __init__(self, name, fridge_name, update_interval=60, **kw):
        super().__init__(name, **kw)
        # self.add_parameter('auto_update_interval',
        #                    parameter_class=ManualParameter

        self._automatic_update_interval = update_interval
        self._temperature_updated = time()


        self.add_parameter('fridge_name', parameter_class=ManualParameter,
                           vals=vals.Enum('LaMaserati', 'LaDucati', 'LaFerrari'))
        self.add_parameter('temperatures',
                           get_cmd=self._get_temperatures)

        self.fridge_name(fridge_name)

        if self.fridge_name() == 'LaMaserati':
            self.url = r'http://dicarlolab.tudelft.nl//' + \
                'wp-content/uploads/LaMaseratiMonitor/'
            self.temperature_names = ['T_MClo', 'T_CMN', 'T_MChi', 'T_50mK',
                                      'T_Still', 'T_3K']

        elif self.fridge_name() == 'LaDucati':
            self.url = r'http://dicarlolab.tudelft.nl//' + \
                'wp-content/uploads/LaDucatiMonitor/'
            self.temperature_names = ['T_MClo', 'T_MChi', 'T_50mK', 'T_Still',
                                      'T_3K']

        elif self.fridge_name() == 'LaFerrari':
            self.url = r'http://dicarlolab.tudelft.nl//' + \
                'wp-content/uploads/LaFerrariMonitor/'
            self.temperature_names = ['T_Sorb', 'T_Still', 'T_MChi', 'T_MCmid',
                                      'T_MClo', 'T_MCStage']

        self._get_temperatures()



    def _get_temperatures(self, doupdate=True):
        last_updated = time() - self._temperature_updated
        if last_updated > (self._automatic_update_interval - 2) \
            or doupdate is True:
            woerterbuch = {}

            try:
                s = urlopen(self.url, timeout=5)
                source = s.read()
                s.close()
                # Extract temperature names with temperature from source code
                temperaturegroups = re.findall(
                    r'<br>(T_[\w_]+(?: \(P\))?) = ([\d\.]+)', str(source))

                woerterbuch = {elem[0]: float(elem[1])
                               for elem in temperaturegroups}
            except:
                print('\nTemperatures could not be extracted from website\n')
                for temperature_name in self.temperature_names:
                    woerterbuch[temperature_name] = 0

            return woerterbuch

