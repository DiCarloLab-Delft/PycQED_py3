from instrument import Instrument
import qt
from . import PyLabVIEW.lv_monitor as lv_monitor
lvmon = lv_monitor.LVmon()
import logging
import numpy as np


class Plot_Monitor(Instrument):
    def __init__(self, name):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Monitor'])
        self.mon = lvmon
        self.add_function('connect_mon')

    def connect_mon(self):
        try:
            self.mon.lb.connect()
        except:
            print('No connection to LabView Plot Server')

    def get_mon(self):
        return self.mon

    def plot2D(self, nmon, data):
        if self.mon.lb.isConnected == 1:
            try:
                # Data conversion, required because plotmon accepts only
                # 64 bit values
                x = data[0]
                y = data[1]
                if type(x[0]) != np.float64:
                    x = x.astype(np.float64)
                    y = y.astype(np.float64)
                    data = [x, y]
            except:
                pass

            try:
                data = [x, y]
                self.mon.plot2D_monitor(nmon, data)
            except:
                print('2D mon not shown')
                self.connect_mon()

    def plot3D(self, nmon, data, axis=(0., 1., 0., 1.)):
        '''
        Sends data to a 3D monitor.
        nmon (int) : id of the monitor
        data (array) : data array of Z-values shaped (x,y)
        axis (tuple) : tuple formatted as (x_start, x_step, y_start, y_step)

        Note: can only render evenly spaced grids correctly.
        '''
        if self.mon.lb.isConnected == 1:
            try:
                self.mon.plot3D_monitor(nmon, data, axis=axis)
            except:
                print('3D mon not shown')
                self.connect_mon()
