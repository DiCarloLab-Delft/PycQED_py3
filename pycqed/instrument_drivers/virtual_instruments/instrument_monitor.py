import logging
import numpy as np
from qcodes.utils.helpers import make_unique, DelegateAttributes
import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class InstrumentMonitor(Instrument):
    """
    Creates a pyqtgraph widget that displays
    """
    proc = None
    rpg = None
    def __init__(self, name, station,
                 figsize=(1000, 600),
                 window_title='', theme=((60, 60, 60), 'w'), show_window=True, remote=True, **kwargs):
        """
        Initializes the plotting window
        """
        super().__init__(name=name, server_name=None)
        self.station = station
        self.add_parameter('plotting_interval',
                           unit='s',
                           vals=vals.Numbers(min_value=0.001),
                           initial_value=1,
                           parameter_class=ManualParameter)
        # self.add_parameter
        if remote:
            if not self.__class__.proc:
                self._init_qt()
        else:
            # overrule the remote pyqtgraph class
            self.rpg = pg

        self.create_tree(figsize=figsize)

    def update(self):
        snapshot = self.station.snapshot()
        self.tree.setData(snapshot)


    def _init_qt(self):
        # starting the process for the pyqtgraph plotting
        # You do not want a new process to be created every time you start a
        # run, so this only starts once and stores the process in the class
        pg.mkQApp()
        self.__class__.proc = pgmp.QtProcess()  # pyqtgraph multiprocessing
        self.__class__.rpg = self.proc._import('pyqtgraph')

    def create_tree(self, figsize=(1000, 600)):
        snapshot = self.station.snapshot()
        # self.tree = self.rpg.DataTreeWidget(data=snapshot)
        self.tree = self.rpg.DataTreeWidget()
        self.update()
        self.tree.show()
        self.tree.setWindowTitle('Instrument Monitor')
        self.tree.resize(*figsize)
