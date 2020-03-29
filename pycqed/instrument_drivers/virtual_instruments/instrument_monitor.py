import time
import PyQt5
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
                 figsize=(600, 600),
                 window_title='', theme=((60, 60, 60), 'w'),
                 show_window=True, remote=True, **kwargs):
        """
        Initializes the plotting window
        """
        super().__init__(name=name)
        self.station = station
        self.add_parameter('update_interval',
                           unit='s',
                           vals=vals.Numbers(min_value=0.001),
                           initial_value=5,
                           parameter_class=ManualParameter)
        # self.add_parameter
        if remote:
            if not self.__class__.proc:
                self._init_qt()
        else:
            # overrule the remote pyqtgraph class
            self.rpg = pg
        # initial value is fake but ensures it will update the first time
        self.last_update_time = 0
        self.create_tree(figsize=figsize)

    def update(self):
        time_since_last_update = time.time()-self.last_update_time
        if time_since_last_update > self.update_interval():
            self.last_update_time = time.time()
            snapshot = self.station.snapshot()
            self.tree.setData(snapshot['instruments'])

    def _init_qt(self):
        # starting the process for the pyqtgraph plotting
        # You do not want a new process to be created every time you start a
        # run, so this only starts once and stores the process in the class
        pg.mkQApp()
        self.__class__.proc = pgmp.QtProcess()  # pyqtgraph multiprocessing
        self.__class__.rpg = self.proc._import('pyqtgraph')
        ins_mon_mod = 'pycqed.instrument_drivers.virtual_instruments.ins_mon.qc_snapshot_widget'
        self.__class__.rpg = self.proc._import(ins_mon_mod)

    def create_tree(self, figsize=(1000, 600)):

        self.tree = self.rpg.QcSnaphotWidget()
        self.update()
        self.tree.show()
        self.tree.setWindowTitle('Instrument Monitor')
        self.tree.resize(*figsize)
