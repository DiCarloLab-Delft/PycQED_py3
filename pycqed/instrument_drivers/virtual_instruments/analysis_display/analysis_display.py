import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp
import pycqed.analysis.analysis_toolbox as a_tools


class AnalysisDisplay:

    def __init__(self, filename=''):
        self.qapp = pg.mkQApp()
        self.proc = pgmp.QtProcess()
        # load required modules in the new process
        self.ada = self.proc._import('pycqed.instrument_drivers.'
                    'virtual_instruments.analysis_display.analysis_display_app')
        # start the app
        self.app = self.ada.AnalysisDisplayApp(filename)
        self.app.set_data_path(a_tools.datadir)
        self.update()

    def update(self):
        self.app.update()
