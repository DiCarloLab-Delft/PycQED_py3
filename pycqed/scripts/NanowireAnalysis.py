import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.scripts import Spectroscopy as Spec


class ResonatorFieldSweep(ResonatorSpectroscopy):
    def __init__(self, t_start,
                 options_dict,
                 t_stop = None,
                 do_fitting = True,
                 extract_only=False,
                 auto=True):
        super(ResonatorFieldSweep, self).__init__(t_start, t_stop=t_stop,
                                             options_dict=options_dict,
                                             do_fitting = True,
                                             extract_only=False,
                                             auto=False)
        self.params_dict['field'] = 'Magnet.field'
        self.numeric_params = ['freq', 'amp', 'phase']
        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(ResonatorFieldSweep, self).process_data()
        self.plot_xvals = self.options_dict.get('xvals',np.array([[tt] for tt in range(len(self.data_dict['timestamps']))]))
        self.data_dict['field'] = np.array(
                        [np.double(val) for val in self.data_dict['field']])
        self.plot_xvals = np.array([[tt*1e3] for tt in self.data_dict['field']])
        self.plot_xlabel = self.options_dict.get('xlabel','Magnetic Field (mT)')
        self.plot_xwidth = self.options_dict.get('xwidth',None)
        if self.plot_xwidth == 'auto':
            x_diff = np.diff(np.ravel(self.plot_xvals))
            dx1 = np.concatenate(([x_diff[0]],x_diff))
            dx2 = np.concatenate((x_diff,[x_diff[-1]]))
            self.plot_xwidth = np.minimum(dx1,dx2)
            self.plot_frequency = np.array([[tt] for tt in self.data_dict['freq']])
            self.plot_phase = np.array([[tt] for tt in self.data_dict['phase']])
            self.plot_amp = np.array([np.array([tt]).transpose() for tt in self.data_dict['amp']])
