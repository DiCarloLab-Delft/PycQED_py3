import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import analysis_toolbox as a_tools
import time
from pycqed.analysis_v2.spectroscopy_analysis import ResonatorSpectroscopy


class AlignmentAnalysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str,
                 options_dict: dict = None,
                 t_stop: str = None,
                 do_fitting: bool = True,
                 extract_only: bool = False,
                 auto: bool = True):
        super().__init__(t_start, t_stop=t_stop,
                                options_dict=options_dict,
                                do_fitting=do_fitting,
                                extract_only=extract_only)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.labels = ['Alignment']
        self.exact_label_match =False


        self.numeric_params = ['xvals','measured_values']
        if auto:
            self.run_analysis()


    def process_data(self):
        proc_dict = self.proc_data_dict
        proc_dict['xunit'] = self.raw_data_dict['xunit'][0][0]
        proc_dict['yunit'] = self.raw_data_dict['value_units'][0]
        proc_dict['xvals'] = self.raw_data_dict['xvals'][0]
        proc_dict['yvals'] = self.raw_data_dict['measured_values'][0]
        # proc_dict['yunit'][0] = 'Hz' ## change this in the qubit object

        # Now prepare the 2-D plot

        options_dict_fine={'scan_label':'Resonator_scan',
                           'exact_label_match':False}
        reso_amps = ResonatorSpectroscopy(t_start=self.t_start, t_stop=self.t_stop,
                              extract_only=True,
                              do_fitting=False,
                              auto=True,
                              options_dict=options_dict_fine)
        proc_dict['reso_analysis_fine'] = reso_amps


    def prepare_plots(self):
        # assumes that value names are unique in an experiment


        for i, val_name in enumerate(self.raw_data_dict['value_names'][0]):
            yvals = self.raw_data_dict['measured_values_ord_dict'][val_name][0]

            self.plot_dicts['{}_vs_iteration'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['xvals'],
                'xlabel': r'B$_{\perp}$',
                'xunit': self.proc_data_dict['xunit'],
                'yvals': self.proc_data_dict['yvals'][i],
                'ylabel': 'Frequency',  ## todo: program this in the measurement in pycqed
                'yunit': self.proc_data_dict['yunit'][i],
                'setdesc':r'f$_{\mathrm{res}}$',
                'setlabel':'',
                'do_legend':True,
                'title': self.raw_data_dict['measurementstring'][0]}

            legend_title = 'timestamp'


        #now the other plot:
        reso_amps = self.proc_data_dict['reso_analysis_fine']
        reso_amps.prepare_plots()
        for key,val in reso_amps.plot_dicts.items():
                self.plot_dicts[key] = val
                self.plot_dicts[key]['xunit'] = self.proc_data_dict['xunit']
                self.plot_dicts[key]['xvals'] = self.proc_data_dict['xvals']
                self.plot_dicts[key]['xlabel'] = r'B$_{\perp}$'
