import h5py
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
import math
import unittest
import json
import os
import pycqed as pq
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis import measurement_analysis as MA
from pycqed.measurement.hdf5_data import write_dict_to_hdf5
import importlib
importlib.reload(ba)


class Test_chevron_analysis(ma.MeasurementAnalysis):

    def __init__(self, t_start: str, t_stop: str, label='arc',
                 options_dict: dict=None,
                 f_demod: float=0, demodulate: bool=False, auto=True):
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict)
        if auto:
            self.run_analysis()

	def set_plot_parameter_values(self, **kw):
        # dpi for plots
        self.dpi = kw.pop('dpi', 300)
        # font sizes
        self.font_size = kw.pop('font_size', 11)
        # line widths connecting data points
        self.line_width = kw.pop('line_width', 2)
        # lw of axes and text boxes
        self.axes_line_width = kw.pop('axes_line_width', 0.5)
        # tick lengths
        self.tick_length = kw.pop('tick_length', 4)
        # tick line widths
        self.tick_width = kw.pop('tick_width', 0.5)
        # marker size for data points
        self.marker_size = kw.pop('marker_size', None)
        # marker size for special points like
        self.marker_size_special = kw.pop('marker_size_special', 8)
        # peak freq., Rabi pi and pi/2 amplitud es
        self.box_props = kw.pop('box_props',
                                dict(boxstyle='Square', facecolor='white',
                                     alpha=0.8, lw=self.axes_line_width))

        self.tick_color = kw.get('tick_color', 'k')
        # tick label color get's updated in savefig
        self.tick_labelcolor = kw.get('tick_labelcolor', 'k')
        self.axes_labelcolor = kw.get('axes_labelcolor', 'k')

        params = {"ytick.color": self.tick_color,
                  "xtick.color": self.tick_color,
                  "axes.labelcolor": self.axes_labelcolor, }
        plt.rcParams.update(params)


    def load_hdf5data(self, folder=None, file_only=False, **kw):
        if folder is None:
            folder = self.folder
        self.h5filepath = a_tools.measureme    nt_filename(folder)
        h5mode = kw.pop('h5mode', 'r+')
        self.data_file = h5py.File(self.h5filepath, h5mode)
        if not file_only:
            for k in list(self.data_file.keys()):
                if type(self.data_file[k]) == h5py.Group:
                    self.name = k
            self.g = self.data_file['Experimental Data']
            self.measurementstring = os.path.split(folder)[1]
            self.timestamp = os.path.split(os.path.split(folder)[0])[1] \
                             + '/' + self.measurementstring[:6]
            self.timestamp_string = os.path.split(os.path.split(folder)[0])[1] \
                                    + '_' + self.measurementstring[:6]
            self.measurementstring = self.measurementstring[7:]
            self.default_plot_title = self.measurementstring
        return self.data_file

    def finish(self, close_file=True, **kw):
        if close_file:
            self.data_file.close()

    def analysis_h5data(self, name='analysis'):
        if not os.path.exists(os.path.join(self.folder, name + '.hdf5')):
            mode = 'w'
        else:
            mode = 'r+'
        return h5py.File(os.path.join(self.folder, name + '.hdf5'), mode)
         



    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels)

        a = ma_old.MeasurementAnalysis(timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        self.raw_data_dict['ncl'] = a.sweep_points
        self.raw_data_dict[''] = a.measured_values[0]
        self.raw_data_dict[''] = a.measured_values[1]
        self.raw_data_dict[''] = a.measured_values[2]
        self.raw_data_dict[''] = a.measured_values[3]
        self.raw_data_dict['data'] = []
        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish() # closes data file


    def process_data(self):
        dd = self.raw_data_dict
        # converting to survival probabilities
        for frac in ['q0_base', 'q1_base', 'q0_inter', 'q1_inter']:
            self.proc_data_dict['p_{}'.format(frac)] = 1-dd[frac]
            self.proc_data_dict['p_{}'.format(frac)] = 1-dd[frac]
        self.proc_data_dict['p_00_base'] = (1-dd['q0_base'])*(1-dd['q1_base'])
        self.proc_data_dict['p_00_inter'] = (1-dd['q0_inter'])*(1-dd['q1_inter'])