import numpy as np
from collections import OrderedDict
from copy import deepcopy
import os
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import measurement_analysis as ma_old
import pygsti
from scipy.stats import sem
from pycqed.measurement.gate_set_tomography.pygsti_helpers import \
    gst_exp_filepath, pygsti_expList_from_dataset


class GST_SingleQubit_DataExtraction(ba.BaseDataAnalysis):
    """
    Analysis class that extracts data from
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 close_figs: bool=True,
                 do_fitting: bool=True, auto=True,
                 ch_idx: int = 0,

                 gst_exp_list_filepath: str=None):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs, extract_only=extract_only,
                         do_fitting=do_fitting)

        self.gst_exp_list_filepath = gst_exp_list_filepath
        self.ch_idx = ch_idx
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)
        self.raw_data_dict['timestamps'] = self.timestamps

        self.timestamp = self.timestamps[0]
        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamp, auto=False, close_file=False)
        a.get_naming_and_values()

        self.raw_data_dict['xvals'] = a.sweep_points
        self.raw_data_dict['xlabel'] = a.parameter_names[0]
        self.raw_data_dict['xunit'] = a.parameter_units[0]

        self.raw_data_dict['bins'] = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value

        if self.gst_exp_list_filepath == None:
            gst_exp_list_filename = a.data_file['Experimental Data'][
                'Experimental Metadata'].attrs['gst_exp_list_filename']

            self.raw_data_dict['gst_exp_list_filepath'] = os.path.join(
                gst_exp_filepath, gst_exp_list_filename)
        else:
            self.raw_data_dict['gst_exp_list_filepath'] = \
                self.gst_exp_list_filepath

        self.raw_data_dict['expList'] = pygsti_expList_from_dataset(
            self.raw_data_dict['gst_exp_list_filepath'])

        self.raw_data_dict['measured_values'] = a.measured_values
        self.raw_data_dict['value_names'] = a.value_names
        self.raw_data_dict['value_units'] = a.value_units
        self.raw_data_dict['measurementstring'] = a.measurementstring
        self.measurementstring = a.measurementstring
        self.raw_data_dict['folder'] = a.folder
        a.finish()

    def process_data(self):
        """
        Involves reshaping the data and writing it to a dataset textfile
        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        bins = self.proc_data_dict['bins']

        self.proc_data_dict['binned_values'] = []
        self.proc_data_dict['binned_values_stderr'] = []

        expList = self.proc_data_dict['expList']
        shots_1 = self.proc_data_dict['measured_values'][self.ch_idx]

        print("Nr bins:", len(bins))
        print("Nr gatestrings", len(expList))
        # Filter out uncompleted iterations
        shots_1 = shots_1[:-(len(shots_1) % len(expList))]
        shots_0 = 1 - shots_1
        counts_1 = np.sum(shots_1.reshape(
            (len(expList), -1),
            order='F'), axis=1)

        counts_0 = np.sum(shots_0.reshape(
            (len(expList), -1),
            order='F'), axis=1)

        # writing to pygsti dataset
        ds = pygsti.objects.DataSet(outcomeLabels=['0', '1'])
        for i, gateString in enumerate(expList):
            ds.add_count_dict(gateString,
                              {'0': counts_0[i],
                               '1': counts_1[i]})
        ds.done_adding_data()

        ds_name = self.measurementstring+self.timestamp+'_counts.txt'
        pygsti.io.write_dataset(
            os.path.join(self.raw_data_dict['folder'], ds_name), ds)
        self.proc_data_dict['dataset'] = ds

        # for i, y in enumerate(self.proc_data_dict['measured_values']):
        #     if len(y) % len(bins) != 0:
        #         y_ext = y[:-(len(y) % len(bins))]
        #     else:
        #         y_ext = y

        #     y_binned = np.nanmean(y_ext.reshape((len(bins), -1),
        #                                         order='F'), axis=1)
        #     y_binned_stderr = sem(
        #         y_ext.reshape((len(bins), -1), order='F'), axis=1,
        #         nan_policy='omit')
        #     self.proc_data_dict['binned_values'].append(y_binned)
        #     self.proc_data_dict['binned_values_stderr'].append(y_binned_stderr)

        #     y_sum = np.sum(y_ext.reshape((len(bins), -1), order='F'), axis=1)

    # def prepare_plots(self):
    #     # assumes that value names are unique in an experiment
    #     # pass
    #     for i, val_name in enumerate(self.raw_data_dict['value_names']):

    #         self.plot_dicts['binned_{}'.format(val_name)] = {
    #             'plotfn': 'plot_errorbar',
    #             'xlabel': self.proc_data_dict['xlabel'],
    #             'xunit': self.proc_data_dict['xunit'],
    #             'ylabel': self.proc_data_dict['value_names'][i],

    #             'yunit': self.proc_data_dict['value_units'][i],
    #             'x': self.proc_data_dict['bins'],
    #             'y': self.proc_data_dict['binned_values'][i],
    #             'yerr': self.proc_data_dict['binned_values_stderr'][i],
    #             'marker': 'o',
    #             'title': "{}\nBinned {}".format(self.timestamp, val_name)}
