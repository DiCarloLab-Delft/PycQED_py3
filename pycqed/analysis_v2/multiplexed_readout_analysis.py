import os
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha
import pycqed.analysis.tools.data_manipulation as dm_tools
from pycqed.utilities.general import int2base
import pycqed.measurement.hdf5_data as h5d
import copy


class Multiplexed_Readout_Analysis(ba.BaseDataAnalysis):
    """
    Multiplexed readout analysis.

    Does data binning and creates histograms of data.
    Threshold is auto determined as the mean of the data.
    Used to construct a assignment probability matris.

    WARNING: Not sure if post selection supports measurement
    data in two quadratures. Should use optimal weights if 
    using post-selection.
    """

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 extract_combinations: bool = False,
                 post_selection: bool = False,
                 post_selec_thresholds: list = None,
                 auto=True):
        """
        Inherits from BaseDataAnalysis.

        extract_combinations (bool):
            if True, tries to extract combinations used in experiment
            from the experimental metadata.
        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.extract_combinations = extract_combinations
        self.post_selection = post_selection
        self.post_selec_thresholds = post_selec_thresholds
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        if self.extract_combinations:
            param_spec['combinations'] = \
                ('Experimental Data/Experimental Metadata/combinations', 'dset')
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

        if self.extract_combinations:
            # For some reason the list is stored a list of length 1 arrays...
            self.raw_data_dict['combinations'] = [
                c[0] for c in self.raw_data_dict['combinations']]

    def process_data(self):
        self.proc_data_dict = {}

        # determine combinations iterated over (e.g., 000, 001, 010, 011 etc.)
        value_names = self.raw_data_dict['value_names']
        nr_qubits = len(value_names)

        if 'combinations' in self.raw_data_dict.keys():
            combinations = self.raw_data_dict['combinations']
        else:
            base = 2
            combinations = [int2base(i, base=base, fixed_length=nr_qubits) for
                            i in range(base**nr_qubits)]
        self.proc_data_dict['combinations'] = combinations

        # Not all experiments are valid calibration points.
        valid_combinations = []
        for c in combinations:
            if set(c) <= {'0', '1'}:
                valid_combinations += [c]
        self.proc_data_dict['valid_combinations'] = valid_combinations

        # Get qubit labels
        qubit_labels = [v[-2:].decode('utf-8').strip() for v in value_names]
        self.proc_data_dict['qubit_labels'] = qubit_labels

        # Bin data in histograms
        raw_shots = self.raw_data_dict['data'][:, 1:]
        hist_data = {}
        
        #############################################
        # Sort post-selection from measurement shots
        #############################################
        if self.post_selection == True:
            self.proc_data_dict['non_selected_shots'] = \
                {ch_name : {} for ch_name in value_names}
            self.proc_data_dict['post_selected_shots'] = \
                {ch_name : {} for ch_name in value_names}
            self.proc_data_dict['post_selecting_shots'] = \
                {ch_name : {} for ch_name in value_names}
        # Loop over qubits
        for i, ch_name in enumerate(value_names):
            ch_data = raw_shots[:, i]  # select per channel
            hist_data[ch_name] = {}
            # Loop over prepared state
            for j, comb in enumerate(combinations):
                if self.post_selection == True:
                    post_selec_shots = ch_data[2*j::len(combinations)*2]
                    shots = ch_data[2*j+1::len(combinations)*2]
                    self.proc_data_dict['non_selected_shots'][ch_name][comb] = shots.copy()
                    self.proc_data_dict['post_selected_shots'][ch_name][comb] = shots
                    self.proc_data_dict['post_selecting_shots'][ch_name][comb] = post_selec_shots
                else:
                    shots = ch_data[j::len(combinations)]
        
                cnts, bin_edges = np.histogram(shots, bins=100,
                    range=(min(ch_data), max(ch_data)))
                    #range=(np.min(ch_data), np.max(ch_data))) # was crashing with np (check if necessary)
                bin_centers = bin_edges[:-1]+(bin_edges[1]-bin_edges[0])/2
                hist_data[ch_name][comb] = (cnts, bin_centers) 

        self.proc_data_dict['hist_data'] = hist_data
        self.proc_data_dict['hist_data_2'] = copy.deepcopy(hist_data)

        #########################
        # Execute post_selection
        #########################
        if self.post_selection == True:
            for comb in combinations: # Loop over prepared states    
                Idxs = []
                '''
                For each prepared state one needs to eliminate every shot 
                if a single qubit fails post selection.
                '''
                for i, ch in enumerate(value_names): # Loop over qubits
                    '''
                    First find all idxs for all qubits. This has to loop
                    over alll qubits before in pre-measurement.
                    '''
                    post_selec_shots = self.proc_data_dict['post_selecting_shots'][ch][comb]
                    post_select_indices = dm_tools.get_post_select_indices(
                        thresholds=[self.post_selec_thresholds[i]], 
                        init_measurements=[post_selec_shots])           
                    Idxs += list(post_select_indices)
                for i, ch in enumerate(value_names): # Loop over qubits
                    '''
                    Now that we have all idxs, we can discard the shots that
                    failed in every qubit.
                    '''
                    shots = self.proc_data_dict['post_selected_shots'][ch][comb]
                
                    shots[Idxs] = np.nan # signal post_selection
                    shots = shots[~np.isnan(shots)] # discard post failed shots
                    self.proc_data_dict['post_selected_shots'][ch][comb] = shots

                    cnts, bin_edges = np.histogram(shots, bins=100,
                        range=(min(raw_shots[:, i]), max(raw_shots[:, i])))
                        #range=(np.min(ch_data), np.max(ch_data))) # was crashing with np (check if necessary)
                    bin_centers = bin_edges[:-1]+(bin_edges[1]-bin_edges[0])/2
                    self.proc_data_dict['hist_data'][ch][comb] = (cnts, bin_centers) 
        
        
        ###############################################
        # Calculate mean voltages (used for threshold)
        ###############################################
        binned_data = {}
        for i, ch_name in enumerate(value_names):
            ch_data = raw_shots[:, i]  # select per channel
            binned_data[ch_name] = {}
            for j, comb in enumerate(combinations):
                if self.post_selection == True:
                    binned_data[ch_name][comb] = np.mean(
                        self.proc_data_dict['post_selected_shots'][ch_name][comb])
                else:
                    binned_data[ch_name][comb] = np.mean(
                        ch_data[j::len(combinations)])  # start at

        mn_voltages = {}
        for i, ch_name in enumerate(value_names):
            ch_data = binned_data[ch_name]  # select per channel
            mn_voltages[ch_name] = {'0': [], '1': []}
            for c in combinations:
                if c in valid_combinations:
                    if c[i] == '0':
                        mn_voltages[ch_name]['0'].append(ch_data[c])
                    elif c[i] == '1':
                        mn_voltages[ch_name]['1'].append(ch_data[c])
            mn_voltages[ch_name]['0'] = np.mean(mn_voltages[ch_name]['0'])
            mn_voltages[ch_name]['1'] = np.mean(mn_voltages[ch_name]['1'])
            mn_voltages[ch_name]['threshold'] = np.mean(
                [mn_voltages[ch_name]['0'],
                 mn_voltages[ch_name]['1']])
        self.proc_data_dict['mn_voltages'] = mn_voltages

        ################
        # Digitize data
        ################
        if self.post_selection == True:
            self.proc_data_dict['post_selected_shots_digitized'] = {ch_name : {} for ch_name in value_names}
            self.proc_data_dict['non_selected_shots_digitized'] = {ch_name : {} for ch_name in value_names}
            for ch in value_names:
                for comb in combinations:
                    self.proc_data_dict['post_selected_shots_digitized'][ch][comb] = np.array( 
                        self.proc_data_dict['post_selected_shots'][ch][comb] > mn_voltages[ch]['threshold'], dtype=int)
                    self.proc_data_dict['non_selected_shots_digitized'][ch][comb] = np.array( 
                        self.proc_data_dict['non_selected_shots'][ch][comb] > mn_voltages[ch]['threshold'], dtype=int)
        else:
            digitized_data = np.zeros(raw_shots.shape)
            for i, vn in enumerate(value_names):
                digitized_data[:, i] = np.array(
                    raw_shots[:, i] > mn_voltages[vn]['threshold'], dtype=int)
        
        # Bin digitized data
        binned_dig_data = {}
        for i, ch_name in enumerate(value_names):
            binned_dig_data[ch_name] = {}
            if self.post_selection == True:
                ch_data = self.proc_data_dict['post_selected_shots_digitized'][ch_name]
                for comb in combinations:
                    binned_dig_data[ch_name][comb] = np.mean(
                        ch_data[comb])  # start at
            else:
                ch_data = digitized_data[:, i]  # select per channel
                for j, comb in enumerate(combinations):
                    binned_dig_data[ch_name][comb] = np.mean(
                        ch_data[j::len(combinations)])  # start at

        self.proc_data_dict['binned_dig_data'] = binned_dig_data

        ##########################################
        # Calculate assignment probability matrix
        ##########################################
        if self.post_selection == True:
            assignment_prob_matrix = calc_assignment_prob_matrix(combinations,
                        self.proc_data_dict['post_selected_shots_digitized'],
                        valid_combinations = valid_combinations,
                        post_selection = True)
            assignment_prob_matrix_2 = calc_assignment_prob_matrix(combinations,
                        self.proc_data_dict['non_selected_shots_digitized'],
                        valid_combinations = valid_combinations,
                        post_selection = True)

            self.proc_data_dict['assignment_prob_matrix_2'] = assignment_prob_matrix_2
            self.proc_data_dict['quantities_of_interest'] = {'assignment_probability_matrix':assignment_prob_matrix,
                                                         'trace':np.trace(assignment_prob_matrix)}
        else:
            assignment_prob_matrix = calc_assignment_prob_matrix(
                combinations, digitized_data, valid_combinations=valid_combinations)
        self.proc_data_dict['assignment_prob_matrix'] = assignment_prob_matrix
        self.proc_data_dict['quantities_of_interest'] = {'assignment_probability_matrix':assignment_prob_matrix,
                                                         'trace':np.trace(assignment_prob_matrix)}

        ##################################
        # Calculate cross-fidelity matrix
        ##################################
        cross_fidelity_matrix = calc_cross_fidelity_matrix(
                                combinations=combinations,
                                assignment_prob_matrix = assignment_prob_matrix,
                                qubit_labels = self.proc_data_dict['qubit_labels'])

        if self.post_selection == True: # Data without post-selection
            cross_fidelity_matrix_2 = calc_cross_fidelity_matrix(
                                      combinations=combinations,
                                      assignment_prob_matrix = assignment_prob_matrix_2,
                                      qubit_labels = self.proc_data_dict['qubit_labels'])
            self.proc_data_dict['cross_fidelity_matrix_2'] = cross_fidelity_matrix_2

        self.proc_data_dict['cross_fidelity_matrix'] = cross_fidelity_matrix
        self.proc_data_dict['quantities_of_interest'] = {'cross_fidelity_matrix': cross_fidelity_matrix,
                                                         'trace': np.trace(cross_fidelity_matrix)}

    def prepare_plots(self):
        if self.post_selection == True:
            self.plot_dicts['assignment_probability_matrix_post_selected'] = {
                'plotfn': plot_assignment_prob_matrix,
                'assignment_prob_matrix':
                    self.proc_data_dict['assignment_prob_matrix'],
                'post_selection': True,
                'combinations': self.proc_data_dict['combinations'],
                'valid_combinations': self.proc_data_dict['valid_combinations'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'plotsize': np.array(np.shape(self.proc_data_dict['assignment_prob_matrix'].T))*.8
            }
            self.plot_dicts['assignment_probability_matrix'] = {
                'plotfn': plot_assignment_prob_matrix,
                'assignment_prob_matrix':
                    self.proc_data_dict['assignment_prob_matrix_2'],
                'post_selection': False,
                'combinations': self.proc_data_dict['combinations'],
                'valid_combinations': self.proc_data_dict['valid_combinations'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'plotsize': np.array(np.shape(self.proc_data_dict['assignment_prob_matrix'].T))*.8
            }
            self.plot_dicts['plot_cross_ass_Fid_matrix_post_selected'] = {
                'plotfn': plot_cross_ass_Fid_matrix,
                'prob_matrix':
                    self.proc_data_dict['cross_fidelity_matrix'],
                'post_selection': True,
                'combinations': self.proc_data_dict['qubit_labels'],
                'valid_combinations': self.proc_data_dict['qubit_labels'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'plotsize': np.array(np.shape(self.proc_data_dict['cross_fidelity_matrix'].T))*.8
            }
            self.plot_dicts['plot_cross_ass_Fid_matrix'] = {
                'plotfn': plot_cross_ass_Fid_matrix,
                'prob_matrix':
                    self.proc_data_dict['cross_fidelity_matrix_2'],
                'post_selection': False,
                'combinations': self.proc_data_dict['qubit_labels'],
                'valid_combinations': self.proc_data_dict['qubit_labels'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'plotsize': np.array(np.shape(self.proc_data_dict['cross_fidelity_matrix_2'].T))*.8
            }
            for i, value_name in enumerate(self.raw_data_dict['value_names']):
                qubit_label = self.proc_data_dict['qubit_labels'][i]

                self.plot_dicts['mux_ssro_histogram_post_selected_{}'.format(qubit_label)] = {
                    'plotfn': plot_mux_ssro_histograms,
                    'hist_data': self.proc_data_dict['hist_data'][value_name],
                    'qubit_idx': i,
                    'value_name': value_name,
                    'combinations': self.proc_data_dict['combinations'],
                    'qubit_labels': self.proc_data_dict['qubit_labels'],
                    'threshold': self.proc_data_dict['mn_voltages'][value_name]['threshold'],
                    'post_selection': True
                }
                self.plot_dicts['mux_ssro_histogram_{}'.format(qubit_label)] = {
                    'plotfn': plot_mux_ssro_histograms,
                    'hist_data': self.proc_data_dict['hist_data_2'][value_name],
                    'qubit_idx': i,
                    'value_name': value_name,
                    'combinations': self.proc_data_dict['combinations'],
                    'qubit_labels': self.proc_data_dict['qubit_labels'],
                    'threshold': self.proc_data_dict['mn_voltages'][value_name]['threshold'],
                    'post_selection': False
                }

        else:
            self.plot_dicts['assignment_probability_matrix'] = {
                'plotfn': plot_assignment_prob_matrix,
                'assignment_prob_matrix':
                    self.proc_data_dict['assignment_prob_matrix'],
                'combinations': self.proc_data_dict['combinations'],
                'valid_combinations': self.proc_data_dict['valid_combinations'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'plotsize': np.array(np.shape(self.proc_data_dict['assignment_prob_matrix'].T))*.8
            }
            self.plot_dicts['plot_cross_ass_Fid_matrix'] = {
                'plotfn': plot_cross_ass_Fid_matrix,
                'prob_matrix':
                    self.proc_data_dict['cross_fidelity_matrix'],
                'combinations': self.proc_data_dict['qubit_labels'],
                'valid_combinations': self.proc_data_dict['qubit_labels'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'plotsize': np.array(np.shape(self.proc_data_dict['cross_fidelity_matrix'].T))*.8
            }
            for i, value_name in enumerate(self.raw_data_dict['value_names']):
                qubit_label = self.proc_data_dict['qubit_labels'][i]

                self.plot_dicts['mux_ssro_histogram_{}'.format(qubit_label)] = {
                    'plotfn': plot_mux_ssro_histograms,
                    'hist_data': self.proc_data_dict['hist_data'][value_name],
                    'qubit_idx': i,
                    'value_name': value_name,
                    'combinations': self.proc_data_dict['combinations'],
                    'qubit_labels': self.proc_data_dict['qubit_labels'],
                    'threshold': self.proc_data_dict['mn_voltages'][value_name]['threshold']
                }
                

class Multiplexed_Readout_Analysis_2(ba.BaseDataAnalysis):
    """
    Multiplexed readout analysis.

    Does data binning and creates histograms of data.
    Threshold is auto determined as the mean of the data.
    Used to construct a assignment probability matris.

    WARNING: Not sure if post selection supports measurement
    data in two quadratures. Should use optimal weights if 
    using post-selection.
    """

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 extract_combinations: bool = False,
                 post_selection: bool = False,
                 post_selec_thresholds: list = None,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.extract_combinations = extract_combinations
        self.post_selection = post_selection
        self.post_selec_thresholds = post_selec_thresholds
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}

        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        
        self.proc_data_dict = {}

        Channels = self.raw_data_dict['value_names'] # Qubit UHF outputs
        nr_qubits = len(Channels)
        combinations = ['{:05b}'.format(i) for i in range(2**nr_qubits)]
        raw_shots = self.raw_data_dict['data'][:, 1:]
        post_selection = self.post_selection
        qubit_labels = [ch[-2:].decode('utf-8').strip() for ch in Channels]
        self.proc_data_dict['combinations'] = combinations
        self.proc_data_dict['qubit_labels'] = qubit_labels

        #############################################
        # Sort post-selection from measurement shots
        #############################################
        self.proc_data_dict['Shots'] = {ch : {} for ch in Channels}

        if self.post_selection == True:
            # Post-selected shots
            self.proc_data_dict['Post_selected_shots'] =\
                {ch : {} for ch in Channels}
            # Pre-measurement shots
            self.proc_data_dict['Pre_measurement_shots'] =\
                {ch : {} for ch in Channels}

        # Loop over all qubits
        for i, ch in enumerate(Channels):
            ch_shots = raw_shots[:, i]
    
            # Loop over prepared states
            for j, comb in enumerate(combinations):
                if post_selection == False:
                    shots = ch_shots[j::len(combinations)]
                    self.proc_data_dict['Shots'][ch][comb] = shots.copy()
                else:
                    pre_meas_shots = ch_shots[2*j::len(combinations)*2]
                    shots = ch_shots[2*j+1::len(combinations)*2]
                    self.proc_data_dict['Shots'][ch][comb] = shots.copy()
                    self.proc_data_dict['Post_selected_shots'][ch][comb] =\
                        shots.copy()
                    self.proc_data_dict['Pre_measurement_shots'][ch][comb] =\
                        pre_meas_shots.copy()

        #########################
        # Execute post_selection
        #########################
        if self.post_selection == True:
            for comb in combinations: # Loop over prepared states    
                Idxs = []
                # For each prepared state one needs to eliminate every shot 
                # if a single qubit fails post selection.
                for i, ch in enumerate(Channels): # Loop over qubits
                    # First, find all idxs for all qubits. This has to loop
                    # over alll qubits before in pre-measurement.
                    pre_meas_shots =\
                        self.proc_data_dict['Pre_measurement_shots'][ch][comb]
                    post_select_indices = dm_tools.get_post_select_indices(
                        thresholds=[self.post_selec_thresholds[i]], 
                        init_measurements=[pre_meas_shots])           
                    Idxs += list(post_select_indices)
                
                for i, ch in enumerate(Channels): # Loop over qubits
                    # Now that we have all idxs, we can discard the shots that
                    # failed in every qubit.
                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
                    shots[Idxs] = np.nan # signal post_selection with nan
                    shots = shots[~np.isnan(shots)] # discard post failed shots
                    self.proc_data_dict['Post_selected_shots'][ch][comb] = shots

        ############################################
        # Histograms, thresholds and digitized data
        ############################################
        self.proc_data_dict['Histogram_data'] = {ch : {} for ch in Channels}
        if post_selection == True:
            self.proc_data_dict['Histogram_data_0'] = {ch : {} for ch in Channels}
        self.proc_data_dict['PDF_data'] = {ch : {} for ch in Channels}
        Shots_digitized = {ch : {} for ch in Channels}
        if post_selection == True:
            Shots_digitized_0 = {ch : {} for ch in Channels}
        for i, ch in enumerate(Channels):
            hist_range = (np.amin(raw_shots[:, i]), np.amax(raw_shots[:, i]))
            Counts_0 = np.zeros(100) # used to store overall counts of a qubit
            Counts_1 = np.zeros(100)
            
            # Histograms
            for comb in combinations:
                if post_selection == True:
                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
                    shots_0 = self.proc_data_dict['Shots'][ch][comb]
                    counts_0, bin_edges_0 = np.histogram(shots_0, bins = 100, 
                                            range=hist_range)
                    bin_centers_0 = (bin_edges_0[1:] + bin_edges_0[:-1])/2
                    self.proc_data_dict['Histogram_data_0'][ch][comb] = \
                        (counts_0, bin_centers_0)
                else:
                    shots = self.proc_data_dict['Shots'][ch][comb]

                counts, bin_edges = np.histogram(shots, bins = 100, 
                                                 range=hist_range)
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                self.proc_data_dict['Histogram_data'][ch][comb] = \
                    (counts, bin_centers)
                if comb[i] == '0':
                    Counts_0 += counts
                else:
                    Counts_1 += counts
            
            # Cumulative sums
            counts_0 = np.cumsum(Counts_0)
            counts_1 = np.cumsum(Counts_1)
            threshold = bin_centers[np.argmax(counts_0-counts_1)]
            self.proc_data_dict['PDF_data'][ch]['0'] = \
                (Counts_0, bin_centers)
            self.proc_data_dict['PDF_data'][ch]['1'] = \
                (Counts_1, bin_centers)
            self.proc_data_dict['PDF_data'][ch]['threshold'] = threshold
            
            # Digitized data
            for comb in combinations:
                if post_selection == True:
                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
                    shots_0 = self.proc_data_dict['Shots'][ch][comb]
                    Shots_digitized_0[ch][comb] = \
                        np.array(shots_0 > threshold, dtype=int)
                else:
                    shots = self.proc_data_dict['Shots'][ch][comb]
                Shots_digitized[ch][comb] = \
                    np.array(shots > threshold, dtype=int)

        ##########################################
        # Calculate assignment probability matrix
        ##########################################
        if post_selection == True:
            assignment_prob_matrix_0 = calc_assignment_prob_matrix(combinations,
                Shots_digitized_0)
            cross_fidelity_matrix_0 = calc_cross_fidelity_matrix(combinations,
                assignment_prob_matrix_0)
            self.proc_data_dict['assignment_prob_matrix_0'] = \
                assignment_prob_matrix_0
            self.proc_data_dict['cross_fidelity_matrix_0'] = \
                cross_fidelity_matrix_0
        
        assignment_prob_matrix = calc_assignment_prob_matrix(combinations,
            Shots_digitized)
        cross_fidelity_matrix = calc_cross_fidelity_matrix(combinations,
            assignment_prob_matrix)
        self.proc_data_dict['assignment_prob_matrix'] = assignment_prob_matrix
        self.proc_data_dict['cross_fidelity_matrix'] = cross_fidelity_matrix

    def prepare_plots(self):

        Channels = self.raw_data_dict['value_names']
        qubit_labels = [ch[-2:].decode('utf-8').strip() for ch in Channels]
        combinations = ['{:05b}'.format(i) for i in range(2**len(Channels))]

        self.plot_dicts['assignment_probability_matrix'] = {
            'plotfn': plot_assignment_prob_matrix,
            'assignment_prob_matrix':
                self.proc_data_dict['assignment_prob_matrix'],
            'combinations': self.proc_data_dict['combinations'],
            'valid_combinations': self.proc_data_dict['combinations'],
            'qubit_labels': qubit_labels,
            'plotsize': np.array(np.shape(\
                self.proc_data_dict['assignment_prob_matrix'].T))*.8
            }
        self.plot_dicts['plot_cross_ass_Fid_matrix'] = {
            'plotfn': plot_cross_ass_Fid_matrix,
            'prob_matrix':
                self.proc_data_dict['cross_fidelity_matrix'],
            'combinations': qubit_labels,
            'valid_combinations': qubit_labels,
            'qubit_labels': qubit_labels,
            'plotsize': np.array(np.shape(\
                self.proc_data_dict['cross_fidelity_matrix'].T))*.8
            } 
        for i, ch in enumerate(Channels):
            qubit_label = qubit_labels[i]
            self.plot_dicts['mux_ssro_histogram_{}'.format(\
                    qubit_label)] = {
                'plotfn': plot_mux_ssro_histograms,
                'hist_data': self.proc_data_dict['Histogram_data'][ch],
                'qubit_idx': i,
                'value_name': ch,
                'combinations': combinations,
                'qubit_labels': qubit_labels,
                'threshold': \
                    self.proc_data_dict['PDF_data'][ch]['threshold'],
                }
            self.plot_dicts['mux_ssro_cumsum_{}'.format(qubit_label)] = {
                'plotfn': plot_cumulative_distribution,
                'data': self.proc_data_dict['PDF_data'][ch],
                'qubit_label': qubit_label
                }

        if self.post_selection == True:
            self.plot_dicts['assignment_probability_matrix_raw'] = {
                'plotfn': plot_assignment_prob_matrix,
                'assignment_prob_matrix':
                    self.proc_data_dict['assignment_prob_matrix_0'],
                'combinations': self.proc_data_dict['combinations'],
                'valid_combinations': self.proc_data_dict['combinations'],
                'qubit_labels': qubit_labels,
                'plotsize': np.array(np.shape(\
                    self.proc_data_dict['assignment_prob_matrix_0'].T))*.8
                }
            self.plot_dicts['plot_cross_ass_Fid_matrix_raw'] = {
                'plotfn': plot_cross_ass_Fid_matrix,
                'prob_matrix':
                    self.proc_data_dict['cross_fidelity_matrix_0'],
                'combinations': qubit_labels,
                'valid_combinations': qubit_labels,
                'qubit_labels': qubit_labels,
                'plotsize': np.array(np.shape(\
                    self.proc_data_dict['cross_fidelity_matrix'].T))*.8
                }
            for i, ch in enumerate(Channels):
                qubit_label = qubit_labels[i]
                self.plot_dicts['mux_ssro_histogram_raw_{}'.format(\
                        qubit_label)] = {
                    'plotfn': plot_mux_ssro_histograms,
                    'hist_data': self.proc_data_dict['Histogram_data_0'][ch],
                    'qubit_idx': i,
                    'value_name': ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                        self.proc_data_dict['PDF_data'][ch]['threshold'],
                    }
                     


       
        
def calc_assignment_prob_matrix(combinations, digitized_data):

    assignment_prob_matrix = np.zeros((len(combinations), len(combinations)))
    
    for i, input_state in enumerate(combinations):
        for j, outcome in enumerate(combinations):
            first_key = next(iter(digitized_data))
            Check = np.ones(len(digitized_data[first_key][input_state]))
            for k, ch in enumerate(digitized_data.keys()):
                check = digitized_data[ch][input_state] == int(outcome[k])
                Check *= check

            assignment_prob_matrix[i][j] = sum(Check)/len(Check)
        
    return assignment_prob_matrix

def calc_cross_fidelity_matrix(combinations,assignment_prob_matrix):
    
    n = int(np.log2(len(combinations)))
    crossFidMat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P_eiIj = 0  # P(e_i|0_j)
            P_giPj = 0  # P(g_i|pi_j)

            # Loop over all entries in the Assignment probability matrix
            for prep_idx, c_prep in enumerate(combinations):
                for decl_idx, c_decl in enumerate(combinations):
                    # Select all entries in the assignment matrix for ei|Ij
                    if (c_decl[i]=='1') and (c_prep[j] == '0'):
                        P_eiIj += assignment_prob_matrix[prep_idx, decl_idx]
                    # Select all entries in the assignment matrix for ei|Ij
                    elif (c_decl[i]=='0') and (c_prep[j] == '1'): # gi|Pj
                        P_giPj += assignment_prob_matrix[prep_idx, decl_idx]

            # Normalize probabilities
            normalization_factor = (len(combinations)/2)

            P_eiIj = P_eiIj/normalization_factor
            P_giPj = P_giPj/normalization_factor

            # Add entry to cross fidelity matrix
            Fc = 1 - P_eiIj - P_giPj
            crossFidMat[i,j] = Fc

    return crossFidMat

def plot_assignment_prob_matrix(assignment_prob_matrix,
                                combinations, qubit_labels, ax=None,
                                valid_combinations=None,
                                **kw):
    if ax is None:
        figsize = np.array(np.shape(assignment_prob_matrix))*.7
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.get_figure()

    if valid_combinations is None:
        valid_combinations = combinations

    alpha_reds = cmap_to_alpha(cmap=pl.cm.Reds)
    colors = [(0.6, 0.76, 0.98), (0, 0, 0)]
    cm = LinearSegmentedColormap.from_list('my_blue', colors)
    alpha_blues = cmap_first_to_alpha(cmap=cm)

    red_im = ax.matshow(assignment_prob_matrix*100,
                        cmap=alpha_reds, clim=(0., 10))
    blue_im = ax.matshow(assignment_prob_matrix*100,
                         cmap=alpha_blues, clim=(50, 100))

    caxb = f.add_axes([0.9, 0.6, 0.02, 0.3])

    caxr = f.add_axes([0.9, 0.15, 0.02, 0.3])
    ax.figure.colorbar(red_im, ax=ax, cax=caxr)
    ax.figure.colorbar(blue_im, ax=ax, cax=caxb)

    rows, cols = np.shape(assignment_prob_matrix)
    for i in range(rows):
        for j in range(cols):
            c = assignment_prob_matrix[i, j]
            if c > .05:
                col = 'white'
            else:
                col = 'black'
            ax.text(j, i, '{:.2f}'.format(c),
                    va='center', ha='center', color=col)

    ax.set_xticklabels(valid_combinations)
    ax.set_xticks(np.arange(len(valid_combinations)))

    ax.set_yticklabels(combinations)
    ax.set_yticks(np.arange(len(combinations)))
    ax.set_ylim(len(combinations)-.5, -.5)
    ax.set_ylabel('Input state')
    ax.set_xlabel('Declared state')
    ax.xaxis.set_label_position('top')

    qubit_labels_str = ', '.join(qubit_labels)
    ax.set_title('Assignment probability matrix\n qubits: [{}]'.format(
        qubit_labels_str))


def plot_mux_ssro_histograms(
        hist_data, combinations,
        qubit_idx, value_name,
        qubit_labels, threshold,
        ax=None,
        **kw):
    if ax is None:
        f, ax = plt.subplots()

    colors_R = pl.cm.Reds
    colors_B = pl.cm.Blues
    colors_G = pl.cm.Greens
    iR = 0.1  # Do not start at the complete white/transparent end
    iB = 0.1
    iG = 0.1
    for i, (key, (cnts, bin_centers)) in enumerate(hist_data.items()):

        if set(key) <= {'0', '1'}:
            if key[qubit_idx] == '0':
                # increment the blue colorscale
                col = colors_B(iB)
                iB += 0.8/(len(combinations)/2)#.8 to not span full colorscale
            elif key[qubit_idx] == '1':
                # Increment the red colorscale
                col = colors_R(iR)
                iR += 0.8/(len(combinations)/2)
            else:
                raise ValueError('{}  {}'.format(
                    combinations, combinations[qubit_idx]))
        else:
            # increment the green colorscale
            col = colors_G(iG)
            iG += 0.8/(len(combinations)/2)  # .8 to not span full colorscale
        ax.plot(bin_centers, cnts, label=key, color=col)

    ax.set_xlabel(value_name.decode('utf-8'))
    ax.set_ylabel('Counts')
    ax.axvline(x=threshold,
               ls='--', color='grey', label='threshold')
    ax.legend(loc=(1.05, .01), title='Prepared state\n{}'.format(
        qubit_labels))

def plot_cross_ass_Fid_matrix(prob_matrix,
                              combinations, qubit_labels, ax=None,
                              valid_combinations=None,
                              **kw):
    if ax is None:
        figsize = np.array(np.shape(prob_matrix))*.7
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.get_figure()

    if valid_combinations is None:
        valid_combinations = combinations

    alpha_reds = cmap_to_alpha(cmap=pl.cm.Reds)
#     colors = [(0.6, 0.76, 0.98), (0, 0, 0)]
    colors = [(0.58, 0.404, 0.741), (0, 0, 0)]

    cm = LinearSegmentedColormap.from_list('my_purple', colors)
    alpha_blues = cmap_first_to_alpha(cmap=cm)

    red_im = ax.matshow(prob_matrix*100,
                        cmap=alpha_reds, clim=(-10., 10))
    red_im = ax.matshow(prob_matrix*100,
                        cmap='RdBu', clim=(-10., 10))

    blue_im = ax.matshow(prob_matrix*100,
                         cmap=alpha_blues, clim=(80, 100))

    caxb = f.add_axes([0.9, 0.6, 0.02, 0.3])

    caxr = f.add_axes([0.9, 0.15, 0.02, 0.3])
    ax.figure.colorbar(red_im, ax=ax, cax=caxr)
    ax.figure.colorbar(blue_im, ax=ax, cax=caxb)

    rows, cols = np.shape(prob_matrix)
    for i in range(rows):
        for j in range(cols):
            c = prob_matrix[i, j]
            if c > .05 or c <-0.05:
                col = 'white'
            else:
                col = 'black'
            ax.text(j, i, '{:.1f}'.format(c*100),
                    va='center', ha='center', color=col)

    ax.set_xticklabels(valid_combinations)
    ax.set_xticks(np.arange(len(valid_combinations)))

    ax.set_yticklabels(combinations)
    ax.set_yticks(np.arange(len(combinations)))
    ax.set_ylim(len(combinations)-.5, -.5)
    # matrix[i,j] => i = column, j = row
    ax.set_ylabel(r'Prepared qubit, $q_i$')
    ax.set_xlabel(r'Classified qubit $q_j$')
    ax.xaxis.set_label_position('top')

    qubit_labels_str = ', '.join(qubit_labels)
#     ax.set_title(r'Cross fidelity $F_{ij}$')

def plot_cross_ass_Fid_matrix(prob_matrix,
                              combinations, qubit_labels, ax=None,
                              valid_combinations=None,
                              **kw):
    if ax is None:
        figsize = np.array(np.shape(prob_matrix))*.7
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.get_figure()

    if valid_combinations is None:
        valid_combinations = combinations

    alpha_reds = cmap_to_alpha(cmap=pl.cm.Reds)
#     colors = [(0.6, 0.76, 0.98), (0, 0, 0)]
    colors = [(0.58, 0.404, 0.741), (0, 0, 0)]

    cm = LinearSegmentedColormap.from_list('my_purple', colors)
    alpha_blues = cmap_first_to_alpha(cmap=cm)

    red_im = ax.matshow(prob_matrix*100,
                        cmap=alpha_reds, clim=(-10., 10))
    red_im = ax.matshow(prob_matrix*100,
                        cmap='RdBu', clim=(-10., 10))

    blue_im = ax.matshow(prob_matrix*100,
                         cmap=alpha_blues, clim=(80, 100))

    caxb = f.add_axes([0.9, 0.6, 0.02, 0.3])

    caxr = f.add_axes([0.9, 0.15, 0.02, 0.3])
    ax.figure.colorbar(red_im, ax=ax, cax=caxr)
    ax.figure.colorbar(blue_im, ax=ax, cax=caxb)

    rows, cols = np.shape(prob_matrix)
    for i in range(rows):
        for j in range(cols):
            c = prob_matrix[i, j]
            if c > .05 or c <-0.05:
                col = 'white'
            else:
                col = 'black'
            ax.text(j, i, '{:.1f}'.format(c*100),
                    va='center', ha='center', color=col)

    ax.set_xticklabels(valid_combinations)
    ax.set_xticks(np.arange(len(valid_combinations)))

    ax.set_yticklabels(combinations)
    ax.set_yticks(np.arange(len(combinations)))
    ax.set_ylim(len(combinations)-.5, -.5)
    # matrix[i,j] => i = column, j = row
    ax.set_ylabel(r'Prepared qubit, $q_i$')
    ax.set_xlabel(r'Classified qubit $q_j$')
    ax.xaxis.set_label_position('top')

    qubit_labels_str = ', '.join(qubit_labels)
#     ax.set_title(r'Cross fidelity $F_{ij}$')

def plot_cumulative_distribution(data, qubit_label, ax, **kw):
    #f, ax = plt.subplots(ncols=2, figsize=(10,4))

    counts_0, bin_centers_0 = data['0']
    counts_1, bin_centers_1 = data['1']
    threshold = data['threshold']
    f = ax.get_figure()
    # Histogram of shots
    #ax[0].bar(bin_centers_0, counts_0,
    #        width=bin_centers_0[1]-bin_centers_0[0],
    #        label=r'$|0\rangle$ state',
    #        color='C0', edgecolor='grey', alpha=.3)
    #ax[0].bar(bin_centers_1, counts_1,
    #        width=bin_centers_1[1]-bin_centers_1[0],
    #        label=r'$|0\rangle$ state',
    #        color='C3', edgecolor='grey', alpha=.3)
    #ax[0].axvline(x=threshold, label='Threshold',
    #            ls='--', color='grey', alpha=.5)
    #ax[0].set_xlim(left=bin_centers_0[0], right=bin_centers_0[-1])
    #ax[0].set_xlabel('Effective voltage (V)')
    #ax[0].set_ylabel('Counts')
    #ax[0].set_title('Histogram of shots "'+qubit_label+'"')

    # Cumulative sum of shots
    ax.plot(bin_centers_0, np.cumsum(counts_0)/sum(counts_0), 
             label=r'$|0\rangle$ state',
             color='C0', alpha=.3)
    ax.plot(bin_centers_1, np.cumsum(counts_1)/sum(counts_1),
             label=r'$|1\rangle$ state',
             color='C3', alpha=.3)
    ax.set_xlabel('Effective voltage (V)')
    ax.set_ylabel('Fraction')
    ax.axvline(x=threshold, label='Threshold',
                ls='--', color='grey', alpha=.5)
    ax.set_xlim(left=bin_centers_0[0], right=bin_centers_0[-1])
    ax.set_ylim(bottom=0)
    ax.set_title('Cumulative sum of shots "'+qubit_label+'"')
    ax.legend(loc=(1.02, .01))

    f.tight_layout()

