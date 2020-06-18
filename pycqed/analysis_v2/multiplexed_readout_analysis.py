import os
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from collections import OrderedDict
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha
import pycqed.analysis.tools.data_manipulation as dm_tools
from pycqed.utilities.general import int2base
import pycqed.measurement.hdf5_data as h5d
import copy
import lmfit
from scipy.optimize import minimize
from pycqed.analysis.fitting_models import ro_gauss, ro_CDF, ro_CDF_discr,\
     gaussian_2D, gauss_2D_guess, gaussianCDF, ro_double_gauss_guess



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

    def __init__(self, nr_qubits: int,
                 t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 extract_combinations: bool = False,
                 post_selection: bool = False,
                 post_selec_thresholds: list = None,
                 q_target=None,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.nr_qubits = nr_qubits
        self.extract_combinations = extract_combinations
        self.post_selection = post_selection
        self.post_selec_thresholds = post_selec_thresholds
        self.q_target = q_target
        self.do_fitting = True
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
        nr_qubits = self.nr_qubits

        # Data in single quadrature
        if len(self.raw_data_dict['value_names']) == nr_qubits:
            self.Channels = self.raw_data_dict['value_names']
            Channels = self.Channels
            raw_shots = self.raw_data_dict['data'][:, 1:]
            qubit_labels = [ch.decode('utf-8').rsplit(' ', 1)[1] for ch in Channels]
        # Data in two quadratures
        elif len(self.raw_data_dict['value_names']) == 2*nr_qubits:
            self.Channels = self.raw_data_dict['value_names'][::2]
            Channels = self.Channels
            raw_shots = self.raw_data_dict['data'][:, 1::2]
            qubit_labels = [ch.decode('utf-8').rsplit(' ', 2)[1] for ch in Channels]
        else:
            raise ValueError('Number of qudratures is not the same for all qubits')

        combinations = \
            ['{:0{}b}'.format(i, nr_qubits) for i in range(2**nr_qubits)]
        post_selection = self.post_selection
        self.proc_data_dict['combinations'] = combinations
        self.proc_data_dict['qubit_labels'] = qubit_labels

        #############################################
        # Sort post-selection from measurement shots
        #############################################
        self.proc_data_dict['Shots'] = {ch : {} for ch in Channels}

        if post_selection == True:
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
        if post_selection == True:
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
        self.proc_data_dict['PDF_data'] = {ch : {} for ch in Channels}
        self.proc_data_dict['CDF_data'] = {ch : {} for ch in Channels}
        Shots_digitized = {ch : {} for ch in Channels}
        if post_selection == True:
            self.proc_data_dict['Post_Histogram_data'] = \
                {ch : {} for ch in Channels}
            self.proc_data_dict['Post_PDF_data'] = {ch : {} for ch in Channels}
            self.proc_data_dict['Post_CDF_data'] = {ch : {} for ch in Channels}
            Post_Shots_digitized = {ch : {} for ch in Channels}

        for i, ch in enumerate(Channels):
            hist_range = (np.amin(raw_shots[:, i]), np.amax(raw_shots[:, i]))
            Shots_0 = [] # used to store overall shots of a qubit
            Shots_1 = []
            if post_selection == True:
                Post_Shots_0 = [] # used to store overall shots of a qubit
                Post_Shots_1 = []

            # Histograms
            for comb in combinations:
                if post_selection == True:
                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
                    # Hitogram data of each prepared_state
                    counts, bin_edges = np.histogram(shots, bins=100,
                                                     range=hist_range)
                    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                    self.proc_data_dict['Post_Histogram_data'][ch][comb]=\
                        (counts, bin_centers)
                    if comb[i] == '0':
                        Post_Shots_0 = np.concatenate((Post_Shots_0, shots))
                    else:
                        Post_Shots_1 = np.concatenate((Post_Shots_1, shots))

                shots = self.proc_data_dict['Shots'][ch][comb]
                # Hitogram data of each prepared_state
                counts, bin_edges = np.histogram(shots, bins=100,
                                                 range=hist_range)
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                self.proc_data_dict['Histogram_data'][ch][comb] = \
                    (counts, bin_centers)

                if comb[i] == '0':
                    Shots_0 = np.concatenate((Shots_0, shots))
                else:
                    Shots_1 = np.concatenate((Shots_1, shots))

            # Cumulative sums
            if post_selection == True:
                # bin data according to unique bins
                ubins_0, ucounts_0 = np.unique(Post_Shots_0, return_counts=True)
                ubins_1, ucounts_1 = np.unique(Post_Shots_1, return_counts=True)
                ucumsum_0 = np.cumsum(ucounts_0)
                ucumsum_1 = np.cumsum(ucounts_1)
                # merge |0> and |1> shot bins
                all_bins = np.unique(np.sort(np.concatenate((ubins_0, ubins_1))))
                # interpolate cumsum for all bins
                int_cumsum_0=np.interp(x=all_bins,xp=ubins_0,fp=ucumsum_0,left=0)
                int_cumsum_1=np.interp(x=all_bins,xp=ubins_1,fp=ucumsum_1,left=0)
                norm_cumsum_0 = int_cumsum_0/np.max(int_cumsum_0)
                norm_cumsum_1 = int_cumsum_1/np.max(int_cumsum_1)
                self.proc_data_dict['Post_CDF_data'][ch]['cumsum_x_ds']=all_bins
                self.proc_data_dict['Post_CDF_data'][ch]['cumsum_y_ds'] = \
                    [int_cumsum_0, int_cumsum_1]
                self.proc_data_dict['Post_CDF_data'][ch]['cumsum_y_ds_n'] = \
                    [norm_cumsum_0, norm_cumsum_1]
                # Calculating threshold
                F_vs_th = (1-(1-abs(norm_cumsum_0-norm_cumsum_1))/2)
                opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
                opt_idx = int(round(np.average(opt_idxs)))
                self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_raw'] = \
                    F_vs_th[opt_idx]
                self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'] = \
                    all_bins[opt_idx]
            # bin data according to unique bins
            ubins_0, ucounts_0 = np.unique(Shots_0, return_counts=True)
            ubins_1, ucounts_1 = np.unique(Shots_1, return_counts=True)
            ucumsum_0 = np.cumsum(ucounts_0)
            ucumsum_1 = np.cumsum(ucounts_1)
            # merge |0> and |1> shot bins
            all_bins = np.unique(np.sort(np.concatenate((ubins_0, ubins_1))))
            # interpolate cumsum for all bins
            int_cumsum_0 = np.interp(x=all_bins,xp=ubins_0,fp=ucumsum_0,left=0)
            int_cumsum_1 = np.interp(x=all_bins,xp=ubins_1,fp=ucumsum_1,left=0)
            norm_cumsum_0 = int_cumsum_0/np.max(int_cumsum_0)
            norm_cumsum_1 = int_cumsum_1/np.max(int_cumsum_1)
            self.proc_data_dict['CDF_data'][ch]['cumsum_x_ds'] = all_bins
            self.proc_data_dict['CDF_data'][ch]['cumsum_y_ds'] = \
                [int_cumsum_0, int_cumsum_1]
            self.proc_data_dict['CDF_data'][ch]['cumsum_y_ds_n'] = \
                [norm_cumsum_0, norm_cumsum_1]
            # Calculating threshold
            F_vs_th = (1-(1-abs(norm_cumsum_0-norm_cumsum_1))/2)
            opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
            opt_idx = int(round(np.average(opt_idxs)))
            self.proc_data_dict['PDF_data'][ch]['F_assignment_raw'] = \
                F_vs_th[opt_idx]
            self.proc_data_dict['PDF_data'][ch]['threshold_raw'] = \
                all_bins[opt_idx]

            # Histogram of overall shots
            if post_selection == True:
                counts_0, bin_edges = np.histogram(Post_Shots_0, bins=100,
                                                   range=hist_range)
                counts_1, bin_edges = np.histogram(Post_Shots_1, bins=100,
                                                   range=hist_range)
                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
                self.proc_data_dict['Post_PDF_data'][ch]['0'] = \
                    (counts_0, bin_centers)
                self.proc_data_dict['Post_PDF_data'][ch]['1'] = \
                    (counts_1, bin_centers)
            counts_0, bin_edges = np.histogram(Shots_0, bins=100,
                                               range=hist_range)
            counts_1, bin_edges = np.histogram(Shots_1, bins=100,
                                               range=hist_range)
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            self.proc_data_dict['PDF_data'][ch]['0'] = \
                (counts_0, bin_centers)
            self.proc_data_dict['PDF_data'][ch]['1'] = \
                (counts_1, bin_centers)

            # Digitized data
            for comb in combinations:
                if post_selection == True:
                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
                    th = self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw']
                    Post_Shots_digitized[ch][comb] = \
                        np.array(shots > th, dtype=int)
                shots = self.proc_data_dict['Shots'][ch][comb]
                th = self.proc_data_dict['PDF_data'][ch]['threshold_raw']
                Shots_digitized[ch][comb] = \
                    np.array(shots > th, dtype=int)

        ##########################################
        # Calculate assignment probability matrix
        ##########################################
        if post_selection == True:
            ass_prob_matrix = calc_assignment_prob_matrix(combinations,
                Post_Shots_digitized)
            cross_fid_matrix = calc_cross_fidelity_matrix(combinations,
                ass_prob_matrix)
            self.proc_data_dict['Post_assignment_prob_matrix'] = ass_prob_matrix
            self.proc_data_dict['Post_cross_fidelity_matrix'] = cross_fid_matrix
        assignment_prob_matrix = calc_assignment_prob_matrix(combinations,
            Shots_digitized)
        cross_fidelity_matrix = calc_cross_fidelity_matrix(combinations,
            assignment_prob_matrix)
        self.proc_data_dict['assignment_prob_matrix'] = assignment_prob_matrix
        self.proc_data_dict['cross_fidelity_matrix'] = cross_fidelity_matrix

    def prepare_fitting(self):
        Channels = self.Channels
        self.fit_dicts = OrderedDict()
        for ch in Channels:
            ###################################
            # Histograms fit (PDF)
            ###################################
            if self.post_selection == True:
                bin_x = self.proc_data_dict['Post_PDF_data'][ch]['0'][1]
                bin_xs = [bin_x, bin_x]
                bin_ys = [self.proc_data_dict['Post_PDF_data'][ch]['0'][0],
                          self.proc_data_dict['Post_PDF_data'][ch]['1'][0]]
                m = lmfit.model.Model(ro_gauss)
                m.guess = ro_double_gauss_guess.__get__(m, m.__class__)
                params = m.guess(x=bin_xs, data=bin_ys,
                         fixed_p01=self.options_dict.get('fixed_p01', False),
                         fixed_p10=self.options_dict.get('fixed_p10', False))
                post_res = m.fit(x=bin_xs, data=bin_ys, params=params)
                self.fit_dicts['Post_PDF_fit_{}'.format(ch)] = {
                    'model': m,
                    'fit_xvals': {'x': bin_xs},
                    'fit_yvals': {'data': bin_ys},
                    'guessfn_pars':
                        {'fixed_p01':self.options_dict.get('fixed_p01', False),
                         'fixed_p10':self.options_dict.get('fixed_p10', False)},
                }
            bin_x = self.proc_data_dict['PDF_data'][ch]['0'][1]
            bin_xs = [bin_x, bin_x]
            bin_ys = [self.proc_data_dict['PDF_data'][ch]['0'][0],
                      self.proc_data_dict['PDF_data'][ch]['1'][0]]
            m = lmfit.model.Model(ro_gauss)
            m.guess = ro_double_gauss_guess.__get__(m, m.__class__)
            params = m.guess(x=bin_xs, data=bin_ys,
                     fixed_p01=self.options_dict.get('fixed_p01', False),
                     fixed_p10=self.options_dict.get('fixed_p10', False))
            res = m.fit(x=bin_xs, data=bin_ys, params=params)
            self.fit_dicts['PDF_fit_{}'.format(ch)] = {
                'model': m,
                'fit_xvals': {'x': bin_xs},
                'fit_yvals': {'data': bin_ys},
                'guessfn_pars':
                    {'fixed_p01': self.options_dict.get('fixed_p01', False),
                     'fixed_p10': self.options_dict.get('fixed_p10', False)},
            }
            ###################################
            #  Fit the CDF                    #
            ###################################
            if self.post_selection == True:
                m_cul = lmfit.model.Model(ro_CDF)
                cdf_xs = self.proc_data_dict['Post_CDF_data'][ch]['cumsum_x_ds']
                cdf_xs = [np.array(cdf_xs), np.array(cdf_xs)]
                cdf_ys = self.proc_data_dict['Post_CDF_data'][ch]['cumsum_y_ds']
                cdf_ys = [np.array(cdf_ys[0]), np.array(cdf_ys[1])]

                cum_params = post_res.params
                cum_params['A_amplitude'].value = np.max(cdf_ys[0])
                cum_params['A_amplitude'].vary = False
                cum_params['B_amplitude'].value = np.max(cdf_ys[1])
                cum_params['A_amplitude'].vary = False # FIXME: check if correct
                self.fit_dicts['Post_CDF_fit_{}'.format(ch)] = {
                    'model': m_cul,
                    'fit_xvals': {'x': cdf_xs},
                    'fit_yvals': {'data': cdf_ys},
                    'guess_pars': cum_params,
                }
            m_cul = lmfit.model.Model(ro_CDF)
            cdf_xs = self.proc_data_dict['CDF_data'][ch]['cumsum_x_ds']
            cdf_xs = [np.array(cdf_xs), np.array(cdf_xs)]
            cdf_ys = self.proc_data_dict['CDF_data'][ch]['cumsum_y_ds']
            cdf_ys = [np.array(cdf_ys[0]), np.array(cdf_ys[1])]

            cum_params = res.params
            cum_params['A_amplitude'].value = np.max(cdf_ys[0])
            cum_params['A_amplitude'].vary = False
            cum_params['B_amplitude'].value = np.max(cdf_ys[1])
            cum_params['A_amplitude'].vary = False # FIXME: check if correct
            self.fit_dicts['CDF_fit_{}'.format(ch)] = {
                'model': m_cul,
                'fit_xvals': {'x': cdf_xs},
                'fit_yvals': {'data': cdf_ys},
                'guess_pars': cum_params,
            }

    def analyze_fit_results(self):
        '''
        This code was taken from single shot readout analysis and adapted to
        mux readout (April 2020).
        '''
        Channels = self.Channels
        self.proc_data_dict['quantities_of_interest'] = \
            {ch : {} for ch in Channels}
        if self.post_selection == True:
            self.proc_data_dict['post_quantities_of_interest'] = \
                {ch : {} for ch in Channels}
        self.qoi = {ch : {} for ch in Channels}
        for ch in Channels:
            if self.post_selection == True:
                # Create a CDF based on the fit functions of both fits.
                post_fr = self.fit_res['Post_CDF_fit_{}'.format(ch)]
                post_bv = post_fr.best_values
                # best values new
                post_bvn = copy.deepcopy(post_bv)
                post_bvn['A_amplitude'] = 1
                post_bvn['B_amplitude'] = 1
                def CDF(x):
                    return ro_CDF(x=x, **post_bvn)
                def CDF_0(x):
                    return CDF(x=[x, x])[0]
                def CDF_1(x):
                    return CDF(x=[x, x])[1]
                def infid_vs_th(x):
                    cdf = ro_CDF(x=[x, x], **post_bvn)
                    return (1-np.abs(cdf[0] - cdf[1]))/2
                self._CDF_0 = CDF_0
                self._CDF_1 = CDF_1
                self._infid_vs_th = infid_vs_th
                post_thr_guess = (3*post_bv['B_center'] - post_bv['A_center'])/2
                opt_fid = minimize(infid_vs_th, post_thr_guess)
                # for some reason the fit sometimes returns a list of values
                if isinstance(opt_fid['fun'], float):
                    self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_fit']=\
                        (1-opt_fid['fun'])
                else:
                    self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_fit']=\
                        (1-opt_fid['fun'])[0]
                self.proc_data_dict['Post_PDF_data'][ch]['threshold_fit']=\
                    opt_fid['x'][0]
            # Create a CDF based on the fit functions of both fits.
            fr = self.fit_res['CDF_fit_{}'.format(ch)]
            bv = fr.best_values
            # best values new
            bvn = copy.deepcopy(bv)
            bvn['A_amplitude'] = 1
            bvn['B_amplitude'] = 1
            def CDF(x):
                return ro_CDF(x=x, **bvn)
            def CDF_0(x):
                return CDF(x=[x, x])[0]
            def CDF_1(x):
                return CDF(x=[x, x])[1]
            def infid_vs_th(x):
                cdf = ro_CDF(x=[x, x], **bvn)
                return (1-np.abs(cdf[0] - cdf[1]))/2
            self._CDF_0 = CDF_0
            self._CDF_1 = CDF_1
            self._infid_vs_th = infid_vs_th
            thr_guess = (3*bv['B_center'] - bv['A_center'])/2
            opt_fid = minimize(infid_vs_th, thr_guess)
            # for some reason the fit sometimes returns a list of values
            if isinstance(opt_fid['fun'], float):
                self.proc_data_dict['PDF_data'][ch]['F_assignment_fit'] = \
                    (1-opt_fid['fun'])
            else:
                self.proc_data_dict['PDF_data'][ch]['F_assignment_fit'] = \
                    (1-opt_fid['fun'])[0]
            self.proc_data_dict['PDF_data'][ch]['threshold_fit'] = \
                opt_fid['x'][0]

            # Calculate the fidelity of both
            ###########################################
            #  Extracting the discrimination fidelity #
            ###########################################
            if self.post_selection == True:
                def CDF_0_discr(x):
                    return gaussianCDF(x, amplitude=1,
                            mu=post_bv['A_center'], sigma=post_bv['A_sigma'])
                def CDF_1_discr(x):
                    return gaussianCDF(x, amplitude=1,
                            mu=post_bv['B_center'], sigma=post_bv['B_sigma'])
                def disc_infid_vs_th(x):
                    cdf0 = gaussianCDF(x, amplitude=1, mu=post_bv['A_center'],
                                       sigma=post_bv['A_sigma'])
                    cdf1 = gaussianCDF(x, amplitude=1, mu=post_bv['B_center'],
                                       sigma=post_bv['B_sigma'])
                    return (1-np.abs(cdf0 - cdf1))/2
                self._CDF_0_discr = CDF_0_discr
                self._CDF_1_discr = CDF_1_discr
                self._disc_infid_vs_th = disc_infid_vs_th
                opt_fid_discr = minimize(disc_infid_vs_th, post_thr_guess)
                # for some reason the fit sometimes returns a list of values
                if isinstance(opt_fid_discr['fun'], float):
                    self.proc_data_dict['Post_PDF_data'][ch]['F_discr'] = \
                        (1-opt_fid_discr['fun'])
                else:
                    self.proc_data_dict['Post_PDF_data'][ch]['F_discr'] = \
                        (1-opt_fid_discr['fun'])[0]
                self.proc_data_dict['Post_PDF_data'][ch]['threshold_discr'] = \
                    opt_fid_discr['x'][0]
                post_fr = self.fit_res['Post_PDF_fit_{}'.format(ch)]
                post_bv = post_fr.params
                A_amp = post_bv['A_spurious'].value
                A_sig = post_bv['A_sigma'].value
                B_amp = post_bv['B_spurious'].value
                B_sig = post_bv['B_sigma'].value
                residual_excitation=A_amp*B_sig/((1-A_amp)*A_sig + A_amp*B_sig)
                relaxation_events = B_amp*A_sig/((1-B_amp)*B_sig + B_amp*A_sig)
                self.proc_data_dict['Post_PDF_data'][ch]['residual_excitation']=\
                    residual_excitation
                self.proc_data_dict['Post_PDF_data'][ch]['relaxation_events']=\
                    relaxation_events
            # No post-selection
            def CDF_0_discr(x):
                return gaussianCDF(x, amplitude=1,
                                   mu=bv['A_center'], sigma=bv['A_sigma'])
            def CDF_1_discr(x):
                return gaussianCDF(x, amplitude=1,
                                   mu=bv['B_center'], sigma=bv['B_sigma'])
            def disc_infid_vs_th(x):
                cdf0 = gaussianCDF(x, amplitude=1, mu=bv['A_center'],
                                   sigma=bv['A_sigma'])
                cdf1 = gaussianCDF(x, amplitude=1, mu=bv['B_center'],
                                   sigma=bv['B_sigma'])
                return (1-np.abs(cdf0 - cdf1))/2
            self._CDF_0_discr = CDF_0_discr
            self._CDF_1_discr = CDF_1_discr
            self._disc_infid_vs_th = disc_infid_vs_th
            opt_fid_discr = minimize(disc_infid_vs_th, thr_guess)
            # for some reason the fit sometimes returns a list of values
            if isinstance(opt_fid_discr['fun'], float):
                self.proc_data_dict['PDF_data'][ch]['F_discr'] = \
                    (1-opt_fid_discr['fun'])
            else:
                self.proc_data_dict['PDF_data'][ch]['F_discr'] = \
                    (1-opt_fid_discr['fun'])[0]
            self.proc_data_dict['PDF_data'][ch]['threshold_discr'] =\
                opt_fid_discr['x'][0]
            fr = self.fit_res['PDF_fit_{}'.format(ch)]
            bv = fr.params
            A_amp = bv['A_spurious'].value
            A_sig = bv['A_sigma'].value
            B_amp = bv['B_spurious'].value
            B_sig = bv['B_sigma'].value
            residual_excitation = A_amp*B_sig/((1-A_amp)*A_sig + A_amp*B_sig)
            relaxation_events = B_amp*A_sig/((1-B_amp)*B_sig + B_amp*A_sig)
            self.proc_data_dict['PDF_data'][ch]['residual_excitation'] = \
                residual_excitation
            self.proc_data_dict['PDF_data'][ch]['relaxation_events'] = \
                relaxation_events

            ###################################
            #  Save quantities of interest.   #
            ###################################
            if self.post_selection == True:
                self.proc_data_dict['post_quantities_of_interest'][ch] = {
                    'Post_SNR': \
                self.fit_res['Post_CDF_fit_{}'.format(ch)].params['SNR'].value,
                    'Post_F_d': \
                self.proc_data_dict['Post_PDF_data'][ch]['F_discr'],
                    'Post_F_a': \
                self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_raw'],
                    'Post_residual_excitation': \
                self.proc_data_dict['Post_PDF_data'][ch]['residual_excitation'],
                    'Post_relaxation_events':
                self.proc_data_dict['Post_PDF_data'][ch]['relaxation_events'],
                    'Post_threshold_raw': \
                    self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
                    'Post_threshold_discr': \
                    self.proc_data_dict['Post_PDF_data'][ch]['threshold_discr']
                }
            self.proc_data_dict['quantities_of_interest'][ch] = {
                'SNR': \
                    self.fit_res['CDF_fit_{}'.format(ch)].params['SNR'].value,
                'F_d': self.proc_data_dict['PDF_data'][ch]['F_discr'],
                'F_a': self.proc_data_dict['PDF_data'][ch]['F_assignment_raw'],
                'residual_excitation': \
                    self.proc_data_dict['PDF_data'][ch]['residual_excitation'],
                'relaxation_events':
                    self.proc_data_dict['PDF_data'][ch]['relaxation_events'],
                'threshold_raw': \
                    self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
                'threshold_discr': \
                    self.proc_data_dict['PDF_data'][ch]['threshold_discr']
            }
            self.qoi[ch] = self.proc_data_dict['quantities_of_interest'][ch]
            if self.post_selection == True:
                self.qoi[ch].update(self.proc_data_dict['post_quantities_of_interest'][ch])

    def prepare_plots(self):

        Channels = self.Channels
        nr_qubits = self.nr_qubits
        qubit_labels = self.proc_data_dict['qubit_labels']
        combinations = \
            ['{:0{}b}'.format(i, nr_qubits) for i in range(2**nr_qubits)]
        self.axs_dict = {}

        if self.q_target == None:
            # Run analysis for all qubits
            if self.post_selection is True:
                self.plot_dicts['assignment_probability_matrix_post'] = {
                    'plotfn': plot_assignment_prob_matrix,
                    'assignment_prob_matrix':
                        self.proc_data_dict['Post_assignment_prob_matrix'],
                    'combinations': self.proc_data_dict['combinations'],
                    'valid_combinations': self.proc_data_dict['combinations'],
                    'qubit_labels': qubit_labels,
                    'plotsize': np.array(np.shape(\
                    self.proc_data_dict['Post_assignment_prob_matrix'].T))*.8,
                    'post_selection': True
                    }
                self.plot_dicts['cross_fid_matrix_post'] = {
                    'plotfn': plot_cross_fid_matrix,
                    'prob_matrix':
                        self.proc_data_dict['Post_cross_fidelity_matrix'],
                    'combinations': qubit_labels,
                    'valid_combinations': qubit_labels,
                    'qubit_labels': qubit_labels,
                    'plotsize': np.array(np.shape(\
                    self.proc_data_dict['Post_cross_fidelity_matrix'].T))*.8,
                    'post_selection': True
                    }
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
            self.plot_dicts['cross_fid_matrix'] = {
                'plotfn': plot_cross_fid_matrix,
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
                # Totalized shots
                if self.post_selection == True:
                    fig, axs = plt.subplots(nrows=2, ncols=3,
                                            figsize=(13,8), dpi=200)
                    axs = axs.ravel()
                else:
                    fig, axs = plt.subplots(ncols=3, figsize=(13,4), dpi=200)
                fig.patch.set_alpha(0)
                self.axs_dict['mux_ssro_totalshots_{}'.format(qubit_label)]=axs
                self.figs['mux_ssro_totalshots_{}'.format(qubit_label)] = fig
                if self.post_selection == True:
                    self.plot_dicts['post_mux_ssro_totalshots_{}'.format(qubit_label)]={
                        'plotfn': plot_single_qubit_histogram,
                        'data': self.proc_data_dict['Post_PDF_data'][ch],
                        'qubit_label': qubit_label,
                        'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
                        'para_hist' : \
                        self.fit_res['Post_PDF_fit_{}'.format(ch)].best_values,
                        'para_cdf' : \
                        self.fit_res['Post_CDF_fit_{}'.format(ch)].best_values,
                        'hist_data': \
                        self.proc_data_dict['Post_Histogram_data'][ch],
                        'qubit_idx': i,
                        'value_name': ch,
                        'combinations': combinations,
                        'qubit_labels': qubit_labels,
                        'threshold': \
                        self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
                        'timestamp': self.timestamp,
                        'qoi': self.qoi[ch],
                        'post_selection': True
                    }
                    self.plot_dicts['post_mux_ssro_cdf_{}'.format(qubit_label)]={
                        'plotfn': plot_single_qubit_CDF,
                        'data': self.proc_data_dict['Post_PDF_data'][ch],
                        'qubit_label': qubit_label,
                        'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
                        'para_hist' : \
                        self.fit_res['Post_PDF_fit_{}'.format(ch)].best_values,
                        'para_cdf' : \
                        self.fit_res['Post_CDF_fit_{}'.format(ch)].best_values,
                        'hist_data': \
                        self.proc_data_dict['Post_Histogram_data'][ch],
                        'qubit_idx': i,
                        'value_name': ch,
                        'combinations': combinations,
                        'qubit_labels': qubit_labels,
                        'threshold': \
                        self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
                        'timestamp': self.timestamp,
                        'qoi': self.qoi[ch],
                        'post_selection': True
                    }
                    self.plot_dicts['post_mux_ssro_crosstalk_{}'.format(qubit_label)]={
                        'plotfn': plot_single_qubit_crosstalk,
                        'data': self.proc_data_dict['Post_PDF_data'][ch],
                        'qubit_label': qubit_label,
                        'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
                        'para_hist' : \
                        self.fit_res['Post_PDF_fit_{}'.format(ch)].best_values,
                        'para_cdf' : \
                        self.fit_res['Post_CDF_fit_{}'.format(ch)].best_values,
                        'hist_data': \
                        self.proc_data_dict['Post_Histogram_data'][ch],
                        'qubit_idx': i,
                        'value_name': ch,
                        'combinations': combinations,
                        'qubit_labels': qubit_labels,
                        'threshold': \
                        self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
                        'timestamp': self.timestamp,
                        'qoi': self.qoi[ch],
                        'post_selection': True
                    }
                self.plot_dicts['mux_ssro_totalshots_{}'.format(qubit_label)]={
                    'plotfn': plot_single_qubit_histogram,
                    'data': self.proc_data_dict['PDF_data'][ch],
                    'qubit_label': qubit_label,
                    'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
                    'para_hist' : \
                        self.fit_res['PDF_fit_{}'.format(ch)].best_values,
                    'para_cdf' : \
                        self.fit_res['CDF_fit_{}'.format(ch)].best_values,
                    'hist_data': self.proc_data_dict['Histogram_data'][ch],
                    'qubit_idx': i,
                    'value_name': ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                        self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
                    'timestamp': self.timestamp,
                    'qoi': self.qoi[ch]
                }
                self.plot_dicts['mux_ssro_cdf_{}'.format(qubit_label)]={
                    'plotfn': plot_single_qubit_CDF,
                    'data': self.proc_data_dict['PDF_data'][ch],
                    'qubit_label': qubit_label,
                    'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
                    'para_hist' : \
                        self.fit_res['PDF_fit_{}'.format(ch)].best_values,
                    'para_cdf' : \
                        self.fit_res['CDF_fit_{}'.format(ch)].best_values,
                    'hist_data': self.proc_data_dict['Histogram_data'][ch],
                    'qubit_idx': i,
                    'value_name': ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                        self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
                    'timestamp': self.timestamp,
                    'qoi': self.qoi[ch]
                }
                self.plot_dicts['mux_ssro_crosstalk_{}'.format(qubit_label)]={
                    'plotfn': plot_single_qubit_crosstalk,
                    'data': self.proc_data_dict['PDF_data'][ch],
                    'qubit_label': qubit_label,
                    'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
                    'para_hist' : \
                        self.fit_res['PDF_fit_{}'.format(ch)].best_values,
                    'para_cdf' : \
                        self.fit_res['CDF_fit_{}'.format(ch)].best_values,
                    'hist_data': self.proc_data_dict['Histogram_data'][ch],
                    'qubit_idx': i,
                    'value_name': ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                        self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
                    'timestamp': self.timestamp,
                    'qoi': self.qoi[ch]
                }

        else:
            # Run analysis on q_target only
            q_target_idx = qubit_labels.index(self.q_target)
            q_target_ch = Channels[q_target_idx]
            if self.post_selection is True:
                fig1, ax1 = plt.subplots(figsize=(5,4), dpi=200)
                fig1.patch.set_alpha(0)
                self.axs_dict['mux_ssro_histogram_{}_post'.format(self.q_target)]=ax1
                self.figs['mux_ssro_histogram_{}_post'.format(self.q_target)]=fig1
                self.plot_dicts['mux_ssro_histogram_{}_post'.format(self.q_target)]={
                    'plotfn': plot_single_qubit_histogram,
                    'data': self.proc_data_dict['Post_PDF_data'][q_target_ch],
                    'qubit_label': self.q_target,
                    'ax_id': 'mux_ssro_histogram_{}_post'.format(self.q_target),
                    'para_hist' : \
                    self.fit_res['Post_PDF_fit_{}'.format(q_target_ch)].best_values,
                    'para_cdf' : \
                    self.fit_res['Post_CDF_fit_{}'.format(q_target_ch)].best_values,
                    'hist_data': \
                    self.proc_data_dict['Post_Histogram_data'][q_target_ch],
                    'qubit_idx': q_target_idx,
                    'value_name': q_target_ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                    self.proc_data_dict['Post_PDF_data'][q_target_ch]['threshold_raw'],
                    'timestamp': self.timestamp,
                    'qoi': self.qoi[q_target_ch],
                    'post_selection':True
                }
                fig2, ax2 = plt.subplots(figsize=(5,4), dpi=200)
                fig2.patch.set_alpha(0)
                self.axs_dict['mux_ssro_cdf_{}_post'.format(self.q_target)]=ax2
                self.figs['mux_ssro_cdf_{}_post'.format(self.q_target)]=fig2
                self.plot_dicts['mux_ssro_cdf_{}_post'.format(self.q_target)]={
                    'plotfn': plot_single_qubit_CDF,
                    'data': self.proc_data_dict['Post_PDF_data'][q_target_ch],
                    'qubit_label': self.q_target,
                    'ax_id': 'mux_ssro_cdf_{}_post'.format(self.q_target),
                    'para_hist' : \
                    self.fit_res['Post_PDF_fit_{}'.format(q_target_ch)].best_values,
                    'para_cdf' : \
                    self.fit_res['Post_CDF_fit_{}'.format(q_target_ch)].best_values,
                    'hist_data': \
                    self.proc_data_dict['Post_Histogram_data'][q_target_ch],
                    'qubit_idx': q_target_idx,
                    'value_name': q_target_ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                    self.proc_data_dict['Post_PDF_data'][q_target_ch]['threshold_raw'],
                    'timestamp': self.timestamp,
                    'qoi': self.qoi[q_target_ch],
                    'post_selection': True
                }
                fig3, ax3 = plt.subplots(figsize=(5,4), dpi=200)
                fig3.patch.set_alpha(0)
                self.axs_dict['mux_ssro_crosstalk_{}_post'.format(self.q_target)]=ax3
                self.figs['mux_ssro_crosstalk_{}_post'.format(self.q_target)]=fig3
                self.plot_dicts['mux_ssro_crosstalk_{}_post'.format(self.q_target)]={
                    'plotfn': plot_single_qubit_crosstalk,
                    'data': self.proc_data_dict['Post_PDF_data'][q_target_ch],
                    'qubit_label': self.q_target,
                    'ax_id': 'mux_ssro_crosstalk_{}_post'.format(self.q_target),
                    'para_hist' : \
                    self.fit_res['Post_PDF_fit_{}'.format(q_target_ch)].best_values,
                    'para_cdf' : \
                    self.fit_res['Post_CDF_fit_{}'.format(q_target_ch)].best_values,
                    'hist_data': \
                    self.proc_data_dict['Post_Histogram_data'][q_target_ch],
                    'qubit_idx': q_target_idx,
                    'value_name': q_target_ch,
                    'combinations': combinations,
                    'qubit_labels': qubit_labels,
                    'threshold': \
                    self.proc_data_dict['Post_PDF_data'][q_target_ch]['threshold_raw'],
                    'timestamp': self.timestamp,
                    'qoi': self.qoi[q_target_ch],
                    'post_selection':True
                }
            fig1, ax1 = plt.subplots(figsize=(5,4), dpi=200)
            fig1.patch.set_alpha(0)
            self.axs_dict['mux_ssro_histogram_{}'.format(self.q_target)]=ax1
            self.figs['mux_ssro_histogram_{}'.format(self.q_target)]=fig1
            self.plot_dicts['mux_ssro_histogram_{}'.format(self.q_target)]={
                'plotfn': plot_single_qubit_histogram,
                'data': self.proc_data_dict['PDF_data'][q_target_ch],
                'qubit_label': self.q_target,
                'ax_id': 'mux_ssro_histogram_{}'.format(self.q_target),
                'para_hist' : \
                    self.fit_res['PDF_fit_{}'.format(q_target_ch)].best_values,
                'para_cdf' : \
                    self.fit_res['CDF_fit_{}'.format(q_target_ch)].best_values,
                'hist_data': \
                    self.proc_data_dict['Histogram_data'][q_target_ch],
                'qubit_idx': q_target_idx,
                'value_name': q_target_ch,
                'combinations': combinations,
                'qubit_labels': qubit_labels,
                'threshold': \
                self.proc_data_dict['PDF_data'][q_target_ch]['threshold_raw'],
                'timestamp': self.timestamp,
                'qoi': self.qoi[q_target_ch]
            }
            fig2, ax2 = plt.subplots(figsize=(5,4), dpi=200)
            fig2.patch.set_alpha(0)
            self.axs_dict['mux_ssro_cdf_{}'.format(self.q_target)]=ax2
            self.figs['mux_ssro_cdf_{}'.format(self.q_target)]=fig2
            self.plot_dicts['mux_ssro_cdf_{}'.format(self.q_target)]={
                'plotfn': plot_single_qubit_CDF,
                'data': self.proc_data_dict['PDF_data'][q_target_ch],
                'qubit_label': self.q_target,
                'ax_id': 'mux_ssro_cdf_{}'.format(self.q_target),
                'para_hist' : \
                    self.fit_res['PDF_fit_{}'.format(q_target_ch)].best_values,
                'para_cdf' : \
                    self.fit_res['CDF_fit_{}'.format(q_target_ch)].best_values,
                'hist_data': \
                    self.proc_data_dict['Histogram_data'][q_target_ch],
                'qubit_idx': q_target_idx,
                'value_name': q_target_ch,
                'combinations': combinations,
                'qubit_labels': qubit_labels,
                'threshold': \
                self.proc_data_dict['PDF_data'][q_target_ch]['threshold_raw'],
                'timestamp': self.timestamp,
                'qoi': self.qoi[q_target_ch]
            }
            fig3, ax3 = plt.subplots(figsize=(5,4), dpi=200)
            fig3.patch.set_alpha(0)
            self.axs_dict['mux_ssro_crosstalk_{}'.format(self.q_target)]=ax3
            self.figs['mux_ssro_crosstalk_{}'.format(self.q_target)]=fig3
            self.plot_dicts['mux_ssro_crosstalk_{}'.format(self.q_target)]={
                'plotfn': plot_single_qubit_crosstalk,
                'data': self.proc_data_dict['PDF_data'][q_target_ch],
                'qubit_label': self.q_target,
                'ax_id': 'mux_ssro_crosstalk_{}'.format(self.q_target),
                'para_hist' : \
                    self.fit_res['PDF_fit_{}'.format(q_target_ch)].best_values,
                'para_cdf' : \
                    self.fit_res['CDF_fit_{}'.format(q_target_ch)].best_values,
                'hist_data': \
                    self.proc_data_dict['Histogram_data'][q_target_ch],
                'qubit_idx': q_target_idx,
                'value_name': q_target_ch,
                'combinations': combinations,
                'qubit_labels': qubit_labels,
                'threshold': \
                self.proc_data_dict['PDF_data'][q_target_ch]['threshold_raw'],
                'timestamp': self.timestamp,
                'qoi': self.qoi[q_target_ch]
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))


class Multiplexed_Transient_Analysis(ba.BaseDataAnalysis):
    """
    Mux transient analysis.
    """

    def __init__(self, q_target: str,
                 t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.q_target = q_target
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

        length = int(len(self.raw_data_dict['data'][:, 0])/2)
        self.proc_data_dict['Time_data'] = np.arange(length)/1.8e9
        self.proc_data_dict['Channel_0_data'] = self.raw_data_dict['data'][:, 1][:length]
        self.proc_data_dict['Channel_1_data'] = self.raw_data_dict['data'][:, 2][:length]


    def prepare_plots(self):

        self.axs_dict = {}
        fig, axs = plt.subplots(nrows=2, sharex='col', figsize=(7, 5), dpi=200)
        fig.patch.set_alpha(0)
        self.axs_dict['MUX_transients'] = axs
        self.figs['MUX_transients'] = fig
        self.plot_dicts['MUX_transients'] = {
            'plotfn': plot_transients,
            'time_data': self.proc_data_dict['Time_data'],
            'data_ch_0': self.proc_data_dict['Channel_0_data'],
            'data_ch_1': self.proc_data_dict['Channel_1_data'],
            'qubit_label': self.q_target,
            'timestamp': self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))


class Multiplexed_Weights_Analysis(ba.BaseDataAnalysis):
    """
    Mux transient analysis.
    """

    def __init__(self, q_target: str,
                 IF: float, pulse_duration: float,
                 A_ground, A_excited,
                 t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.q_target = q_target
        self.IF = IF
        self.pulse_duration= pulse_duration
        self.A_ground = A_ground
        self.A_excited= A_excited
        if auto:
            self.run_analysis()

    def extract_data(self):
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict={}
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):

        Time = self.A_ground.proc_data_dict['Time_data']

        I_e = self.A_excited.proc_data_dict['Channel_0_data']
        I_g = self.A_ground.proc_data_dict['Channel_0_data']

        Q_e = self.A_excited.proc_data_dict['Channel_1_data']
        Q_g = self.A_ground.proc_data_dict['Channel_1_data']

        pulse_start = Time[get_pulse_start(Time, Q_g)]
        pulse_stop  = pulse_start+self.pulse_duration

        W_I = I_e - I_g
        W_Q = Q_e - Q_g

        #normalize weights
        W_I = W_I/np.max(W_I)
        W_Q = W_Q/np.max(W_Q)

        C = W_I + 1j*W_Q

        dW_I = np.real(np.exp(1j*2*np.pi*self.IF*Time)*C)
        dW_Q = np.imag(np.exp(1j*2*np.pi*self.IF*Time)*C)

        ps_I = np.abs(np.fft.fft(W_I))**2
        ps_Q = np.abs(np.fft.fft(W_Q))**2
        time_step = Time[1]
        Freqs = np.fft.fftfreq(W_I.size, time_step)
        idx = np.argsort(Freqs)
        Freqs = Freqs[idx]
        ps_I = ps_I[idx]
        ps_Q = ps_Q[idx]

        self.proc_data_dict['Time'] = Time
        self.proc_data_dict['I_e'] = I_e
        self.proc_data_dict['I_g'] = I_g
        self.proc_data_dict['Q_e'] = Q_e
        self.proc_data_dict['Q_g'] = Q_g
        self.proc_data_dict['W_I'] = W_I
        self.proc_data_dict['W_Q'] = W_Q
        self.proc_data_dict['dW_I'] = dW_I
        self.proc_data_dict['dW_Q'] = dW_Q
        self.proc_data_dict['Freqs'] = Freqs
        self.proc_data_dict['ps_I'] = ps_I
        self.proc_data_dict['ps_Q'] = ps_Q
        self.proc_data_dict['pulse_start'] = pulse_start
        self.proc_data_dict['pulse_stop'] = pulse_stop

        self.qoi = {}
        self.qoi = {'W_I': W_I,
                    'W_Q': W_Q}

    def prepare_plots(self):

        self.axs_dict = {}

        fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row', figsize=(9,5))
        axs = axs.flatten()
        fig.patch.set_alpha(0)
        self.axs_dict['MUX_transients_combined'] = axs
        self.figs['MUX_transients_combined'] = fig
        self.plot_dicts['MUX_transients_combined'] = {
            'plotfn': plot_mux_transients_optimal,
            'Time': self.proc_data_dict['Time'],
            'I_g': self.proc_data_dict['I_g'],
            'I_e': self.proc_data_dict['I_e'],
            'Q_g': self.proc_data_dict['Q_g'],
            'Q_e': self.proc_data_dict['Q_e'],
            'pulse_start': self.proc_data_dict['pulse_start'],
            'pulse_stop': self.proc_data_dict['pulse_stop'],
            'qubit_label': self.q_target
        }
        # Set up axis grid
        fig, axs = plt.subplots(ncols=2, nrows=3, sharey='row', figsize=(9, 7))
        axs = axs.flatten()
        gs = GridSpec(3, 2)
        [ax.remove() for ax in axs[-2:]]
        axs[4] = fig.add_subplot(gs[2,0:])
        fig.patch.set_alpha(0)
        self.axs_dict['MUX_optimal_weights'] = axs
        self.figs['MUX_optimal_weights'] = fig
        self.plot_dicts['MUX_optimal_weights'] = {
            'plotfn': plot_mux_weights,
            'Time': self.proc_data_dict['Time'],
            'W_I': self.proc_data_dict['W_I'],
            'W_Q': self.proc_data_dict['W_Q'],
            'dW_I': self.proc_data_dict['dW_I'],
            'dW_Q': self.proc_data_dict['dW_Q'],
            'Freqs': self.proc_data_dict['Freqs'],
            'ps_I': self.proc_data_dict['ps_I'],
            'ps_Q': self.proc_data_dict['ps_Q'],
            'pulse_start': self.proc_data_dict['pulse_start'],
            'pulse_stop': self.proc_data_dict['pulse_stop'],
            'IF': self.IF,
            'qubit_label': self.q_target
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))


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

def calc_cross_fidelity_matrix(combinations, assignment_prob_matrix):

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
                                post_selection=False, **kw):
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
    if post_selection is True:
        txtstr = 'Post-selected assignment probability matrix\n qubits: [{}]'.format(qubit_labels_str)
    else:
        txtstr = 'Assignment probability matrix\n qubits: [{}]'.format(
            qubit_labels_str)
    ax.set_title(txtstr, fontsize=24)


def plot_cross_fid_matrix(prob_matrix,
                          combinations, qubit_labels, ax=None,
                          valid_combinations=None,
                          post_selection=False, **kw):
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
    if post_selection:
        txtstr = 'Post-selected cross fidelity matrix'
    else:
        txtstr = 'Cross fidelity matrix'
    ax.text(.5, 1.25, txtstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', horizontalalignment='center')

def plot_single_qubit_histogram(data, ax, para_hist,
                                para_cdf, timestamp,
                                hist_data, combinations,
                                qubit_idx, value_name,
                                qubit_labels, threshold,
                                qoi, post_selection=False,
                                **kw):
    counts_0, bin_centers_0 = data['0']
    counts_1, bin_centers_1 = data['1']
    qubit_label = qubit_labels[qubit_idx]
    flag = False
    if type(ax) is np.ndarray:
        idx = int(3*post_selection)
        ax = ax[idx]
        flag=True
    f = ax.get_figure()
    ########################################
    # Histogram of shots
    ########################################
    ax.bar(bin_centers_0, counts_0,
           width=bin_centers_0[1]-bin_centers_0[0],
           label=r'$|g\rangle$ shots',
           color='C0', edgecolor='C0', alpha=.4)
    ax.bar(bin_centers_1, counts_1,
           width=bin_centers_1[1]-bin_centers_1[0],
           label=r'$|e\rangle$ shots',
           color='C3', edgecolor='C3', alpha=.3)
    # Plot Fit results
    x = np.linspace(bin_centers_0[0], bin_centers_0[-1], 150)
    ro_g = ro_gauss(x=[x, x], **para_hist)
    ax.plot(x, ro_g[0], color='C0', label=r'$|g\rangle$ fit')
    ax.plot(x, ro_g[1], color='C3', label=r'$|e\rangle$ fit')
    # Plot Threshold
    ax.axvline(x=threshold, label=r'$\mathrm{threshold}_{assign}$',
               ls='--', linewidth=1., color='black', alpha=.5)

    ax.set_xlim(left=bin_centers_0[0], right=bin_centers_0[-1])
    ax.set_xlabel('Effective voltage (V)')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of shots "'+qubit_label+'"')
    ax.legend(loc=0, fontsize=5)
    # Text box with quantities of interest
    if post_selection is True:
        textstr = '\n'.join((
            r'SNR    :       %.2f' % \
                (qoi['Post_SNR'], ),
            r'$F_{assign}$  :    %.2f%%       p(g|$\pi$) : %.2f%%' % \
                (qoi['Post_F_a']*1e2, qoi['Post_relaxation_events']*1e2, ),
            r'$F_{discr}$    :    %.2f%%       p(e|$0$) : %.2f%%' % \
                (qoi['Post_F_d']*1e2,  qoi['Post_residual_excitation']*1e2, )))
    else:
        textstr = '\n'.join((
            r'SNR    :       %.2f' % \
                (qoi['SNR'], ),
            r'$F_{assign}$  :    %.2f%%       p(g|$\pi$) : %.2f%%' % \
                (qoi['F_a']*1e2, qoi['relaxation_events']*1e2, ),
            r'$F_{discr}$    :    %.2f%%       p(e|$0$) : %.2f%%' % \
                (qoi['F_d']*1e2,  qoi['residual_excitation']*1e2, )))
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=1)
    ax.text(0.01, 1.35, textstr, transform=ax.transAxes, fontsize= 9,
           verticalalignment='top', bbox=props)

    f.suptitle('Mux_ssro_{}_{}'.format(qubit_label, timestamp))
    if flag == False:
        ax.legend(loc=0, fontsize=7)
        if post_selection is True:
            f.suptitle('Post-selected mux_ssro_{}_{}'.format(qubit_label, timestamp))

    f.tight_layout()

def plot_single_qubit_CDF(data, ax, para_hist,
                          para_cdf, timestamp,
                          hist_data, combinations,
                          qubit_idx, value_name,
                          qubit_labels, threshold,
                          qoi, post_selection=False,
                          **kw):

    counts_0, bin_centers_0 = data['0']
    counts_1, bin_centers_1 = data['1']
    qubit_label = qubit_labels[qubit_idx]
    flag = False
    if type(ax) is np.ndarray:
        idx = int(1+3*post_selection)
        ax = ax[idx]
        flag = True
        ax.set_title('Cumulative sum of shots "{}"'.format(qubit_label))
        if post_selection is True:
            ax.text(.5, 1.3, 'Post-selected Shots', transform=ax.transAxes,
            fontsize= 20, verticalalignment='top', horizontalalignment='center')
    f = ax.get_figure()
    ########################################
    # Cumulative sum of shots
    ########################################
    ax.plot(bin_centers_0, np.cumsum(counts_0)/sum(counts_0),
             label=r'$|g\rangle$ shots',
             color='C0', alpha=.75)
    ax.plot(bin_centers_1, np.cumsum(counts_1)/sum(counts_1),
             label=r'$|e\rangle$ shots',
             color='C3', alpha=.75)
    # Plot Fit results
    x = np.linspace(bin_centers_0[0], bin_centers_0[-1], 150)
    ro_c = ro_CDF(x=[x, x], **para_cdf)
    ax.plot(x, ro_c[0]/np.max(ro_c[0]), '--C0', linewidth=1,
        label=r'$|g\rangle$ fit')
    ax.plot(x, ro_c[1]/np.max(ro_c[1]), '--C3', linewidth=1,
        label=r'$|e\rangle$ fit')
    # Plot thresholds
    ax.axvline(x=threshold, label=r'$\mathrm{threshold}_{assign}$',
               ls='--', linewidth=1., color='black', alpha=.5)

    ax.set_xlim(left=bin_centers_0[0], right=bin_centers_0[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Effective voltage (V)')
    ax.set_ylabel('Fraction')
    ax.legend(loc=0, fontsize=5)

    if flag == False:
        if post_selection:
            ax.set_title('Post-selected mux_ssro_{}_{}'.format(qubit_label, timestamp))
        else:
            ax.set_title('Mux_ssro_{}_{}'.format(qubit_label, timestamp))
        ax.legend(loc=0, fontsize=7)
    f.tight_layout()

def plot_single_qubit_crosstalk(data, ax, para_hist,
                                para_cdf, timestamp,
                                hist_data, combinations,
                                qubit_idx, value_name,
                                qubit_labels, threshold,
                                qoi, post_selection=False,
                                 **kw):

    qubit_label = qubit_labels[qubit_idx]
    flag = False
    if type(ax) is np.ndarray:
        idx = int(2+3*post_selection)
        ax = ax[idx]
        flag = True
        ax.set_title('Histogram vs Prepared state "'+qubit_label+'"')

    f = ax.get_figure()
    ########################################
    # cross talk
    ########################################
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
    ax.axvline(x=threshold, label=r'$\mathrm{threshold}_{assign}$',
               ls='--', linewidth=1., color='black', alpha=.75)
    ax.set_xlabel(value_name.decode('utf-8'))
    ax.set_ylabel('Counts')
    l = ax.legend(loc=(1.05, .01), title='Prepared state\n{}'.format(
        qubit_labels), prop={'size': 4})
    l.get_title().set_fontsize('5')

    if flag == False:
        if post_selection is True:
            ax.set_title('Post-selected mux_ssro_{}_{}'.format(qubit_label, timestamp))
        else:
            ax.set_title('Mux_ssro_{}_{}'.format(qubit_label, timestamp))
        l = ax.legend(loc=(1.05, .01),
                      title='Prepared state\n{}'.format(qubit_labels),
                      prop={'size': 4})
        l.get_title().set_fontsize('4')

    f.tight_layout()

def get_pulse_start(x, y, tolerance=2):
    '''
    The start of the pulse is estimated in three steps:
        1. Evaluate signal standard deviation in a certain interval as
           function of time: f(t).
        2. Calculate the derivative of the aforementioned data: f'(t).
        3. Evaluate when the derivative exceeds a threshold. This
           threshold is defined as max(f'(t))/5.
    This approach is more tolerant to noisy signals.
    '''
    pulse_baseline = np.mean(y) # get pulse baseline
    pulse_std      = np.std(y)  # get pulse standard deviation

    nr_points_interval = 5        # number of points in the interval
    aux = int(nr_points_interval/2)

    iteration_idx = np.arange(-aux, len(y)+aux)     # mask for circular array
    aux_list = [ y[i%len(y)] for i in iteration_idx] # circular array

    # Calculate standard deviation for each interval
    y_std = []
    for i in range(len(y)):
        interval = aux_list[i : i+nr_points_interval]
        y_std.append( np.std(interval) )

    y_std_derivative = np.gradient(y_std[:-aux])# calculate derivative
    threshold = max(y_std_derivative)/10        # define threshold
    start_index = np.where( y_std_derivative > threshold )[0][0] + aux

    return start_index-tolerance


def plot_transients(time_data,
                    data_ch_0, data_ch_1,
                    qubit_label,
                    timestamp,
                    ax, **kw):
    fig = ax[0].get_figure()

    ax[0].plot(time_data, data_ch_0, '-', color='C0', linewidth=1)
    ax[0].set_xlim(left=0, right=time_data[-1])
    set_ylabel(ax[0], 'Channel_0 amplitude', 'a.u.')

    ax[1].plot(time_data, data_ch_1, '-', color='indianred', linewidth=1)
    set_ylabel(ax[1], 'Channel_1 amplitude', 'a.u.')
    set_xlabel(ax[1], 'Time', 's')

    fig.suptitle('{} Mux_transients_{}'.format(timestamp, qubit_label),
        y=1.05)
    fig.tight_layout()


def plot_mux_weights(Time,
                     W_I, W_Q,
                     dW_I, dW_Q,
                     ps_I, ps_Q,
                     pulse_start, pulse_stop,
                     IF, Freqs,
                     qubit_label,
                     ax, **kw):

    fig = ax[0].get_figure()

    for axis in ax[:4]:
        axis.axvspan(pulse_start, pulse_stop, alpha=0.15, color='yellow')
        axis.axvline(pulse_start, ls='--', color='black', linewidth=1)
        axis.axvline(pulse_stop, ls='--', color='black', linewidth=1)

    ax[0].plot(Time, W_I, 'forestgreen', linewidth=1)
    ax[2].plot(Time, dW_I, 'forestgreen', linewidth=1)
    ax[1].plot(Time, W_Q, 'darkseagreen', linewidth=1)
    ax[3].plot(Time, dW_Q, 'darkseagreen', linewidth=1)

    ax[0].set_xlim(left=0, right=Time[-1])
    ax[1].set_xlim(left=0, right=Time[-1])
    ax[2].set_xlim(left=0, right=Time[-1])
    ax[3].set_xlim(left=0, right=Time[-1])

    ax[0].set_title('Channel 0')
    ax[1].set_title('Channel 1')
    ax[2].set_title('Channel 0 (demodulated)')
    ax[3].set_title('Channel 1 (demodulated)')
    set_xlabel(ax[0], 'Time', 's')
    set_xlabel(ax[1], 'Time', 's')
    set_xlabel(ax[2], 'Time', 's')
    set_xlabel(ax[3], 'Time', 's')
    set_ylabel(ax[0], 'Amplitude', 'a.u.')
    set_ylabel(ax[2], 'Amplitude', 'a.u.')

    ax[4].axvline(abs(IF), ls='--', color='black', linewidth=1, label='IF = {:0.1f} MHz'.format(IF*1e-6))
    ax[4].plot(Freqs, ps_I, linewidth=1, color='forestgreen', label='Channel 0')
    ax[4].plot(Freqs, ps_Q, linewidth=1, color='darkseagreen', label='Channel 1')

    ax[4].set_xlim(0, Freqs[-1])
    ax[4].legend()
    ax[4].set_title('Power spectrum')
    set_xlabel(ax[4], 'Frequency', 'Hz')
    set_ylabel(ax[4], 'S($f$)', 'a.u.')

    fig.suptitle('Optimal integration weights {}'.format(qubit_label),
                 y=1.05, fontsize=16)

    fig.tight_layout()


def plot_mux_transients_optimal(Time,
                                I_g, I_e,
                                Q_g, Q_e,
                                pulse_start, pulse_stop,
                                qubit_label,
                                ax, **kw):

    fig = ax[0].get_figure()

    for axis in ax:
        axis.axvline(pulse_start, ls='--', color='black', linewidth=1)
        axis.axvline(pulse_stop, ls='--', color='black', linewidth=1)
        axis.axvspan(pulse_start, pulse_stop, alpha=0.15, color='yellow')

    ax[0].plot(Time, I_g, 'C0', linewidth=1, label='ground')
    ax[2].plot(Time, I_e, 'indianred', linewidth=1, label='excited')
    ax[1].plot(Time, Q_g, 'C0', linewidth=1, label='ground')
    ax[3].plot(Time, Q_e, 'indianred', linewidth=1, label='excited')

    ax[0].set_xlim(left=0, right=Time[-1])
    ax[1].set_xlim(left=0, right=Time[-1])
    ax[2].set_xlim(left=0, right=Time[-1])
    ax[3].set_xlim(left=0, right=Time[-1])

    ax[0].set_title('Channel 0')
    ax[1].set_title('Channel 1')
    set_xlabel(ax[0], 'Time', 's')
    set_xlabel(ax[1], 'Time', 's')
    set_xlabel(ax[2], 'Time', 's')
    set_xlabel(ax[3], 'Time', 's')
    set_ylabel(ax[0], 'Amplitude', 'a.u.')
    set_ylabel(ax[2], 'Amplitude', 'a.u.')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    fig.suptitle('Multiplexed transients {}'.format(qubit_label),
                 y=1.05, fontsize=16)

    fig.tight_layout()



#import os
#import matplotlib.pylab as pl
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
#import numpy as np
#from collections import OrderedDict
#import pycqed.analysis_v2.base_analysis as ba
#from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
#from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
#    cmap_to_alpha, cmap_first_to_alpha
#import pycqed.analysis.tools.data_manipulation as dm_tools
#from pycqed.utilities.general import int2base
#import pycqed.measurement.hdf5_data as h5d
#import copy
#import lmfit
#from scipy.optimize import minimize
#from pycqed.analysis.fitting_models import ro_gauss, ro_CDF, ro_CDF_discr,\
#     gaussian_2D, gauss_2D_guess, gaussianCDF, ro_double_gauss_guess
#
#
#
#class Multiplexed_Readout_Analysis(ba.BaseDataAnalysis):
#    """
#    Multiplexed readout analysis.
#
#    Does data binning and creates histograms of data.
#    Threshold is auto determined as the mean of the data.
#    Used to construct a assignment probability matris.
#
#    WARNING: Not sure if post selection supports measurement
#    data in two quadratures. Should use optimal weights if
#    using post-selection.
#    """
#
#    def __init__(self, t_start: str = None, t_stop: str = None,
#                 label: str = '',
#                 options_dict: dict = None, extract_only: bool = False,
#                 extract_combinations: bool = False,
#                 post_selection: bool = False,
#                 post_selec_thresholds: list = None,
#                 q_target=None,
#                 auto=True):
#
#        super().__init__(t_start=t_start, t_stop=t_stop,
#                         label=label,
#                         options_dict=options_dict,
#                         extract_only=extract_only)
#
#        self.extract_combinations = extract_combinations
#        self.post_selection = post_selection
#        self.post_selec_thresholds = post_selec_thresholds
#        self.q_target = q_target
#        self.do_fitting = True
#        if auto:
#            self.run_analysis()
#
#    def extract_data(self):
#        """
#        This is a new style (sept 2019) data extraction.
#        This could at some point move to a higher level class.
#        """
#        self.get_timestamps()
#        self.timestamp = self.timestamps[0]
#
#        data_fp = get_datafilepath_from_timestamp(self.timestamp)
#        param_spec = {'data': ('Experimental Data/Data', 'dset'),
#                      'value_names': ('Experimental Data', 'attr:value_names')}
#
#        self.raw_data_dict = h5d.extract_pars_from_datafile(
#            data_fp, param_spec)
#
#        # Parts added to be compatible with base analysis data requirements
#        self.raw_data_dict['timestamps'] = self.timestamps
#        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
#
#    def process_data(self):
#
#        self.proc_data_dict = {}
#
#        Channels = self.raw_data_dict['value_names'] # Qubit UHF outputs
#        nr_qubits = len(Channels)
#        combinations = \
#            ['{:0{}b}'.format(i, nr_qubits) for i in range(2**nr_qubits)]
#        raw_shots = self.raw_data_dict['data'][:, 1:]
#        post_selection = self.post_selection
#        qubit_labels = [ch.decode('utf-8').rsplit(' ', 1)[1] for ch in Channels]
#        self.proc_data_dict['combinations'] = combinations
#        self.proc_data_dict['qubit_labels'] = qubit_labels
#
#        #############################################
#        # Sort post-selection from measurement shots
#        #############################################
#        self.proc_data_dict['Shots'] = {ch : {} for ch in Channels}
#
#        if post_selection == True:
#            # Post-selected shots
#            self.proc_data_dict['Post_selected_shots'] =\
#                {ch : {} for ch in Channels}
#            # Pre-measurement shots
#            self.proc_data_dict['Pre_measurement_shots'] =\
#                {ch : {} for ch in Channels}
#
#        # Loop over all qubits
#        for i, ch in enumerate(Channels):
#            ch_shots = raw_shots[:, i]
#
#            # Loop over prepared states
#            for j, comb in enumerate(combinations):
#                if post_selection == False:
#                    shots = ch_shots[j::len(combinations)]
#                    self.proc_data_dict['Shots'][ch][comb] = shots.copy()
#                else:
#                    pre_meas_shots = ch_shots[2*j::len(combinations)*2]
#                    shots = ch_shots[2*j+1::len(combinations)*2]
#                    self.proc_data_dict['Shots'][ch][comb] = shots.copy()
#                    self.proc_data_dict['Post_selected_shots'][ch][comb] =\
#                        shots.copy()
#                    self.proc_data_dict['Pre_measurement_shots'][ch][comb] =\
#                        pre_meas_shots.copy()
#
#        #########################
#        # Execute post_selection
#        #########################
#        if post_selection == True:
#            for comb in combinations: # Loop over prepared states
#                Idxs = []
#                # For each prepared state one needs to eliminate every shot
#                # if a single qubit fails post selection.
#                for i, ch in enumerate(Channels): # Loop over qubits
#                    # First, find all idxs for all qubits. This has to loop
#                    # over alll qubits before in pre-measurement.
#                    pre_meas_shots =\
#                        self.proc_data_dict['Pre_measurement_shots'][ch][comb]
#                    post_select_indices = dm_tools.get_post_select_indices(
#                        thresholds=[self.post_selec_thresholds[i]],
#                        init_measurements=[pre_meas_shots])
#                    Idxs += list(post_select_indices)
#
#                for i, ch in enumerate(Channels): # Loop over qubits
#                    # Now that we have all idxs, we can discard the shots that
#                    # failed in every qubit.
#                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
#                    shots[Idxs] = np.nan # signal post_selection with nan
#                    shots = shots[~np.isnan(shots)] # discard post failed shots
#                    self.proc_data_dict['Post_selected_shots'][ch][comb] = shots
#
#        ############################################
#        # Histograms, thresholds and digitized data
#        ############################################
#        self.proc_data_dict['Histogram_data'] = {ch : {} for ch in Channels}
#        self.proc_data_dict['PDF_data'] = {ch : {} for ch in Channels}
#        self.proc_data_dict['CDF_data'] = {ch : {} for ch in Channels}
#        Shots_digitized = {ch : {} for ch in Channels}
#        if post_selection == True:
#            self.proc_data_dict['Post_Histogram_data'] = \
#                {ch : {} for ch in Channels}
#            self.proc_data_dict['Post_PDF_data'] = {ch : {} for ch in Channels}
#            self.proc_data_dict['Post_CDF_data'] = {ch : {} for ch in Channels}
#            Post_Shots_digitized = {ch : {} for ch in Channels}
#
#        for i, ch in enumerate(Channels):
#            hist_range = (np.amin(raw_shots[:, i]), np.amax(raw_shots[:, i]))
#            Shots_0 = [] # used to store overall shots of a qubit
#            Shots_1 = []
#            if post_selection == True:
#                Post_Shots_0 = [] # used to store overall shots of a qubit
#                Post_Shots_1 = []
#
#            # Histograms
#            for comb in combinations:
#                if post_selection == True:
#                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
#                    # Hitogram data of each prepared_state
#                    counts, bin_edges = np.histogram(shots, bins=100,
#                                                     range=hist_range)
#                    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
#                    self.proc_data_dict['Post_Histogram_data'][ch][comb]=\
#                        (counts, bin_centers)
#                    if comb[i] == '0':
#                        Post_Shots_0 = np.concatenate((Post_Shots_0, shots))
#                    else:
#                        Post_Shots_1 = np.concatenate((Post_Shots_1, shots))
#
#                shots = self.proc_data_dict['Shots'][ch][comb]
#                # Hitogram data of each prepared_state
#                counts, bin_edges = np.histogram(shots, bins=100,
#                                                 range=hist_range)
#                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
#                self.proc_data_dict['Histogram_data'][ch][comb] = \
#                    (counts, bin_centers)
#
#                if comb[i] == '0':
#                    Shots_0 = np.concatenate((Shots_0, shots))
#                else:
#                    Shots_1 = np.concatenate((Shots_1, shots))
#
#            # Cumulative sums
#            if post_selection == True:
#                # bin data according to unique bins
#                ubins_0, ucounts_0 = np.unique(Post_Shots_0, return_counts=True)
#                ubins_1, ucounts_1 = np.unique(Post_Shots_1, return_counts=True)
#                ucumsum_0 = np.cumsum(ucounts_0)
#                ucumsum_1 = np.cumsum(ucounts_1)
#                # merge |0> and |1> shot bins
#                all_bins = np.unique(np.sort(np.concatenate((ubins_0, ubins_1))))
#                # interpolate cumsum for all bins
#                int_cumsum_0=np.interp(x=all_bins,xp=ubins_0,fp=ucumsum_0,left=0)
#                int_cumsum_1=np.interp(x=all_bins,xp=ubins_1,fp=ucumsum_1,left=0)
#                norm_cumsum_0 = int_cumsum_0/np.max(int_cumsum_0)
#                norm_cumsum_1 = int_cumsum_1/np.max(int_cumsum_1)
#                self.proc_data_dict['Post_CDF_data'][ch]['cumsum_x_ds']=all_bins
#                self.proc_data_dict['Post_CDF_data'][ch]['cumsum_y_ds'] = \
#                    [int_cumsum_0, int_cumsum_1]
#                self.proc_data_dict['Post_CDF_data'][ch]['cumsum_y_ds_n'] = \
#                    [norm_cumsum_0, norm_cumsum_1]
#                # Calculating threshold
#                F_vs_th = (1-(1-abs(norm_cumsum_0-norm_cumsum_1))/2)
#                opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
#                opt_idx = int(round(np.average(opt_idxs)))
#                self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_raw'] = \
#                    F_vs_th[opt_idx]
#                self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'] = \
#                    all_bins[opt_idx]
#            # bin data according to unique bins
#            ubins_0, ucounts_0 = np.unique(Shots_0, return_counts=True)
#            ubins_1, ucounts_1 = np.unique(Shots_1, return_counts=True)
#            ucumsum_0 = np.cumsum(ucounts_0)
#            ucumsum_1 = np.cumsum(ucounts_1)
#            # merge |0> and |1> shot bins
#            all_bins = np.unique(np.sort(np.concatenate((ubins_0, ubins_1))))
#            # interpolate cumsum for all bins
#            int_cumsum_0 = np.interp(x=all_bins,xp=ubins_0,fp=ucumsum_0,left=0)
#            int_cumsum_1 = np.interp(x=all_bins,xp=ubins_1,fp=ucumsum_1,left=0)
#            norm_cumsum_0 = int_cumsum_0/np.max(int_cumsum_0)
#            norm_cumsum_1 = int_cumsum_1/np.max(int_cumsum_1)
#            self.proc_data_dict['CDF_data'][ch]['cumsum_x_ds'] = all_bins
#            self.proc_data_dict['CDF_data'][ch]['cumsum_y_ds'] = \
#                [int_cumsum_0, int_cumsum_1]
#            self.proc_data_dict['CDF_data'][ch]['cumsum_y_ds_n'] = \
#                [norm_cumsum_0, norm_cumsum_1]
#            # Calculating threshold
#            F_vs_th = (1-(1-abs(norm_cumsum_0-norm_cumsum_1))/2)
#            opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
#            opt_idx = int(round(np.average(opt_idxs)))
#            self.proc_data_dict['PDF_data'][ch]['F_assignment_raw'] = \
#                F_vs_th[opt_idx]
#            self.proc_data_dict['PDF_data'][ch]['threshold_raw'] = \
#                all_bins[opt_idx]
#            
#            # Histogram of overall shots
#            if post_selection == True:
#                counts_0, bin_edges = np.histogram(Post_Shots_0, bins=100,
#                                                   range=hist_range)
#                counts_1, bin_edges = np.histogram(Post_Shots_1, bins=100,
#                                                   range=hist_range)
#                bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
#                self.proc_data_dict['Post_PDF_data'][ch]['0'] = \
#                    (counts_0, bin_centers)
#                self.proc_data_dict['Post_PDF_data'][ch]['1'] = \
#                    (counts_1, bin_centers)
#            counts_0, bin_edges = np.histogram(Shots_0, bins=100,
#                                               range=hist_range)
#            counts_1, bin_edges = np.histogram(Shots_1, bins=100,
#                                               range=hist_range)
#            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
#            self.proc_data_dict['PDF_data'][ch]['0'] = \
#                (counts_0, bin_centers)
#            self.proc_data_dict['PDF_data'][ch]['1'] = \
#                (counts_1, bin_centers)
#
#            # Digitized data
#            for comb in combinations:
#                if post_selection == True:
#                    shots = self.proc_data_dict['Post_selected_shots'][ch][comb]
#                    th = self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw']
#                    Post_Shots_digitized[ch][comb] = \
#                        np.array(shots > th, dtype=int)
#                shots = self.proc_data_dict['Shots'][ch][comb]
#                th = self.proc_data_dict['PDF_data'][ch]['threshold_raw']
#                Shots_digitized[ch][comb] = \
#                    np.array(shots > th, dtype=int)
#
#        ##########################################
#        # Calculate assignment probability matrix
#        ##########################################
#        if post_selection == True:
#            ass_prob_matrix = calc_assignment_prob_matrix(combinations,
#                Post_Shots_digitized)
#            cross_fid_matrix = calc_cross_fidelity_matrix(combinations,
#                ass_prob_matrix)
#            self.proc_data_dict['Post_assignment_prob_matrix'] = ass_prob_matrix
#            self.proc_data_dict['Post_cross_fidelity_matrix'] = cross_fid_matrix
#        assignment_prob_matrix = calc_assignment_prob_matrix(combinations,
#            Shots_digitized)
#        cross_fidelity_matrix = calc_cross_fidelity_matrix(combinations,
#            assignment_prob_matrix)
#        self.proc_data_dict['assignment_prob_matrix'] = assignment_prob_matrix
#        self.proc_data_dict['cross_fidelity_matrix'] = cross_fidelity_matrix
#
#    def prepare_fitting(self):
#        Channels = self.raw_data_dict['value_names']
#        self.fit_dicts = OrderedDict()
#        for ch in Channels:
#            ###################################
#            # Histograms fit (PDF)
#            ###################################
#            if self.post_selection == True:
#                bin_x = self.proc_data_dict['Post_PDF_data'][ch]['0'][1]
#                bin_xs = [bin_x, bin_x]
#                bin_ys = [self.proc_data_dict['Post_PDF_data'][ch]['0'][0],
#                          self.proc_data_dict['Post_PDF_data'][ch]['1'][0]]
#                m = lmfit.model.Model(ro_gauss)
#                m.guess = ro_double_gauss_guess.__get__(m, m.__class__)
#                params = m.guess(x=bin_xs, data=bin_ys,
#                         fixed_p01=self.options_dict.get('fixed_p01', False),
#                         fixed_p10=self.options_dict.get('fixed_p10', False))
#                post_res = m.fit(x=bin_xs, data=bin_ys, params=params)
#                self.fit_dicts['Post_PDF_fit_{}'.format(ch)] = {
#                    'model': m,
#                    'fit_xvals': {'x': bin_xs},
#                    'fit_yvals': {'data': bin_ys},
#                    'guessfn_pars':
#                        {'fixed_p01':self.options_dict.get('fixed_p01', False),
#                         'fixed_p10':self.options_dict.get('fixed_p10', False)},
#                }
#            bin_x = self.proc_data_dict['PDF_data'][ch]['0'][1]
#            bin_xs = [bin_x, bin_x]
#            bin_ys = [self.proc_data_dict['PDF_data'][ch]['0'][0],
#                      self.proc_data_dict['PDF_data'][ch]['1'][0]]
#            m = lmfit.model.Model(ro_gauss)
#            m.guess = ro_double_gauss_guess.__get__(m, m.__class__)
#            params = m.guess(x=bin_xs, data=bin_ys,
#                     fixed_p01=self.options_dict.get('fixed_p01', False),
#                     fixed_p10=self.options_dict.get('fixed_p10', False))
#            res = m.fit(x=bin_xs, data=bin_ys, params=params)
#            self.fit_dicts['PDF_fit_{}'.format(ch)] = {
#                'model': m,
#                'fit_xvals': {'x': bin_xs},
#                'fit_yvals': {'data': bin_ys},
#                'guessfn_pars':
#                    {'fixed_p01': self.options_dict.get('fixed_p01', False),
#                     'fixed_p10': self.options_dict.get('fixed_p10', False)},
#            }
#            ###################################
#            #  Fit the CDF                    #
#            ###################################
#            if self.post_selection == True:
#                m_cul = lmfit.model.Model(ro_CDF)
#                cdf_xs = self.proc_data_dict['Post_CDF_data'][ch]['cumsum_x_ds']
#                cdf_xs = [np.array(cdf_xs), np.array(cdf_xs)]
#                cdf_ys = self.proc_data_dict['Post_CDF_data'][ch]['cumsum_y_ds']
#                cdf_ys = [np.array(cdf_ys[0]), np.array(cdf_ys[1])]
#
#                cum_params = post_res.params
#                cum_params['A_amplitude'].value = np.max(cdf_ys[0])
#                cum_params['A_amplitude'].vary = False
#                cum_params['B_amplitude'].value = np.max(cdf_ys[1])
#                cum_params['A_amplitude'].vary = False # FIXME: check if correct
#                self.fit_dicts['Post_CDF_fit_{}'.format(ch)] = {
#                    'model': m_cul,
#                    'fit_xvals': {'x': cdf_xs},
#                    'fit_yvals': {'data': cdf_ys},
#                    'guess_pars': cum_params,
#                }
#            m_cul = lmfit.model.Model(ro_CDF)
#            cdf_xs = self.proc_data_dict['CDF_data'][ch]['cumsum_x_ds']
#            cdf_xs = [np.array(cdf_xs), np.array(cdf_xs)]
#            cdf_ys = self.proc_data_dict['CDF_data'][ch]['cumsum_y_ds']
#            cdf_ys = [np.array(cdf_ys[0]), np.array(cdf_ys[1])]
#
#            cum_params = res.params
#            cum_params['A_amplitude'].value = np.max(cdf_ys[0])
#            cum_params['A_amplitude'].vary = False
#            cum_params['B_amplitude'].value = np.max(cdf_ys[1])
#            cum_params['A_amplitude'].vary = False # FIXME: check if correct
#            self.fit_dicts['CDF_fit_{}'.format(ch)] = {
#                'model': m_cul,
#                'fit_xvals': {'x': cdf_xs},
#                'fit_yvals': {'data': cdf_ys},
#                'guess_pars': cum_params,
#            }
#
#    def analyze_fit_results(self):
#        '''
#        This code was taken from single shot readout analysis and adapted to
#        mux readout (April 2020).
#        '''
#        Channels = self.raw_data_dict['value_names']
#        self.proc_data_dict['quantities_of_interest'] = \
#            {ch : {} for ch in Channels}
#        if self.post_selection == True:
#            self.proc_data_dict['post_quantities_of_interest'] = \
#                {ch : {} for ch in Channels}
#        self.qoi = {ch : {} for ch in Channels}
#        for ch in Channels:
#            if self.post_selection == True:
#                # Create a CDF based on the fit functions of both fits.
#                post_fr = self.fit_res['Post_CDF_fit_{}'.format(ch)]
#                post_bv = post_fr.best_values
#                # best values new
#                post_bvn = copy.deepcopy(post_bv)
#                post_bvn['A_amplitude'] = 1
#                post_bvn['B_amplitude'] = 1
#                def CDF(x):
#                    return ro_CDF(x=x, **post_bvn)
#                def CDF_0(x):
#                    return CDF(x=[x, x])[0]
#                def CDF_1(x):
#                    return CDF(x=[x, x])[1]
#                def infid_vs_th(x):
#                    cdf = ro_CDF(x=[x, x], **post_bvn)
#                    return (1-np.abs(cdf[0] - cdf[1]))/2
#                self._CDF_0 = CDF_0
#                self._CDF_1 = CDF_1
#                self._infid_vs_th = infid_vs_th
#                post_thr_guess = (3*post_bv['B_center'] - post_bv['A_center'])/2
#                opt_fid = minimize(infid_vs_th, post_thr_guess)
#                # for some reason the fit sometimes returns a list of values
#                if isinstance(opt_fid['fun'], float):
#                    self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_fit']=\
#                        (1-opt_fid['fun'])
#                else:
#                    self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_fit']=\
#                        (1-opt_fid['fun'])[0]
#                self.proc_data_dict['Post_PDF_data'][ch]['threshold_fit']=\
#                    opt_fid['x'][0]
#            # Create a CDF based on the fit functions of both fits.
#            fr = self.fit_res['CDF_fit_{}'.format(ch)]
#            bv = fr.best_values
#            # best values new
#            bvn = copy.deepcopy(bv)
#            bvn['A_amplitude'] = 1
#            bvn['B_amplitude'] = 1
#            def CDF(x):
#                return ro_CDF(x=x, **bvn)
#            def CDF_0(x):
#                return CDF(x=[x, x])[0]
#            def CDF_1(x):
#                return CDF(x=[x, x])[1]
#            def infid_vs_th(x):
#                cdf = ro_CDF(x=[x, x], **bvn)
#                return (1-np.abs(cdf[0] - cdf[1]))/2
#            self._CDF_0 = CDF_0
#            self._CDF_1 = CDF_1
#            self._infid_vs_th = infid_vs_th
#            thr_guess = (3*bv['B_center'] - bv['A_center'])/2
#            opt_fid = minimize(infid_vs_th, thr_guess)
#            # for some reason the fit sometimes returns a list of values
#            if isinstance(opt_fid['fun'], float):
#                self.proc_data_dict['PDF_data'][ch]['F_assignment_fit'] = \
#                    (1-opt_fid['fun'])
#            else:
#                self.proc_data_dict['PDF_data'][ch]['F_assignment_fit'] = \
#                    (1-opt_fid['fun'])[0]
#            self.proc_data_dict['PDF_data'][ch]['threshold_fit'] = \
#                opt_fid['x'][0]
#
#            # Calculate the fidelity of both
#            ###########################################
#            #  Extracting the discrimination fidelity #
#            ###########################################
#            if self.post_selection == True:
#                def CDF_0_discr(x):
#                    return gaussianCDF(x, amplitude=1,
#                            mu=post_bv['A_center'], sigma=post_bv['A_sigma'])
#                def CDF_1_discr(x):
#                    return gaussianCDF(x, amplitude=1,
#                            mu=post_bv['B_center'], sigma=post_bv['B_sigma'])
#                def disc_infid_vs_th(x):
#                    cdf0 = gaussianCDF(x, amplitude=1, mu=post_bv['A_center'],
#                                       sigma=post_bv['A_sigma'])
#                    cdf1 = gaussianCDF(x, amplitude=1, mu=post_bv['B_center'],
#                                       sigma=post_bv['B_sigma'])
#                    return (1-np.abs(cdf0 - cdf1))/2
#                self._CDF_0_discr = CDF_0_discr
#                self._CDF_1_discr = CDF_1_discr
#                self._disc_infid_vs_th = disc_infid_vs_th
#                opt_fid_discr = minimize(disc_infid_vs_th, post_thr_guess)
#                # for some reason the fit sometimes returns a list of values
#                if isinstance(opt_fid_discr['fun'], float):
#                    self.proc_data_dict['Post_PDF_data'][ch]['F_discr'] = \
#                        (1-opt_fid_discr['fun'])
#                else:
#                    self.proc_data_dict['Post_PDF_data'][ch]['F_discr'] = \
#                        (1-opt_fid_discr['fun'])[0]
#                self.proc_data_dict['Post_PDF_data'][ch]['threshold_discr'] = \
#                    opt_fid_discr['x'][0]
#                post_fr = self.fit_res['Post_CDF_fit_{}'.format(ch)]
#                post_bv = post_fr.params
#                # self.proc_data_dict['PDF_data'][ch]['residual_excitation'] = \
#                #     bv['A_spurious'].value
#                # self.proc_data_dict['PDF_data'][ch]['relaxation_events'] = \
#                #     bv['B_spurious'].value
#                A_amp = post_bv['A_spurious'].value
#                A_sig = post_bv['A_sigma'].value
#                B_amp = post_bv['B_spurious'].value
#                B_sig = post_bv['B_sigma'].value
#                residual_excitation=A_amp*B_sig/((1-A_amp)*A_sig + A_amp*B_sig)
#                relaxation_events = B_amp*A_sig/((1-B_amp)*B_sig + B_amp*A_sig)
#                self.proc_data_dict['Post_PDF_data'][ch]['residual_excitation']=\
#                    residual_excitation
#                self.proc_data_dict['Post_PDF_data'][ch]['relaxation_events']=\
#                    relaxation_events
#            # No post-selection
#            def CDF_0_discr(x):
#                return gaussianCDF(x, amplitude=1,
#                                   mu=bv['A_center'], sigma=bv['A_sigma'])
#            def CDF_1_discr(x):
#                return gaussianCDF(x, amplitude=1,
#                                   mu=bv['B_center'], sigma=bv['B_sigma'])
#            def disc_infid_vs_th(x):
#                cdf0 = gaussianCDF(x, amplitude=1, mu=bv['A_center'],
#                                   sigma=bv['A_sigma'])
#                cdf1 = gaussianCDF(x, amplitude=1, mu=bv['B_center'],
#                                   sigma=bv['B_sigma'])
#                return (1-np.abs(cdf0 - cdf1))/2
#            self._CDF_0_discr = CDF_0_discr
#            self._CDF_1_discr = CDF_1_discr
#            self._disc_infid_vs_th = disc_infid_vs_th
#            opt_fid_discr = minimize(disc_infid_vs_th, thr_guess)
#            # for some reason the fit sometimes returns a list of values
#            if isinstance(opt_fid_discr['fun'], float):
#                self.proc_data_dict['PDF_data'][ch]['F_discr'] = \
#                    (1-opt_fid_discr['fun'])
#            else:
#                self.proc_data_dict['PDF_data'][ch]['F_discr'] = \
#                    (1-opt_fid_discr['fun'])[0]
#            self.proc_data_dict['PDF_data'][ch]['threshold_discr'] =\
#                opt_fid_discr['x'][0]
#            fr = self.fit_res['CDF_fit_{}'.format(ch)]
#            bv = fr.params
#            # self.proc_data_dict['PDF_data'][ch]['residual_excitation'] = \
#            #     bv['A_spurious'].value
#            # self.proc_data_dict['PDF_data'][ch]['relaxation_events'] = \
#            #     bv['B_spurious'].value
#            A_amp = bv['A_spurious'].value
#            A_sig = bv['A_sigma'].value
#            B_amp = bv['B_spurious'].value
#            B_sig = bv['B_sigma'].value
#            residual_excitation = A_amp*B_sig/((1-A_amp)*A_sig + A_amp*B_sig)
#            relaxation_events = B_amp*A_sig/((1-B_amp)*B_sig + B_amp*A_sig)
#            self.proc_data_dict['PDF_data'][ch]['residual_excitation'] = \
#                residual_excitation
#            self.proc_data_dict['PDF_data'][ch]['relaxation_events'] = \
#                relaxation_events
#
#            ###################################
#            #  Save quantities of interest.   #
#            ###################################
#            if self.post_selection == True:
#                self.proc_data_dict['post_quantities_of_interest'][ch] = {
#                    'Post_SNR': \
#                self.fit_res['Post_CDF_fit_{}'.format(ch)].params['SNR'].value,
#                    'Post_F_d': \
#                self.proc_data_dict['Post_PDF_data'][ch]['F_discr'],
#                    'Post_F_a': \
#                self.proc_data_dict['Post_PDF_data'][ch]['F_assignment_raw'],
#                    'Post_residual_excitation': \
#                self.proc_data_dict['Post_PDF_data'][ch]['residual_excitation'],
#                    'Post_relaxation_events':
#                self.proc_data_dict['Post_PDF_data'][ch]['relaxation_events'],
#                    'Post_threshold_raw': \
#                    self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
#                    'Post_threshold_discr': \
#                    self.proc_data_dict['Post_PDF_data'][ch]['threshold_discr']
#                }
#            self.proc_data_dict['quantities_of_interest'][ch] = {
#                'SNR': \
#                    self.fit_res['CDF_fit_{}'.format(ch)].params['SNR'].value,
#                'F_d': self.proc_data_dict['PDF_data'][ch]['F_discr'],
#                'F_a': self.proc_data_dict['PDF_data'][ch]['F_assignment_raw'],
#                'residual_excitation': \
#                    self.proc_data_dict['PDF_data'][ch]['residual_excitation'],
#                'relaxation_events':
#                    self.proc_data_dict['PDF_data'][ch]['relaxation_events'],
#                'threshold_raw': \
#                    self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
#                'threshold_discr': \
#                    self.proc_data_dict['PDF_data'][ch]['threshold_discr']
#            }
#            self.qoi[ch] = self.proc_data_dict['quantities_of_interest'][ch]
#            if self.post_selection == True:
#                self.qoi[ch].update(self.proc_data_dict['post_quantities_of_interest'][ch])
#    
#    def prepare_plots(self):
#
#        Channels = self.raw_data_dict['value_names']
#        nr_qubits = len(Channels)
#        qubit_labels = self.proc_data_dict['qubit_labels']
#        combinations = \
#            ['{:0{}b}'.format(i, nr_qubits) for i in range(2**nr_qubits)]
#        self.axs_dict = {}
#
#        if self.q_target == None:
#            # Run analysis for all qubits
#            if self.post_selection is True:
#                self.plot_dicts['assignment_probability_matrix_post'] = {
#                    'plotfn': plot_assignment_prob_matrix,
#                    'assignment_prob_matrix':
#                        self.proc_data_dict['Post_assignment_prob_matrix'],
#                    'combinations': self.proc_data_dict['combinations'],
#                    'valid_combinations': self.proc_data_dict['combinations'],
#                    'qubit_labels': qubit_labels,
#                    'plotsize': np.array(np.shape(\
#                    self.proc_data_dict['Post_assignment_prob_matrix'].T))*.8,
#                    'post_selection': True
#                    }
#                self.plot_dicts['cross_fid_matrix_post'] = {
#                    'plotfn': plot_cross_fid_matrix,
#                    'prob_matrix':
#                        self.proc_data_dict['Post_cross_fidelity_matrix'],
#                    'combinations': qubit_labels,
#                    'valid_combinations': qubit_labels,
#                    'qubit_labels': qubit_labels,
#                    'plotsize': np.array(np.shape(\
#                    self.proc_data_dict['Post_cross_fidelity_matrix'].T))*.8,
#                    'post_selection': True
#                    }
#            self.plot_dicts['assignment_probability_matrix'] = {
#                'plotfn': plot_assignment_prob_matrix,
#                'assignment_prob_matrix':
#                    self.proc_data_dict['assignment_prob_matrix'],
#                'combinations': self.proc_data_dict['combinations'],
#                'valid_combinations': self.proc_data_dict['combinations'],
#                'qubit_labels': qubit_labels,
#                'plotsize': np.array(np.shape(\
#                    self.proc_data_dict['assignment_prob_matrix'].T))*.8
#                }
#            self.plot_dicts['cross_fid_matrix'] = {
#                'plotfn': plot_cross_fid_matrix,
#                'prob_matrix':
#                    self.proc_data_dict['cross_fidelity_matrix'],
#                'combinations': qubit_labels,
#                'valid_combinations': qubit_labels,
#                'qubit_labels': qubit_labels,
#                'plotsize': np.array(np.shape(\
#                    self.proc_data_dict['cross_fidelity_matrix'].T))*.8
#                }
#            for i, ch in enumerate(Channels):
#                qubit_label = qubit_labels[i]
#                # Totalized shots
#                if self.post_selection == True:
#                    fig, axs = plt.subplots(nrows=2, ncols=3, 
#                                            figsize=(13,8), dpi=200)
#                    axs = axs.ravel()
#                else:
#                    fig, axs = plt.subplots(ncols=3, figsize=(13,4), dpi=200)
#                fig.patch.set_alpha(0)
#                self.axs_dict['mux_ssro_totalshots_{}'.format(qubit_label)]=axs
#                self.figs['mux_ssro_totalshots_{}'.format(qubit_label)] = fig
#                if self.post_selection == True:
#                    self.plot_dicts['post_mux_ssro_totalshots_{}'.format(qubit_label)]={
#                        'plotfn': plot_single_qubit_histogram,
#                        'data': self.proc_data_dict['Post_PDF_data'][ch],
#                        'qubit_label': qubit_label,
#                        'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
#                        'para_hist' : \
#                        self.fit_res['Post_PDF_fit_{}'.format(ch)].best_values,
#                        'para_cdf' : \
#                        self.fit_res['Post_CDF_fit_{}'.format(ch)].best_values,
#                        'hist_data': \
#                        self.proc_data_dict['Post_Histogram_data'][ch],
#                        'qubit_idx': i,
#                        'value_name': ch,
#                        'combinations': combinations,
#                        'qubit_labels': qubit_labels,
#                        'threshold': \
#                        self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
#                        'timestamp': self.timestamp,
#                        'qoi': self.qoi[ch],
#                        'post_selection': True
#                    }
#                    self.plot_dicts['post_mux_ssro_cdf_{}'.format(qubit_label)]={
#                        'plotfn': plot_single_qubit_CDF,
#                        'data': self.proc_data_dict['Post_PDF_data'][ch],
#                        'qubit_label': qubit_label,
#                        'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
#                        'para_hist' : \
#                        self.fit_res['Post_PDF_fit_{}'.format(ch)].best_values,
#                        'para_cdf' : \
#                        self.fit_res['Post_CDF_fit_{}'.format(ch)].best_values,
#                        'hist_data': \
#                        self.proc_data_dict['Post_Histogram_data'][ch],
#                        'qubit_idx': i,
#                        'value_name': ch,
#                        'combinations': combinations,
#                        'qubit_labels': qubit_labels,
#                        'threshold': \
#                        self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
#                        'timestamp': self.timestamp,
#                        'qoi': self.qoi[ch],
#                        'post_selection': True
#                    }
#                    self.plot_dicts['post_mux_ssro_crosstalk_{}'.format(qubit_label)]={
#                        'plotfn': plot_single_qubit_crosstalk,
#                        'data': self.proc_data_dict['Post_PDF_data'][ch],
#                        'qubit_label': qubit_label,
#                        'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
#                        'para_hist' : \
#                        self.fit_res['Post_PDF_fit_{}'.format(ch)].best_values,
#                        'para_cdf' : \
#                        self.fit_res['Post_CDF_fit_{}'.format(ch)].best_values,
#                        'hist_data': \
#                        self.proc_data_dict['Post_Histogram_data'][ch],
#                        'qubit_idx': i,
#                        'value_name': ch,
#                        'combinations': combinations,
#                        'qubit_labels': qubit_labels,
#                        'threshold': \
#                        self.proc_data_dict['Post_PDF_data'][ch]['threshold_raw'],
#                        'timestamp': self.timestamp,
#                        'qoi': self.qoi[ch],
#                        'post_selection': True
#                    }
#                self.plot_dicts['mux_ssro_totalshots_{}'.format(qubit_label)]={
#                    'plotfn': plot_single_qubit_histogram,
#                    'data': self.proc_data_dict['PDF_data'][ch],
#                    'qubit_label': qubit_label,
#                    'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
#                    'para_hist' : \
#                        self.fit_res['PDF_fit_{}'.format(ch)].best_values,
#                    'para_cdf' : \
#                        self.fit_res['CDF_fit_{}'.format(ch)].best_values,
#                    'hist_data': self.proc_data_dict['Histogram_data'][ch],
#                    'qubit_idx': i,
#                    'value_name': ch,
#                    'combinations': combinations,
#                    'qubit_labels': qubit_labels,
#                    'threshold': \
#                        self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
#                    'timestamp': self.timestamp,
#                    'qoi': self.qoi[ch]
#                }
#                self.plot_dicts['mux_ssro_cdf_{}'.format(qubit_label)]={
#                    'plotfn': plot_single_qubit_CDF,
#                    'data': self.proc_data_dict['PDF_data'][ch],
#                    'qubit_label': qubit_label,
#                    'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
#                    'para_hist' : \
#                        self.fit_res['PDF_fit_{}'.format(ch)].best_values,
#                    'para_cdf' : \
#                        self.fit_res['CDF_fit_{}'.format(ch)].best_values,
#                    'hist_data': self.proc_data_dict['Histogram_data'][ch],
#                    'qubit_idx': i,
#                    'value_name': ch,
#                    'combinations': combinations,
#                    'qubit_labels': qubit_labels,
#                    'threshold': \
#                        self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
#                    'timestamp': self.timestamp,
#                    'qoi': self.qoi[ch]
#                }
#                self.plot_dicts['mux_ssro_crosstalk_{}'.format(qubit_label)]={
#                    'plotfn': plot_single_qubit_crosstalk,
#                    'data': self.proc_data_dict['PDF_data'][ch],
#                    'qubit_label': qubit_label,
#                    'ax_id': 'mux_ssro_totalshots_{}'.format(qubit_label),
#                    'para_hist' : \
#                        self.fit_res['PDF_fit_{}'.format(ch)].best_values,
#                    'para_cdf' : \
#                        self.fit_res['CDF_fit_{}'.format(ch)].best_values,
#                    'hist_data': self.proc_data_dict['Histogram_data'][ch],
#                    'qubit_idx': i,
#                    'value_name': ch,
#                    'combinations': combinations,
#                    'qubit_labels': qubit_labels,
#                    'threshold': \
#                        self.proc_data_dict['PDF_data'][ch]['threshold_raw'],
#                    'timestamp': self.timestamp,
#                    'qoi': self.qoi[ch]
#                }
#
#        else:
#            # Run analysis on q_target only
#            q_target_idx = qubit_labels.index(self.q_target)
#            q_target_ch = Channels[q_target_idx]
#            if self.post_selection is True:
#                fig1, ax1 = plt.subplots(figsize=(5,4), dpi=200)
#                fig1.patch.set_alpha(0)
#                self.axs_dict['mux_ssro_histogram_{}_post'.format(self.q_target)]=ax1
#                self.figs['mux_ssro_histogram_{}_post'.format(self.q_target)]=fig1
#                self.plot_dicts['mux_ssro_histogram_{}_post'.format(self.q_target)]={
#                    'plotfn': plot_single_qubit_histogram,
#                    'data': self.proc_data_dict['Post_PDF_data'][q_target_ch],
#                    'qubit_label': self.q_target,
#                    'ax_id': 'mux_ssro_histogram_{}_post'.format(self.q_target),
#                    'para_hist' : \
#                    self.fit_res['Post_PDF_fit_{}'.format(q_target_ch)].best_values,
#                    'para_cdf' : \
#                    self.fit_res['Post_CDF_fit_{}'.format(q_target_ch)].best_values,
#                    'hist_data': \
#                    self.proc_data_dict['Post_Histogram_data'][q_target_ch],
#                    'qubit_idx': q_target_idx,
#                    'value_name': q_target_ch,
#                    'combinations': combinations,
#                    'qubit_labels': qubit_labels,
#                    'threshold': \
#                    self.proc_data_dict['Post_PDF_data'][q_target_ch]['threshold_raw'],
#                    'timestamp': self.timestamp,
#                    'qoi': self.qoi[q_target_ch],
#                    'post_selection':True
#                }
#                fig2, ax2 = plt.subplots(figsize=(5,4), dpi=200)
#                fig2.patch.set_alpha(0)
#                self.axs_dict['mux_ssro_cdf_{}_post'.format(self.q_target)]=ax2
#                self.figs['mux_ssro_cdf_{}_post'.format(self.q_target)]=fig2
#                self.plot_dicts['mux_ssro_cdf_{}_post'.format(self.q_target)]={
#                    'plotfn': plot_single_qubit_CDF,
#                    'data': self.proc_data_dict['Post_PDF_data'][q_target_ch],
#                    'qubit_label': self.q_target,
#                    'ax_id': 'mux_ssro_cdf_{}_post'.format(self.q_target),
#                    'para_hist' : \
#                    self.fit_res['Post_PDF_fit_{}'.format(q_target_ch)].best_values,
#                    'para_cdf' : \
#                    self.fit_res['Post_CDF_fit_{}'.format(q_target_ch)].best_values,
#                    'hist_data': \
#                    self.proc_data_dict['Post_Histogram_data'][q_target_ch],
#                    'qubit_idx': q_target_idx,
#                    'value_name': q_target_ch,
#                    'combinations': combinations,
#                    'qubit_labels': qubit_labels,
#                    'threshold': \
#                    self.proc_data_dict['Post_PDF_data'][q_target_ch]['threshold_raw'],
#                    'timestamp': self.timestamp,
#                    'qoi': self.qoi[q_target_ch],
#                    'post_selection': True
#                }
#                fig3, ax3 = plt.subplots(figsize=(5,4), dpi=200)
#                fig3.patch.set_alpha(0)
#                self.axs_dict['mux_ssro_crosstalk_{}_post'.format(self.q_target)]=ax3
#                self.figs['mux_ssro_crosstalk_{}_post'.format(self.q_target)]=fig3
#                self.plot_dicts['mux_ssro_crosstalk_{}_post'.format(self.q_target)]={
#                    'plotfn': plot_single_qubit_crosstalk,
#                    'data': self.proc_data_dict['Post_PDF_data'][q_target_ch],
#                    'qubit_label': self.q_target,
#                    'ax_id': 'mux_ssro_crosstalk_{}_post'.format(self.q_target),
#                    'para_hist' : \
#                    self.fit_res['Post_PDF_fit_{}'.format(q_target_ch)].best_values,
#                    'para_cdf' : \
#                    self.fit_res['Post_CDF_fit_{}'.format(q_target_ch)].best_values,
#                    'hist_data': \
#                    self.proc_data_dict['Post_Histogram_data'][q_target_ch],
#                    'qubit_idx': q_target_idx,
#                    'value_name': q_target_ch,
#                    'combinations': combinations,
#                    'qubit_labels': qubit_labels,
#                    'threshold': \
#                    self.proc_data_dict['Post_PDF_data'][q_target_ch]['threshold_raw'],
#                    'timestamp': self.timestamp,
#                    'qoi': self.qoi[q_target_ch],
#                    'post_selection':True
#                }
#            fig1, ax1 = plt.subplots(figsize=(5,4), dpi=200)
#            fig1.patch.set_alpha(0)
#            self.axs_dict['mux_ssro_histogram_{}'.format(self.q_target)]=ax1
#            self.figs['mux_ssro_histogram_{}'.format(self.q_target)]=fig1
#            self.plot_dicts['mux_ssro_histogram_{}'.format(self.q_target)]={
#                'plotfn': plot_single_qubit_histogram,
#                'data': self.proc_data_dict['PDF_data'][q_target_ch],
#                'qubit_label': self.q_target,
#                'ax_id': 'mux_ssro_histogram_{}'.format(self.q_target),
#                'para_hist' : \
#                    self.fit_res['PDF_fit_{}'.format(q_target_ch)].best_values,
#                'para_cdf' : \
#                    self.fit_res['CDF_fit_{}'.format(q_target_ch)].best_values,
#                'hist_data': \
#                    self.proc_data_dict['Histogram_data'][q_target_ch],
#                'qubit_idx': q_target_idx,
#                'value_name': q_target_ch,
#                'combinations': combinations,
#                'qubit_labels': qubit_labels,
#                'threshold': \
#                self.proc_data_dict['PDF_data'][q_target_ch]['threshold_raw'],
#                'timestamp': self.timestamp,
#                'qoi': self.qoi[q_target_ch]
#            }
#            fig2, ax2 = plt.subplots(figsize=(5,4), dpi=200)
#            fig2.patch.set_alpha(0)
#            self.axs_dict['mux_ssro_cdf_{}'.format(self.q_target)]=ax2
#            self.figs['mux_ssro_cdf_{}'.format(self.q_target)]=fig2
#            self.plot_dicts['mux_ssro_cdf_{}'.format(self.q_target)]={
#                'plotfn': plot_single_qubit_CDF,
#                'data': self.proc_data_dict['PDF_data'][q_target_ch],
#                'qubit_label': self.q_target,
#                'ax_id': 'mux_ssro_cdf_{}'.format(self.q_target),
#                'para_hist' : \
#                    self.fit_res['PDF_fit_{}'.format(q_target_ch)].best_values,
#                'para_cdf' : \
#                    self.fit_res['CDF_fit_{}'.format(q_target_ch)].best_values,
#                'hist_data': \
#                    self.proc_data_dict['Histogram_data'][q_target_ch],
#                'qubit_idx': q_target_idx,
#                'value_name': q_target_ch,
#                'combinations': combinations,
#                'qubit_labels': qubit_labels,
#                'threshold': \
#                self.proc_data_dict['PDF_data'][q_target_ch]['threshold_raw'],
#                'timestamp': self.timestamp,
#                'qoi': self.qoi[q_target_ch]
#            }
#            fig3, ax3 = plt.subplots(figsize=(5,4), dpi=200)
#            fig3.patch.set_alpha(0)
#            self.axs_dict['mux_ssro_crosstalk_{}'.format(self.q_target)]=ax3
#            self.figs['mux_ssro_crosstalk_{}'.format(self.q_target)]=fig3
#            self.plot_dicts['mux_ssro_crosstalk_{}'.format(self.q_target)]={
#                'plotfn': plot_single_qubit_crosstalk,
#                'data': self.proc_data_dict['PDF_data'][q_target_ch],
#                'qubit_label': self.q_target,
#                'ax_id': 'mux_ssro_crosstalk_{}'.format(self.q_target),
#                'para_hist' : \
#                    self.fit_res['PDF_fit_{}'.format(q_target_ch)].best_values,
#                'para_cdf' : \
#                    self.fit_res['CDF_fit_{}'.format(q_target_ch)].best_values,
#                'hist_data': \
#                    self.proc_data_dict['Histogram_data'][q_target_ch],
#                'qubit_idx': q_target_idx,
#                'value_name': q_target_ch,
#                'combinations': combinations,
#                'qubit_labels': qubit_labels,
#                'threshold': \
#                self.proc_data_dict['PDF_data'][q_target_ch]['threshold_raw'],
#                'timestamp': self.timestamp,
#                'qoi': self.qoi[q_target_ch]
#            }
#
#    def run_post_extract(self):
#        self.prepare_plots()  # specify default plots
#        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
#        if self.options_dict.get('save_figs', False):
#            self.save_figures(
#                close_figs=self.options_dict.get('close_figs', True),
#                tag_tstamp=self.options_dict.get('tag_tstamp', True))
#
#
#
#def calc_assignment_prob_matrix(combinations, digitized_data):
#
#    assignment_prob_matrix = np.zeros((len(combinations), len(combinations)))
#
#    for i, input_state in enumerate(combinations):
#        for j, outcome in enumerate(combinations):
#            first_key = next(iter(digitized_data))
#            Check = np.ones(len(digitized_data[first_key][input_state]))
#            for k, ch in enumerate(digitized_data.keys()):
#                check = digitized_data[ch][input_state] == int(outcome[k])
#                Check *= check
#
#            assignment_prob_matrix[i][j] = sum(Check)/len(Check)
#
#    return assignment_prob_matrix
#
#def calc_cross_fidelity_matrix(combinations,assignment_prob_matrix):
#
#    n = int(np.log2(len(combinations)))
#    crossFidMat = np.zeros((n, n))
#    for i in range(n):
#        for j in range(n):
#            P_eiIj = 0  # P(e_i|0_j)
#            P_giPj = 0  # P(g_i|pi_j)
#
#            # Loop over all entries in the Assignment probability matrix
#            for prep_idx, c_prep in enumerate(combinations):
#                for decl_idx, c_decl in enumerate(combinations):
#                    # Select all entries in the assignment matrix for ei|Ij
#                    if (c_decl[i]=='1') and (c_prep[j] == '0'):
#                        P_eiIj += assignment_prob_matrix[prep_idx, decl_idx]
#                    # Select all entries in the assignment matrix for ei|Ij
#                    elif (c_decl[i]=='0') and (c_prep[j] == '1'): # gi|Pj
#                        P_giPj += assignment_prob_matrix[prep_idx, decl_idx]
#
#            # Normalize probabilities
#            normalization_factor = (len(combinations)/2)
#
#            P_eiIj = P_eiIj/normalization_factor
#            P_giPj = P_giPj/normalization_factor
#
#            # Add entry to cross fidelity matrix
#            Fc = 1 - P_eiIj - P_giPj
#            crossFidMat[i,j] = Fc
#
#    return crossFidMat
#
#def plot_assignment_prob_matrix(assignment_prob_matrix,
#                                combinations, qubit_labels, ax=None,
#                                valid_combinations=None,
#                                post_selection=False, **kw):
#    if ax is None:
#        figsize = np.array(np.shape(assignment_prob_matrix))*.7
#        f, ax = plt.subplots(figsize=figsize)
#    else:
#        f = ax.get_figure()
#
#    if valid_combinations is None:
#        valid_combinations = combinations
#
#    alpha_reds = cmap_to_alpha(cmap=pl.cm.Reds)
#    colors = [(0.6, 0.76, 0.98), (0, 0, 0)]
#    cm = LinearSegmentedColormap.from_list('my_blue', colors)
#    alpha_blues = cmap_first_to_alpha(cmap=cm)
#
#    red_im = ax.matshow(assignment_prob_matrix*100,
#                        cmap=alpha_reds, clim=(0., 10))
#    blue_im = ax.matshow(assignment_prob_matrix*100,
#                         cmap=alpha_blues, clim=(50, 100))
#
#    caxb = f.add_axes([0.9, 0.6, 0.02, 0.3])
#
#    caxr = f.add_axes([0.9, 0.15, 0.02, 0.3])
#    ax.figure.colorbar(red_im, ax=ax, cax=caxr)
#    ax.figure.colorbar(blue_im, ax=ax, cax=caxb)
#
#    rows, cols = np.shape(assignment_prob_matrix)
#    for i in range(rows):
#        for j in range(cols):
#            c = assignment_prob_matrix[i, j]
#            if c > .05:
#                col = 'white'
#            else:
#                col = 'black'
#            ax.text(j, i, '{:.2f}'.format(c),
#                    va='center', ha='center', color=col)
#
#    ax.set_xticklabels(valid_combinations)
#    ax.set_xticks(np.arange(len(valid_combinations)))
#
#    ax.set_yticklabels(combinations)
#    ax.set_yticks(np.arange(len(combinations)))
#    ax.set_ylim(len(combinations)-.5, -.5)
#    ax.set_ylabel('Input state')
#    ax.set_xlabel('Declared state')
#    ax.xaxis.set_label_position('top')
#
#    qubit_labels_str = ', '.join(qubit_labels)
#    if post_selection is True:
#        txtstr = 'Post-selected assignment probability matrix\n qubits: [{}]'.format(qubit_labels_str)
#    else:
#        txtstr = 'Assignment probability matrix\n qubits: [{}]'.format(
#            qubit_labels_str)
#    ax.set_title(txtstr, fontsize=24)
#
#
#def plot_cross_fid_matrix(prob_matrix,
#                          combinations, qubit_labels, ax=None,
#                          valid_combinations=None,
#                          post_selection=False, **kw):
#    if ax is None:
#        figsize = np.array(np.shape(prob_matrix))*.7
#        f, ax = plt.subplots(figsize=figsize)
#    else:
#        f = ax.get_figure()
#
#    if valid_combinations is None:
#        valid_combinations = combinations
#
#    alpha_reds = cmap_to_alpha(cmap=pl.cm.Reds)
##     colors = [(0.6, 0.76, 0.98), (0, 0, 0)]
#    colors = [(0.58, 0.404, 0.741), (0, 0, 0)]
#
#    cm = LinearSegmentedColormap.from_list('my_purple', colors)
#    alpha_blues = cmap_first_to_alpha(cmap=cm)
#
#    red_im = ax.matshow(prob_matrix*100,
#                        cmap=alpha_reds, clim=(-10., 10))
#    red_im = ax.matshow(prob_matrix*100,
#                        cmap='RdBu', clim=(-10., 10))
#
#    blue_im = ax.matshow(prob_matrix*100,
#                         cmap=alpha_blues, clim=(80, 100))
#
#    caxb = f.add_axes([0.9, 0.6, 0.02, 0.3])
#
#    caxr = f.add_axes([0.9, 0.15, 0.02, 0.3])
#    ax.figure.colorbar(red_im, ax=ax, cax=caxr)
#    ax.figure.colorbar(blue_im, ax=ax, cax=caxb)
#
#    rows, cols = np.shape(prob_matrix)
#    for i in range(rows):
#        for j in range(cols):
#            c = prob_matrix[i, j]
#            if c > .05 or c <-0.05:
#                col = 'white'
#            else:
#                col = 'black'
#            ax.text(j, i, '{:.1f}'.format(c*100),
#                    va='center', ha='center', color=col)
#
#    ax.set_xticklabels(valid_combinations)
#    ax.set_xticks(np.arange(len(valid_combinations)))
#
#    ax.set_yticklabels(combinations)
#    ax.set_yticks(np.arange(len(combinations)))
#    ax.set_ylim(len(combinations)-.5, -.5)
#    # matrix[i,j] => i = column, j = row
#    ax.set_ylabel(r'Prepared qubit, $q_i$')
#    ax.set_xlabel(r'Classified qubit $q_j$')
#    ax.xaxis.set_label_position('top')
#
#    qubit_labels_str = ', '.join(qubit_labels)
#    if post_selection:
#        txtstr = 'Post-selected cross fidelity matrix'
#    else:
#        txtstr = 'Cross fidelity matrix'
#    ax.text(.5, 1.25, txtstr, transform=ax.transAxes, fontsize=15,
#            verticalalignment='top', horizontalalignment='center')
#
#def plot_single_qubit_histogram(data, ax, para_hist,
#                                para_cdf, timestamp,
#                                hist_data, combinations,
#                                qubit_idx, value_name,
#                                qubit_labels, threshold,
#                                qoi, post_selection=False,
#                                **kw):
#    counts_0, bin_centers_0 = data['0']
#    counts_1, bin_centers_1 = data['1']
#    qubit_label = qubit_labels[qubit_idx]
#    flag = False
#    if type(ax) is np.ndarray:
#        idx = int(3*post_selection)
#        ax = ax[idx]
#        flag=True
#    f = ax.get_figure()
#    ########################################
#    # Histogram of shots
#    ########################################
#    ax.bar(bin_centers_0, counts_0,
#           width=bin_centers_0[1]-bin_centers_0[0],
#           label=r'$|g\rangle$ shots',
#           color='C0', edgecolor='C0', alpha=.4)
#    ax.bar(bin_centers_1, counts_1,
#           width=bin_centers_1[1]-bin_centers_1[0],
#           label=r'$|e\rangle$ shots',
#           color='C3', edgecolor='C3', alpha=.3)
#    # Plot Fit results
#    x = np.linspace(bin_centers_0[0], bin_centers_0[-1], 150)
#    ro_g = ro_gauss(x=[x, x], **para_hist)
#    ax.plot(x, ro_g[0], color='C0', label=r'$|g\rangle$ fit')
#    ax.plot(x, ro_g[1], color='C3', label=r'$|e\rangle$ fit')
#    # Plot Threshold
#    ax.axvline(x=threshold, label=r'$\mathrm{threshold}_{assign}$',
#               ls='--', linewidth=1., color='black', alpha=.5)
#
#    ax.set_xlim(left=bin_centers_0[0], right=bin_centers_0[-1])
#    ax.set_xlabel('Effective voltage (V)')
#    ax.set_ylabel('Counts')
#    ax.set_title('Histogram of shots "'+qubit_label+'"')
#    ax.legend(loc=0, fontsize=5)
#    # Text box with quantities of interest
#    if post_selection is True:
#        textstr = '\n'.join((
#            r'SNR    :       %.2f' % \
#                (qoi['Post_SNR'], ),
#            r'$F_{assign}$  :    %.2f%%       p(g|$\pi$) : %.2f%%' % \
#                (qoi['Post_F_a']*1e2, qoi['Post_relaxation_events']*1e2, ),
#            r'$F_{discr}$    :    %.2f%%       p(e|$0$) : %.2f%%' % \
#                (qoi['Post_F_d']*1e2,  qoi['Post_residual_excitation']*1e2, ))) 
#    else:
#        textstr = '\n'.join((
#            r'SNR    :       %.2f' % \
#                (qoi['SNR'], ),
#            r'$F_{assign}$  :    %.2f%%       p(g|$\pi$) : %.2f%%' % \
#                (qoi['F_a']*1e2, qoi['relaxation_events']*1e2, ),
#            r'$F_{discr}$    :    %.2f%%       p(e|$0$) : %.2f%%' % \
#                (qoi['F_d']*1e2,  qoi['residual_excitation']*1e2, )))
#    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=1)
#    ax.text(0.01, 1.35, textstr, transform=ax.transAxes, fontsize= 9,
#           verticalalignment='top', bbox=props)
#
#    f.suptitle('Mux_ssro_{}_{}'.format(qubit_label, timestamp))
#    if flag == False:
#        ax.legend(loc=0, fontsize=7)
#        if post_selection is True:
#            f.suptitle('Post-selected mux_ssro_{}_{}'.format(qubit_label, timestamp))
#    
#    f.tight_layout()
#
#def plot_single_qubit_CDF(data, ax, para_hist,
#                          para_cdf, timestamp,
#                          hist_data, combinations,
#                          qubit_idx, value_name,
#                          qubit_labels, threshold,
#                          qoi, post_selection=False,
#                          **kw):
#
#    counts_0, bin_centers_0 = data['0']
#    counts_1, bin_centers_1 = data['1']
#    qubit_label = qubit_labels[qubit_idx]
#    flag = False
#    if type(ax) is np.ndarray:
#        idx = int(1+3*post_selection)
#        ax = ax[idx]
#        flag = True
#        ax.set_title('Cumulative sum of shots "{}"'.format(qubit_label))
#        if post_selection is True:
#            ax.text(.5, 1.3, 'Post-selected Shots', transform=ax.transAxes, 
#            fontsize= 20, verticalalignment='top', horizontalalignment='center')
#    f = ax.get_figure()
#    ########################################
#    # Cumulative sum of shots
#    ########################################
#    ax.plot(bin_centers_0, np.cumsum(counts_0)/sum(counts_0),
#             label=r'$|g\rangle$ shots',
#             color='C0', alpha=.75)
#    ax.plot(bin_centers_1, np.cumsum(counts_1)/sum(counts_1),
#             label=r'$|e\rangle$ shots',
#             color='C3', alpha=.75)
#    # Plot Fit results
#    x = np.linspace(bin_centers_0[0], bin_centers_0[-1], 150)
#    ro_c = ro_CDF(x=[x, x], **para_cdf)
#    ax.plot(x, ro_c[0]/np.max(ro_c[0]), '--C0', linewidth=1,
#        label=r'$|g\rangle$ fit')
#    ax.plot(x, ro_c[1]/np.max(ro_c[1]), '--C3', linewidth=1,
#        label=r'$|e\rangle$ fit')
#    # Plot thresholds
#    ax.axvline(x=threshold, label=r'$\mathrm{threshold}_{assign}$',
#               ls='--', linewidth=1., color='black', alpha=.5)
#
#    ax.set_xlim(left=bin_centers_0[0], right=bin_centers_0[-1])
#    ax.set_ylim(bottom=0)
#    ax.set_xlabel('Effective voltage (V)')
#    ax.set_ylabel('Fraction')
#    ax.legend(loc=0, fontsize=5) 
#
#    if flag == False:
#        if post_selection:
#            ax.set_title('Post-selected mux_ssro_{}_{}'.format(qubit_label, timestamp))
#        else:
#            ax.set_title('Mux_ssro_{}_{}'.format(qubit_label, timestamp))
#        ax.legend(loc=0, fontsize=7)
#    f.tight_layout()
#
#def plot_single_qubit_crosstalk(data, ax, para_hist,
#                                para_cdf, timestamp,
#                                hist_data, combinations,
#                                qubit_idx, value_name,
#                                qubit_labels, threshold,
#                                qoi, post_selection=False,
#                                 **kw):
#
#    qubit_label = qubit_labels[qubit_idx]
#    flag = False
#    if type(ax) is np.ndarray:
#        idx = int(2+3*post_selection)
#        ax = ax[idx]
#        flag = True
#        ax.set_title('Histogram vs Prepared state "'+qubit_label+'"')
#
#    f = ax.get_figure()
#    ########################################
#    # cross talk
#    ########################################
#    colors_R = pl.cm.Reds
#    colors_B = pl.cm.Blues
#    colors_G = pl.cm.Greens
#    iR = 0.1  # Do not start at the complete white/transparent end
#    iB = 0.1
#    iG = 0.1
#    for i, (key, (cnts, bin_centers)) in enumerate(hist_data.items()):
#
#        if set(key) <= {'0', '1'}:
#            if key[qubit_idx] == '0':
#                # increment the blue colorscale
#                col = colors_B(iB)
#                iB += 0.8/(len(combinations)/2)#.8 to not span full colorscale
#            elif key[qubit_idx] == '1':
#                # Increment the red colorscale
#                col = colors_R(iR)
#                iR += 0.8/(len(combinations)/2)
#            else:
#                raise ValueError('{}  {}'.format(
#                    combinations, combinations[qubit_idx]))
#        else:
#            # increment the green colorscale
#            col = colors_G(iG)
#            iG += 0.8/(len(combinations)/2)  # .8 to not span full colorscale
#        ax.plot(bin_centers, cnts, label=key, color=col)
#    ax.axvline(x=threshold, label=r'$\mathrm{threshold}_{assign}$',
#               ls='--', linewidth=1., color='black', alpha=.75)
#    ax.set_xlabel(value_name.decode('utf-8'))
#    ax.set_ylabel('Counts')
#    l = ax.legend(loc=(1.05, .01), title='Prepared state\n{}'.format(
#        qubit_labels), prop={'size': 4})
#    l.get_title().set_fontsize('5')
#
#    if flag == False:
#        if post_selection is True:
#            ax.set_title('Post-selected mux_ssro_{}_{}'.format(qubit_label, timestamp))
#        else:
#            ax.set_title('Mux_ssro_{}_{}'.format(qubit_label, timestamp))
#        l = ax.legend(loc=(1.05, .01),
#                      title='Prepared state\n{}'.format(qubit_labels),
#                      prop={'size': 4})
#        l.get_title().set_fontsize('4')
#
#    f.tight_layout()
#
#