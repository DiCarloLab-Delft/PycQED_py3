import os
from uncertainties import ufloat, UFloat
import lmfit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
from pprint import pprint
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict
import itertools as itt
import h5py
from scipy.optimize import curve_fit

import pycqed.measurement.hdf5_data as hd5
from pycqed.measurement import optimization as opt

import pycqed.analysis.measurement_analysis as ma
from pycqed.analysis_v2 import  measurement_analysis as ma2
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis_v2.base_analysis import BaseDataAnalysis


class Parity_Check_Analysis_OLD():

    def __init__(self, label, ancilla_qubits, data_qubits, parking_qubits=None, 
                folder=None, timestamp=None, cases=None, plotting=False):
        self.result = self.parity_check_analysis(label, ancilla_qubits, data_qubits, parking_qubits,
                                                folder=folder, timestamp=timestamp, cases=cases, plotting=plotting)
        return

    def parity_check_analysis(self, label, ancilla_qubits, data_qubits, 
                                parking_qubits=None, folder=None, timestamp=None, cases=None, plotting=False):    
        res_dict = {}
        all_qubits = ancilla_qubits + data_qubits
        if parking_qubits:
            all_qubits += parking_qubits
        a = ma.MeasurementAnalysis(label=label, auto=False, close_file=False)
        a.get_naming_and_values()
        res_dict['label'] = label
        res_dict['value_names'] = a.value_names

        if not cases:
            cases = np.array(['{:0{}b}'.format(i, len(data_qubits)) for i in range(2**len(data_qubits))])
        else:
            cases = np.array(cases)
        res_dict['cases'] = cases

        angles = np.arange(0, 341, 20)
        n = len(ancilla_qubits+data_qubits)
        cal_pts = ['{:0{}b}'.format(i, n) for i in range(2**n)]
        # dummy indices for calibration points to match x axis with angles
        cal_num = 2**n
        cal_pnt_idx = np.arange(360, 360 + cal_num * 20, 20)

        # extract measured data for each qubits, preserving measurement order
        qubit_msmt_order = [x.split(' ')[-1] for x in a.value_names]
        qubit_indices = [qubit_msmt_order.index(qb) for qb in all_qubits]
        data_ordered = a.measured_values[qubit_indices]

        raw_cal_data = {}
        normalized_cal_data = {}
        normalized_data = [None] * len(data_ordered)
        for i, d in enumerate(data_ordered):
            msmt = d[:-2**n]
            msmt = msmt.reshape(len(cases), len(angles))
            
            cal_ind = i - len(parking_qubits) if parking_qubits else i

            cal_data = d[-2**n:]
            raw_cal_data[cal_ind] = cal_data 
            zero_levels = [cal_data[k] for k,cal_pt in enumerate(cal_pts) if int(cal_pt[cal_ind]) == 0]
            one_levels = [cal_data[k] for k,cal_pt in enumerate(cal_pts) if int(cal_pt[cal_ind]) == 1]
            normalized_cal_data[cal_ind] = (cal_data - np.mean(zero_levels)) / (np.mean(one_levels) - np.mean(zero_levels))

            normalized_data[cal_ind] = {}
            for j, case in enumerate(cases): 
                normalized_data[cal_ind][case] = (msmt[j] - np.mean(zero_levels)) / (np.mean(one_levels) - np.mean(zero_levels))

        res_dict['raw_cal_data'] = raw_cal_data
        res_dict['normalized_data']  = normalized_data
        res_dict['normalized_cal_data'] = normalized_cal_data

        params = {}
        fit_res = {} 
        best_vals = {}
        fit = {}
        phi_phases = {}
        missing_fractions = {}

        if plotting: 
            # plotting params
            if len(cases) > 4:
                rc_params = {'lines.linewidth': 1,
                            'lines.markersize': 6,
                            'legend.fontsize': 8,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 6}
                colors = [mpl.cm.hsv(x) for x in np.linspace(0,1,len(cases))]
            else:
                rc_params = {'lines.linewidth': 2,
                            'lines.markersize': 14,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 12,
                            'axes.labelsize': 12}   
                colors = ['blue','orange','green','red']             

            fig, ax = plt.subplots(n, 1, sharex=True, dpi=120, figsize=(8,8))
            fig_mf, ax_mf = plt.subplots(len(data_qubits), 1, sharex=True, sharey=True, dpi=120, figsize=(8,8))
            if len(data_qubits) == 1:
                ax_mf = [ax_mf]

        for q, qubit in enumerate(ancilla_qubits + data_qubits):

            if plotting:
                with plt.rc_context(rc_params):
                    ax[q].plot(cal_pnt_idx, normalized_cal_data[q], linestyle='-', marker='o', alpha=0.6)
                    angles_label = np.arange(0,360,60)
                    ax[q].set_xticks(np.concatenate([angles_label, cal_pnt_idx]))
                    deg_sign = u"\N{DEGREE SIGN}"
                    ax[q].set_xticklabels(["{:3.0f}".format(ang) + deg_sign for ang in angles_label] 
                                            + [fr"$\vert{{{case}}} \rangle$" for case in cal_pts])
                    ax[q].tick_params(axis="x", labelrotation=45)

            for i,case in enumerate(cases):
                if 0 == q:
                    cos_mod0 = lmfit.Model(fit_mods.CosFunc)
                    cos_mod0.guess = fit_mods.Cos_guess.__get__(cos_mod0, cos_mod0.__class__)
                    params[f'{case}'] = cos_mod0.guess(data=normalized_data[q][case], t=angles)
                    fit_res[f'{case}'] = cos_mod0.fit(data=normalized_data[q][case], t=angles, params=params[f'{case}'])
                    best_vals[f'{case}'] = fit_res[f'{case}'].best_values
                    fit[f'{case}'] = fit_mods.CosFunc(angles,
                                                    frequency=best_vals[f'{case}']['frequency'],
                                                    phase=best_vals[f'{case}']['phase'],
                                                    amplitude=best_vals[f'{case}']['amplitude'],
                                                    offset=best_vals[f'{case}']['offset'])

                    phi = ufloat(
                            np.rad2deg(fit_res[f'{case}'].params['phase'].value),
                            np.rad2deg(
                                fit_res[f'{case}'].params['phase'].stderr
                                if fit_res[f'{case}'].params['phase'].stderr is not None
                                else np.nan
                            ),
                        )
                    phi_phases[f'{case}'] = np.rad2deg(fit_res[f'{case}'].params['phase'].value) % 360

                    if plotting:
                        with plt.rc_context(rc_params):
                            ax[0].plot(angles, fit[f'{case}'], color=colors[i], alpha=0.4, linestyle='dashed')
                            ax[0].plot(angles, normalized_data[q][case], color=colors[i], alpha=0.4, marker='.', linestyle='none',
                                        label=fr'$\vert{cases[i]}\rangle$ : {phi % 360}')
                            ax[0].set_ylabel(f"Population {qubit}", fontsize=12)
                            ax[0].legend(bbox_to_anchor=(1.05, .98), loc='upper left', frameon=False, fontsize=12)
                            # ax[0].text(1.1, 0.98, fr"$|{{{','.join(data_qubits)}}}\rangle$", transform=ax[0].transAxes, fontsize=12)
                
                else:
                    mean_miss_frac = np.mean(normalized_data[q][cases[-1]]) - np.mean(normalized_data[q][cases[0]])
                    missing_fractions[qubit] = mean_miss_frac

                    if plotting:  
                        with plt.rc_context(rc_params):
                            ax[q].plot(angles, normalized_data[q][case], 
                                       color=colors[i], alpha=0.4, marker='.')
                            ax[q].set_ylabel(f"Population {qubit}", fontsize=12)
                            hline = ax[q].hlines(mean_miss_frac, angles[0], angles[-1], colors='k', linestyles='--')
                            # ax[q].text(1.1, 0.9, 'Missing frac : {:.1f} %'.format(mean_miss_frac*100), 
                            #            transform=ax[q].transAxes, fontsize=12)
                            ax[q].legend([hline], [f"Missing frac : {mean_miss_frac*100:.1f} %"], loc=0)

                            # separate figure for missing fractions per case
                            ax_mf[q-1].plot([fr"$\vert{{{case}}} \rangle$" for case in cases], 
                                            100 * np.array([np.mean(normalized_data[q][case]) for case in cases]),
                                            color='b', alpha=0.4, marker='o')
                            ax_mf[q-1].set_ylabel(fr"$P_{{exc}}$ of {qubit} (%)", fontsize=12)
                            # ax_mf[q-1].set_yticks(np.arange(0, ax_mf[q-1].get_ylim()[-1], int(0.25*ax_mf[q-1].get_ylim()[-1])) )
                            # ax_mf[q-1].sharey(True)
                            ax_mf[q-1].tick_params(axis="x", labelrotation=45)
                            ax_mf[q-1].grid(True, linestyle=':', linewidth=0.9, alpha=0.8)

        res_dict['phi_osc'] = phi_phases 
        res_dict['missing_frac'] = missing_fractions

        if plotting:
            with plt.rc_context(rc_params):
                # missing fraction figure
                ax_mf[-1].set_xlabel(fr"Cases ($\vert {','.join(data_qubits)} \rangle$)", fontsize=12)
                fig_mf.suptitle(t = f"{a.timestamp}\nParity check {ancilla_qubits} with data qubits {data_qubits}\nMissing fractions",
                            #x = 0.38, y = 1.04,
                            fontsize=14)        
                fn = os.path.join(a.folder, label + '_missing_fractions.png')
                fig_mf.savefig(fn, dpi=120, bbox_inches='tight', format='png')
                plt.close(fig_mf)

                # show phase differences in main plot: (can become too crowded)
                # phase_diff_str = fr"Diff $\varphi_{{{cases[0]}}}$ to:" + '\n' \
                #      + '\n'.join([fr"$\phi_{{{case}}}$ : {((res_dict['phi_osc'][f'{cases[0]}'] - res_dict['phi_osc'][f'{case}']) % 360):.1f}" 
                #                  for case in cases[1:]])
                # ax[0].text(1.1, 0.04, phase_diff_str, transform=ax[0].transAxes, fontsize=12)
                ax[-1].set_xlabel(r"Phase (deg), Calibration points ($\vert Q_a, [Q_d] \rangle$)", fontsize=12)
            
                fig.suptitle(t = f"{a.timestamp}\nParity check {ancilla_qubits} with data qubits {data_qubits}",
                            #x = 0.38, y = 1.04,
                            fontsize=14)        
                fn = os.path.join(a.folder, label + '.png')
                fig.savefig(fn, dpi=120, bbox_inches='tight', format='png')
                plt.close(fig)

                # show phase differences in separate figure:
                wrapped_phases = deepcopy(res_dict['phi_osc'])
                for case, phase in wrapped_phases.items():
                    if not bool(case.count('1') % 2) and np.isclose(phase, 360, atol=60):
                        wrapped_phases[case] -= 360
                phase_diff_by_case = [wrapped_phases[f'{case}'] for case in cases] 
                
                fig, ax = plt.subplots(1, 1, dpi=120, figsize=(6,4))                 
                ax.plot(cases, phase_diff_by_case, linestyle='-', marker='o', alpha=0.7)
                
                # mark the target phases
                ax.hlines([0,180], cases[0], cases[-1], colors='k', linestyles='--', alpha=0.6)
                mask_odd = np.array([bool(case.count('1') % 2) for case in cases])
                ax.plot(cases[mask_odd], [180]*len(cases[mask_odd]), 
                    color='r', marker='*', markersize=2+mpl.rcParams['lines.markersize'], linestyle='None', alpha=0.6)
                ax.plot(cases[~mask_odd], [0]*len(cases[~mask_odd]), 
                    color='r', marker='*', markersize=2+mpl.rcParams['lines.markersize'], linestyle='None', alpha=0.6)

                ax.set_xlabel(r"Case ($\vert [Q_d] \rangle$)", fontsize=12)
                ax.set_xticklabels([fr"$\vert{{{case}}} \rangle$" for case in cases])
                ax.tick_params(axis="x", labelrotation=45)
                ax.set_ylabel(r"Conditional phase (deg)", fontsize=12)
                ax.set_yticks(np.arange(0,211,30))
                ax.set_title(f"{a.timestamp}\nParity check {ancilla_qubits} with data qubits {data_qubits}")
                ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)

                fn = os.path.join(a.folder, label + 'phase_diffs' + '.png')
                fig.savefig(fn, dpi=120, bbox_inches='tight', format='png')
                plt.close(fig)

        return res_dict

class Parity_Check_Analysis(BaseDataAnalysis):
    """
    Analysis for a parity check measured using conditional oscillation 
    experiments for each control preparation case.
    Contains parity check model analysis that extracts phase error of all n-body terms of a parity check.
    """

    def __init__(
            self,
            t_start: str = None,
            t_stop: str = None,
            label: str = "Parity_check_flux_dance",
            data_file_path: str = None,
            target_qubit: str = None,
            analyze_parity_model: bool = True,
            options_dict: dict = None,
            extract_only: bool = False,
            auto: bool = True,
        ):
        """

        """
        # if options_dict:
        #     options_dict['scan_label'] = label
        # else:
        #     options_dict = {'scan_label': label}

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            data_file_path=data_file_path,
            options_dict=options_dict,
            close_figs=True,
            extract_only=extract_only,
            save_qois=True,
            do_fitting=True,
        )

        # self.single_timestamp = False
        self.target_qubit = target_qubit
        self.analyze_parity_model = analyze_parity_model

        # self.params_dict = {
        #     "xlabel": "sweep_name",
        #     "xunit": "sweep_unit",
        #     "xvals": "sweep_points",
        #     "measurementstring": "measurementstring",
        #     "value_names": "value_names",
        #     "value_units": "value_units",
        #     "measured_values": "measured_values",
        # }

        self.numeric_params = []

        if auto:
            self.run_analysis()

        return

    def save_quantities_of_interest(self):
        print(self.proc_data_dict['quantities_of_interest'])

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = ma.a_tools.get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names'),
                      'measurement_name': ('MC settings', 'attr:measurement_name')}

        self.raw_data_dict = hd5.extract_pars_from_datafile(data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
        return

    def process_data(self):
        print(list(self.raw_data_dict.keys()))
        self.proc_data_dict = OrderedDict()
        # values stored in quantities of interest will be saved in the data file
        self.proc_data_dict['quantities_of_interest'] = {}
        qoi = self.proc_data_dict['quantities_of_interest']

        # get all measured qubits from value_names (should be at least two)
        measured_qubits = [x.decode('utf-8').split(' ')[-1] for x in self.raw_data_dict['value_names']]
        print(measured_qubits)
        print(self.raw_data_dict['data'].shape, self.raw_data_dict['data'])

        assert len(measured_qubits) >= 2
        qoi['parity_check_weight'] = len(measured_qubits) - 1

        qubit_indices = [measured_qubits.index(qb) for qb in measured_qubits]
        data_ordered = np.array(self.raw_data_dict['data'][0])[qubit_indices]

        # a = ma.MeasurementAnalysis(label=self.labels[0], auto=False, close_file=False)
        # a.get_naming_and_values()
        # # extract measured data for each qubits, preserving measurement order
        # qubit_msmt_order = [x.split(' ')[-1] for x in a.value_names]
        # print(qubit_msmt_order)
        # qubit_indices = [qubit_msmt_order.index(qb) for qb in ['X2', 'D6', 'D5', 'D3', 'D2']]
        # data_ordered = a.data[qubit_indices]
        # print(qubit_indices)

        # qubit_indices_2 = [qubit_msmt_order.index(qb) for qb in measured_qubits]
        # print(qubit_indices_2)
        # print(type(a.data))

        if not self.target_qubit:
            # if ramseyed target qubit was not given, assume the first measured qubit is the ramsey qubit
            # and all following ones the control qubits
            self.target_qubit = measured_qubits[0]
            self.control_qubits = measured_qubits[1:] # ['D6', 'D5', 'D3', 'D2'] #  
        else:
            # if target qubit was given, assume all other qubits are control qubits
            self.control_qubits = [qb for qb in measured_qubits if qb != self.target_qubit]

        # save qubits
        qoi['measured_qubit_order'] = measured_qubits
        qoi['target_qubit'] = self.target_qubit
        qoi['control_qubits'] = self.control_qubits
        # save labels
        qoi['label'] = self.labels[0]
        qoi['measurementstring'] = self.raw_data_dict['measurement_name'] #.decode('utf-8')
        # NOTE: assumed that measured angles are the default rotation steps.
        #       This is necessary because the sweep points in the parity check experiment are cases, 
        #       so we cannot find the angles in the datafile
        qoi['measured_angles'] = np.arange(0, 341, 20)
        # generate measurement cases of control qubits, labeled by binary strings
        qoi['control_cases'] = np.array(['{:0{}b}'.format(i, qoi['parity_check_weight'])
                                         for i in range(2**qoi['parity_check_weight'])])
        qoi['control_labels'] = np.array([rf"$\vert {case} \rangle$" for case in qoi['control_cases']])

        # prepare various ordered dicts for saving analysis data
        qoi['oscillation_phases'] = OrderedDict().fromkeys(qoi['control_cases'])
        qoi['wrapped_phases'] = OrderedDict().fromkeys(qoi['control_cases'])
        qoi['mean_population_per_qubit_per_case'] = OrderedDict().fromkeys(qoi['control_qubits'])
        qoi['mean_missing_fraction_per_qubit'] = OrderedDict().fromkeys(qoi['control_qubits'])

        # save target conditional phases for each case in the parity check (0 if even excitation, 180 if odd)
        qoi['target_phases'] = OrderedDict(zip(qoi['control_cases'],
                                               180*np.array([case.count('1') % 2 
                                                            for case in qoi['control_cases']])) )

        # generate cases of calibration points, which involve all qubits including the target qubit
        self.proc_data_dict['calibration_point_cases'] \
            = np.array(['{:0{}b}'.format(i, len(measured_qubits)) for i in range(2**len(measured_qubits))])
        self.proc_data_dict['calibration_point_plotting_indices'] \
            = np.arange(360, 360 + 20*len(self.proc_data_dict['calibration_point_cases']), 20)
        n_cal_pts = len(self.proc_data_dict['calibration_point_cases'])

        # prepare ordered dicts for data
        self.proc_data_dict['raw_cal_data'] = OrderedDict().fromkeys(qoi['measured_qubit_order'])
        self.proc_data_dict['normalized_cal_data'] = OrderedDict().fromkeys(qoi['measured_qubit_order'])
        self.proc_data_dict['raw_data'] = OrderedDict().fromkeys(qoi['measured_qubit_order'])
        self.proc_data_dict['normalized_data'] = OrderedDict().fromkeys(qoi['measured_qubit_order'])

        # normalize oscillation data and calibration points separately
        for i, (qb, data) in enumerate(zip(measured_qubits, self.raw_data_dict['data'][:, 1:].T)):
        #     qb = qoi['measured_qubit_order'][i]
        # for i, qb in enumerate(qoi['measured_qubit_order']):
        # for qb, data in zip(['X2', 'D6', 'D5', 'D3', 'D2'], data_ordered):
        # for name, data in self.raw_data_dict['measured_values_ord_dict'].items():
            # qb = measured_qubits[qb]
            # qb = name.split(' ')[-1]
            # data = np.array(data[0])
            # data = self.raw_data_dict['measured_values'][0][i]
            # print(qb, data)
            # separate data array into calibration points and measurement (oscillation) data
            cal_data = data[-n_cal_pts:]
            msmt_data = data[:-n_cal_pts].reshape( (len(qoi['control_cases']), len(qoi['measured_angles'])) )

            # compute calibration levels of zero and one states as average of all lower (upper) points
            # i = ['X2', 'D6', 'D5', 'D3', 'D2'].index(qb)
            cal_zero_level = np.nanmean([cal_data[k] 
                                        for k,cal_pt in enumerate(self.proc_data_dict['calibration_point_cases'])
                                        if cal_pt[i] == '0'])
            cal_one_level = np.nanmean([cal_data[k] 
                                        for k,cal_pt in enumerate(self.proc_data_dict['calibration_point_cases'])
                                        if cal_pt[i] == '1'])

            # normalize calibration points
            self.proc_data_dict['raw_cal_data'][qb] = cal_data
            self.proc_data_dict['normalized_cal_data'][qb] \
                = (cal_data - cal_zero_level) / (cal_one_level - cal_zero_level)

            # normalize oscillation data for each case separately
            self.proc_data_dict['raw_data'] = msmt_data
            self.proc_data_dict['normalized_data'][qb] = OrderedDict.fromkeys(qoi['control_cases'])
            for j, case in enumerate(qoi['control_cases']):
                self.proc_data_dict['normalized_data'][qb][case] \
                    = (msmt_data[j] - cal_zero_level) / (cal_one_level - cal_zero_level)

            print(self.proc_data_dict['normalized_data'].keys())
        return


    def prepare_fitting(self):
        cases = self.proc_data_dict['quantities_of_interest']['control_cases']
        self.fit_dicts = OrderedDict().fromkeys(cases)

        for case in cases:
            cos_mod = lmfit.Model(fit_mods.CosFunc)
            cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
            self.fit_dicts[case] = {
                'model': cos_mod,
                'guess_dict': {"frequency": {"value": 1 / 360, "vary": False}},
                'fit_xvals': {"t": self.proc_data_dict['quantities_of_interest']['measured_angles']},
                'fit_yvals': {"data": self.proc_data_dict['normalized_data'][self.target_qubit][case]}
                }

        return

    def analyze_fit_results(self):
        qoi = self.proc_data_dict['quantities_of_interest']
        data = self.proc_data_dict['normalized_data']
        cases = qoi['control_cases']

        # extract phases from fit results
        for case, fit in self.fit_res.items():
            qoi['oscillation_phases'][case] \
                = ufloat(nominal_value=np.rad2deg(fit.params['phase'].value) % 360,
                        std_dev=np.rad2deg(fit.params['phase'].stderr)
                                 if fit.params['phase'].stderr is not None
                                 else np.nan)
            # save another representation of phases, wrapped such that all phases are between 0 and 180,
            # while phases close to 360 (within 60 deg) are represented as negative.
            # wrapping is applied only to phases of odd cases, even cases are left as is (close to 180)
            if not bool(case.count('1') % 2) and np.isclose(qoi['oscillation_phases'][case].n, 360, atol=60):
                qoi['wrapped_phases'][case] = qoi['oscillation_phases'][case] - 360
            else:
                qoi['wrapped_phases'][case] = qoi['oscillation_phases'][case]

        # mean missing fraction of each control qubit is taken as the difference of means of its population 
        # for the the case of all control qubits on and all control qubits off
        for qb in qoi['control_qubits']:
            level_on = ufloat(nominal_value=np.nanmean(data[qb][cases[-1]]),
                              std_dev=np.nanstd(data[qb][cases[-1]]) / np.sqrt(len(data[qb][cases[-1]])) )
            level_off = ufloat(nominal_value=np.nanmean(data[qb][cases[0]]),
                              std_dev=np.nanstd(data[qb][cases[0]]) / np.sqrt(len(data[qb][cases[0]])) )
            qoi['mean_missing_fraction_per_qubit'][qb] = level_on - level_off

            # also save the population of each control qubit as a mean of each case
            qoi['mean_population_per_qubit_per_case'][qb] = OrderedDict.fromkeys(cases)
            for case in cases:
                qoi['mean_population_per_qubit_per_case'][qb][case] \
                    = ufloat(nominal_value=np.nanmean(data[qb][case]),
                             std_dev=np.nanstd(data[qb][case]) / np.sqrt(len(data[qb][case])))

        if self.analyze_parity_model:
            self._analyze_parity_model()

        return


    def _analyze_parity_model(self):
        qoi = self.proc_data_dict['quantities_of_interest']
        weight = qoi['parity_check_weight']
        cases = qoi['control_cases']

        # save raw phases (already only positive and max 360 due to modulo operation in previous step)
        phases_raw = qoi['oscillation_phases']
        # save phases for further manipulation
        phases_measured = np.array(list(phases_raw.values()), dtype=UFloat) #h5py.special_dtype(vlen=str))
        # shift everything down by the phase of the all-zero control case (global phase offset)
        phases_measured -= phases_raw[qoi['control_cases'][0]]

        # M = self._generate_model_conversion_matrix(weight, cases)

        if weight == 2:
            for i, (case, phase) in enumerate(zip(cases, phases_measured)):
                if phase < 0:
                    phases_measured[i] += 360
                PhaseB = phases_measured[2**1]
                PhaseA = phases_measured[2**0]
                bitB = np.floor(i/2)%2
                bitA = np.floor(i/1)%2
                refPhase = PhaseB*bitB+PhaseA*bitA
                phases_measured[i] += np.round((refPhase-phases_measured[i]).n/360)*360

            # # plot raw phases
            # if ax:
            #     ax.plot(cases, phases_measured, marker='.', color=color)
            #     ax.tick_params(axis="x", labelrotation=90)

            # mark the target phases
            phase_vector_ideal = np.zeros(len(cases))
            mask_odd = np.array([bool(case.count('1') % 2) for case in cases])
            phase_vector_ideal[mask_odd] = 180
            phase_vector_ideal[-1] += 360

            ## Z matrix
            ZM = np.array([[+1,+1,+1,+1],
                            [+1,+1,-1,-1],
                            [+1,-1,+1,-1],
                            [+1,-1,-1,+1]
                            ]).T

        elif weight == 4:
            for i, (case, phase) in enumerate(zip(cases, phases_measured)):
                if phase < 0:
                    phases_measured[i] += 360
                PhaseD = phases_measured[2**3]
                PhaseC = phases_measured[2**2]
                PhaseB = phases_measured[2**1]
                PhaseA = phases_measured[2**0]
                bitD = np.floor(i/8)%2
                bitC = np.floor(i/4)%2
                bitB = np.floor(i/2)%2
                bitA = np.floor(i/1)%2
                refPhase = PhaseD*bitD+PhaseC*bitC+PhaseB*bitB+PhaseA*bitA
                phases_measured[i] += np.round((refPhase-phases_measured[i]).n/360)*360

            # # plot raw phases
            # if ax:
            #     ax.plot(cases, phases_measured, marker='.', color=color)
            #     ax.tick_params(axis="x", labelrotation=90)
            
            # mark the target phases
            phase_vector_ideal = np.zeros(len(cases))
            mask_odd = np.array([bool(case.count('1') % 2) for case in cases])
            phase_vector_ideal[mask_odd] = 180
            # phase_vector_ideal[3:15] += 360
            # phase_vector_ideal[15] += 720

            phase_vector_ideal[3] += 360
            phase_vector_ideal[5] += 360
            phase_vector_ideal[6] += 360
            phase_vector_ideal[7] += 360
            phase_vector_ideal[9] += 360
            phase_vector_ideal[10] += 360
            phase_vector_ideal[11] += 360
            phase_vector_ideal[12] += 360
            phase_vector_ideal[13] += 360
            phase_vector_ideal[14] += 360
            phase_vector_ideal[15] += 720

            ## Z matrix
            ZM = np.array([[+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1],
                          [+1,+1,+1,+1,+1,+1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
                          [+1,+1,+1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,-1,-1,-1],
                          [+1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1],
                          [+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1],
                          [+1,+1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,+1,+1,+1,+1],
                          [+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1],
                          [+1,+1,-1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,+1,+1],
                          [+1,-1,+1,-1,+1,-1,+1,-1,-1,+1,-1,+1,-1,+1,-1,+1],
                          [+1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,-1,-1,-1,+1,+1],
                          [+1,-1,+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,-1,+1],
                          [+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,+1,+1,+1,+1,-1,-1],
                          [+1,-1,+1,-1,-1,+1,-1,+1,-1,+1,-1,+1,+1,-1,+1,-1],
                          [+1,-1,-1,+1,+1,-1,-1,+1,-1,+1,+1,-1,-1,+1,+1,-1],
                          [+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1],
                          [+1,-1,-1,+1,-1,+1,+1,-1,-1,+1,+1,-1,+1,-1,-1,+1],
                        ]).T
        else:
            raise NotImplementedError(f"Parity model analysis not implemented for weight-{weight} parity checks, only weights 2 and 4 supported!")

        phase_estimate = np.dot(np.linalg.inv(ZM), phases_measured)
        phase_estimate_ideal = np.dot(np.linalg.inv(ZM), phase_vector_ideal)

        # all these steps are needed to generate a list of labels for the model terms
        # terms_nested = [list(combinations(''.join([str(i) for i in range(0,weight)]), j)) 
        #                 for j in range(1,weight+1)]
        # terms_flattened = [list(p) for i, term in enumerate(terms_nested) 
        #                     for _, p in enumerate(term)]

        # model_terms = [qoi['target_qubit']]
        # for term in terms_flattened:
        #     model_terms += [qoi['target_qubit'] + ',' 
        #                     + ','.join([qoi['control_qubits'][int(p)] for p in term]) ] 

        # compute model terms to use for labels
        control_combinations = [elem for k in range(1, weight+1) 
                                        for elem in itt.combinations(qoi['control_qubits'], k)]
        model_terms = [qoi['target_qubit']]
        model_terms += [ qoi['target_qubit'] + ',' + qbs 
                        for qbs in [','.join(comb) for comb in control_combinations] ]

        qoi['parity_model'] = dict()
        qoi['parity_model']['phases_measured'] = phases_measured
        qoi['parity_model']['phases_model'] = phase_estimate
        qoi['parity_model']['phases_model_ideal'] = phase_estimate_ideal
        qoi['parity_model']['model_terms'] = model_terms
        qoi['parity_model']['model_errors'] = phase_estimate - phase_estimate_ideal
        return


    @staticmethod
    def _generate_model_conversion_matrix(
            n_control_qubits: int,
            control_cases: List[str] = None,
            show_matrix: bool = False,
        ) -> Tuple[np.ndarray, bool]:
        """
        Generate matrix of +1's and -1's that converts model phases to measured phases.

        Args:
            control_qubits:
                List of measured control qubits, not including target (ramsey-/ancilla-) qubit.
            control_cases:
                List of binary strings representing the measured control qubit cases.
                A '1' the string means that the control qubit in the corresponding position in `control_qubits` is on.
            show_matrix:
                Print and plot resulting matrix, for verification.

        Returns:
            ZM: Numpy matrix of +1's and -1's that converts model phases to measured phases
            is_invertible: Boolean indicating whether the resulting matrix is invertible or not
        """
        if not control_cases:
            control_cases = ['{:0{}b}'.format(i, n_control_qubits) for i in range(2**n_control_qubits)]

        positions = ''.join(map(str, range(n_control_qubits)))
        terms = [list(''.join(x) for x in sorted(combinations(positions, i))) for i in range(0, n_control_qubits+1)]

        ZM = np.zeros((n_control_qubits,)*2)

        for i,case in list(enumerate(control_cases))[1:]:
            element = np.zeros(len(control_cases))
            it = 0
            for j,term in enumerate(terms):
                for k,qbs in enumerate(term):
                    Z1 = [-1 if case[int(s)] == '1' else 1 for s in qbs ]
        #             print(Z1)
        #             element.append(np.prod(Z1))
        #             print(it, i,j,k)
                    element[it] = np.prod(Z1)
                    it += 1

        #     print(i, element)
            ZM[:,i] = element

        ZM[:,0] = np.ones(len(control_cases))

        is_invertible = np.all(np.dot(np.linalg.inv(ZM), ZM) == np.eye(len(control_cases)))

        TRUE = np.array([[+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1],
              [+1,+1,+1,+1,+1,+1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
              [+1,+1,+1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,-1,-1,-1],
              [+1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1],
              [+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1],
              [+1,+1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,+1,+1,+1,+1],
              [+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,-1,+1],
              [+1,+1,-1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,+1,+1],
              [+1,-1,+1,-1,+1,-1,+1,-1,-1,+1,-1,+1,-1,+1,-1,+1],
              [+1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,-1,-1,-1,+1,+1],
              [+1,-1,+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,-1,+1],
              [+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,+1,+1,+1,+1,-1,-1],
              [+1,-1,+1,-1,-1,+1,-1,+1,-1,+1,-1,+1,+1,-1,+1,-1],
              [+1,-1,-1,+1,+1,-1,-1,+1,-1,+1,+1,-1,-1,+1,+1,-1],
              [+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,-1,+1,+1,-1],
              [+1,-1,-1,+1,-1,+1,+1,-1,-1,+1,+1,-1,+1,-1,-1,+1],
            ]).T

        if show_matrix:
            print(ZM)
            plt.imshow(ZM)
            plt.colorbar()
            plt.show()
            plt.imshow(np.abs(TRUE - ZM))
            plt.colorbar()
            plt.show()

        return ZM, is_invertible


    def run_post_extract(self):
        # NOTE: we need to overload run_post_extract here 
        #       to specify our own axs_dict to the plot method
        self.prepare_plots()
        self.plot(key_list='auto', axs_dict=self.axs_dict)
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True)
                )
        return


    def prepare_plots(self):
        self.axs_dict = dict()
        qoi = self.proc_data_dict['quantities_of_interest']
        
        fig, axs = plt.subplots(len(qoi['measured_qubit_order']), 1, sharex=True, dpi=120, figsize=(8,8))
        self.axs_dict['main'] = axs
        self.figs['main'] = fig
        self.plot_dicts['main'] = {
            'plotfn': self._plot_main_figure 
        }

        fig, axs = plt.subplots(len(qoi['control_qubits']), 1, sharex=True, sharey=True, dpi=120, figsize=(8,8))
        self.axs_dict['control_qubit_excitations'] = axs
        self.figs['control_qubit_excitations'] = fig
        self.plot_dicts['control_qubit_excitations'] = {
            'plotfn': self._plot_control_qubit_excitations_figure 
        }

        fig, axs = plt.subplots(1, 1, dpi=120, figsize=(6,4))
        self.axs_dict['phase_differences'] = axs
        self.figs['phase_differences'] = fig
        self.plot_dicts['phase_differences'] = {
            'plotfn': self._plot_phase_differences_figure 
        }

        if self.analyze_parity_model:
            fig, axs = plt.subplots(1, 1, dpi=120, figsize=(6,4))
            self.axs_dict['parity_model'] = axs
            self.figs['parity_model'] = fig
            self.plot_dicts['parity_model'] = {
                'plotfn': self._plot_parity_model_figure 
            }

            fig, axs = plt.subplots(1, 1, dpi=120, figsize=(6,4))
            self.axs_dict['parity_model_error'] = axs
            self.figs['parity_model_error'] = fig
            self.plot_dicts['parity_model_error'] = {
                'plotfn': self._plot_parity_model_error_figure 
            }


    def _plot_main_figure(self, ax, **kw):
        fig = ax[0].get_figure()
        qoi = self.proc_data_dict['quantities_of_interest']
        cases = qoi['control_cases']
        angles = qoi['measured_angles']
        colors = [mpl.cm.hsv(x) for x in np.linspace(0,1,len(cases)+1)]

        # adjust sizes based on number of lines to plot
        if len(qoi['control_cases']) > 4:
            context_dict = {'lines.linewidth': 1, 
                            'lines.markersize': 6,
                            'legend.fontsize': 8,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 16,
                            'errorbar.capsize': 2
                            }
        else:
            context_dict = {'lines.linewidth': 2, 
                            'lines.markersize': 14,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 12,
                            'axes.labelsize': 12,
                            'errorbar.capsize': 2
                            }

        with plt.rc_context(context_dict):
            for i, qb in enumerate([qoi['target_qubit']] + qoi['control_qubits']):
                self._plot_cal_pnts(
                        ax[i], 
                        self.proc_data_dict['calibration_point_plotting_indices'], 
                        [fr"$\vert{{{case}}} \rangle$" for case in self.proc_data_dict['calibration_point_cases']],
                        self.proc_data_dict['normalized_cal_data'][qb],
                        fontsize=8
                    )

                for j, case in enumerate(cases):
                    if 0 == i:
                        ax[i].plot(angles, self.fit_res[case].best_fit, 
                            color=colors[j], alpha=0.3, marker=None, linestyle='--')
                        ax[i].plot(angles, self.proc_data_dict['normalized_data'][qb][case], 
                            label=fr"$\varphi(\vert{cases[j]}\rangle)$: {qoi['oscillation_phases'][case]}", 
                            color=colors[j], alpha=0.4, marker='.', linestyle=None)
                        ax[i].legend(bbox_to_anchor=(1.05, .98), loc='upper left', frameon=False, fontsize=14)
                        # ax[0].text(1.1, 0.98, fr"$|{{{','.join(qoi['control_qubits'])}}}\rangle$", transform=ax[0].transAxes, fontsize=12)
                    else:
                        ax[i].plot(angles, self.proc_data_dict['normalized_data'][qb][case],
                            color=colors[j], alpha=0.4, marker='.')
                        hline = ax[i].hlines(qoi['mean_missing_fraction_per_qubit'][qb].n, angles[0], angles[-1], 
                            colors='k', linestyles='--', alpha=0.8)
                        # ax[i].text(1.1, 0.9, 'Missing frac : {:.1f} %'.format(mean_miss_frac*100), 
                        #            transform=ax[i].transAxes, fontsize=12)
                        ax[i].legend([hline], [f"Missing fraction: {qoi['mean_missing_fraction_per_qubit'][qb]*100} %"], loc='upper left')

                    ax[i].set_ylabel(fr"$P_{{exc}}$ of {qb} (%)")
                    ax[i].grid(True, linestyle=':', linewidth=0.9, alpha=0.8)

            ax[-1].set_xlabel(rf"Phase (deg), Calibration points ($\vert {qoi['target_qubit'] + ',' + ','.join(qoi['control_qubits'])} \rangle$)")
            fig.suptitle(f"{self.t_start}\nParity check {qoi['target_qubit']} - {qoi['control_qubits']}",
                #x = 0.38, y = 1.04,
                fontsize=14)
            fig.tight_layout()
        return


    def _plot_control_qubit_excitations_figure(self, ax, **kw):
        qoi = self.proc_data_dict['quantities_of_interest']
        if len(qoi['control_qubits']) == 1:
            ax = [ax]
        fig = ax[0].get_figure()

        with plt.rc_context({'lines.linewidth': 1, 
                            'lines.markersize': 8,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 10,
                            'axes.labelsize': 18,
                            'errorbar.capsize': 2
                            }):
            for i, qb in enumerate(qoi['control_qubits']):
                ax[i].errorbar([fr"$\vert{{{case}}} \rangle$" for case in qoi['control_cases']],
                    100*np.array([qoi['mean_population_per_qubit_per_case'][qb][case].n for case in qoi['control_cases']]),
                    yerr=100*np.array([qoi['mean_population_per_qubit_per_case'][qb][case].s for case in qoi['control_cases']]),
                    linestyle='-', marker='o', color='b', alpha=0.6)
                ax[i].set_ylabel(fr"$P_{{exc}}$ of {qb} (%)")
                # ax[i].set_yticks(np.arange(0, np.ceil(ax[i].get_ylim()[-1]), 2))
                # ax[i].sharey(True)
                ax[i].tick_params(axis='x', labelrotation=45)
                # ax[i].grid(True, linestyle=':', linewidth=0.9, alpha=0.7)

                # missing fraction figure
                ax[-1].set_xlabel(fr"Cases ($\vert {','.join(qoi['control_qubits'])} \rangle$)")
                fig.suptitle(f"{self.t_start}\nParity check {qoi['target_qubit']} - {qoi['control_qubits']}\nControl qubit excitations by case",
                    #x = 0.38, y = 1.04,
                    fontsize=14)
        return

    def _plot_phase_differences_figure(self, ax, **kw):
        qoi = self.proc_data_dict['quantities_of_interest']
        cases = qoi['control_cases']

        with plt.rc_context({'lines.linewidth': 1, 
                            'lines.markersize': 8,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 14,
                            'errorbar.capsize': 2
                            }):
            ax.errorbar(cases, 
                [qoi['wrapped_phases'][case].n for case in cases], 
                yerr=[qoi['wrapped_phases'][case].s for case in cases],
                marker='o', markersize=8, linestyle='-', alpha=0.7)

            # mark the target phases
            ax.hlines([0,180], cases[0], cases[-1], 
                colors='k', linestyles='--', alpha=0.5)
            ax.plot(cases, qoi['target_phases'].values(),
                color='r', marker='*', markersize=8, linestyle='None', alpha=0.6)

            ax.set_xlabel(rf"Cases ($\vert {','.join(qoi['control_qubits'])} \rangle$)")
            ax.set_xticklabels([fr"$\vert{{{case}}} \rangle$" for case in cases])
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_ylabel(r"Conditional phase (deg)")
            ax.set_yticks(np.arange(-30, ax.get_ylim()[-1], 30))
            ax.set_title(f"{self.t_start}\nParity check {qoi['target_qubit']} - {qoi['control_qubits']}\nPhase differences by case")
            ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)
        return

    def _plot_parity_model_figure(self, ax, **kw):
        qoi = self.proc_data_dict['quantities_of_interest']

        with plt.rc_context({'lines.linewidth': 1, 
                            'lines.markersize': 6,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 14,
                            'errorbar.capsize': 2
                            }):
            ax.errorbar(qoi['parity_model']['model_terms'], 
                [x.n for x in qoi['parity_model']['phases_model']], 
                yerr=[x.s for x in qoi['parity_model']['phases_model']], 
                linestyle='-', marker='o', color='b', alpha=0.7, markersize=8, linewidth=2, zorder=0, label='Estimated')
            ax.plot(qoi['parity_model']['model_terms'], 
                qoi['parity_model']['phases_model_ideal'], 
                linestyle='-', marker='*', color='r', alpha=0.5, zorder=1, label='Ideal')
            # ax.set_xticklabels(qoi['parity_model']['model_terms'])
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_ylabel(r"Parity model phase (deg)")
            ax.set_xlabel(r"Parity check model term")
            ax.set_yticks(np.arange(-90,360.01,45))
            ax.set_title(f"{self.t_start}\nParity check {qoi['target_qubit']} - {qoi['control_qubits']}\nParity model phases")
            ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)
            ax.legend(loc=0)
        return

    def _plot_parity_model_error_figure(self, ax, **kw):
        qoi = self.proc_data_dict['quantities_of_interest']

        with plt.rc_context({'lines.linewidth': 2, 
                            'lines.markersize': 6,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 14,
                            'errorbar.capsize': 4
                            }):
            ax.errorbar(qoi['parity_model']['model_terms'], 
                [x.n for x in qoi['parity_model']['model_errors']],
                yerr=[x.s for x in qoi['parity_model']['model_errors']],
                marker='o', linestyle='-', color='b', alpha=0.5)
            ax.set_xticklabels(qoi['parity_model']['model_terms'])
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0]), np.ceil(ax.get_ylim()[-1]), 1))
            ax.set_ylabel(r"Phase error (deg)")
            ax.set_xlabel(r"Parity check model term")
            ax.set_title(f"{self.t_start}\nParity check {qoi['target_qubit']} - {qoi['control_qubits']}\nParity model phase errors")
            ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)
        return

    def _plot_cal_pnts(self, ax, x_inds, x_labels, cal_data, fontsize):
        deg_sign = u"\N{DEGREE SIGN}"
        angles = self.proc_data_dict['quantities_of_interest']['measured_angles']
        ax.plot(x_inds, cal_data, linestyle='-', marker='o', alpha=0.7)
        ax.set_xticks(np.concatenate([angles, x_inds]))
        ax.set_xticklabels(["{:3.0f}".format(ang) + deg_sign for ang in angles]
                           + x_labels, fontsize=fontsize)
        ax.tick_params(axis='x', labelrotation=45)
        return


class Parity_Model_Optimization_Analysis(BaseDataAnalysis):

    def __init__(
            self,
            label: str = "Parity_model_optimization",
            t_start: str = None,
            t_stop: str = None,
            options_dict: dict = None,
            extract_only: bool = False,
            auto: bool = True
        ):

        super().__init__(
            t_start=t_start,
            t_stop=t_stop,
            label=label,
            options_dict=options_dict,
            close_figs=True,
            extract_only=extract_only,
            save_qois=True,
            do_fitting=True,
        )

        self.params_dict = {
            # "xlabels": "parameter_names",
            # "xunits": "parameter_units",
            # "sweep_points": "sweep_points",
            # "measurementstring": "measurementstring",
            # "value_names": "value_names",
            # "value_units": "value_units",
            # "measured_values": "measured_values",
        }

        for extra_key in ['target_qubit',
                        'control_qubits',
                        'sweep_qubit',
                        # 'sweep_points',
                        'old_B_value',
                        'model_terms',
                        ]:
            self.params_dict[extra_key] = f"Experimental Data.Experimental Metadata.{extra_key}"

        if auto:
            self.run_analysis()

        return

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = ma.a_tools.get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names'),
                      'measurement_name': ('MC settings', 'attr:measurement_name'),
                      'metadata': ('Experimental Data/Experimental Metadata', 'group')
                      }

        # explicit metadata keys:
        # for extra_key in ['target_qubit',
        #                 'control_qubits',
        #                 'sweep_qubit',
        #                 'old_B_value',
        #                 'model_terms',
        #                 ]:
        #     param_spec[extra_key] = (f'Experimental Data/Experimental Metadata/{extra_key}', 'dset') 

        self.raw_data_dict = hd5.extract_pars_from_datafile(data_fp, param_spec)

        # parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
        return

    def process_data(self):
        self.proc_data_dict = dict()
        # values stored in quantities of interest will be saved in the data file
        self.proc_data_dict['quantities_of_interest'] = dict()
        qoi = self.proc_data_dict['quantities_of_interest']
        self.proc_data_dict['sweep_points'] = self.raw_data_dict['data'][:, 0]
        self.proc_data_dict['measured_model_errors'] = self.raw_data_dict['data'][:, 1:]

        # these values are saved as extra metadata of the experiment data, and are nested deeply
        qoi['sweep_qubit'] = self.raw_data_dict['metadata']['sweep_qubit'][0]
        qoi['target_qubit'] = self.raw_data_dict['metadata']['target_qubit'][0]
        qoi['old_B_value'] = self.raw_data_dict['metadata']['old_B_value'][0]
        qoi['control_qubits'] = self.raw_data_dict['metadata']['control_qubits']

        qoi['model_terms'] = [x.decode('utf-8').split('_')[-1] for x in self.raw_data_dict['value_names']]
        qoi['sweep_qubit_index'] = qoi['model_terms'].index(qoi['target_qubit']+','+qoi['sweep_qubit'])
        qoi['sweep_qubit_model_errors'] = self.proc_data_dict['measured_model_errors'][:, qoi['sweep_qubit_index']]
        return


    def process_data_old(self):
        print(self.raw_data_dict)
        figdir = r'D:\Experiments\202012_Aurelius\Data\Figures'
        qb = self.raw_data_dict['sweep_qubit'][0][0][0]
        target_qubit = self.raw_data_dict['target_qubit'][0][0][0]
        qubit_readout_order = self.raw_data_dict['control_qubits'][0].flatten()
        print(qubit_readout_order)
        trange = self.timestamps

        # get index of sweep qubit within the data saved according to readout order
        # dqb_ind = qubit_readout_order.index(qb) #len(qubit_sweep_order) - 1 - j # - 1 if list_timestamps is for single sweep
        # arr_dqb_ind = dqb_ind + 1
        # value_names format is 'model_error_X1,D1,..'
        arr_dqb_ind = [x.split('_')[-1] for x in self.raw_data_dict['value_names'][0]].index(target_qubit+','+qb)
        print(arr_dqb_ind)
        B = self.raw_data_dict['sweep_points'][0]
        n_sweep_points = len(B)
        old_B = self.raw_data_dict['old_B_value'][0][0]

        # model_phases = np.empty((len(B), 2**len(qubit_readout_order)))
        # ideal_phases = np.empty(2**len(qubit_readout_order)) 

        model_phase_errors = self.raw_data_dict['measured_values'][0].T
        print(model_phase_errors)
        xticklabels = self.raw_data_dict['value_names'][0]

        # for ts in trange:
        #     a = ma2.Parity_Check_Analysis(ts, target_qubit=target_qubit, extract_only=True)
        #     qubit_readout_order = a.proc_data_dict['quantities_of_interest']['measured_qubit_order']
        #     model_phases[i] = a.proc_data_dict['quantities_of_interest']['parity_model']['phases_model']
        #     ideal_phases = a.proc_data_dict['quantities_of_interest']['parity_model']['phases_model']
        #     xticklabels = a.proc_data_dict['quantities_of_interest']['parity_model']['model_terms']

        cmap = plt.get_cmap("viridis", n_sweep_points)
        # colors = [mpl.cm.viridis(x) for x in np.linspace(0,1,len(B))]
        # cmap = plt.get_cmap("viridis", len(B))
        # norm = mpl.colors.BoundaryNorm(B, len(B))
        norm = mpl.colors.BoundaryNorm(np.arange(n_sweep_points+1)+0.5, n_sweep_points)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        # phase error figure
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6,4))
        for i, phase_vector in enumerate(model_phase_errors):
            ax.plot(xticklabels, phase_vector, 
                marker='o', markersize=8, alpha=0.55, color=cmap(i), label=f"B={B[i]:.2f}")
        
        ax.axvline(xticklabels[arr_dqb_ind], linestyle='--', color='k', alpha=0.4)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_ylabel(r"Phase error (deg)", fontsize=12)
        ax.set_title(f"{trange[0]}-{trange[-1]}\nParity check {target_qubit} - {qubit_readout_order}\nSweeping B({qb})")
        # ax.legend(bbox_to_anchor=(1.02,1.1), loc='upper left', frameon=False, fontsize=10)
        # fig.colorbar(sm, ticks=np.around(B+np.diff(B).mean()/2, 3), label=f"B({qb})", extend='both') 
        cbar = fig.colorbar(sm, ticks=np.arange(1, len(B)+1), extend='both')
        cbar.ax.set_yticklabels([f"{x:.2f}" for x in B], fontsize=8)
        cbar.ax.set_title(f"B({qb})")
        
        ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)
        # ax.text(0.55,0.9,"  Optimal B = {:.6f}\n@ {}".format(B[np.abs(model_phases[:,arr_dqb_ind]+90).argmin()],
        #                                                      trange[np.abs(model_phases[:,arr_dqb_ind]+90).argmin()]),
        #        transform = ax.transAxes)

        fn = join(figdir, f"parity_check_error_vs_B({qb})_{target_qubit[0]}_{qubit_readout_order}_{trange[0]}_{trange[-1]}.png")
        fig.savefig(fn, dpi=300, bbox_inches='tight', format='png')
        
        # B lever arm figure
        # model_phase_errors = model_phases[:,arr_dqb_ind] - ideal_phases[arr_dqb_ind]
        # fit = scipy.optimize.curve_fit(lambda x,a,b: a*x+b,  B,  ll)
        def func(x, a, b):
            return a * x + b
        popt, pcov = curve_fit(func, B, model_phase_errors[:,arr_dqb_ind])
        B_opt = -popt[-1]/popt[0]

        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6,4))
        ax.set_title(f"{trange[0]}-{trange[-1]}\nParity check {target_qubit} - {qubit_readout_order}\nSweeping B({qb})")
        ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)
        ax.plot(B, func(B, *popt), 'r-',label='fit')
        ax.plot(B, model_phase_errors[:,arr_dqb_ind] , marker='o', markersize=8, alpha=0.6, color='b', label=f"Slope: {popt[0]:.4f} deg/a.u.\nOffset: {popt[1]:.4f} deg")
        ax.axhline(0, linestyle='--', color='k', alpha=0.7)
        ax.axvline(B_opt, linestyle='--', color='k', alpha=0.7)
        ax.plot(B_opt, func(B_opt, *popt), marker='x', markersize=8, color='k', label=f"Optimal B = {B_opt:.8f}")
        ax.set_ylabel(r"Phase error (deg)", fontsize=12)
        ax.set_xlabel(r"B (a.u.)", fontsize=12)
        ax.legend(loc=0, frameon=False, fontsize=10)
        print(B_opt)

        fn = join(figdir, f"parity_check_error_vs_B_slope({qb})_{target_qubit}_{qubit_readout_order}_{trange[0]}_{trange[-1]}.png")
        fig.savefig(fn, dpi=300, bbox_inches='tight', format='png')

        print("B_opt", B_opt)
        print("B_old", old_B)
        print(f"Relative change: {(np.array(B_opt)/np.array(old_B)-1)*100} %")

        self.proc_data_dict['quantities_of_interest'] = {}
        self.proc_data_dict['quantities_of_interest']['optimal_B'] = B_opt
        return

    def prepare_fitting(self):
        qoi = self.proc_data_dict['quantities_of_interest']
        self.fit_dicts = dict()
        lin_mod = lmfit.models.LinearModel()

        self.fit_dicts['optimal_B_fit'] = {
            'model': lin_mod,
            'guess_dict': {'intercept': {'value': 0}, 'slope': {'value': 1/qoi['old_B_value'] if qoi['old_B_value'] != 0 else 1}},
            'fit_xvals': {'x': self.proc_data_dict['sweep_points']},
            'fit_yvals': {'data': qoi['sweep_qubit_model_errors']},
            # 'weights': np.array([1/x.s for x in qoi['sweep_qubit_model_errors']])
        }
        return

    def analyze_fit_results(self):
        qoi = self.proc_data_dict['quantities_of_interest']
        fit = self.fit_res['optimal_B_fit']

        fit_slope = ufloat(fit.params['slope'].value, fit.params['slope'].stderr)
        fit_intercept = ufloat(fit.params['intercept'].value, fit.params['intercept'].stderr)
        qoi['optimal_B_value'] = -1 * fit_intercept/fit_slope
        qoi['relative_B_change'] = 100 * (qoi['optimal_B_value']/qoi['old_B_value'] - 1)
        return

    def run_post_extract(self):
        # NOTE: we need to overload run_post_extract here
        #       to specify our own axs_dict to the plot method
        self.prepare_plots()
        self.plot(key_list='auto', axs_dict=self.axs_dict)
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True)
                )
        return

    def prepare_plots(self):
        self.axs_dict = dict()
        qoi = self.proc_data_dict['quantities_of_interest']

        fig, axs = plt.subplots(1, 1, dpi=120, figsize=(6,4))
        self.axs_dict['model_errors_vs_B_sweep'] = axs
        self.figs['model_errors_vs_B_sweep'] = fig
        self.plot_dicts['model_errors_vs_B_sweep'] = {
            'plotfn': self._plot_sweep_figure
        }
        fig, axs = plt.subplots(1, 1, dpi=120, figsize=(6,4))
        self.axs_dict['optimal_B_fit'] = axs
        self.figs['optimal_B_fit'] = fig
        self.plot_dicts['optimal_B_fit'] = {
            'plotfn': self._plot_fit_figure
        }
        return

    def _plot_sweep_figure(self, ax, **kw):
        fig = ax.get_figure()
        qoi = self.proc_data_dict['quantities_of_interest']
        n_sweep_points = len(self.proc_data_dict['sweep_points'])

        # set up discrete colormap
        cmap = plt.get_cmap("viridis", n_sweep_points)
        norm = mpl.colors.BoundaryNorm(np.arange(n_sweep_points+1)+0.5, n_sweep_points)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        with plt.rc_context({'lines.linewidth': 1,
                            'lines.markersize': 8,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 14,
                            'errorbar.capsize': 2
                            }):
            for i, phase_vector in enumerate(self.proc_data_dict['measured_model_errors']):
                ax.plot(qoi['model_terms'], phase_vector,
                    marker='o', alpha=0.55, color=cmap(i),
                    label=f"B={self.proc_data_dict['sweep_points'][i]:.2f}")

            ax.axvline(qoi['model_terms'][qoi['sweep_qubit_index']], linestyle='--', color='k', alpha=0.5)
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_ylabel(r"Phase error (deg)", fontsize=12)
            # ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0]), np.ceil(ax.get_ylim()[-1]), 5))
            ax.set_title(f"{self.timestamps[0]}"
                        f"\nParity check {qoi['target_qubit']} - {qoi['control_qubits']} model errors"
                        f"\nSweeping B of {qoi['target_qubit']}-{qoi['sweep_qubit']}")
            ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)
            # label discrete colorbar
            cbar = fig.colorbar(sm, ticks=np.arange(1, n_sweep_points+1), extend='both')
            cbar.ax.set_yticklabels([f"{x:.2f}" for x in self.proc_data_dict['sweep_points']], fontsize=8)
            cbar.ax.set_title(f"B({qoi['sweep_qubit']})")

        return

    def _plot_fit_figure(self, ax ,**kw):
        fig = ax.get_figure()
        qoi = self.proc_data_dict['quantities_of_interest']
        fit = self.fit_res['optimal_B_fit']

        with plt.rc_context({'lines.linewidth': 2,
                            'lines.markersize': 8,
                            'legend.fontsize': 12,
                            'xtick.labelsize': 8,
                            'axes.labelsize': 14,
                            'errorbar.capsize': 2
                            }):
            ax.plot(self.proc_data_dict['sweep_points'], fit.best_fit,
                marker=None, alpha=0.8, color='r', 
                label="Linear fit,"
                        f"\nSlope: {fit.best_values['slope']:.2f} deg/a.u."
                        f"\nOffset: {fit.best_values['intercept']:.2f} deg")

            ax.plot(self.proc_data_dict['sweep_points'], qoi['sweep_qubit_model_errors'],
                marker='o', alpha=0.6, color='b',
                label="Measured model error")
            ax.axhline(0, linestyle='--', color='k', alpha=0.7)
            ax.axvline(qoi['optimal_B_value'].n, linestyle='--', color='k', alpha=0.7)
            ax.plot(qoi['optimal_B_value'].n, 
                fit.eval(x=qoi['optimal_B_value'].n, **fit.best_values),
                marker='x', color='k', label=f"Optimal B = {qoi['optimal_B_value']:.8f}")

            ax.set_xlabel(r"B (a.u.)")
            ax.set_ylabel(r"Phase error (deg)")
            # ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0]), np.ceil(ax.get_ylim()[-1]), 5))
            # ax.set_xticks(np.arange(np.floor(ax.get_xlim()[0]), np.ceil(ax.get_xlim()[-1]), 0.1))
            ax.set_title(f"{self.timestamps[0]}"
                        f"\nParity check {qoi['target_qubit']} - {qoi['control_qubits']} model errors"
                        f"\nSweeping B of {qoi['target_qubit']}-{qoi['sweep_qubit']}")
            ax.legend(loc=0, frameon=False)
            ax.grid(True, linestyle=':', linewidth=0.9, alpha=0.8)

        return


class Parity_Check_Fidelity_Analysis():

    def __init__(
            self, 
            timestamp: str = None, 
            label: str = 'Parity_check_fidelity', 
            ro_threshold_corrections: dict = None,
            ps_threshold_corrections: dict = None
            ):
        self.label = label
        if timestamp:
            self.timestamp = timestamp
        else:            
            self.timestamp = ma.a_tools.latest_data(self.label, return_timestamp=True)[0]
        
        self.file_path = ma.a_tools.get_datafilepath_from_timestamp(self.timestamp)
        self.ro_threshold_corrections = ro_threshold_corrections
        self.ps_threshold_corrections = ps_threshold_corrections
        
        self.data = hd5.extract_pars_from_datafile(
            self.file_path, 
            param_spec={'data': ('Experimental Data/Data', 'dset'),
                        'value_names': ('Experimental Data', 'attr:value_names'),
                        })

        self.qubits = [ name.split(' ')[-1].decode('utf-8') for name in self.data['value_names'] ]
        self.ro_thresholds = hd5.extract_pars_from_datafile(
            self.file_path, 
            param_spec={
                f'threshold_{qb}': (f'Instrument settings/{qb}', 'attr:ro_acq_threshold') 
                for qb in self.qubits
            })
        
        self.result = self.fidelity_analysis()
        return

    def fidelity_analysis(self):
        res_dict = {}
        # Sort raw shots
        names = [ name[-2:].decode() for name in self.data['value_names'] ]
        Shots_data = {}
        Shots_init = {}
        for i, name in enumerate(names):
            Shots_init[name] = self.data['data'][:,i+1][0::2]
            Shots_data[name] = self.data['data'][:,i+1][1::2]
            
        # Correct thresholds
        ro_thresholds = np.array([ float(self.ro_thresholds[f'threshold_{name}']) for name in names ])
        ps_thresholds = np.array([ float(self.ro_thresholds[f'threshold_{name}']) for name in names ])
        
        if self.ro_threshold_corrections:
            th_corr = np.array([self.ro_threshold_corrections.get(name, 0) for name in names])
            ro_thresholds += th_corr
        
        if self.ps_threshold_corrections:
            th_corr_ps = np.array([self.ps_threshold_corrections.get(name, 0) for name in names])
            ps_thresholds += th_corr_ps
        
        res_dict['ro_thresholds'] = ro_thresholds
        res_dict['ps_thresholds'] = ps_thresholds

        # Plot histograms
        with plt.rc_context({"axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.titlesize": 20, "grid.alpha": 0.5, "axes.grid": True}):
            fig, axs = plt.subplots(nrows=2, ncols=len(names), sharex='col', sharey='row', figsize=(4*len(names),8))
            for i, name in enumerate(names):
                axs[0,i].hist(Shots_init[name], bins=100, alpha=0.8)
                axs[0,i].axvline(ps_thresholds[i], color='r', ls='--', lw=2)
                axs[1,i].hist(Shots_data[name], bins=100, alpha=0.8)
                axs[1,i].axvline(ro_thresholds[i], color='r', ls='--', lw=2)
                axs[0,i].set_title(name)
                axs[1,i].set_xlabel("Readout signal (V)")
            axs[0,0].set_ylabel("Shots")
            axs[1,0].set_ylabel("Shots")
            fig.tight_layout()
            fig.savefig(Path(self.file_path).parents[0].joinpath(f"{self.label}_shots_{names[0]}-{names[1:]}_{self.timestamp}.png"), 
                        bbox_inches='tight', format='png', dpi=120)
                
        # Digitize shots
        shots_init_dig = {}
        shots_data_dig = {}
        mask = np.ones(len(shots_init[names[0]]))
        for i, name in enumerate(names):
            shots_init_dig[name] = np.array([ +1 if shot < ps_thresholds[i] else np.nan for shot in shots_init[name] ], dtype=float)
            shots_data_dig[name] = np.array([ +1 if shot < ro_thresholds[i] else -1 for shot in shots_data[name] ], dtype=float)
            mask *= shots_init_dig[name]
        
        shots_data_dig[names[0]] *= mask
        fraction_discarded = np.sum(np.isnan(mask))/len(mask)
        res_dict['ps_frac'] = 100 * (1-fraction_discarded)
        
        # Sort shots by states nad calculate fidelity from difference to ideal case
        n_data_qubits = len(names) - 1
        Results = dict.fromkeys(['{:0{}b}'.format(i, n_data_qubits) for i in range(2**n_data_qubits)]) 
        p_msmt = np.zeros(len(Results))
        p_ideal = np.array([1 if case.count('1') % 2 else 0 for case in Results.keys()])
        print("Ideal probabilities per control state: ", p_ideal)
        
        for i, state in enumerate(Results.keys()):
            Results[state] = shots_data_dig[names[0]][i::2**n_data_qubits]
            p_msmt[i] = np.nanmean((1 - Results[state])/2)
        Fidelity = 1 - np.mean(np.abs(p_msmt - p_ideal))
        res_dict['F_assign'] = 100 * Fidelity      
        
        # Plot main state probability plot
        with plt.rc_context({"axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "axes.titlesize": 24,  "figure.titlesize": 22}):
            fig, ax = plt.subplots(figsize=(1.5*len(Results.keys()),6))
            for i, state in enumerate(Results.keys()):
                ax.bar(state, p_msmt[i], color='C0', alpha=0.8, lw=0)
                ax.bar(state, p_ideal[i], fc=(0,0,0,0), edgecolor='k', lw=3, ls='-')

            ax.set_ylabel(fr'{names[0]} P$(|1\rangle)$')
            ax.set_xticklabels(Results.keys(), rotation=45)
            ax.set_yticks(np.arange(0,1.01,0.1))
            ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0,1.01,0.1)])
            ax.set_title(fr"$F_{{assign}} = {{{res_dict['F_assign']:.2f}}}$%, "
                         fr"post-sel. frac.: {res_dict['ps_frac']:.1f}%")
    #         fig.suptitle(f"{names[0]} parity check fidelity, {ts}")
            fig.savefig(Path(self.file_path).parents[0].joinpath(f"{self.label}_{names[0]}-{names[1:]}_{self.timestamp}.png"), 
                        bbox_inches='tight', format='png', dpi=120)
        
        print(f"Post-selected fraction : {res_dict['ps_frac']:.2f} %")
        print(f"Parity assignment fidelity : {res_dict['F_assign']:.2f} %")

        return res_dict
