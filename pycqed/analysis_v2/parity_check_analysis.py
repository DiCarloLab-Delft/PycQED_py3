from os.path import join
from uncertainties import ufloat
import lmfit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
from pprint import pprint
from copy import deepcopy

import pycqed.measurement.hdf5_data as hd5
from pycqed.measurement import optimization as opt

import pycqed.analysis.measurement_analysis as ma
from pycqed.analysis_v2 import  measurement_analysis as ma2
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import fitting_models as fit_mods

class Parity_Check_Analysis():

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
        res_dict['label'] = label
        a.get_naming_and_values()
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
                fn = join(a.folder, label + '_missing_fractions.png')
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
                fn = join(a.folder, label + '.png')
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

                fn = join(a.folder, label + 'phase_diffs' + '.png')
                fig.savefig(fn, dpi=120, bbox_inches='tight', format='png')
                plt.close(fig)

        return res_dict


class Parity_Check_Fidelity_Analysis():

    def __init__(self):

        return
