import os
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
import pycqed.measurement.hdf5_data as h5d
from scipy.optimize import minimize, curve_fit



class Weight2_conditional_oscillation_analysis(ba.BaseDataAnalysis):
    """
    """

    def __init__(self,
                 q_ramsey_idx: int,
                 Ramsey_qubit: str,
                 t_start: str = None, t_stop: str = None,
                 options_dict = None,
                 label: str = '',
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.q_ramsey_idx = q_ramsey_idx
        self.Ramsey_qubit = Ramsey_qubit
        self.do_fitting = True
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) d ata extraction.
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

        Angles = self.raw_data_dict['data'][:-8:4, 0]
        Phases = self.raw_data_dict['data'][:-8, self.q_ramsey_idx]
        Cal_points = self.raw_data_dict['data'][-8:, self.q_ramsey_idx]

        Phase_00 = Phases[0::4]
        Phase_01 = Phases[1::4]
        Phase_10 = Phases[2::4]
        Phase_11 = Phases[3::4]

        if self.q_ramsey_idx == 1:
            zero_lvl = np.mean(Cal_points[[0,1,4,5]])
            one_lvl  = np.mean(Cal_points[[2,3,6,7]])
        elif self.q_ramsey_idx == 2:
            zero_lvl = np.mean(Cal_points[:4])
            one_lvl  = np.mean(Cal_points[4:])
        elif self.q_ramsey_idx == 3:
            zero_lvl = np.mean(Cal_points[[0,2,4,6]])
            one_lvl  = np.mean(Cal_points[[1,3,5,7]])

        Phase_00 = (Phase_00-zero_lvl)/(one_lvl-zero_lvl)
        Phase_01 = (Phase_01-zero_lvl)/(one_lvl-zero_lvl)
        Phase_10 = (Phase_10-zero_lvl)/(one_lvl-zero_lvl)
        Phase_11 = (Phase_11-zero_lvl)/(one_lvl-zero_lvl)
        Cal_points = (Cal_points-zero_lvl)/(one_lvl-zero_lvl)

        self.proc_data_dict['Angles'] = Angles
        self.proc_data_dict['Curve_00'] = Phase_00
        self.proc_data_dict['Curve_01'] = Phase_01
        self.proc_data_dict['Curve_10'] = Phase_10
        self.proc_data_dict['Curve_11'] = Phase_11
        self.proc_data_dict['Cal_points'] = Cal_points

        #############################
        # Fitting data
        #############################
        def func(a, phi, A, B):
            return A*(np.cos(a*np.pi/180 + phi*np.pi/180) + B)/2

        popt_00, pcov_00 = curve_fit(func, Angles, Phase_00)
        popt_01, pcov_01 = curve_fit(func, Angles, Phase_01)
        popt_10, pcov_10 = curve_fit(func, Angles, Phase_10)
        popt_11, pcov_11 = curve_fit(func, Angles, Phase_11)

        if popt_00[0] < -30:
            popt_00[0] += 360
        if popt_01[0] < -30:
            popt_01[0] += 360
        if popt_10[0] < -30:
            popt_10[0] += 360
        if popt_11[0] < -30:
            popt_11[0] += 360

        self.proc_data_dict['popt_00'] = popt_00
        self.proc_data_dict['popt_01'] = popt_01
        self.proc_data_dict['popt_10'] = popt_10
        self.proc_data_dict['popt_11'] = popt_11

        self.qoi = {}
        self.qoi['phase_00'] = popt_00[0]
        self.qoi['phase_01'] = popt_01[0]
        self.qoi['phase_10'] = popt_10[0]
        self.qoi['phase_11'] = popt_11[0]

    def prepare_plots(self):


        self.axs_dict = {}

        fig, axs = plt.subplots(nrows=2,figsize=(6,6), dpi=100)
        axs = axs.ravel()
        fig.patch.set_alpha(0)
        self.axs_dict['Conditional_oscillation']=axs
        self.figs['Conditional_oscillation'] = fig
        self.plot_dicts['Conditional_oscillation']={
                        'plotfn': plot_weight2_cond_oscillation,
                        'Angles': self.proc_data_dict['Angles'],
                        'Phase_00': self.proc_data_dict['Curve_00'],
                        'Phase_01': self.proc_data_dict['Curve_01'],
                        'Phase_10': self.proc_data_dict['Curve_10'],
                        'Phase_11': self.proc_data_dict['Curve_11'],
                        'Cal_points': self.proc_data_dict['Cal_points'],
                        'popt_00': self.proc_data_dict['popt_00'],
                        'popt_01': self.proc_data_dict['popt_01'],
                        'popt_10': self.proc_data_dict['popt_10'],
                        'popt_11': self.proc_data_dict['popt_11'],
                        'Ramsey_qubit': self.Ramsey_qubit,
                        'ts': self.raw_data_dict['timestamps'][0]
                    }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))



class Weight4_conditional_oscillation_analysis(ba.BaseDataAnalysis):
    """
    """

    def __init__(self,
                 q_ramsey_idx: int,
                 Ramsey_qubit: str,
                 t_start: str = None, t_stop: str = None,
                 options_dict = None,
                 label: str = '',
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.q_ramsey_idx = q_ramsey_idx
        self.Ramsey_qubit = Ramsey_qubit
        self.do_fitting = True
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) d ata extraction.
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

        Angles = self.raw_data_dict['data'][::16,0]
        Phases = self.raw_data_dict['data'][:, self.q_ramsey_idx]
        Cal_points = self.raw_data_dict['data'][-8:, self.q_ramsey_idx]

        Phase_curves = { '{:04b}'.format(i) : 0 for i in range(16) }
        for i, state in enumerate(Phase_curves.keys()):
            Phase_curves[state] = Phases[i::16]

        #############################
        # Fitting data
        #############################
        def func(a, phi, A, B):
            return A*(np.cos(a*np.pi/180 + phi*np.pi/180) + B)/2

        Popt = { '{:04b}'.format(i) : 0 for i in range(16) }
        Pcov = { '{:04b}'.format(i) : 0 for i in range(16) }

        for i, state in enumerate(Popt.keys()):
            Popt[state], Pcov[state] = curve_fit(func, Angles, Phase_curves[state])
            if Popt[state][0] < -30 or Popt[state][0] > 360:
                Popt[state][0] = np.mod(Popt[state][0], 360)



        self.proc_data_dict['Angles'] = Angles
        self.proc_data_dict['Curves'] = Phase_curves
        self.proc_data_dict['Popt'] = Popt

    def prepare_plots(self):


        self.axs_dict = {}

        fig, axs = plt.subplots(nrows=2,figsize=(9,6), dpi=100)
        axs = axs.ravel()
        fig.patch.set_alpha(0)
        self.axs_dict['Conditional_oscillation']=axs
        self.figs['Conditional_oscillation'] = fig
        self.plot_dicts['Conditional_oscillation']={
                        'plotfn': plot_weight4_cond_oscillation,
                        'Angles': self.proc_data_dict['Angles'],
                        'Curves': self.proc_data_dict['Curves'],
                        'Popt': self.proc_data_dict['Popt'],
                        'Ramsey_qubit': self.Ramsey_qubit,
                        'ts': self.raw_data_dict['timestamps'][0]
                    }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))


class Single_Oscillation_analysis(ba.BaseDataAnalysis):
    """
    """

    def __init__(self,
                 Ramsey_qubit: str,
                 nr_shots: int,
                 t_start: str = None, t_stop: str = None,
                 options_dict = None,
                 label: str = '',
                 extract_only: bool = False,
                 auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.Ramsey_qubit = Ramsey_qubit
        self.nr_shots = nr_shots
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) d ata extraction.
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

        Angles = self.raw_data_dict['data'][:-4:self.nr_shots, 0]
        Phase = self.raw_data_dict['data'][self.nr_shots//2:-4:self.nr_shots, 2]
        Cal_points = self.raw_data_dict['data'][-4:,2]

        zer_lvl = np.mean(Cal_points[:2])
        one_lvl = np.mean(Cal_points[2:])

        Phase = (Phase-zer_lvl)/(one_lvl-zer_lvl)
        Cal_points = (Cal_points-zer_lvl)/(one_lvl-zer_lvl)

        #############################
        # Fitting data
        #############################
        def func(a, phi, A, B):
            return A*(np.cos(a*np.pi/180 + phi*np.pi/180) + B)/2

        popt, pcov = curve_fit(func, Angles, Phase)

        self.proc_data_dict['Angles'] = Angles
        self.proc_data_dict['Phase'] = Phase
        self.proc_data_dict['Cal_points'] = Cal_points
        self.proc_data_dict['popt'] = popt

        self.qoi = {'Phase': np.mod(popt[0], 360)}

    def prepare_plots(self):

        self.axs_dict = {}

        fig, axs = plt.subplots(figsize=(6,4), dpi=100)
        fig.patch.set_alpha(0)
        self.axs_dict['Conditional_oscillation']=axs
        self.figs['Conditional_oscillation'] = fig
        self.plot_dicts['Conditional_oscillation']={
                        'plotfn': plot_single_oscillation,
                        'Angles': self.proc_data_dict['Angles'],
                        'Phase': self.proc_data_dict['Phase'],
                        'Cal_points': self.proc_data_dict['Cal_points'],
                        'popt': self.proc_data_dict['popt'],
                        'Ramsey_qubit': self.Ramsey_qubit,
                        'ts': self.raw_data_dict['timestamps'][0]
                    }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))


def plot_single_oscillation(Angles,
                            Phase,
                            Cal_points,
                            popt,
                            Ramsey_qubit,
                            ts,
                            ax, **kw):

    def func(a, phi, A, B):
            return A*(np.cos(a*np.pi/180 + phi*np.pi/180) + B)/2

    ax.plot(Angles, func(Angles, *popt),'C0--', label=r'Phase=${:.2f}^o$'.format(\
            np.mod(popt[0], 360)))
    ax.plot(Angles, Phase,'C0o')
    ax.legend(bbox_to_anchor=(1.025, 1), loc='upper left')

    ax.plot(np.arange(360, 440, 20), Cal_points, 'C2o-')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel(r'Prob $|1\rangle$ (deg)')
    ax.set_xticks(np.arange(0, 360, 60))
    ax.set_xticklabels([r'${}^o$'.format(i) for i in np.arange(0, 360, 60)])
    ax.set_title('Qubit {} Oscillation'.format(Ramsey_qubit))
    fig = ax.get_figure()
    fig.suptitle(ts)

def plot_weight2_cond_oscillation(Angles,
                                  Phase_00, Phase_01,
                                  Phase_10, Phase_11,
                                  Cal_points,
                                  popt_00, popt_01,
                                  popt_10, popt_11,
                                  Ramsey_qubit,
                                  ts,
                                  ax, **kw):

    fig = ax[0].get_figure()

    def func(a, phi, A, B):
        return A*(np.cos(a*np.pi/180 + phi*np.pi/180) + B)/2

    ax[0].plot(Angles, func(Angles, *popt_00),'C0--',
               label=r'00 | $\phi={:.1f}^o$'.format(popt_00[0] if popt_00[0]>0
                                                    else popt_00[0]+360))
    ax[0].plot(Angles, func(Angles, *popt_01),'C1--',
               label=r'01 | $\phi={:.1f}^o$'.format(popt_01[0] if popt_01[0]>0
                                                    else popt_01[0]+360))
    ax[0].plot(Angles, func(Angles, *popt_10),'C2--',
               label=r'10 | $\phi={:.1f}^o$'.format(popt_10[0] if popt_10[0]>0
                                                    else popt_10[0]+360))
    ax[0].plot(Angles, func(Angles, *popt_11),'C3--',
               label=r'11 | $\phi={:.1f}^o$'.format(popt_11[0] if popt_11[0]>0
                                                    else popt_11[0]+360))
    ax[0].plot(Angles, Phase_00, 'C0o')
    ax[0].plot(Angles, Phase_01, 'C1o')
    ax[0].plot(Angles, Phase_10, 'C2o')
    ax[0].plot(Angles, Phase_11, 'C3o')
    Cal_pos = [361, 380, 399, 418, 437, 456, 475, 494]
    ax[0].plot(Cal_pos, Cal_points, 'o-')

    ax[0].set_title('Conditional oscillations {}'.format(Ramsey_qubit))
    ax[0].set_xticks(np.concatenate((np.arange(0, 341, 60), Cal_pos)))
    ax[0].set_xticklabels(['{:}'.format(i) \
            for i in np.arange(0, 341, 60) ]+['000', '001', '010', '011',
                                              '100', '101', '110', '111'],
            rotation=60)

    ax[0].legend(bbox_to_anchor=(1.025, 1), loc='upper left')
    ax[1].set_title('Curve phases')
    ax[1].axhline(180, color='black', linestyle='--', alpha=.5)
    ax[1].axhline(0, color='black', linestyle='--', alpha=.5)
    ax[1].plot([popt_00[0], popt_01[0], popt_10[0], popt_11[0]], 'o-')
    ax[1].set_xticks([0,1,2,3])
    ax[1].set_xticklabels(['00', '01', '10', '11'])

    fig.suptitle('Conditional oscillations {}'.format(ts), y=1.05)
    fig.tight_layout()

def plot_weight4_cond_oscillation(Angles,
                                  Curves,
                                  Popt,
                                  Ramsey_qubit,
                                  ts,
                                  ax, **kw):

    fig = ax[0].get_figure()

    def func(a, phi, A, B):
        return A*(np.cos(a*np.pi/180 + phi*np.pi/180) + B)/2


    ax[1].axhline(180, color='black', linestyle='--', alpha=.5)
    ax[1].axhline(0, color='black', linestyle='--', alpha=.5)

    for i, state in enumerate(Popt.keys()):
        ax[0].plot(Angles, func(Angles, *Popt[state]),'--', color='C{}'.format(i),
            label=r'{} | $\phi={:.1f}^o$'.format(state, Popt[state][0]))
        ax[0].plot(Angles, Curves[state], 'o', color='C{}'.format(i))


    ax[0].set_title('Conditional oscillations {}'.format(Ramsey_qubit))
    ax[0].set_xticks(np.arange(0, 341, 60))
    ax[0].set_xticklabels(['{:}'.format(i) \
            for i in np.arange(0, 341, 60) ], rotation=60)
    ax[0].legend(bbox_to_anchor=(1.025, 1), loc='upper left', ncol=2)
    ax[1].set_title('Curve phases')
    ax[1].plot([Popt[state][0] for state in Popt.keys() ], 'o-')
    ax[1].set_xticks(np.arange(0, 16))
    ax[1].set_xticklabels([state for state in Popt.keys()], rotation=60)

    fig.suptitle('Conditional oscillations {}'.format(ts), y=1.05)
    fig.tight_layout()

