import os
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools


class SingleQubitGBT_analysis(ba.BaseDataAnalysis):
	"""
	Analysis for Chevron routine
	"""
	def __init__(self,
				 Qubits: list,
				 t_start: str = None,
				 t_stop: str = None,
				 label: str = '',
				 options_dict: dict = None, 
				 extract_only: bool = False,
				 auto=True):

		super().__init__(t_start=t_start, 
						 t_stop=t_stop,
						 label=label,
						 options_dict=options_dict,
						 extract_only=extract_only)
		self.Qubits = Qubits
		if auto:
			self.run_analysis()

	def extract_data(self):
		self.raw_data_dict = {}
		self.get_timestamps()
		self.timestamp = self.timestamps[0]
		data_fp = get_datafilepath_from_timestamp(self.timestamp)
		self.raw_data_dict['timestamps'] = self.timestamps
		self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
		# Extract last measured metrics
		for q in self.Qubits:
			self.raw_data_dict[q] = {}
			# Get AllXY data
			fp_allxy = a_tools.latest_data(contains=f"AllXY_{q}",
										   older_than=self.timestamp)
			label_allxy = fp_allxy.split('\\')[-1]
			allxy = ma.AllXY_Analysis(label=label_allxy,
									  extract_only=True)
			self.raw_data_dict[q]['allxy'] = allxy.corr_data
			self.raw_data_dict[q]['allxy_err'] = allxy.deviation_total
			# Analyse RB experiment
			fp_rb = a_tools.latest_data(contains=f"seeds_{q}",
										older_than=self.timestamp)
			label_rb = fp_rb.split('\\')[-1]
			rb = ma2.RandomizedBenchmarking_SingleQubit_Analysis(
					label=label_rb,
					rates_I_quad_ch_idx=0,
					cal_pnts_in_dset=np.repeat(["0", "1", "2"], 2),
					extract_only=True)
			N_clf = rb.proc_data_dict['ncl']
			M0 = rb.proc_data_dict['M0']
			X1 = rb.proc_data_dict['X1']
			_err_key = [k for k in rb.proc_data_dict['quantities_of_interest'].keys()\
						if 'eps_g' in k ][0]
			_L1_key = [k for k in rb.proc_data_dict['quantities_of_interest'].keys()\
					   if 'L1' in k ][0]
			SQG_err = rb.proc_data_dict['quantities_of_interest'][_err_key]
			L1_err = rb.proc_data_dict['quantities_of_interest'][_L1_key]
			self.raw_data_dict[q]['N_clf'] = N_clf
			self.raw_data_dict[q]['M0'] = list(M0.values())[0]
			self.raw_data_dict[q]['X1'] = list(X1.values())[0]
			self.raw_data_dict[q]['SQG_err'] = SQG_err.nominal_value
			self.raw_data_dict[q]['L1_err'] = L1_err.nominal_value
			# Analyse SSRO experiment
			fp_ssro = a_tools.latest_data(contains=f"SSRO_f_{q}",
										  older_than=self.timestamp)
			label_ssro = fp_ssro.split('\\')[-1]
			ssro = ma2.ra.Singleshot_Readout_Analysis(
						label=label_ssro,
						qubit=q,
						qubit_freq=6e9,
						heralded_init=True,
						f_state=True,
						extract_only=True)
			self.raw_data_dict[q]['RO_err'] = 1-ssro.qoi['F_a']
			self.raw_data_dict[q]['RO_n0'] = ssro.proc_data_dict['h0']
			self.raw_data_dict[q]['RO_n1'] = ssro.proc_data_dict['h1']
			# Analyse T1
			fp_t1 = a_tools.latest_data(contains=f"T1_{q}",
										older_than=self.timestamp)
			label_t1 = fp_t1.split('\\')[-1]
			t1 = ma.T1_Analysis(label = label_t1,
				auto=True, close_fig=True, extract_only=True)
			self.raw_data_dict[q]['t_T1'] = t1.sweep_points[:-4]
			self.raw_data_dict[q]['p_T1'] = t1.normalized_data_points
			self.raw_data_dict[q]['T1'] = t1.T1
			# Analyse T2 echo
			fp_t2 = a_tools.latest_data(contains=f"echo_{q}",
										older_than=self.timestamp)
			label_t2 = fp_t2.split('\\')[-1]
			t2 = ma.Echo_analysis_V15(label=label_t2,
				auto=True, close_fig=True, extract_only=True)
			self.raw_data_dict[q]['t_T2'] = t2.sweep_points[:-4]
			self.raw_data_dict[q]['p_T2'] = t2.normalized_data_points
			self.raw_data_dict[q]['T2'] = t2.fit_res.params['tau'].value

	def process_data(self):
		pass

	def prepare_plots(self):
		self.axs_dict = {}
		fig = plt.figure(figsize=(3,3), dpi=150)
		axs = {}
		n_metrics = 6 # [SQG err, SQG leakage, allxy, T1, T2]
		_n = 1.4*n_metrics
		self.figs['Single_Qubit_performance_overview'] = fig
		for q in self.Qubits:
			ax = _add_singleQ_plot(q, _n, fig, axs)
			self.axs_dict[f'{q}'] = ax
			# Plot single qubit gate error
			self.plot_dicts[f'{q}_SQG_er']={
				'plotfn': _plot_SQG_error,
				'ax_id': f'{q}',
				'N_clf': self.raw_data_dict[q]['N_clf'],
				'M0': self.raw_data_dict[q]['M0'],
				'SQG_err': self.raw_data_dict[q]['SQG_err'],
				'row': 0,
				'n': _n,
			}
			# Plot single_qubit gate leakage
			self.plot_dicts[f'{q}_leakage']={
				'plotfn': _plot_SQG_leakage,
				'ax_id': f'{q}',
				'N_clf': self.raw_data_dict[q]['N_clf'],
				'X1': self.raw_data_dict[q]['X1'],
				'SQG_leak': self.raw_data_dict[q]['L1_err'],
				'row': 1,
				'n': _n,
			}
			# Plot single_qubit gate leakage
			self.plot_dicts[f'{q}_allxy']={
				'plotfn': _plot_allxy,
				'ax_id': f'{q}',
				'allxy': self.raw_data_dict[q]['allxy'],
				'allxy_err': self.raw_data_dict[q]['allxy_err'],
				'row': 2,
				'n': _n,
			}
			# Plot single_qubit gate leakage
			self.plot_dicts[f'{q}_T1']={
				'plotfn': _plot_T1,
				'ax_id': f'{q}',
				't': self.raw_data_dict[q]['t_T1'],
				'p': self.raw_data_dict[q]['p_T1'],
				'T1': self.raw_data_dict[q]['T1'],
				'row': 3,
				'n': _n,
			}
			# Plot single_qubit gate leakage
			self.plot_dicts[f'{q}_T2']={
				'plotfn': _plot_T2,
				'ax_id': f'{q}',
				't': self.raw_data_dict[q]['t_T2'],
				'p': self.raw_data_dict[q]['p_T2'],
				'T2': self.raw_data_dict[q]['T2'],
				'row': 4,
				'n': _n,
			}
			# Plot single_qubit gate leakage
			self.plot_dicts[f'{q}_SSRO']={
				'plotfn': _plot_SSRO,
				'ax_id': f'{q}',
				'n0': self.raw_data_dict[q]['RO_n0'],
				'n1': self.raw_data_dict[q]['RO_n1'],
				'RO_err': self.raw_data_dict[q]['RO_err'],
				'row': 5,
				'n': _n,
			}
		n_plts = len(self.Qubits)
		fig.suptitle(f'{self.timestamp}\nSingle-qubit performance', x=0.125+n_plts/2*0.775*1.3, y=2.8, size=18)


	def run_post_extract(self):
		self.prepare_plots()  # specify default plots
		self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
		if self.options_dict.get('save_figs', False):
			self.save_figures(
				close_figs=self.options_dict.get('close_figs', True),
				tag_tstamp=self.options_dict.get('tag_tstamp', True))

def _add_singleQ_plot(qubit, n, fig, axs):
    n_plts = len(axs.keys()) 
    ax = fig.add_subplot(10,10,n_plts+1)
    _pos = ax.get_position()
    # pos = [_pos.x0 + n_plts*_pos.width*1.3, _pos.y0, _pos.width, _pos.width/3*n]
    pos = [0.125 + n_plts*0.775*1.3, 0.11, 0.775, 0.775/3*n]
    ax.set_position(pos)
    axs[qubit] = ax
    axs[qubit].text(0, n/3+.2, f'$\\mathrm{{{qubit[0]}_{qubit[1]}}}$',
                    va='center', ha='left', size=40)
    ax.set_xlim(0,1)
    ax.set_ylim(0,n/3)
    ax.axis('off')
    ax.patch.set_alpha(0)
    return ax

def _plot_SQG_error(
    ax,
    N_clf,
	M0,
	SQG_err,
	n,
	row,
	**kw):
    # Assess pereformance level
    if SQG_err < 0.0025:
        _color = 'C2'
    elif SQG_err < 0.005:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Single qubit gate', ha='left', size=11.5)
    ax.text(.86, (n-row*1.4)/3-.13, 'err.', ha='left', size=11.5)
    ax.text(.375, (n-row*1.4)/3-.3, f'{SQG_err*100:2.1f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{SQG_err*100:2.1f}', ha='left', size=50,
    		color=_color, alpha=.65) # Overlay to give color
    ax.text(.85, (n-row*1.4)/3-.3, '%', ha='left', size=25)
    # RB decay plot
    _x = N_clf/N_clf[-1]
    _y = M0
    ax.plot(_x*0.3+.025,
            (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='-', color=_color, clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}',
                    alpha=.1, lw=0)
    ax.plot(_x*0.3+.025,
            [(.5+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3 for x in _x],
            f'k--', lw=.5)

def _plot_SQG_leakage(
    ax,
    N_clf,
	X1,
	SQG_leak,
	n,
	row,
	**kw):
    # Assess pereformance level
    if SQG_leak < 0.0015:
        _color = 'C2'
    elif SQG_leak < 0.003:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Single qubit gate', ha='left', size=11.5)
    ax.text(.86, (n-row*1.4)/3-.13, 'leak.', ha='left', size=11.5)
    ax.text(.375, (n-row*1.4)/3-.3, f'{abs(SQG_leak)*1e3:2.1f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{abs(SQG_leak)*1e3:2.1f}', ha='left', size=50,
    		color=_color, alpha=.65) # Overlay to give color
    ax.text(0.985, (n-row*1.4)/3-.25, f'-', ha='left', size=15)
    ax.text(.85, (n-row*1.4)/3-.3, f'$10^{{\\:\\:3}}$', ha='left', size=17)
    # RB decay plot
    _x = N_clf/N_clf[-1]
    _y = (1-X1)*2
    ax.plot(_x*0.3+.025,
            (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='-', color=_color, clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}', alpha=.1, lw=0)

def _plot_allxy(
    ax,
    allxy,
    allxy_err,
	n,
	row,
	**kw):
    # Assess pereformance level
    if allxy_err < 0.01:
        _color = 'C2'
    elif allxy_err < 0.02:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Single qubit AllXY', ha='left', size=11.5)
    ax.text(.86, (n-row*1.4)/3-.13, 'err.', ha='left', size=11.5)
    ax.text(.375, (n-row*1.4)/3-.3, f'{abs(allxy_err)*1e2:2.1f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{abs(allxy_err)*1e2:2.1f}', ha='left', size=50,
    		color=_color, alpha=.65) # Overlay to give color
    ax.text(.85, (n-row*1.4)/3-.3, '%', ha='left', size=25)
    # AllXY plot
    _x = np.arange(42)/41
    _y = allxy
    _y_id = np.array([0]*10 + [.5]*24 + [1]*8)
    ax.plot(_x*0.3+.025,
            (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='-', color=_color, clip_on=False)
    ax.plot(_x*0.3+.025,
            (_y_id+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='--', lw=.5, color='k', clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}', alpha=.1, lw=0)

def _plot_T1(
    ax,
    t,
	p,
	T1,
	n,
	row,
	**kw):
    # Assess pereformance level
    if T1 > 10e-6:
        _color = 'C2'
    elif T1 > 5e-6:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Relaxation time', ha='left', size=11.5)
    ax.text(.8, (n-row*1.4)/3-.17, '$\\mathrm{{T_1}}$', ha='left', size=20)
    ax.text(.375, (n-row*1.4)/3-.3, f'{T1*1e6:2.0f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{T1*1e6:2.0f}', ha='left', size=50,
    		color=_color, alpha=.65) # Overlay to give color
    ax.text(.8, (n-row*1.4)/3-.3, '$\\mathrm{{\\mu s}}$', ha='left', size=25)
    # T1 decay plot
    _x = t/t[-1]
    _y = p
    ax.plot(_x*0.3+.025,
            (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='-', color=_color, clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}',
                    alpha=.1, lw=0)

def _plot_T2(
    ax,
    t,
	p,
	T2,
	n,
	row,
	**kw):
    # Assess pereformance level
    if T2 > 15e-6:
        _color = 'C2'
    elif T2 > 7.5e-6:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Echo time', ha='left', size=11.5)
    ax.text(.8, (n-row*1.4)/3-.17, '$\\mathrm{{T_2}}$', ha='left', size=20)
    ax.text(.375, (n-row*1.4)/3-.3, f'{T2*1e6:2.0f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{T2*1e6:2.0f}', ha='left', size=50,
			color=_color, alpha=.65) # Overlay to give color
    ax.text(.8, (n-row*1.4)/3-.3, '$\\mathrm{{\\mu s}}$', ha='left', size=25)
    # T2 decay plot
    _x = t/t[-1]
    _y = 1-p
    _y_env = (1+np.exp(-t/T2))/2
    ax.plot(_x*0.3+.025,
            (_y+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='-', color=_color, clip_on=False)
    ax.plot(_x*0.3+.025,
            (_y_env+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='--', lw=.5, color=_color, clip_on=False)
    ax.plot(_x*0.3+.025,
            (1-_y_env+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='--', lw=.5, color=_color, clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_y_env+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (1-_y_env+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}',
                    alpha=.1, lw=0)

def _plot_SSRO(
        ax,
        RO_err,
        n0, n1,
		n,
		row,
		**kw):
    if RO_err < .015:
        _color = 'C2'
    elif RO_err < .025:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Readout error', ha='left', size=11.5)
    ax.text(.375, (n-row*1.4)/3-.3, f'{RO_err*100:.1f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{RO_err*100:.1f}', ha='left', size=50,
    		color=_color, alpha=0.65) # color overlay
    ax.text(.85, (n-row*1.4)/3-.3, '%', ha='left', size=25)
    # Plot
    _max = max(np.max(n0), np.max(n1))
    _n0 = n0/_max
    _n1 = n1/_max
    _x = np.arange(len(n0))/(len(n0)-1)
    ax.fill_between(_x*0.33+.02,
                    (_n0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    color=_color, alpha=.1)
    ax.fill_between(_x*0.33+.02,
                    (_n1+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    color=_color, alpha=.1)
    ax.fill_between(_x*0.33+.02,
                    (_n0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    color=_color,
                    fc='None',
                    hatch='///', alpha=.3)
    ax.fill_between(_x*0.33+.02,
                    (_n1+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    color=_color,
                    fc='None',
                    hatch='\\\\\\', alpha=.3)
    ax.plot(_x*0.33+.02,
            (_n0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, f'{_color}')
    ax.plot(_x*0.33+.02,
            (_n1+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, f'{_color}')    


class TwoQubitGBT_analysis(ba.BaseDataAnalysis):
	"""
	Analysis for Chevron routine
	"""
	def __init__(self,
				 Qubit_pairs: list,
				 t_start: str = None,
				 t_stop: str = None,
				 label: str = '',
				 options_dict: dict = None, 
				 extract_only: bool = False,
				 auto=True):

		super().__init__(t_start=t_start, 
						 t_stop=t_stop,
						 label=label,
						 options_dict=options_dict,
						 extract_only=extract_only)
		self.Qubit_pairs = Qubit_pairs
		if auto:
			self.run_analysis()

	def extract_data(self):
		self.raw_data_dict = {}
		self.get_timestamps()
		self.timestamp = self.timestamps[0]
		data_fp = get_datafilepath_from_timestamp(self.timestamp)
		self.raw_data_dict['timestamps'] = self.timestamps
		self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
		# Extract last measured metrics
		for q0, q1 in self.Qubit_pairs:
			# Analyse IRB experiment
			fp_base = a_tools.latest_data(contains=f"icl[None]_{q0}_{q1}",
								  		     older_than=self.timestamp)
			fp_int = a_tools.latest_data(contains=f"icl[104368]_{q0}_{q1}",
								  		    older_than=self.timestamp)
			label_base = fp_base.split('\\')[-1]
			label_int = fp_int.split('\\')[-1]
			a = ma2.InterleavedRandomizedBenchmarkingAnalysis(
				        label_base=label_base,
				        label_int=label_int,
						# label_base=f"icl[None]_{q0}_{q1}",
						# label_int=f"icl[104368]_{q0}_{q1}",
						extract_only=True)
			N_clf = a.raw_data_dict['analyses']['base'].proc_data_dict['ncl']
			M_ref = a.raw_data_dict['analyses']['base'].proc_data_dict['M0']['2Q']
			M_int = a.raw_data_dict['analyses']['int'].proc_data_dict['M0']['2Q']
			X1_ref = a.raw_data_dict['analyses']['base'].proc_data_dict['X1']['2Q']
			X1_int = a.raw_data_dict['analyses']['int'].proc_data_dict['X1']['2Q']
			TQG_err = a.proc_data_dict['quantities_of_interest']['eps_CZ_simple'].nominal_value
			L1_err = a.proc_data_dict['quantities_of_interest']['L1_CZ'].nominal_value
			self.raw_data_dict[(q0, q1)] = {'N_clf':N_clf, 'M_ref':M_ref, 'M_int':M_int, 'TQG_err':abs(TQG_err),
											'X1_ref':X1_ref, 'X1_int':X1_int, 'L1_err':L1_err,}

	def process_data(self):
		pass

	def prepare_plots(self):
		self.axs_dict = {}
		fig = plt.figure(figsize=(3,3), dpi=100)
		axs = {}
		n_metrics = 2 # [IRB, leakage]
		_n = 1.4*n_metrics
		self.figs['Two_Qubit_performance_overview'] = fig
		for Qubits in self.Qubit_pairs:
			Qubits = tuple(Qubits)
			ax = _add_twoQ_plot(Qubits, _n, fig, axs)
			self.axs_dict[f'{Qubits}'] = ax
			# Plot two_qubit gate error
			self.plot_dicts[f'{Qubits}_error']={
				'plotfn': _plot_TQG_error,
				'ax_id': f'{Qubits}',
				'N_clf': self.raw_data_dict[Qubits]['N_clf'],
				'M_ref': self.raw_data_dict[Qubits]['M_ref'],
				'M_int': self.raw_data_dict[Qubits]['M_int'],
				'TQG_err': self.raw_data_dict[Qubits]['TQG_err'],
				'row': 0,
				'n': _n,
			}
			# Plot two_qubit gate leakage
			self.plot_dicts[f'{Qubits}_leakage']={
				'plotfn': _plot_TQG_leakage,
				'ax_id': f'{Qubits}',
				'N_clf': self.raw_data_dict[Qubits]['N_clf'],
				'X1_ref': self.raw_data_dict[Qubits]['X1_ref'],
				'X1_int': self.raw_data_dict[Qubits]['X1_int'],
				'TQG_leak': self.raw_data_dict[Qubits]['L1_err'],
				'row': 1,
				'n': _n,
			}
		n_plts = len(self.Qubit_pairs)
		fig.suptitle(f'{self.timestamp}\nTwo-qubit gate performance', x=0.125+n_plts/2*0.775*1.3, y=1.35, size=18)

	def run_post_extract(self):
		self.prepare_plots()  # specify default plots
		self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
		if self.options_dict.get('save_figs', False):
			self.save_figures(
				close_figs=self.options_dict.get('close_figs', True),
				tag_tstamp=self.options_dict.get('tag_tstamp', True))

def _add_twoQ_plot(Qubits, n, fig, axs):
    n_plts = len(axs.keys()) 
    ax = fig.add_subplot(10,10,n_plts+1)
    _pos = ax.get_position()
    pos = [0.125 + n_plts*0.775*1.3, 0.11, 0.775, 0.775/3*n]
    ax.set_position(pos)
    axs[Qubits] = ax
    axs[Qubits].text(0, n/3+.2, f'$\\mathrm{{{Qubits[0][0]}_{Qubits[0][1]}, {Qubits[1][0]}_{Qubits[1][1]}}}$',
                    va='center', ha='left', size=40)
    ax.set_xlim(0,1)
    ax.set_ylim(0,n/3)
    ax.axis('off')
    ax.patch.set_alpha(0)
    return ax

def _plot_TQG_error(
    ax,
    N_clf,
	M_ref,
	M_int,
	TQG_err,
	n,
	row,
	**kw):
    # fig = ax.get_figure()
    # ax = fig.get_axes()[-1]
    # Assess pereformance level
    if TQG_err < 0.02:
        _color = 'C2'
    elif TQG_err < 0.04:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Two qubit gate', ha='left', size=11.5)
    ax.text(.86, (n-row*1.4)/3-.13, 'err.', ha='left', size=11.5)
    ax.text(.375, (n-row*1.4)/3-.3, f'{TQG_err*100:2.1f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{TQG_err*100:2.1f}', ha='left', size=50,
    		color=_color, alpha=.65) # Overlay to give color
    ax.text(.85, (n-row*1.4)/3-.3, '%', ha='left', size=25)
    # RB decay plot
    _x = N_clf/N_clf[-1]
    _yref = (M_ref)
    _yint = (M_int)
    ax.plot(_x*0.3+.025,
            (_yref+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='--', color=_color, clip_on=False)
    ax.plot(_x*0.3+.025,
            (_yint+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            color=_color, clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_yref+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}', alpha=.1, lw=0)
    ax.plot(_x*0.3+.025,
            [(.25+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3 for x in _x],
            f'k--', lw=.5)

def _plot_TQG_leakage(
    ax,
    N_clf,
	X1_ref,
	X1_int,
	TQG_leak,
	n,
	row,
	**kw):
    # fig = ax.get_figure()
    # ax = fig.get_axes()[-1]
    # Assess pereformance level
    if TQG_leak < 0.005:
        _color = 'C2'
    elif TQG_leak < 0.01:
        _color = 'goldenrod'
    else:
        _color = 'C3'
    # Label
    ax.text(.4, (n-row*1.4)/3-.05, 'Two qubit gate', ha='left', size=11.5)
    ax.text(.86, (n-row*1.4)/3-.13, 'leak.', ha='left', size=11.5)
    ax.text(.375, (n-row*1.4)/3-.3, f'{abs(TQG_leak)*100:2.1f}', ha='left', size=50)
    ax.text(.375, (n-row*1.4)/3-.3, f'{abs(TQG_leak)*100:2.1f}', ha='left', size=50,
    		color=_color, alpha=.65) # Overlay to give color
    ax.text(.85, (n-row*1.4)/3-.3, '%', ha='left', size=25)
    # RB decay plot
    _x = N_clf/N_clf[-1]
    _yref = (1-X1_ref)
    _yint = (1-X1_int)
    ax.plot(_x*0.3+.025,
            (_yref+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            ls='--', color=_color, clip_on=False)
    ax.plot(_x*0.3+.025,
            (_yint+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
            color=_color, clip_on=False)
    ax.fill_between(_x*0.3+.025,
                    (_yref+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3,
                    (0+.1)*(1/3)/(1+2*.1)+(n-1-row*1.4)/3, color=f'{_color}', alpha=.1, lw=0)


class ParityCheckGBT_analysis(ba.BaseDataAnalysis):
	"""
	Analysis for Chevron routine
	"""
	def __init__(self,
				 Stabilizers: list,
				 t_start: str = None,
				 t_stop: str = None,
				 label: str = '',
				 options_dict: dict = None, 
				 extract_only: bool = False,
				 auto=True):

		super().__init__(t_start=t_start, 
						 t_stop=t_stop,
						 label=label,
						 options_dict=options_dict,
						 extract_only=extract_only)
		self.Stabilizers = Stabilizers
		if auto:
			self.run_analysis()

	def extract_data(self):
		self.raw_data_dict = {}
		self.get_timestamps()
		self.timestamp = self.timestamps[0]
		data_fp = get_datafilepath_from_timestamp(self.timestamp)
		self.raw_data_dict['timestamps'] = self.timestamps
		self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
		# Extract last measured metrics
		for stab in self.Stabilizers:
			self.raw_data_dict[stab] = {}
			# Analyse parity check ramsey
			fp_ramsey = ma.a_tools.latest_data(contains=f"Parity_check_ramsey_{stab}",
											   older_than=self.timestamp)
			label_ramsey = fp_ramsey.split('\\')[-1]
			Data_qubits = label_ramsey.split(f'ramsey_{stab}_')[-1]
			Data_qubits = Data_qubits.split('_device')[0]
			Data_qubits = Data_qubits.split('_')
			control_cases = ['{:0{}b}'.format(i, len(Data_qubits))\
							 for i in range(2**len(Data_qubits))]
			a = ma2.tqg.Parity_check_ramsey_analysis(
				label=label_ramsey,
				Q_target = [stab],
				Q_control = Data_qubits,
				Q_spectator = [],
				control_cases = control_cases,
				angles = np.arange(0, 341, 20),
				solve_for_phase_gate_model = False,
				extract_only = True)
			self.raw_data_dict[stab]['ramsey_curves'] = a.proc_data_dict['Ramsey_curves'][stab]
			self.raw_data_dict[stab]['ramsey_fit'] = a.proc_data_dict['Fit_res'][stab]
			self.raw_data_dict[stab]['missing_fraction'] = a.proc_data_dict['Missing_fraction']
			# Analyse parity assignment fidelity
			fp_fidelity = ma.a_tools.latest_data(contains=f"Parity_check_fidelity_{stab}",
												 older_than=self.timestamp)
			label_fidelity = fp_fidelity.split('\\')[-1]
			Data_qubits = fp_fidelity.split(f'fidelity_{stab}_')[-1]
			Data_qubits = Data_qubits.split('_device')[0]
			Data_qubits = Data_qubits.split('_')
			control_cases = ['{:0{}b}'.format(i, len(Data_qubits))\
							 for i in range(2**len(Data_qubits))]
			a = ma2.tqg.Parity_check_fidelity_analysis(
				label=label_fidelity,
				Q_ancilla=stab,
				Q_control=Data_qubits,
				control_cases=control_cases,
				post_selection=False,
				extract_only=True)
			self.raw_data_dict[stab]['P_dist'] = a.proc_data_dict['P']
			self.raw_data_dict[stab]['P_dist_id'] = a.proc_data_dict['P_ideal']
			self.raw_data_dict[stab]['assignment_fidelity'] = a.qoi['fidelity']
			# Analyse parity check tomography
			fp_tomo = ma.a_tools.latest_data(contains=f"parity_tomography_{stab}",
											 older_than=self.timestamp)
			label_tomo = fp_tomo.split('\\')[-1]
			Data_qubits = fp_tomo.split(f'tomography_{stab}_')[-1]
			Data_qubits = Data_qubits.split('_sim')[0]
			Data_qubits = Data_qubits.split('_')
			if stab == 'Z1':
				exc_qubits = ['D1']
			elif stab == 'Z2':
				exc_qubits = ['D3']
			elif stab == 'Z3':
				exc_qubits = []
			elif stab == 'Z4':
				exc_qubits = ['D5']
			else:
			    raise ValueError('exception qubits must be given to analysis')
			a = ma2.pba.Weight_n_parity_tomography(
				label = label_tomo,
				sim_measurement=True,
				n_rounds=2,
				exception_qubits=exc_qubits,
				post_selection=False,
				extract_only=True)
			self.raw_data_dict[stab]['rho_0'] = a.proc_data_dict['rho_0']
			self.raw_data_dict[stab]['rho_1'] = a.proc_data_dict['rho_1']
			self.raw_data_dict[stab]['fidelity_0'] = a.proc_data_dict['Fid_0']
			self.raw_data_dict[stab]['fidelity_1'] = a.proc_data_dict['Fid_1']
			self.raw_data_dict[stab]['M1'] = a.proc_data_dict['M1']
			self.raw_data_dict[stab]['M2'] = a.proc_data_dict['M2']
			self.raw_data_dict[stab]['repeatability'] = a.proc_data_dict['repeatability']

	def process_data(self):
		pass

	def prepare_plots(self):
		self.axs_dict = {}
		fig = plt.figure(figsize=(3,3), dpi=120)
		axs = {}
		n_metrics = 4 # [Ramsey, ass. fid., tomox2, repeatability]
		_n = 1.4*n_metrics
		self.figs['Parity_check_performance_overview'] = fig
		for stab in self.Stabilizers:
			ax = _add_PC_plot(stab, _n, fig, axs)
			self.axs_dict[f'{stab}'] = ax
			# Plot ancilla ramsey curves
			self.plot_dicts[f'{stab}_ramsey']={
				'plotfn': _plot_ramsey_curves,
				'ax_id': f'{stab}',
				'qubit': stab,
				'Ramsey_curves': self.raw_data_dict[stab]['ramsey_curves'],
				'Fit_res': self.raw_data_dict[stab]['ramsey_fit'],
				'Missing_fraction': self.raw_data_dict[stab]['missing_fraction'],
				'row': 0,
				'_n': _n,
			}
			# Plot parity check assignment fidelity
			self.plot_dicts[f'{stab}_fidelity']={
				'plotfn': _plot_parity_fidelity,
				'ax_id': f'{stab}',
				'qubit': stab,
				'P_dist': self.raw_data_dict[stab]['P_dist'],
				'P_dist_ideal': self.raw_data_dict[stab]['P_dist_id'],
				'fidelity': self.raw_data_dict[stab]['assignment_fidelity'],
				'row': 1,
				'_n': _n,
			}
			# Plot data-qubit tomography
			self.plot_dicts[f'{stab}_tomography']={
				'plotfn': _plot_tomo_fidelity,
				'ax_id': f'{stab}',
				'qubit': stab,
			    'rho_0': self.raw_data_dict[stab]['rho_0'],
			    'rho_1': self.raw_data_dict[stab]['rho_1'],
			    'fidelity_0': self.raw_data_dict[stab]['fidelity_0'],
			    'fidelity_1': self.raw_data_dict[stab]['fidelity_1'],
				'row': 2,
				'_n': _n,
			}
			# Plot repeatability
			self.plot_dicts[f'{stab}_repeatability']={
				'plotfn': _plot_repeatability,
				'ax_id': f'{stab}',
				'qubit': stab,
			    'M1' : self.raw_data_dict[stab]['M1'],
			    'M2' : self.raw_data_dict[stab]['M2'],
			    'repeatability' : self.raw_data_dict[stab]['repeatability'],
				'row': 4,
				'_n': _n,
			}
		n_plts = len(self.Stabilizers)
		fig.suptitle(f'{self.timestamp}\nParity check performance', x=0.125+n_plts/2*0.775*1.3, y=2, size=14)

	def run_post_extract(self):
		self.prepare_plots()  # specify default plots
		self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
		if self.options_dict.get('save_figs', False):
			self.save_figures(
				close_figs=self.options_dict.get('close_figs', True),
				tag_tstamp=self.options_dict.get('tag_tstamp', True))

def _add_PC_plot(stabilizer, n, fig, axs):
    n_plts = len(axs.keys()) 
    ax = fig.add_subplot(10,10,n_plts+1)
    _pos = ax.get_position()
    pos = [0.125 + n_plts*0.775*1.3, 0.11, 0.775, 0.775/3*n]
    ax.set_position(pos)
    axs[stabilizer] = ax
    axs[stabilizer].text(0, n/3+.2, f'$\\mathrm{{{stabilizer[0]}_{stabilizer[1]}}}$',
	                     va='center', ha='left', size=40)
    ax.set_xlim(0,1)
    ax.set_ylim(0,n/3)
    ax.axis('off')
    ax.patch.set_alpha(0)
    return ax
    
def _plot_ramsey_curves(
	ax,
    qubit,
    Ramsey_curves,
    Fit_res,
    Missing_fraction,
    _n,
    row = 0,
	**kw):
	# State conditions
	mf = np.array(list(Missing_fraction.values()))
	if np.all(mf < .035):
		_color = 'C2'
	elif np.all(mf < .07):
		_color = 'goldenrod'
	else:
		_color = 'C3'
	# definitions
	def func(x, phi, A, B):
		return A*(np.cos( (x+phi)/360 *2*np.pi )+1)/2 + B
	# Label
	ax.text(.4, (_n-row*1.4)/3, 'Missing fraction', ha='left', size=11.5)
	# Ramsey plot
	for state in Ramsey_curves.keys():
		# Plot ramsey data
		_x = np.linspace(0.02, .35, 51)
		_y = func(np.linspace(0,340,51), *Fit_res[state])
		ax.plot(_x, (_y+.1)*(1.2/3)/(1+2*.1)+(_n-1-row*1.4)/3,
						marker='', ls='--', color=_color, lw=1, alpha=.5,
						clip_on=False)
		# Plot ramsey fits
		_x = np.linspace(0.02, .35, 18)
		_y = Ramsey_curves[state]
		ax.plot(_x, (_y+.1)*(1.2/3)/(1+2*.1)+(_n-1-row*1.4)/3,
						marker='.', ls='', markersize=1, color=_color,
						clip_on=False)
	# Plot Missing fraction
	from matplotlib.colors import to_rgba
	n = len(Missing_fraction.keys())
	for _x, q in zip(np.linspace(.41, .94, 2*n+1)[1::2], Missing_fraction.keys()):
		# Qubit labels and missing fraction values
		ax.text(_x, (_n-row*1.4)/3-.265, f'$\\mathrm{{{q[0]}_{q[1]}}}$', ha='center', size=9)
		ax.text(_x, (_n-row*1.4)/3-.15, f'{Missing_fraction[q]*100:.1f}',
						ha='center', size=8)
	# Bar plot for missing fractions
	_x = np.linspace(.41, .94, 2*n+1)[1::2]
	_y = np.array(list(Missing_fraction.values()))*10
	ax.bar(_x, (_y)*(1/3)/(1+2*.1),
					bottom = (.2)*(1/3)/(1+2*.1)+(_n-1-row*1.4)/3,
					width = (.5-.025*n)/(n),
					ec=_color, fc=to_rgba(_color,.1),
					clip_on=False)

def _plot_parity_fidelity(
		ax,
		qubit,
		P_dist,
		P_dist_ideal,
		fidelity,
		_n,
		row = 1,
		**kw):
	# weight-4 stabilizer condition
	if len(P_dist) == 16:
		if 1-fidelity < 0.10:
			_color = 'C2'
		elif 1-fidelity < 0.125:
			_color = 'goldenrod'
		else:
			_color = 'C3'
	# weight-2 stabilizer condition
	elif len(P_dist) == 4:
		if 1-fidelity < 0.05:
			_color = 'C2'
		elif 1-fidelity < 0.075:
			_color = 'goldenrod'
		else:
			_color = 'C3'
	# definitions
	def func(x, phi, A, B):
		return A*(np.cos( (x+phi)/360 *2*np.pi )+1)/2 + B
	# Label
	ax.text(.4, (_n-row*1.4)/3-.05, 'Avg. parity assign.', ha='left', size=11.5)
	ax.text(.86, (_n-row*1.4)/3-.13, 'error', ha='left', size=11.5)
	ax.text(.375+.475, (_n-row*1.4)/3-.3, f'{(1-fidelity)*100:.3g}', ha='right', size=35)
	ax.text(.375+.475, (_n-row*1.4)/3-.3, f'{(1-fidelity)*100:.3g}', ha='right', size=35,
    		color=_color, alpha=.65) # Overlay to give color
	ax.text(.85, (_n-row*1.4)/3-.3, '%', ha='left', size=20)
	# Bar plot
	idx_sort = np.argsort([ s.count('1') for s in P_dist.keys() ])
	from matplotlib.colors import to_rgba
	n = len(P_dist.keys())
	# Bar plot for missing fractions
	_x = np.linspace(0, .35, 2*n+1)[1::2]
	_y = np.array(list(P_dist.values()))[idx_sort]
	ax.bar(_x, (_y)*(1/3)/(1+2*.1),
			bottom = .1*(1/3)/(1+2*.1)+(_n-1-row*1.4)/3,
			width = 0.016 if n == 16 else .07,
			clip_on=False,
			ec=None, fc=to_rgba(_color,.4))
	_y = np.array(list(P_dist_ideal.values()))[idx_sort]
	ax.bar(_x, (_y)*(1/3)/(1+2*.1),
			bottom = .1*(1/3)/(1+2*.1)+(_n-1-row*1.4)/3,
			width = 0.016 if n == 16 else .07,
			clip_on=False,
			ec=_color, fc=to_rgba('k', 0), ls='-', lw=.5)
    
def _plot_tomo_fidelity(
	ax,
	qubit,
    rho_0,
    rho_1,
    fidelity_0,
    fidelity_1,
    _n,
    row=2,
    **kw):
	if qubit in ['Z2', 'Z3', 'X1', 'X4']:
		if 1-np.mean([fidelity_0,fidelity_1]) < .20:
			_color = 'C2'
		elif 1-np.mean([fidelity_0,fidelity_1]) < .25:
			_color = 'goldenrod'
		else:
			_color = 'C3'
	else:
		if 1-np.mean([fidelity_0,fidelity_1]) < .30:
			_color = 'C2'
		elif 1-np.mean([fidelity_0,fidelity_1]) < .40:
			_color = 'goldenrod'
		else:
			_color = 'C3'
	# Label
	ax.text(.4, (_n-row*1.4)/3-.05, 'GHZ state infidelity', ha='left', size=11.5)
	ax.text(.55, (_n-row*1.4)/3-.15, 'ancilla state $|0\\rangle$', ha='left', size=10)
	ax.text(.375+.15, (_n-row*1.4)/3-.4, f'{(1-fidelity_0)*100:.2g}', ha='left', size=40)
	ax.text(.375+.15, (_n-row*1.4)/3-.4, f'{(1-fidelity_0)*100:.2g}', ha='left', size=40,
			color=_color, alpha=.65)
	ax.text(.85, (_n-row*1.4)/3-.4, '%', ha='left', size=20)
	ax.text(.55, (_n-row*1.4)/3-.15-.375, 'ancilla state $|1\\rangle$', ha='left', size=10)
	ax.text(.375+.15, (_n-row*1.4)/3-.4-.375, f'{(1-fidelity_1)*100:.2g}', ha='left', size=40)
	ax.text(.375+.15, (_n-row*1.4)/3-.4-.375, f'{(1-fidelity_1)*100:.2g}', ha='left', size=40,
			color=_color, alpha=.65)
	ax.text(.85, (_n-row*1.4)/3-.4-.375, '%', ha='left', size=20)
	# Ideal density matrices
	rho_id_0 = np.zeros(rho_0.shape)
	rho_id_0[ 0, 0] = .5
	rho_id_0[ 0,-1] = .5
	rho_id_0[-1, 0] = .5
	rho_id_0[-1,-1] = .5
	rho_id_1 = np.zeros(rho_1.shape)
	rho_id_1[ 0, 0] = .5
	rho_id_1[ 0,-1] = -.5
	rho_id_1[-1, 0] = -.5
	rho_id_1[-1,-1] = .5
	for i, rho, rho_id in zip([0,1], [rho_0, rho_1], [rho_id_0, rho_id_1]):
		# Tomography axis
		fig = ax.get_figure()
		pos = ax.get_position()
		axt = fig.add_subplot(111, projection='3d', azim=-35, elev=30)
		_pos = axt.get_position()
		axt.set_position([pos.x0, _pos.y0-(i-1)*.325+.075, _pos.width/2, _pos.height/2])
		# turn background transparent
		axt.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		axt.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		axt.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		axt.patch.set_alpha(0)
		# make the grid lines transparent
		axt.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		axt.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		axt.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		# Plot density matrix
		n = len(rho)
		xedges = np.linspace(0, 1, n+1)
		yedges = np.linspace(0, 1, n+1)
		xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
		xpos = xpos.ravel()
		ypos = ypos.ravel()
		zpos = 0
		dx = dy = 1/n*0.8
		dz = np.abs(rho).ravel()
		import matplotlib
		from matplotlib.colors import to_rgba
		cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",'antiquewhite',"C3"])
		norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
		color=cmap(norm([np.angle(e) for e in rho.ravel()]))
		color_id=cmap(norm([np.angle(e) for e in rho_id.ravel()]))
		dz1 = np.abs(rho_id).ravel()
		axt.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='max',
				 color=color, alpha=1, edgecolor='black', linewidth=.1)
		# selector
		s = [k for k in range(len(dz1)) if dz1[k] > .15]
		colors = [ to_rgba(color_id[k], .25) if dz1[k] > dz[k] else to_rgba(color_id[k], 1) for k in s ]
		Z = [ dz[k] if dz1[k] > dz[k] else dz1[k] for k in s ]
		DZ= [ dz1[k]-dz[k] if dz1[k] > dz[k] else dz[k]-dz1[k] for k in s ]
		axt.bar3d(xpos[s], ypos[s], Z, dx, dy, dz=DZ, zsort='min',
				 color=colors, edgecolor=to_rgba('black', .25), linewidth=.4)
		axt.set_xticklabels([])
		axt.set_yticklabels([])
		axt.set_zticklabels([])
        
def _plot_repeatability(
	ax,
    qubit,
    M1,
    M2,
    repeatability,
    _n,
    state='good',
    row = 4,
    **kw):
	if qubit in ['Z2', 'Z3', 'X1', 'X4']:
		if 1-repeatability < .10:
			_color = 'C2'
		elif 1-repeatability < .15:
			_color = 'goldenrod'
		else:
			_color = 'C3'
	else:
		if 1-repeatability < .20:
			_color = 'C2'
		elif 1-repeatability < .25:
			_color = 'goldenrod'
		else:
			_color = 'C3'
	# definitions
	def func(x, phi, A, B):
		return A*(np.cos( (x+phi)/360 *2*np.pi )+1)/2 + B
	# Label
	ax.text(.4, (_n-row*1.4)/3-.05, 'Repeatability err.', ha='left', size=11.5)
	ax.text(.375+.5, (_n-row*1.4)/3-.275, f'{(1-repeatability)*100:.1f}', ha='right', size=35)
	ax.text(.375+.5, (_n-row*1.4)/3-.275, f'{(1-repeatability)*100:.1f}', ha='right', size=35,
    		color=_color, alpha=.65)
	ax.text(.875, (_n-row*1.4)/3-.275, '%', ha='left', size=20)
	# Bar plot for missing fractions
	from matplotlib.colors import to_rgba
	_x = np.linspace(0, .35, 5)[1::2]
	_y = (np.array([M1, M2])+1)/2
	ax.bar(_x, (_y+.2-.7)*(1/3)/(1+2*.1),
			bottom = (.7)*(1/3)/(1+2*.1)+(_n-1-row*1.4)/3,
			width = .15,
			ec=None, fc=to_rgba(_color,.5),
			clip_on=False)

	_y = (np.array([0, 1])+1)/2
	ax.bar(_x, (_y+.2-.7)*(1/3)/(1+2*.1),
			bottom = (.7)*(1/3)/(1+2*.1)+(_n-1-row*1.4)/3,
			width = .15,
			ec=_color, fc=to_rgba(_color,0),
			clip_on=False)
