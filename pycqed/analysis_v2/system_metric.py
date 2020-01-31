import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

class System_Metric(ba.BaseDataAnalysis):
	"""
	System analysis plots data from several qubit objects to visualize total system metric.

	"""	
	def __init__(self, qubit_objects = None, qubit_list: list = None, t_start: str = None, metric: str = None, label: str = '',
				options_dict: dict = None, 
				auto=True):
		
		super().__init__(t_start=t_start,
						 label=label,
						 options_dict=options_dict)
		self.qubit_objects = qubit_objects
		self.qubit_list = qubit_list
		if auto:
			self.run_analysis()

	def extract_data(self):
		self.raw_data_dict = {}
		if t_start is not None:
			data_fp = get_datafilepath_from_timestamp(ts)
			if self.qubit_objects is not None and self.qubit_list is None:
				qubit_list = [qubit.getname for qubit in qubit_objects]
			else:
				qubit_list = self.qubit_list
			for qubit in qubit_list:
				self.raw_data_dict[qubit] = {}
				param_spec = {
					'T1': ('Instrument settings/{}'.format(qubit), 'attr:T1'),
					'T2_echo': ('Instrument settings/{}'.format(qubit), 'attr:T2_echo'),
					'T2_star': ('Instrument settings/{}'.format(qubit), 'attr:T2_star'),
					'freq_max': ('Instrument settings/{}'.format(qubit), 'attr:freq_max')}
				self.raw_data_dict[qubit] = h5d.extract_pars_from_datafile(data_fp, param_spec)

		# Parts added to be compatible with base analysis data requirements
		self.raw_data_dict['timestamps'] = self.t_start
		self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

	def process_data(self):
		self.proc_data_dict = self.raw_data_dict.copy()
		coords = [(0,1), (2,1), (-1,0), (1,0), (3,0), (0,-1), (2,-1)]
		for i, qubit in enumerate(self.qubit_list):
			self.proc_data_dict[qubit]['coords'] = coords[i]
		self.proc_data_frame = pd.DataFrame.from_dict(self.proc_data_dict).T

	
	def prepare_plots(self):
		self.plot_data_dict = self.proc_data_dict.copy()
		del self.plot_data_dict['timestamps']
		
		self.plot_dicts['metric'] = {
							'plotfn': plot_system_metric,
							'1Q_dict': self.proc_data_dict
		}




def plot_system_metric(ax, 1Q_dict, 2Q_dict = None, vmin=1.5e-4, vmax=5e-2, 
					 plot:str='gate', main_color='black'):
	"""
	Plots device performance
	
	plot (str): 
		{"leakage", "gate", "readout"}
	"""
	val_fmt_str = '{0:1.1e}'
	norm = LogNorm()

	if plot=='leakage':
		plot_key = 'L1'
		cmap='hot'
		clabel= 'Leakage'
	elif plot=='gate': 
		plot_key = 'gate_infid'
		cmap='viridis'
		clabel= 'Gate infidelity'
	elif plot == 'readout':
#		 cmap='cividis'
		cmap = 'ocean'
		clabel= 'Readout infidelity'
		plot_key = 'ro_fid'
	elif plot == 'readout_QND':
		cmap='cividis'
		clabel= 'Readout QNDness'
		plot_key = 'ro_QND'
	elif plot == 'freq_max':
		cmap='nipy_spectral_r'
		clabel= 'Frequency (GHz)'
		plot_key = 'freq_max_GHz'
		val_fmt_str = '{:.3f}'
		norm = None
	elif plot == 'freq_target':
		cmap='nipy_spectral_r'
		clabel= 'Frequency (GHz)'
		plot_key = 'freq_target_GHz'
		val_fmt_str = '{:.3f}'
		norm = None
	elif plot == 'coherence_times':
		cmap='nipy_spectral_r'
		clabel= 'T1'
		plot_key = 'T1'
		val_fmt_str = '{:.3f}'
		norm = None
	else: 
		raise KeyError

	#Plot qubit locations
	x = [c[0] for c in 1Q_dict['coords']]
	y = [c[1] for c in 1Q_dict['coords']]
	ax.scatter(x, y, s=1500, edgecolors=main_color, color='None')

	value = [1Q_dict[qubit][plot_key] for qubit in qubit_list]
	if 'times' in plot:
#		 sc = ax.scatter(x, y, s=1500, c=df[plot_key], vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm())
		sc = ax.scatter(x, y, s=1500, c=1Q_dict[plot_key]/1e6, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
	else:
		sc = ax.scatter(x, y, s=1500, c=1Q_dict[plot_key], vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
	
	cax = f.add_axes([.95, 0.13, .03, .75])
	plt.colorbar(sc, cax=cax)
	cax.set_ylabel(clabel)

	for ind, row in df.iterrows():
		ax.text(row['coords'][0], row['coords'][1]+.4, s=ind, va='center', ha='center', color=main_color)
		ax.text(row['coords'][0], row['coords'][1], s=val_fmt_str.format(row[plot_key]), 
				color='white',
				va='center', ha='center')

	ax.set_ylim(-1.5, 1.6)
	ax.set_xlim(-1.5, 3.5)
	ax.set_aspect('equal')

	set_axeslabel_color(cax, main_color)
	
	# Two qubit part 
	if 2Q_dict is not None:

		x = np.array([c[0] for c in 2Q_dict['coords']])
		y = np.array([c[1] for c in 2Q_dict['coords']])


		ax.scatter(x, y, s=2000, edgecolors=main_color, color='None', 
				   marker=(4, 0, i*90),)
		if plot in {'gate', 'leakage'}:
			sc = ax.scatter(x, y, s=2000, c=2Q_dict[plot_key], vmin=vmin, vmax=vmax, cmap=cmap, 
							marker=(4, 0, i*90),
							norm=norm)
			for ind, row in 2Q_dict.iterrows():
				if row[plot_key]>1e-3 and plot=='leakage':
					c = 'black'
				else: 
					c='white'
				ax.text(row['coords'][0], row['coords'][1], s=val_fmt_str.format(row[plot_key]), 
						color=c,
						va='center', ha='center')