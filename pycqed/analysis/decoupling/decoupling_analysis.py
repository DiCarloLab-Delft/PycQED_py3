import numpy as np
import pycqed.analysis.composite_analysis as RA
import pycqed.analysis.analysis_toolbox as a_tools
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

"""
    Inputs:
            N                       Number of qubits
            scan_start
            scan_stop
            qubit_scan_labels
            dac_mapping
            M                       Number of points
"""


class DecouplingAnalysis:

    def __init__(self, N, scan_start, scan_stop, qubit_scan_labels,
                 dac_mapping, num_points):
        assert(N == len(qubit_scan_labels))
        self.N = N
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.qubit_scan_labels = qubit_scan_labels
        self.dac_mapping = dac_mapping
        self.num_points = num_points
        self.filter_mask_vector = np.zeros((self.N, self.N,
                                            self.num_points),
                                           dtype=np.bool)
        self.filter_mask_vector[:, :, :] = False

    def extract_data(self):
        self.spec_scans = [None] * self.N

        pdict = {'amp': 'amp',
                 'frequencies': 'sweep_points'}
        nparams = ['amp', 'frequencies']

        # adds the dac channels to the extraction dictionary
        for d in self.dac_mapping:
            pdict.update({'dac%d' % d: 'IVVI.dac%d' % d})
            nparams.append('dac%d' % d)

        # extracts the data
        for i, q_label in enumerate(self.qubit_scan_labels):
            opt_dict = {'scan_label':  q_label,
                        'exact_label_match':True}

            self.spec_scans[i] = RA.quick_analysis(t_start=self.scan_start,
                                                   t_stop=self.scan_stop,
                                                   options_dict=opt_dict,
                                                   params_dict_TD=pdict,
                                                   numeric_params=nparams)


    def plot_freq_vs_dac_raw(self):
        fig, ax = plt.subplots(self.N, self.N, figsize=(15, 15))

        # for i in range(1, self.N+1):
        for d, i in zip(self.dac_mapping, range(len(self.dac_mapping))):
            for q in range(self.N):
                start_slice = self.num_points*(i)
                stop_slice = self.num_points*(i+1)
                dac_key = 'dac%d' % d

                this_ax = ax[q, i]
                dac_vector = self.spec_scans[q].TD_dict[
                    dac_key][start_slice:stop_slice]


                freq_vector = self.spec_scans[q].TD_dict[
                    'frequencies'][start_slice:stop_slice]

                z_vector = self.spec_scans[q].TD_dict[
                    'amp'][start_slice:stop_slice]

                pk_vec = get_peaks(dac_vector,
                                   freq_vector,
                                   z_vector)

                plot_scan(
                    self.spec_scans[q], this_ax, dac_vector, freq_vector, z_vector, d)
                this_ax.plot(dac_vector, pk_vec, 'o')

    def plot_fit_freq_vs_dac(self):
        fig, ax = plt.subplots(self.N, self.N, figsize=(15, 15))

        ylims = []
        xlims = []
        fits_freq = np.zeros((self.N, self.N, 2))
        for d, i in zip(self.dac_mapping, range(len(self.dac_mapping))):
            for q in range(self.N):
                start_slice = self.num_points*(i)
                stop_slice = self.num_points*(i+1)
                dac_key = 'dac%d' % d
                this_ax = ax[q, i]
                filter_points = ~self.filter_mask_vector[i, q, :]
                dac_vector = self.spec_scans[q].TD_dict[
                    dac_key][start_slice:stop_slice]
                freq_vector = self.spec_scans[q].TD_dict[
                    'frequencies'][start_slice:stop_slice]

                z_vector = self.spec_scans[q].TD_dict[
                    'amp'][start_slice:stop_slice]
                pk_vec = get_peaks(dac_vector,
                                   freq_vector,
                                   z_vector)
                freqs = np.where(filter_points, pk_vec, False)
                if i == (q+1):
                    ylims.append(
                        [freqs[filter_points].min()*0.99, freqs[filter_points].max()*1.01])
                    xlims.append([dac_vector.min(), dac_vector.max()])
        #         plot_scan_flux(self.spec_scans[q], this_ax, dac_vector, freqs, z_vector, i)
                this_ax.plot(
                    dac_vector[filter_points], freqs[filter_points], 'o')
                fits_freq[
                    q, i, :] = np.polyfit(dac_vector[filter_points], freqs[filter_points], 1).flatten()
                this_ax.plot(dac_vector[filter_points], np.polyval(
                    fits_freq[q, i, :], dac_vector[filter_points]))
        for i in range(self.N):
            for q in range(self.N):
                #         ax[i,q].set_ylim(ylims[i][0],ylims[i][1])
                #         ax[i,q].set_xlim(xlims[i][0],xlims[i][1])
                ax[i, q].ticklabel_format(useOffset=False)
        matrix = fits_freq[:, :, 0]
        diagonal = np.diagonal(matrix)
        matrix_normalized = np.transpose(np.transpose(matrix)/diagonal)
        return matrix, matrix_normalized

    def plot_fit_flux_vs_dac(self):
        # Need to adapt the function to the class design
        raise NotImplementedError
        """
        fig, ax = plt.subplots(5,5, figsize=(15,15))
        ylims = []
        xlims = []
        fits = np.zeros((5,5,2))
        for i in range(1,6):
            for q in range(5):
                start_slice = 9*(i-1)
                stop_slice = 9*i
                dac_key = 'dac%d'%i
                this_ax = ax[q,i-1]
                filter_points = ~filter_mask_vector[i-1,q,:]
                dac_vector = spec_scans[q].TD_dict[dac_key][start_slice:stop_slice]
                freq_vector = spec_scans[q].TD_dict['frequencies'][start_slice:stop_slice]

                z_vector = spec_scans[q].TD_dict['phase'][start_slice:stop_slice]
                pk_vec = get_peaks(dac_vector,
                                   freq_vector,
                                   z_vector)
                freqs = np.where(filter_points,pk_vec,False)
                flux_vector = np.array([find_freq(f*1e-9,f_flux[q]) for f in freqs])
                if i==(q+1):
                    ylims.append([flux_vector[filter_points].min(),flux_vector[filter_points].max()])
                    xlims.append([dac_vector.min(),dac_vector.max()])
        #         plot_scan_flux(spec_scans[q], this_ax, dac_vector, flux_vector, z_vector, i)
                this_ax.plot(dac_vector[filter_points],flux_vector[filter_points],'o')
                fits[q,i-1,:] = np.polyfit(dac_vector[filter_points],flux_vector[filter_points],1).flatten()
                this_ax.plot(dac_vector[filter_points],np.polyval(fits[q,i-1,:],dac_vector[filter_points]))
        for i in range(5):
            for q in range(5):
                ax[i,q].set_ylim(ylims[i][0],ylims[i][1])
                ax[i,q].set_xlim(xlims[i][0],xlims[i][1])
        # fits[:,:,0]
        """


def plot_scan(spec_scan_obj, ax, xvals, yvals, zvals, idx_dac):
    spec_scan_obj.plot_dicts['arches'] = {'plotfn': spec_scan_obj.plot_colorx,
                                          'xvals': xvals,
                                          'yvals': yvals,
                                          'zvals': zvals.transpose(),
                                          #                 'zvals': peak_finder_v2spec_scans.TD_dict['amp'].transpose(),
                                          'title': '',
                                          #                 'title': '%s - %s: Qubit Arches'%(spec_scan_obj.TD_timestamps[0],spec_scans.TD_timestamps[-1]),
                                          'xlabel': r'Dac %d value (mV)' % idx_dac,
                                          'ylabel': r'Frequency (GHz)',
                                          'zlabel': 'Homodyne amplitude (mV)',
                                          'zrange': [0.5, 1],
                                          'plotsize': (8, 8),
                                          'cmap': 'YlGn_r'
                                          }
    spec_scan_obj.axs['arches'] = ax
    spec_scan_obj.plot()


def plot_scan_flux(spec_scan_obj, ax, xvals, yvals, zvals, idx_dac):
    # Need to adapt the function to the module design
    raise NotImplementedError
    """
    #     print(spec_scan_obj)
    spec_scan_obj.plot_dicts['arches'] = {'plotfn': spec_scan_obj.plot_colorx,
                                          'xvals': xvals,
                                          'yvals': yvals,
                                          'zvals': zvals.transpose(),
                                          #                 'zvals': peak_finder_v2spec_scans.TD_dict['amp'].transpose(),
                                          'title': '',
                                          #                 'title': '%s - %s: Qubit Arches'%(spec_scan_obj.TD_timestamps[0],spec_scans.TD_timestamps[-1]),
                                          'xlabel': r'Flux %d ($\phi_0$)' % idx_dac,
                                          'ylabel': r'Frequency (GHz)',
                                          'zlabel': 'Homodyne amplitude (mV)',
                                          'zrange': [0.5, 1],
                                          'plotsize': (8, 8),
                                          'cmap': 'YlGn_r'
                                          }
    spec_scan_obj.axs['arches'] = ax
    spec_scan_obj.plot()
    """


def get_peaks(dac_vector, f_vector, z_vector):
    plot_z = np.zeros(np.array(z_vector).shape)
    int_interval = np.arange(20)
    for i in range(len(dac_vector)):
        plot_z[i][:] = a_tools.smooth(
            z_vector[i][:], 11)-np.mean(z_vector[i][1:10])
    peaks = np.zeros(len(dac_vector))
    for i in range(len(dac_vector)):
        p_dict = a_tools.peak_finder(f_vector[i][1:-2], plot_z[i][1:-2])
        if (np.mean(plot_z[i][1:-2])-np.min(plot_z[i][1:-2])) > (np.max(plot_z[i][1:-2])-np.mean(plot_z[i][1:-2])):
            peaks[i] = p_dict['dip']
        else:
            peaks[i] = p_dict['peak']
    return peaks


def find_freq(freq, func, x0=0.2):
    # Need to adapt the function to the module design
    raise NotImplementedError
    """
    if freq is False:
        return 0.
    else:
        func1 = lambda d: func(d) - freq
        return fsolve(func1, x0)
    """