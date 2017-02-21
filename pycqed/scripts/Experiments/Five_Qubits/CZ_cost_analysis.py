import numpy as np
import pycqed.analysis.measurement_analysis as ma


class CPhase_2Q_amp_cost_analysis(ma.Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True, **kw):

        self.get_naming_and_values()

        cal_0I = np.mean([self.measured_values[0][-4],
                          self.measured_values[0][-3]])
        cal_1I = np.mean([self.measured_values[0][-2],
                          self.measured_values[0][-1]])

        cal_0Q = np.mean([self.measured_values[1][-4],
                          self.measured_values[1][-2]])
        cal_1Q = np.mean([self.measured_values[1][-3],
                          self.measured_values[1][-1]])

        self.measured_values[0][:] = (
            self.measured_values[0] - cal_0I)/(cal_1I-cal_0I)
        self.measured_values[1][:] = (
            self.measured_values[1] - cal_0Q)/(cal_1Q-cal_0Q)
        # self.measured_values = self.measured_values
        self.sweep_points = self.sweep_points
        self.calculate_cost_func(**kw)
        self.make_figures(**kw)

        if close_file:
            self.data_file.close()

    def calculate_cost_func(self, **kw):
        num_points = len(self.sweep_points)-4

        id_dat_swp = self.measured_values[1][:num_points//2]
        ex_dat_swp = self.measured_values[1][num_points//2:-4]

        id_dat_cp = self.measured_values[0][:num_points//2]
        ex_dat_cp = self.measured_values[0][num_points//2:-4]

        maximum_difference = max((id_dat_cp-ex_dat_cp))
        # I think the labels are wrong in excited and identity but the value
        # we get is correct
        missing_swap_pop = np.mean(ex_dat_swp- id_dat_swp)
        self.cost_func_val = maximum_difference, missing_swap_pop

    def make_figures(self, **kw):
        self.fig, self.axs = ma.plt.subplots(2, 1, figsize=(5, 6))
        self.ylabels = ['q_CP', 'q_S']
        for i in [0, 1]:
            if i == 0:
                plot_title = kw.pop('plot_title', ma.textwrap.fill(
                                    self.timestamp_string + '_' +
                                    self.measurementstring, 40))
            else:
                plot_title = ''
            self.axs[i].ticklabel_format(useOffset=False)
            self.plot_results_vs_sweepparam(x=self.sweep_points,
                                            y=self.measured_values[i],
                                            fig=self.fig, ax=self.axs[i],
                                            xlabel=self.xlabel,
                                            ylabel=self.ylabels[i],
                                            save=False,
                                            plot_title=plot_title, marker='--o')

        self.save_fig(self.fig, fig_tight=False, **kw)

