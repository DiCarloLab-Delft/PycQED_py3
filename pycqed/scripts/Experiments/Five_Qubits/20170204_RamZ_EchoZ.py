qubits = [AncT, DataT]
for q in qubits:
    q.RO_acq_averages(512)
    MC.soft_avg(10)
    times = np.arange(0.5e-6, 80e-6, 1e-6)
    q.measure_T1(times)
    q.measure_echo(times, artificial_detuning=4/times[-1])
    q.measure_ramsey(times, artificial_detuning=4/times[-1])
    q.measure_allxy()


for q in qubits:
    q.RO_acq_averages(512)
    MC.soft_avg(10)
    times = np.arange(0.5e-6, 40e-6, .5e-6)
    q.measure_echo(times, artificial_detuning=4/times[-1])
    q.measure_ramsey(times, artificial_detuning=4/times[-1])
    q.measure_allxy()

################
# Adding the Z parameter
####################

AncT.add_operation('Z')
AncT.link_param_to_operation('Z', 'fluxing_operation_type', 'operation_type')
AncT.link_param_to_operation('Z', 'fluxing_channel', 'channel')

AncT.link_param_to_operation('Z', 'CZ_refpoint', 'refpoint')

AncT.add_pulse_parameter('Z', 'Z_amp', 'amplitude', .1)
AncT.add_pulse_parameter('Z', 'Z_length',
                         'length', 10e-9)
#
AncT.add_pulse_parameter('Z', 'Z_pulse_type', 'pulse_type',
                         initial_value='SquarePulse',
                         vals=vals.Strings())
AncT.add_pulse_parameter('Z', 'Z_pulse_delay',
                         'pulse_delay', 0)

DataT.add_operation('Z')
DataT.link_param_to_operation('Z', 'fluxing_operation_type', 'operation_type')
DataT.link_param_to_operation('Z', 'fluxing_channel', 'channel')

DataT.link_param_to_operation('Z', 'SWAP_refpoint', 'refpoint')

DataT.add_pulse_parameter('Z', 'Z_amp', 'amplitude', .1)
DataT.add_pulse_parameter('Z', 'Z_length',
                          'length', 10e-9)
#
DataT.add_pulse_parameter('Z', 'Z_pulse_type', 'pulse_type',
                          initial_value='SquarePulse',
                          vals=vals.Strings())
DataT.add_pulse_parameter('Z', 'Z_pulse_delay',
                          'pulse_delay', 0)


reload_mod_stuff()
int_avg_det = det.UHFQC_integrated_average_detector(
    UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
    channels=[DataT.RO_acq_weight_function_I(),
              AncT.RO_acq_weight_function_I()],
    nr_averages=AncT.RO_acq_averages(),
    integration_length=AncT.RO_acq_integration_length(),
    cross_talk_suppression=True)

operation_dict = S5.get_operation_dict()

ram_Z_sweep = awg_swf.awg_seq_swf(fsqs.Ram_Z_seq,
                                  awg_seq_func_kwargs={'operation_dict': operation_dict, 'q0': 'AncT',
                                                       'operation_name': 'Z',
                                                       'distortion_dict': AncT.dist_dict()},
                                  parameter_name='times')


MC.set_sweep_function(ram_Z_sweep)
MC.set_sweep_points(np.arange(0, 20e-6, .5e-6))
MC.set_detector_function(int_avg_det)
MC.run('Ram_Z_AncT')
ma.MeasurementAnalysis()


###########
# Echo Z control experiment
###############
reload_mod_stuff()
times = [10e-6] * 40

for Z_amp in np.linspace(0.00, 0.05, 11):

    AncT.Z_amp(Z_amp)
    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
        channels=[DataT.RO_acq_weight_function_I(),
                  AncT.RO_acq_weight_function_I()],
        nr_averages=AncT.RO_acq_averages(),
        integration_length=AncT.RO_acq_integration_length(),
        cross_talk_suppression=True)

    operation_dict = S5.get_operation_dict()

    Echo_Z_seq = awg_swf.awg_seq_swf(fsqs.Echo_Z_seq,
                                     awg_seq_func_kwargs={'operation_dict': operation_dict, 'q0': 'AncT',
                                                          'operation_name': 'Z',
                                                          'artificial_detuning': 4/times[-1],
                                                          'distortion_dict': AncT.dist_dict()},
                                     parameter_name='times')

    MC.set_sweep_function(Echo_Z_seq)
    MC.set_sweep_points(times)
    MC.set_detector_function(int_avg_det)
    MC.run('Echo_Z_AncT_{}'.format(AncT.Z_amp()))
    ma.MeasurementAnalysis()

###################
# Echo Z
for Z_amp in np.linspace(0.00, 0.05, 11):

    AncT.Z_amp(Z_amp)
    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
        channels=[DataT.RO_acq_weight_function_I(),
                  AncT.RO_acq_weight_function_I()],
        nr_averages=AncT.RO_acq_averages(),
        integration_length=AncT.RO_acq_integration_length(),
        cross_talk_suppression=True)

    operation_dict = S5.get_operation_dict()
    recovery_phases = np.linspace(0, 720, 41)
    Echo_Z_seq = awg_swf.awg_seq_swf(fsqs.Echo_Z_seq,
                                     awg_seq_func_kwargs={'operation_dict': operation_dict, 'q0': 'AncT',
                                                          'operation_name': 'Z',
                                                          'times': [10e-6],
                                                          'artificial_detuning': None,
                                                          'distortion_dict': AncT.dist_dict()},
                                     parameter_name='recovery_phases', unit='degree')

    MC.set_sweep_function(Echo_Z_seq)
    MC.set_sweep_points(recovery_phases)
    MC.set_detector_function(int_avg_det)
    MC.run('Echo_Z_AncT_{}'.format(AncT.Z_amp()))
    ma.MeasurementAnalysis()


##########################################
# Echo Z variants
##########################################
reload_mod_stuff()
times = [5e-6]
recovery_phases = np.linspace(0, 720, 41)

Z_signs_lst = [[+1, +1]]*2 + [[+1, -1]]*2
echo_MW_pulses = [False, True, False, True]
Flux_amps = np.linspace(0.001, 1, 11)

AncT.Z_amp(0.1)

int_avg_det = det.UHFQC_integrated_average_detector(
    UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
    channels=[AncT.RO_acq_weight_function_I()],
    nr_averages=AncT.RO_acq_averages(),
    integration_length=AncT.RO_acq_integration_length(),
    cross_talk_suppression=True)
Flux_amp = .5

seq_list = [None]*10
elts_list = [None]*10

for i, (Z_signs, echo_MW_pulse) in enumerate(zip(Z_signs_lst, echo_MW_pulses)):
    print(i, Flux_amp, Z_signs, echo_MW_pulse)
    AWG.ch3_amp(Flux_amp)

    operation_dict = S5.get_operation_dict()
    awg_seq_func_kwargs = {'operation_dict': operation_dict, 'q0': 'AncT',
                           'operation_name': 'Z',
                           'times': times,
                           'Z_signs': Z_signs,
                           'echo_MW_pulse': echo_MW_pulse,
                           'artificial_detuning': None,
                           'distortion_dict': AncT.dist_dict()}

    seq, elts = fsqs.Echo_Z_seq(recovery_phases=recovery_phases,
                                **awg_seq_func_kwargs, upload=False)
    seq_list[i] = seq
    elts_list[i] = elts

# for Flux_amp in [0.02, 0.1, .5]:

#     for i, (Z_signs, echo_MW_pulse) in enumerate(zip(Z_signs_lst, echo_MW_pulses)):
#         Echo_Z_seq = awg_swf.awg_seq_swf(
#             fsqs.Echo_Z_seq,
#             awg_seq_func_kwargs=awg_seq_func_kwargs,
#             upload=False,
#             parameter_name='recovery_phases', unit='degree')

#         old_vals = AWG.get('ch3_amp')
#         AWG.set('ch3_amp', 2)
#         station.components['AWG'].stop()
#         station.pulsar.program_awg(seq_list[i], *elts_list[i],
#                                    verbose=False)
#         AWG.set('ch3_amp', Flux_amp)


#         MC.set_sweep_function(Echo_Z_seq)
#         MC.set_sweep_points(recovery_phases)
#         MC.set_detector_function(int_avg_det)
#         MC.run('Echo_Z_AncT_tau{}_amp{}_signs{}_echo{}'.format(
#             times[0], AWG.ch3_amp(), Z_signs, echo_MW_pulse))
#         a=oscillation_analysis()
#         osc_amps[i] = a.osc_amp_0

########################
# A
########################

class oscillation_analysis(ma.Rabi_Analysis):

    def __init__(self, label='', **kw):
        super().__init__(label=label, **kw)

    def run_default_analysis(self, close_file=True, **kw):

        self.get_naming_and_values()
        cal_0I = self.measured_values[0][-4]
        cal_1I = self.measured_values[0][-1]

        self.measured_values[0][:] = (
            self.measured_values[0] - cal_0I)/(cal_1I-cal_0I)

        self.measured_values = self.measured_values
        self.sweep_points = self.sweep_points
        self.fit_data(**kw)
        self.make_figures(**kw)

        self.osc_amp_0 = self.fit_res.best_values['amplitude']

        if close_file:
            self.data_file.close()

    def fit_data(self, **kw):
        self.add_analysis_datagroup_to_file()
        self.fit_res = ['']
        # It would be best to do 1 fit to both datasets but since it is
        # easier to do just one fit we stick to that.

        num_points = len(self.sweep_points)-4
        xvals = self.sweep_points[:-4]
        yvals = self.measured_values[0][:-4]
        #self.cost_func_val = 2-(max(id_dat-ex_dat) + max(ex_dat-id_dat))
        # we can calculate the cost function disreclty from the fit-parameters

        model = lmfit.Model(ma.fit_mods.CosFunc)
        params = simple_cos_guess(model, data=yvals, t=xvals)
        self.fit_res = model.fit(data=yvals, t=xvals, params=params)
        self.save_fitted_parameters(fit_res=self.fit_res,
                                    var_name='id_fit')

        self.x_fine = np.linspace(min(self.sweep_points[:-4]), max(self.sweep_points[:-4]),
                             1000)
        self.fine_fit_0 = self.fit_res.model.func(
                self.x_fine, **self.fit_res.best_values)


    def make_figures(self, **kw):
        show_guess = kw.pop('show_guess', False)
        self.fig, self.ax = ma.plt.subplots()
        plot_title = kw.pop('plot_title', ma.textwrap.fill(
                            self.timestamp_string + '_' +
                            self.measurementstring, 40))
        self.plot_results_vs_sweepparam(x=self.sweep_points,
                                        y=self.measured_values[0],
                                        fig=self.fig, ax=self.ax,
                                        xlabel=self.xlabel,
                                        ylabel=self.ylabels[0],
                                        save=False,
                                        plot_title=plot_title, marker='--o')

        fine_fit = self.fit_res.model.func(
            self.x_fine, **self.fit_res.best_values)
        self.ax.plot(self.x_fine, fine_fit, label='fit')
        if show_guess:
            fine_fit = self.fit_res.model.func(
                self.x_fine, **self.fit_res.init_values)
            self.ax.plot(self.x_fine, fine_fit, label='guess')
            self.ax.legend(loc='best')
        self.save_fig(self.fig, fig_tight=False, **kw)


def simple_cos(t, amplitude, frequency, phase, offset):
    return amplitude*np.cos(frequency*t+phase)+offset


def simple_cos_guess(model, data, t):
    '''
    Guess for a cosine fit using FFT, only works for evenly spaced points
    '''
    amp_guess = abs(max(data)-min(data))/2  # amp is positive by convention
    offs_guess = np.mean(data)

    freq_guess = 1/360
    ph_guess = -2*np.pi*t[data == max(data)]/360

    model.set_param_hint('period', expr='1/frequency')
    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp
    return params



class echo_Z_cost_det(det.Soft_Detector):
    #function ment to only tune-up CPhase pulse
    def __init__(self, seq_list, elts_list, MC):
        self.name = 'CPhase_cost_func_det'
        self.detector_control = 'soft'
        self.MC = MC
        self.seq_list = seq_list
        self.elts_list = elts_list
        self.value_names = ['amp +Z, I, +Z',
                            'amp +Z, pi, -Z',
                            'amp +Z, I, +Z',
                            'amp +Z, pi, -Z']
        self.value_units = ['a.u.']*4


    def measure_echo_Z_osc_amps(self):

        Z_signs_lst = [[+1, +1]]*2 + [[+1, -1]]*2
        echo_MW_pulses = [False, True, False, True]
        osc_amps = np.zeros(len(echo_MW_pulses))
        for i, (Z_signs, echo_MW_pulse) in enumerate(zip(Z_signs_lst,
                                                         echo_MW_pulses)):
            Echo_Z_seq = awg_swf.awg_seq_swf(
                fsqs.Echo_Z_seq,
                awg_seq_func_kwargs=awg_seq_func_kwargs,
                upload=False,
                parameter_name='recovery_phases', unit='degree')

            old_val = AWG.get('ch3_amp')
            AWG.set('ch3_amp', 2)
            station.components['AWG'].stop()
            station.pulsar.program_awg(self.seq_list[i], *self.elts_list[i],
                                       verbose=False)
            AWG.set('ch3_amp', old_val)

            self.MC.set_sweep_function(Echo_Z_seq)
            self.MC.set_sweep_points(recovery_phases)
            self.MC.set_detector_function(int_avg_det)
            self.MC.run('Echo_Z_AncT_tau{}_amp{}_signs{}_echo{}'.format(
                times[0], AWG.ch3_amp(), Z_signs, echo_MW_pulse))
            a=oscillation_analysis()
            osc_amps[i] = a.osc_amp_0
        return osc_amps

    def acquire_data_point(self, **kw):
        return self.measure_echo_Z_osc_amps()

d = echo_Z_cost_det(seq_list, elts_list, MC=nested_MC)



MC.set_sweep_function(AWG.ch3_amp)
MC.set_sweep_points(np.arange(0.02, 1.5, 0.08))
MC.set_detector_function(d)
MC.run('Echo_Z_variants')
ma.MeasurementAnalysis()