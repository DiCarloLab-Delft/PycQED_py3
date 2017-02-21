######################################
# Flux track "Cross"
######################################

amps = np.arange(1.05, 1.25, 0.01)
fluxes = np.arange(-0.03, -0.01, 0.002)

q0 = S5.qubits()['DataT']

operation_dict = S5.get_operation_dict()
AWG = q0.AWG

FluxTrack_det = cdet.FluxTrack(qubit=q0,
                               device=S5,
                               MC=nested_MC,
                               AWG=AWG, cal_points=True)

MC.set_sweep_function(AWG.ch4_amp)
MC.set_sweep_points(amps)

MC.set_sweep_function_2D(Flux_Control.flux2)
MC.set_sweep_points_2D(fluxes)
MC.set_detector_function(FluxTrack_det)
# MC.run('Flux_Track_linescan')
MC.run('FluxTrack_%s' % q0.name, mode='2D')
ma.TwoD_Analysis(auto=True, label='FluxTrack_DataT')


##############################
# Two D resonator scan
##############################
freqs = np.arange(DataT.f_res()-5e6, DataT.f_res()+5e6, 250e3)
fluxes = np.arange(-0.03, -0.01, 0.005)
DataT.prepare_for_continuous_wave()
MC.set_sweep_function(pw.wrap_par_to_swf(
                              DataT.heterodyne_instr.frequency))
MC.set_sweep_points(freqs)
MC.set_detector_function(
    det.Heterodyne_probe(DataT.heterodyne_instr,
                         trigger_separation=4e-6))
MC.set_sweep_function_2D(Flux_Control.flux2)
MC.set_sweep_points_2D(fluxes)
MC.run(name='Resonator_scan_2D'+DataT.msmt_suffix, mode='2D')


##############################
# Two D pulsed spec scan
##############################
freqs = np.arange(DataT.f_qubit()-20e6, DataT.f_qubit()+20e6, .5e6)
DataT.prepare_for_pulsed_spec()
DataT.heterodyne_instr._disable_auto_seq_loading = True

DataT.cw_source.pulsemod_state.set('On')
DataT.cw_source.power.set(DataT.spec_pow_pulsed.get())
DataT.cw_source.on()

if MC is None:
    MC = DataT.MC

spec_pars, RO_pars = DataT.get_spec_pars()
# Upload the AWG sequence
RO_pars['f_RO_mod']= DataT.f_RO_mod()
sq.Pulsed_spec_seq(spec_pars, RO_pars)

DataT.AWG.start()


MC.set_sweep_function(DataT.cw_source.frequency)
MC.set_sweep_points(freqs)
MC.set_sweep_function_2D(Flux_Control.flux2)
MC.set_sweep_points_2D(fluxes)
MC.set_detector_function(
    det.Heterodyne_probe(DataT.heterodyne_instr))
MC.run(name='pulsed-spec'+DataT.msmt_suffix, mode='2D')