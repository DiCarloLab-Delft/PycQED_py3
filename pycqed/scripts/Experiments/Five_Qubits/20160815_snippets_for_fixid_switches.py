def prepare_for_continuous_wave(qubit):
    # Heterodyne tone configuration
    qubit.prepare_for_continuous_wave()
    qubit.heterodyne_instr._disable_auto_seq_loading = False
    qubit.heterodyne_instr.RF.on()
    qubit.heterodyne_instr.LO.on()
    if hasattr(qubit.heterodyne_instr, 'mod_amp'):
        qubit.heterodyne_instr.set('mod_amp', qubit.mod_amp_cw.get())
    else:
        qubit.heterodyne_instr.RF_power(qubit.RO_power_cw())
    qubit.heterodyne_instr.set('IF', qubit.f_RO_mod.get())
    qubit.heterodyne_instr.frequency.set(qubit.f_RO.get())
    qubit.heterodyne_instr.RF.power(qubit.RO_power_cw())
    qubit.heterodyne_instr.RF_power(qubit.RO_power_cw())

    # Turning of TD source
    qubit.td_source.off()

    # Updating Spec source
    qubit.cw_source.power(qubit.spec_pow())
    qubit.cw_source.frequency(qubit.f_qubit())
    if hasattr(qubit.cw_source, 'pulsemod_state'):
        qubit.cw_source.pulsemod_state('off')
    qubit.rf_RO_source.pulsemod_state('Off')

def prepare_for_two_tone_spec(qubit):
    prepare_for_continuous_wave(qubit)
    qubit.cw_source.on()

def find_frequency(qubit,freqs):
    prepare_for_continuous_wave(qubit)
    qubit.cw_source.on()
    qubit.measure_spectroscopy(freqs=freqs)
    qubit.cw_source.off()

def prepare_for_TD(qubit):
    qubit.rf_RO_source.pulsemod_state('on')
    qubit.prepare_for_timedomain()
    qubit.rf_RO_source.power(AncB.RO_pulse_power())
    qubit.rf_RO_source.on()

Spec_source.on()
prepare_for_PS(AncT)
HS.frequency(AncT.f_RO())
CBox.nr_averages(8192*2)
AncT.measure_pulsed_spectroscopy(freqs=np.arange(5.922e9,5.94e9,0.2e6))