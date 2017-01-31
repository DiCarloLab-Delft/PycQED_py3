def set_AWG_limits(station, lim):
    AWG = station.AWG
    AWG.ch1_amp(lim)
    AWG.ch2_amp(lim)
    AWG.ch3_amp(lim)
    AWG.ch4_amp(lim)

def config_cw(qubit, cw_dict):
    """
    configures the parameters of a qubit RO pulse from a dictionary
    """
    if cw_dict['I_channel']:
        qubit.RO_I_channel(cw_dict['I_channel'])
    if cw_dict['Q_channel']:
        qubit.RO_Q_channel(cw_dict['Q_channel'])
    if cw_dict['RO_pulse_marker_channel']:
        qubit.RO_pulse_marker_channel(cw_dict['RO_pulse_marker_channel'])
    if cw_dict['acq_marker_channel']:
        qubit.RO_acq_marker_channel(cw_dict['acq_marker_channel'])
    if not(cw_dict['acq_marker_delay'] is None):
        qubit.RO_acq_marker_delay(cw_dict['acq_marker_delay'])
    if cw_dict['amplitude']:
        qubit.RO_amp(cw_dict['amplitude'])
    if cw_dict['length']:
        qubit.RO_pulse_length(cw_dict['length'])
    if not(cw_dict['pulse_delay'] is None):
        qubit.RO_pulse_delay(cw_dict['pulse_delay'])
    if cw_dict['pulse_type']:
        qubit.RO_pulse_type(cw_dict['pulse_type'])
    if cw_dict['mod_frequency']:
        qubit.f_RO_mod(cw_dict['mod_frequency'])
        # qubit.RO = cw_dict['phase']
    if cw_dict['RO_power_cw']:
        qubit.RO_power_cw(cw_dict['RO_power_cw'])
    if cw_dict['spec_power']:
        qubit.spec_pow(cw_dict['spec_power'])

def copy_settings_cw(qubit_src,qubit_dst):
    RO_dict = qubit_src.get_pulse_pars()[1]
    RO_dict['RO_power_cw'] = qubit_src.RO_power_cw()
    RO_dict['spec_power'] = qubit_src.spec_pow()
    config_cw(qubit_dst,RO_dict)


def config_ps(qubit, ps_ro_dict, ps_spec_dict):
    """
    configures the parameters of a qubit RO pulse from a dictionary
    """
    if ps_ro_dict['I_channel']:
        qubit.RO_I_channel(ps_ro_dict['I_channel'])
    if ps_ro_dict['Q_channel']:
        qubit.RO_Q_channel(ps_ro_dict['Q_channel'])
    if ps_ro_dict['RO_pulse_marker_channel']:
        qubit.RO_pulse_marker_channel(ps_ro_dict['RO_pulse_marker_channel'])
    if ps_ro_dict['acq_marker_channel']:
        qubit.RO_acq_marker_channel(ps_ro_dict['acq_marker_channel'])
    if not(ps_ro_dict['acq_marker_delay'] is None):
        qubit.RO_acq_marker_delay(ps_ro_dict['acq_marker_delay'])
    if ps_ro_dict['amplitude']:
        qubit.RO_amp(ps_ro_dict['amplitude'])
    if ps_ro_dict['length']:
        qubit.RO_pulse_length(ps_ro_dict['length'])
    if not(ps_ro_dict['pulse_delay'] is None):
        qubit.RO_pulse_delay(ps_ro_dict['pulse_delay'])
    if ps_ro_dict['pulse_type']:
        qubit.RO_pulse_type(ps_ro_dict['pulse_type'])
    if ps_ro_dict['mod_frequency']:
        qubit.f_RO_mod(ps_ro_dict['mod_frequency'])
        # qubit.RO = ps_ro_dict['phase']
    if ps_ro_dict['RO_power_cw']:
        qubit.RO_power_cw(ps_ro_dict['RO_power_cw'])
    if ps_ro_dict['RO_pulse_power']:
        qubit.RO_pulse_power(ps_ro_dict['RO_pulse_power'])
    if ps_spec_dict['channel']:
        qubit.spec_pulse_marker_channel(ps_spec_dict['channel'])
    if ps_spec_dict['length']:
        qubit.spec_pulse_length(ps_spec_dict['length'])
    if ps_spec_dict['spec_pulse_depletion_time']:
        qubit.spec_pulse_depletion_time(ps_spec_dict['spec_pulse_depletion_time'])
    if ps_spec_dict['spec_power']:
        qubit.spec_pow_pulsed(ps_spec_dict['spec_power'])

def copy_settings_ps(qubit_src, qubit_dst):
    spec_dict, RO_dict = qubit_src.get_spec_pars()
    RO_dict['RO_power_cw'] = qubit_src.RO_power_cw()
    RO_dict['RO_pulse_power'] = qubit_src.RO_pulse_power()
    spec_dict['spec_power'] = qubit_src.spec_pow_pulsed()
    spec_dict['spec_pulse_depletion_time'] = qubit_src.spec_pulse_depletion_time()
    config_ps(qubit_dst, RO_dict, spec_dict)



def config_td(qubit, ro_dict, pulse_dict):
    """
    configures the parameters of a qubit RO pulse from a dictionary
    """
    if ro_dict['I_channel']:
        qubit.RO_I_channel(ro_dict['I_channel'])
    if ro_dict['Q_channel']:
        qubit.RO_Q_channel(ro_dict['Q_channel'])
    if ro_dict['RO_pulse_marker_channel']:
        qubit.RO_pulse_marker_channel(ro_dict['RO_pulse_marker_channel'])
    if ro_dict['acq_marker_channel']:
        qubit.RO_acq_marker_channel(ro_dict['acq_marker_channel'])
    if not(ro_dict['acq_marker_delay'] is None):
        qubit.RO_acq_marker_delay(ro_dict['acq_marker_delay'])
    if ro_dict['amplitude']:
        qubit.RO_amp(ro_dict['amplitude'])
    if ro_dict['length']:
        qubit.RO_pulse_length(ro_dict['length'])
    if not(ro_dict['pulse_delay'] is None):
        qubit.RO_pulse_delay(ro_dict['pulse_delay'])
    if ro_dict['pulse_type']:
        qubit.RO_pulse_type(ro_dict['pulse_type'])
    if ro_dict['mod_frequency']:
        qubit.f_RO_mod(ro_dict['mod_frequency'])
    if ro_dict['RO_pulse_power']:
        qubit.RO_pulse_power(ro_dict['RO_pulse_power'])
        # qubit.RO = ro_dict['phase']
    if pulse_dict['I_channel']:
        qubit.pulse_I_channel(pulse_dict['I_channel'])
    if pulse_dict['Q_channel']:
        qubit.pulse_Q_channel(pulse_dict['Q_channel'])
    if pulse_dict['alpha']:
        qubit.alpha(pulse_dict['alpha'])
    if pulse_dict['amp90_scale']:
        qubit.amp90_scale(pulse_dict['amp90_scale'])
    if pulse_dict['amplitude']:
        qubit.amp180(pulse_dict['amplitude'])
    if pulse_dict['mod_frequency']:
        qubit.f_pulse_mod(pulse_dict['mod_frequency'])
    if pulse_dict['motzoi']:
        qubit.motzoi(pulse_dict['motzoi'])
    if pulse_dict['pulse_delay']:
        qubit.pulse_delay(pulse_dict['pulse_delay'])
    if pulse_dict['phi_skew']:
        qubit.phi_skew(pulse_dict['phi_skew'])
    if pulse_dict['sigma']:
        qubit.gauss_sigma(pulse_dict['sigma'])
    if pulse_dict['pulse_power']:
        qubit.td_source_pow(pulse_dict['pulse_power'])

def copy_settings_td(qubit_src,qubit_dst):
    pulse_dict, RO_dict = qubit_src.get_pulse_pars()
    pulse_dict['pulse_power'] = qubit_src.td_source_pow()
    RO_dict['RO_pulse_power'] = qubit_src.RO_pulse_power()
    config_td(qubit_dst, RO_dict, pulse_dict)