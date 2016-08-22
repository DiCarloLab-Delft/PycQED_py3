# for st_seqs
def Pulsed_spec_seq_RF_gated(RO_pars, spec_pulse_length=1e-6):
    seq_name = 'Pulsed_spec_with_RF_gated'
    seq = sequence.Sequence(seq_name)
    el_list = []
    R
    for i in range(2):
        el = st_elts.pulsed_spec_elt_with_RF_gated(
            i, station, RO_pars,spec_pulse_length=spec_pulse_length)
        el_list.append(el)
        seq.append_element(el, trigger_wait=False) # Ensures a continuously running sequence
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=False)

# for st_elts
def pulsed_spec_elt_with_RF_gated(i, station, RO_pars,
                                spec_pulse_length=1e-6,
                                marker_interval=4e-6):

    acq_marker = RO_pars['acq_marker']
    RO_marker = RO_pars['RO_marker']
    spec_marker = RO_pars['spec_marker']
    RO_trigger_delay = RO_pars['RO_trigger_delay']
    RO_pulse_length = RO_pars['RO_pulse_length']
    RO_trigger_delay = RO_pars['RO_trigger_delay']

    el = element.Element(name=('el %s' % i),
                         pulsar=station.pulsar)

    # This pulse ensures that the total length of the element is exactly 200us
    ref_length_pulse = el.add(pulse.SquarePulse(name='refpulse_0',
                              channel='ch2',
                              amplitude=0, length=200e-6,
                              start=0))
    # This pulse is used as a reference
    refpulse = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                      amplitude=0, length=100e-9,
                      start=10e-9))
    # a marker pulse
    acq_p = pulse.SquarePulse(name='CBox-pulse-trigger',
                            channel=acq_marker,
                            amplitude=1, length=15e-9)
    RO_p = pulse.SquarePulse(name='RO-pulse-marker',
                            channel=RO_marker,
                            amplitude=1, length=RO_pulse_length)
    spec_p = pulse.SquarePulse(name='RO-pulse-marker',
                            channel=spec_marker,
                            amplitude=1, length=RO_pulse_length)

    number_of_pulses = int(200*1e-6/marker_interval)

    for i in range(number_of_pulses):
        # spec pulse marker
        el.add(spec_p,
               name='Spec-marker-{}'.format(i),
               start=i*marker_interval,
               refpulse=refpulse, refpoint='start')
        # RO modulation tone
        el.add(RO_p,
               name='RO-marker-{}'.format(i),
               start=i*marker_interval,
               refpulse=refpulse, refpoint='start')
        for j in range(2):
            # RO acquisition marker
            el.add(acq_p,
                   name='acq-marker-{}{}'.format(i, j),
                   start=RO_trigger_delay,
                   refpulse='RO-marker-{}'.format(i), refpoint='start')
    return el





def fit_last_spec():
    def get_last_timestamp():
        splitted_str = (a_tools.get_folder()).split('\\')
        return splitted_str[-2]+'_'+splitted_str[-1][:6]

    pdict = {'y':'amp',
             'x':'sweep_points'}
    opt_dict = {'scan_label':''}
    nparams = ['y', 'x']
    spec = RA.quick_analysis(t_start=get_last_timestamp(),t_stop=get_last_timestamp(), options_dict=opt_dict,
                      params_dict_TD=pdict,numeric_params=nparams)

    peaks_dict = a_tools.peak_finder(x=spec.TD_dict['x'][0],y=a_tools.smooth(spec.TD_dict['y'][0], window_len=11))
    peaks_dict['peak']